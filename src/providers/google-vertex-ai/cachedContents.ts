import { Params } from '../../types/requestBody';

/**
 * Vertex AI explicit context caching (cachedContents).
 *
 * Gemini implicit caching is best-effort with no TTL/SLA guarantee, so a caller
 * that resends a large static prefix rarely gets a cache hit. This module lets a
 * caller opt into *explicit* caching using the same Anthropic-style
 * `cache_control: {type:"ephemeral"}` markers that OpenRouter/LiteLLM expose:
 *
 *   1. detect the last cache_control marker in the incoming messages,
 *   2. ensure a Vertex `cachedContents` resource exists for the marked prefix
 *      (list-and-match by a deterministic displayName, else create with a TTL),
 *   3. rewrite the outgoing generateContent body to reference that resource via
 *      `cachedContent` and carry only the non-cached suffix in `contents`.
 *
 * Any failure in the cache path (below-min-token 400, permissions, network)
 * leaves the transformed body untouched so the request proceeds uncached.
 */

const CACHE_TTL = '3600s';

type VertexPart = { text?: string; [k: string]: any };
type VertexContent = { role: string; parts: VertexPart[] };
type TransformedBody = {
  contents?: VertexContent[];
  systemInstruction?: { role: string; parts: VertexPart[] };
  tools?: any;
  toolConfig?: any;
  cachedContent?: string;
  [k: string]: any;
};

/**
 * Cheap scan of the incoming (gateway-format) request for any cache_control
 * marker on a content block or a whole message.
 */
export function hasCacheControl(params: Params | undefined): boolean {
  const messages = params?.messages;
  if (!Array.isArray(messages)) return false;
  for (const msg of messages) {
    if ((msg as any)?.cache_control) return true;
    const content = (msg as any)?.content;
    if (Array.isArray(content)) {
      for (const item of content) {
        if (item?.cache_control) return true;
      }
    }
  }
  return false;
}

/**
 * Find the index of the last message that carries (or contains) a cache_control
 * marker. Markers are honored at message granularity (the whole marked message
 * and everything before it becomes the cached prefix), matching OpenRouter's
 * "only the last breakpoint" behavior. Returns -1 if none.
 */
export function lastCacheControlMessageIndex(params: Params): number {
  const messages = params.messages;
  if (!Array.isArray(messages)) return -1;
  let idx = -1;
  messages.forEach((msg: any, i: number) => {
    if (msg?.cache_control) idx = i;
    else if (Array.isArray(msg?.content)) {
      if (msg.content.some((item: any) => item?.cache_control)) idx = i;
    }
  });
  return idx;
}

/**
 * Number of non-system messages up to and including `markerIndex`. The vertex
 * transform drops system messages from `contents` (they go to
 * systemInstruction), so this maps a gateway message index onto a `contents`
 * offset. Consecutive same-role merging (chatComplete.ts) can fold several
 * gateway messages into one contents entry; we clamp to the available length.
 */
function contentsPrefixLength(params: Params, markerIndex: number): number {
  const messages = params.messages as any[];
  let n = 0;
  for (let i = 0; i <= markerIndex && i < messages.length; i++) {
    if (messages[i]?.role !== 'system') n++;
  }
  return n;
}

/**
 * Split an already-transformed vertex body at the marker into a cacheable
 * prefix (systemInstruction + leading contents) and the remaining suffix.
 */
export function splitTransformedBody(
  params: Params,
  body: TransformedBody
): { prefix: TransformedBody; suffix: VertexContent[] } | null {
  const markerIndex = lastCacheControlMessageIndex(params);
  if (markerIndex < 0) return null;

  const allContents = Array.isArray(body.contents) ? body.contents : [];
  const prefixLen = Math.min(
    contentsPrefixLength(params, markerIndex),
    allContents.length
  );

  const prefixContents = allContents.slice(0, prefixLen);
  const suffix = allContents.slice(prefixLen);

  const prefix: TransformedBody = {};
  if (body.systemInstruction) prefix.systemInstruction = body.systemInstruction;
  if (prefixContents.length) prefix.contents = prefixContents;
  if (body.tools) prefix.tools = body.tools;
  if (body.toolConfig) prefix.toolConfig = body.toolConfig;

  // Nothing worth caching (no system prompt and no leading turns).
  if (!prefix.systemInstruction && !prefix.contents?.length) return null;

  return { prefix, suffix };
}

async function sha256Hex(input: string): Promise<string> {
  const data = new TextEncoder().encode(input);
  const digest = await crypto.subtle.digest('SHA-256', data);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

export async function deriveCacheKey(
  model: string,
  prefix: TransformedBody
): Promise<string> {
  const hex = await sha256Hex(JSON.stringify({ model, prefix }));
  return `portkey:${hex}`;
}

interface EnsureArgs {
  baseURL: string;
  projectId: string;
  region: string;
  model: string;
  prefix: TransformedBody;
  authToken: string;
}

/**
 * Return the resource name of a cachedContents entry for this prefix, creating
 * it if one with the matching displayName does not already exist. Throws on any
 * non-2xx so the caller can fall back to an uncached request.
 */
export async function ensureCachedContent(args: EnsureArgs): Promise<string> {
  const { baseURL, projectId, region, model, prefix, authToken } = args;
  const displayName = await deriveCacheKey(model, prefix);
  const listBase = `${baseURL}/v1/projects/${projectId}/locations/${region}/cachedContents`;
  const authHeaders = {
    Authorization: `Bearer ${authToken}`,
    'Content-Type': 'application/json',
  };

  // List and match by displayName (no server-side filter param exists).
  let pageToken = '';
  for (let page = 0; page < 5; page++) {
    const url = `${listBase}?pageSize=1000${pageToken ? `&pageToken=${pageToken}` : ''}`;
    const res = await fetch(url, { method: 'GET', headers: authHeaders });
    if (!res.ok) throw new Error(`cachedContents list failed: ${res.status}`);
    const json: any = await res.json();
    const match = (json.cachedContents ?? []).find(
      (cc: any) => cc.displayName === displayName
    );
    if (match?.name) return match.name;
    if (!json.nextPageToken) break;
    pageToken = json.nextPageToken;
  }

  // Not found: create it.
  const createBody: Record<string, any> = {
    model: `projects/${projectId}/locations/${region}/publishers/google/models/${model}`,
    displayName,
    ttl: CACHE_TTL,
  };
  if (prefix.systemInstruction)
    createBody.systemInstruction = prefix.systemInstruction;
  if (prefix.contents) createBody.contents = prefix.contents;
  if (prefix.tools) createBody.tools = prefix.tools;
  if (prefix.toolConfig) createBody.toolConfig = prefix.toolConfig;

  const res = await fetch(listBase, {
    method: 'POST',
    headers: authHeaders,
    body: JSON.stringify(createBody),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(
      `cachedContents create failed: ${res.status} ${text.slice(0, 200)}`
    );
  }
  const created: any = await res.json();
  if (!created?.name) throw new Error('cachedContents create returned no name');
  return created.name;
}

/**
 * Rewrite the transformed body in place to use a cached resource. Vertex 400s
 * if systemInstruction/tools/toolConfig coexist with cachedContent, so those
 * (now absorbed by the cache) are removed.
 */
export function applyCachedContent(
  body: TransformedBody,
  resourceName: string,
  suffix: VertexContent[]
): void {
  body.cachedContent = resourceName;
  body.contents = suffix;
  delete body.systemInstruction;
  delete body.tools;
  delete body.toolConfig;
}
