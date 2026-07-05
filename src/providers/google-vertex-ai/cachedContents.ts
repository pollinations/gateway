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
 *
 * Supported shapes (everything else falls back to uncached): the vertex
 * transform merges consecutive same-role messages and drops system messages
 * into `systemInstruction`, so a cache_control marker only maps cleanly onto a
 * `contents` boundary when
 *   - it sits on the system message, or
 *   - it sits on a message that is the last of its role-run (i.e. the marked
 *     message is not merged with a following same-role message — typically it
 *     is followed by a message of a different role), and
 *   - within that message the marker is at message level or on the message's
 *     last content block.
 * Anything else — consecutive same-role messages that the transform merges, or
 * an Anthropic-style block-level breakpoint on a non-final block with trailing
 * (likely dynamic) blocks in the same message — would misalign the split or
 * fold dynamic content into the cache key, so `splitTransformedBody` returns
 * null and the request proceeds uncached.
 *
 * Billing note (create-vs-match): the request that CREATES a cachedContents
 * resource still has the whole prefix counted by Vertex under
 * `usageMetadata.cachedContentTokenCount`, even though Google bills those
 * tokens to us at the standard input rate on the create. Downstream billing
 * derives non-cached prompt tokens as `prompt_tokens - cached_tokens` and
 * discounts the cached portion, which would under-bill a create. To keep the
 * create billing at the standard input rate, `ensureCachedContent` reports
 * whether it created the resource (`created: true`) or matched an existing one
 * (`created: false`); the api.ts caching hook records that on the request's
 * messages array (see `markCacheCreated`), and the response transforms zero
 * `prompt_tokens_details.cached_tokens` for creates. Matched requests keep the
 * reported cached_tokens (and its discount).
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
 * Explicit caching is only applied when the request opts in via cache_control
 * AND involves no tools. Tool messages map to the `function` role, whose
 * consecutive-entry acceptance in `contents` is unverified, and tools interact
 * awkwardly with cachedContent (systemInstruction/tools/toolConfig cannot
 * coexist with cachedContent). Excluding tools keeps the split exact and the
 * boundary well-defined.
 */
export function isExplicitCachingEligible(params: Params | undefined): boolean {
  if (!hasCacheControl(params)) return false;
  const tools = (params as any)?.tools;
  if (Array.isArray(tools) && tools.length > 0) return false;
  if ((params as any)?.tool_choice) return false;
  const messages = params?.messages;
  if (Array.isArray(messages)) {
    for (const msg of messages) {
      if ((msg as any)?.role === 'tool') return false;
      if ((msg as any)?.tool_calls) return false;
    }
  }
  return true;
}

/**
 * Channel for the create-vs-match billing flag between the api.ts caching hook
 * (which knows whether a cache was created this request) and the response
 * transforms (which zero cached_tokens on creates). Keyed on the request's
 * `messages` array: the params object is re-spread on every RequestContext
 * access so its identity is unstable, but `messages` is copied by reference in
 * that shallow spread, so the same array instance is observed by both the
 * headers hook (`gatewayRequestBody.messages`) and the response transforms
 * (`gatewayRequest.messages`). A WeakMap avoids leaking entries.
 */
const cacheCreatedByMessages = new WeakMap<object, boolean>();

export function markCacheCreated(params: Params | undefined): void {
  const messages = params?.messages;
  if (Array.isArray(messages)) {
    cacheCreatedByMessages.set(messages, true);
  }
}

export function wasCacheCreated(params: Params | undefined): boolean {
  const messages = params?.messages;
  if (Array.isArray(messages)) {
    return cacheCreatedByMessages.get(messages) === true;
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
 * offset — but only when no same-role merging happened in the prefix. The count
 * guard in `splitTransformedBody` rejects requests where merging occurred, so
 * on the paths that reach the split each non-system message is 1:1 with a
 * contents entry.
 */
function contentsPrefixLength(params: Params, markerIndex: number): number {
  const messages = params.messages as any[];
  let n = 0;
  for (let i = 0; i <= markerIndex && i < messages.length; i++) {
    if (messages[i]?.role !== 'system') n++;
  }
  return n;
}

/** Total number of non-system gateway messages. When this equals
 * `body.contents.length` no same-role merging occurred, so each non-system
 * message maps 1:1 onto a contents entry (see the count guard). */
function nonSystemMessageCount(params: Params): number {
  const messages = params.messages;
  if (!Array.isArray(messages)) return 0;
  return contentsPrefixLength(params, messages.length - 1);
}

/**
 * The last cache_control marker must land at message granularity: either on the
 * whole message, or on the LAST content block of the marked message. A marker
 * on a non-final block (Anthropic-style block-level breakpoint with trailing
 * dynamic blocks in the same message) is rejected — we cache at message
 * boundaries only, and folding the trailing blocks into the cache key would
 * churn a fresh resource per request. Returns true when the marker position is
 * cacheable.
 */
function lastMarkerIsMessageFinal(
  params: Params,
  markerIndex: number
): boolean {
  const msg = (params.messages as any[])?.[markerIndex];
  if (!msg) return false;
  // Message-level marker is always fine.
  if (msg.cache_control) return true;
  const content = msg.content;
  if (!Array.isArray(content)) return true;
  const lastMarked = content.reduce(
    (acc: number, item: any, i: number) => (item?.cache_control ? i : acc),
    -1
  );
  // Marker only counts as final when it is on the last content block.
  return lastMarked === content.length - 1;
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

  // Final-block guard: honor markers at message granularity only. If the last
  // marker sits on a non-final content block of its message, the trailing
  // (likely dynamic) blocks would fold into the cache key and churn — bail out
  // so the request proceeds uncached.
  if (!lastMarkerIsMessageFinal(params, markerIndex)) return null;

  const allContents = Array.isArray(body.contents) ? body.contents : [];

  // Count guard: the vertex transform merges consecutive same-role messages.
  // When it hasn't, each non-system gateway message maps 1:1 onto a contents
  // entry and the message-index split is exact. If the counts disagree
  // (merging or dropping occurred somewhere), the split would be misaligned and
  // caching would churn a new resource per request — bail out so the request
  // proceeds uncached rather than caching the wrong prefix.
  if (nonSystemMessageCount(params) !== allContents.length) return null;

  const prefixLen = Math.min(
    contentsPrefixLength(params, markerIndex),
    allContents.length
  );

  const prefixContents = allContents.slice(0, prefixLen);
  const suffix = allContents.slice(prefixLen);

  const prefix: TransformedBody = {};
  if (body.systemInstruction) prefix.systemInstruction = body.systemInstruction;
  if (prefixContents.length) prefix.contents = prefixContents;

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

/** Milliseconds of headroom required before a matched cache's expireTime; a
 * cache about to expire is treated as absent so we create a fresh one. */
const EXPIRE_BUFFER_MS = 60_000;

/**
 * Return the resource name of a cachedContents entry for this prefix, creating
 * it if one with the matching displayName does not already exist (or is about
 * to expire). `created` is true when a new resource was POSTed, false on a list
 * match — the caller uses it to adjust create-vs-match billing. Throws on any
 * non-2xx so the caller can fall back to an uncached request.
 */
export async function ensureCachedContent(
  args: EnsureArgs
): Promise<{ name: string; created: boolean }> {
  const { baseURL, projectId, region, model, prefix, authToken } = args;
  const displayName = await deriveCacheKey(model, prefix);
  const listBase = `${baseURL}/v1/projects/${projectId}/locations/${region}/cachedContents`;
  const authHeaders = {
    Authorization: `Bearer ${authToken}`,
    'Content-Type': 'application/json',
  };

  // List and match by displayName (no server-side filter param exists). Skip
  // entries that are about to expire so we don't reference a cache that Vertex
  // drops mid-request.
  let pageToken = '';
  for (let page = 0; page < 5; page++) {
    const url = `${listBase}?pageSize=1000${pageToken ? `&pageToken=${pageToken}` : ''}`;
    const res = await fetch(url, { method: 'GET', headers: authHeaders });
    if (!res.ok) throw new Error(`cachedContents list failed: ${res.status}`);
    const json: any = await res.json();
    const match = (json.cachedContents ?? []).find(
      (cc: any) =>
        cc.displayName === displayName &&
        (!cc.expireTime ||
          Date.parse(cc.expireTime) > Date.now() + EXPIRE_BUFFER_MS)
    );
    if (match?.name) return { name: match.name, created: false };
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
  return { name: created.name, created: true };
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
