import {
  applyCachedContent,
  deriveCacheKey,
  ensureCachedContent,
  hasCacheControl,
  isExplicitCachingEligible,
  lastCacheControlMessageIndex,
  markCacheCreated,
  splitTransformedBody,
  wasCacheCreated,
} from './cachedContents';
import {
  GoogleChatCompleteResponseTransform,
  GoogleChatCompleteStreamChunkTransform,
} from './chatComplete';

const marker = { type: 'ephemeral' as const };

describe('hasCacheControl', () => {
  it('detects a marker on a content block', () => {
    expect(
      hasCacheControl({
        messages: [
          {
            role: 'user',
            content: [{ type: 'text', text: 'x', cache_control: marker }],
          },
        ],
      } as any)
    ).toBe(true);
  });

  it('detects a message-level marker', () => {
    expect(
      hasCacheControl({
        messages: [{ role: 'system', content: 'x', cache_control: marker }],
      } as any)
    ).toBe(true);
  });

  it('returns false with no marker', () => {
    expect(
      hasCacheControl({
        messages: [{ role: 'user', content: 'hi' }],
      } as any)
    ).toBe(false);
  });
});

describe('lastCacheControlMessageIndex', () => {
  it('returns the last marked message index', () => {
    const params = {
      messages: [
        { role: 'system', content: 'sys', cache_control: marker },
        { role: 'user', content: 'q1' },
        {
          role: 'user',
          content: [{ type: 'text', text: 'big', cache_control: marker }],
        },
        { role: 'user', content: 'tail' },
      ],
    } as any;
    expect(lastCacheControlMessageIndex(params)).toBe(2);
  });
});

describe('splitTransformedBody', () => {
  const params = {
    messages: [
      { role: 'system', content: 'sys', cache_control: marker },
      {
        role: 'user',
        content: [{ type: 'text', text: 'ctx', cache_control: marker }],
      },
      { role: 'user', content: 'question' },
    ],
  } as any;

  const body = {
    systemInstruction: { role: 'system', parts: [{ text: 'sys' }] },
    contents: [
      { role: 'user', parts: [{ text: 'ctx' }] },
      { role: 'user', parts: [{ text: 'question' }] },
    ],
  };

  it('splits system + leading contents into the prefix', () => {
    const split = splitTransformedBody(params, body);
    expect(split).not.toBeNull();
    expect(split!.prefix.systemInstruction).toEqual(body.systemInstruction);
    expect(split!.prefix.contents).toEqual([
      { role: 'user', parts: [{ text: 'ctx' }] },
    ]);
    expect(split!.suffix).toEqual([
      { role: 'user', parts: [{ text: 'question' }] },
    ]);
  });

  it('returns null when there is no marker', () => {
    expect(
      splitTransformedBody(
        { messages: [{ role: 'user', content: 'hi' }] } as any,
        body
      )
    ).toBeNull();
  });
});

describe('deriveCacheKey', () => {
  it('is deterministic and prefixed', async () => {
    const p = { systemInstruction: { role: 'system', parts: [{ text: 'a' }] } };
    const k1 = await deriveCacheKey('gemini-2.5-flash-lite', p);
    const k2 = await deriveCacheKey('gemini-2.5-flash-lite', p);
    expect(k1).toBe(k2);
    expect(k1).toMatch(/^portkey:[0-9a-f]{64}$/);
  });

  it('differs by model', async () => {
    const p = { systemInstruction: { role: 'system', parts: [{ text: 'a' }] } };
    expect(await deriveCacheKey('m1', p)).not.toBe(
      await deriveCacheKey('m2', p)
    );
  });
});

describe('applyCachedContent', () => {
  it('sets cachedContent, replaces contents, strips cache-incompatible fields', () => {
    const body: any = {
      systemInstruction: { role: 'system', parts: [{ text: 'sys' }] },
      tools: [{}],
      toolConfig: {},
      contents: [{ role: 'user', parts: [{ text: 'ctx' }] }],
    };
    const suffix = [{ role: 'user', parts: [{ text: 'q' }] }];
    applyCachedContent(
      body,
      'projects/1/locations/global/cachedContents/9',
      suffix
    );
    expect(body.cachedContent).toBe(
      'projects/1/locations/global/cachedContents/9'
    );
    expect(body.contents).toEqual(suffix);
    expect(body.systemInstruction).toBeUndefined();
    expect(body.tools).toBeUndefined();
    expect(body.toolConfig).toBeUndefined();
  });
});

describe('ensureCachedContent', () => {
  const base = {
    baseURL: 'https://aiplatform.googleapis.com',
    projectId: 'proj',
    region: 'global',
    model: 'gemini-2.5-flash-lite',
    prefix: { systemInstruction: { role: 'system', parts: [{ text: 'sys' }] } },
    authToken: 'tok',
  };

  afterEach(() => jest.restoreAllMocks());

  it('reuses an existing resource matched by displayName', async () => {
    const displayName = await deriveCacheKey(base.model, base.prefix);
    jest.spyOn(global, 'fetch').mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        cachedContents: [
          {
            name: 'projects/1/locations/global/cachedContents/existing',
            displayName,
          },
        ],
      }),
    } as any);
    const result = await ensureCachedContent(base);
    expect(result).toEqual({
      name: 'projects/1/locations/global/cachedContents/existing',
      created: false,
    });
    expect(global.fetch).toHaveBeenCalledTimes(1); // list only, no create
  });

  it('creates when no match exists', async () => {
    jest
      .spyOn(global, 'fetch')
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ cachedContents: [] }),
      } as any)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          name: 'projects/1/locations/global/cachedContents/new',
        }),
      } as any);
    const result = await ensureCachedContent(base);
    expect(result).toEqual({
      name: 'projects/1/locations/global/cachedContents/new',
      created: true,
    });
    expect(global.fetch).toHaveBeenCalledTimes(2);
  });

  it('ignores a matching entry that is about to expire and creates fresh', async () => {
    const displayName = await deriveCacheKey(base.model, base.prefix);
    jest
      .spyOn(global, 'fetch')
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          cachedContents: [
            {
              name: 'projects/1/locations/global/cachedContents/stale',
              displayName,
              // expires in 10s — inside the 60s buffer, treated as absent
              expireTime: new Date(Date.now() + 10_000).toISOString(),
            },
          ],
        }),
      } as any)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          name: 'projects/1/locations/global/cachedContents/fresh',
        }),
      } as any);
    const result = await ensureCachedContent(base);
    expect(result).toEqual({
      name: 'projects/1/locations/global/cachedContents/fresh',
      created: true,
    });
    expect(global.fetch).toHaveBeenCalledTimes(2); // list missed, then create
  });

  it('throws on a below-min-token create failure (so caller falls back)', async () => {
    jest
      .spyOn(global, 'fetch')
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ cachedContents: [] }),
      } as any)
      .mockResolvedValueOnce({
        ok: false,
        status: 400,
        text: async () => 'The minimum token count to start caching is 2048.',
      } as any);
    await expect(ensureCachedContent(base)).rejects.toThrow(
      /create failed: 400/
    );
  });
});

describe('isExplicitCachingEligible', () => {
  it('is true with a marker and no tools', () => {
    expect(
      isExplicitCachingEligible({
        messages: [{ role: 'user', content: 'hi', cache_control: marker }],
      } as any)
    ).toBe(true);
  });

  it('is false when tools are present', () => {
    expect(
      isExplicitCachingEligible({
        messages: [{ role: 'user', content: 'hi', cache_control: marker }],
        tools: [{ type: 'function', function: { name: 'f' } }],
      } as any)
    ).toBe(false);
  });

  it('is false when tool_choice is present', () => {
    expect(
      isExplicitCachingEligible({
        messages: [{ role: 'user', content: 'hi', cache_control: marker }],
        tool_choice: 'auto',
      } as any)
    ).toBe(false);
  });

  it('is false when a tool-role message is present', () => {
    expect(
      isExplicitCachingEligible({
        messages: [
          { role: 'user', content: 'hi', cache_control: marker },
          { role: 'tool', content: 'result', name: 'f' },
        ],
      } as any)
    ).toBe(false);
  });

  it('is false when an assistant message has tool_calls', () => {
    expect(
      isExplicitCachingEligible({
        messages: [
          { role: 'user', content: 'hi', cache_control: marker },
          {
            role: 'assistant',
            content: '',
            tool_calls: [{ function: { name: 'f', arguments: '{}' } }],
          },
        ],
      } as any)
    ).toBe(false);
  });

  it('is false without a marker', () => {
    expect(
      isExplicitCachingEligible({
        messages: [{ role: 'user', content: 'hi' }],
      } as any)
    ).toBe(false);
  });
});

describe('splitTransformedBody count guard', () => {
  it('returns null when contents length disagrees with message count', () => {
    // 3 non-system messages, but only 2 contents entries (as if merged).
    const params = {
      messages: [
        { role: 'user', content: 'a', cache_control: marker },
        { role: 'user', content: 'b' },
        { role: 'user', content: 'c' },
      ],
    } as any;
    const body = {
      contents: [
        { role: 'user', parts: [{ text: 'a' }] },
        { role: 'user', parts: [{ text: 'bc' }] },
      ],
    };
    expect(splitTransformedBody(params, body)).toBeNull();
  });

  it('returns null for a merged consecutive same-role shape (uncached fallback)', () => {
    // Two consecutive user messages the vertex transform merges into one
    // contents entry; the marker is on the first. contents.length (1) < 2
    // non-system messages, so caching bails out.
    const params = {
      messages: [
        { role: 'user', content: 'ctx', cache_control: marker },
        { role: 'user', content: 'question' },
      ],
    } as any;
    const body = {
      contents: [
        { role: 'user', parts: [{ text: 'ctx' }, { text: 'question' }] },
      ],
    };
    expect(splitTransformedBody(params, body)).toBeNull();
  });
});

describe('splitTransformedBody final-block guard', () => {
  it('returns null when the last marker is on a non-final content block', () => {
    // Marker sits on the first of two blocks in the same message; the trailing
    // block would fold into the cache key, so caching bails out.
    const params = {
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: 'static', cache_control: marker },
            { type: 'text', text: 'dynamic' },
          ],
        },
      ],
    } as any;
    const body = {
      contents: [
        { role: 'user', parts: [{ text: 'static' }, { text: 'dynamic' }] },
      ],
    };
    expect(splitTransformedBody(params, body)).toBeNull();
  });

  it('allows a marker on the last content block of a message', () => {
    const params = {
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: 'a' },
            { type: 'text', text: 'b', cache_control: marker },
          ],
        },
        { role: 'assistant', content: 'answer' },
      ],
    } as any;
    const body = {
      contents: [
        { role: 'user', parts: [{ text: 'a' }, { text: 'b' }] },
        { role: 'model', parts: [{ text: 'answer' }] },
      ],
    };
    const split = splitTransformedBody(params, body);
    expect(split).not.toBeNull();
    expect(split!.prefix.contents).toEqual([
      { role: 'user', parts: [{ text: 'a' }, { text: 'b' }] },
    ]);
    expect(split!.suffix).toEqual([
      { role: 'model', parts: [{ text: 'answer' }] },
    ]);
  });
});

describe('splitTransformedBody supported positives', () => {
  it('splits with a marker on the system message and a separate user tail', () => {
    const params = {
      messages: [
        { role: 'system', content: 'sys', cache_control: marker },
        { role: 'user', content: 'question' },
      ],
    } as any;
    const body = {
      systemInstruction: { role: 'system', parts: [{ text: 'sys' }] },
      contents: [{ role: 'user', parts: [{ text: 'question' }] }],
    };
    const split = splitTransformedBody(params, body);
    expect(split).not.toBeNull();
    expect(split!.prefix.systemInstruction).toEqual(body.systemInstruction);
    // Marker is on the system message (index 0), so no non-system message is in
    // the prefix; the whole user turn is the suffix.
    expect(split!.prefix.contents).toBeUndefined();
    expect(split!.suffix).toEqual([
      { role: 'user', parts: [{ text: 'question' }] },
    ]);
  });

  it('splits with a marker on a user message followed by an assistant message', () => {
    const params = {
      messages: [
        { role: 'user', content: 'ctx', cache_control: marker },
        { role: 'assistant', content: 'ack' },
      ],
    } as any;
    const body = {
      contents: [
        { role: 'user', parts: [{ text: 'ctx' }] },
        { role: 'model', parts: [{ text: 'ack' }] },
      ],
    };
    const split = splitTransformedBody(params, body);
    expect(split).not.toBeNull();
    expect(split!.prefix.contents).toEqual([
      { role: 'user', parts: [{ text: 'ctx' }] },
    ]);
    expect(split!.suffix).toEqual([
      { role: 'model', parts: [{ text: 'ack' }] },
    ]);
  });
});

describe('response transform cached_tokens billing', () => {
  const usageMetadata = {
    promptTokenCount: 20000,
    candidatesTokenCount: 100,
    totalTokenCount: 20100,
    cachedContentTokenCount: 18000,
  };
  const nonStreamResponse = {
    candidates: [
      {
        content: { parts: [{ text: 'hi' }] },
        finishReason: 'STOP',
      },
    ],
    usageMetadata,
    modelVersion: 'gemini-2.5-flash-lite',
  };

  afterEach(() => {
    // markCacheCreated writes to a module WeakMap keyed on the messages array;
    // each test uses a fresh array, so no cross-test leakage.
  });

  it('passes cached_tokens through when the cache was matched (not created)', () => {
    const gatewayRequest = {
      messages: [{ role: 'user', content: 'x', cache_control: marker }],
    } as any;
    const result: any = GoogleChatCompleteResponseTransform(
      nonStreamResponse as any,
      200,
      new Headers(),
      true,
      'https://example',
      gatewayRequest
    );
    expect(result.usage.prompt_tokens_details.cached_tokens).toBe(18000);
    expect(result.usage.prompt_tokens).toBe(20000);
  });

  it('zeroes cached_tokens when the request created the cache (non-stream)', () => {
    const gatewayRequest = {
      messages: [{ role: 'user', content: 'x', cache_control: marker }],
    } as any;
    markCacheCreated(gatewayRequest);
    expect(wasCacheCreated(gatewayRequest)).toBe(true);
    const result: any = GoogleChatCompleteResponseTransform(
      nonStreamResponse as any,
      200,
      new Headers(),
      true,
      'https://example',
      gatewayRequest
    );
    expect(result.usage.prompt_tokens_details.cached_tokens).toBe(0);
    expect(result.usage.prompt_tokens).toBe(20000);
  });

  const parseStreamUsage = (out: string) => {
    const line = out
      .split('\n')
      .find((l) => l.startsWith('data: ') && l.includes('usage'));
    return JSON.parse(line!.replace(/^data: /, '')).usage;
  };

  const streamChunk = JSON.stringify({
    candidates: [{ content: { parts: [{ text: 'hi' }] }, index: 0 }],
    usageMetadata,
    modelVersion: 'gemini-2.5-flash-lite',
  });

  it('passes cached_tokens through in stream chunks when matched', () => {
    const gatewayRequest = {
      messages: [{ role: 'user', content: 'x', cache_control: marker }],
    } as any;
    const out = GoogleChatCompleteStreamChunkTransform(
      streamChunk,
      'id',
      {},
      true,
      gatewayRequest
    );
    expect(parseStreamUsage(out).prompt_tokens_details.cached_tokens).toBe(
      18000
    );
  });

  it('zeroes cached_tokens in stream chunks when created', () => {
    const gatewayRequest = {
      messages: [{ role: 'user', content: 'x', cache_control: marker }],
    } as any;
    markCacheCreated(gatewayRequest);
    const out = GoogleChatCompleteStreamChunkTransform(
      streamChunk,
      'id',
      {},
      true,
      gatewayRequest
    );
    expect(parseStreamUsage(out).prompt_tokens_details.cached_tokens).toBe(0);
  });
});
