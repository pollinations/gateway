import {
  applyCachedContent,
  deriveCacheKey,
  ensureCachedContent,
  hasCacheControl,
  lastCacheControlMessageIndex,
  splitTransformedBody,
} from './cachedContents';

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
    const name = await ensureCachedContent(base);
    expect(name).toBe('projects/1/locations/global/cachedContents/existing');
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
    const name = await ensureCachedContent(base);
    expect(name).toBe('projects/1/locations/global/cachedContents/new');
    expect(global.fetch).toHaveBeenCalledTimes(2);
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
