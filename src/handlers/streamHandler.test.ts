import {
  BEDROCK,
  GOOGLE,
  GOOGLE_VERTEX_AI,
  OPEN_AI,
  PERPLEXITY_AI,
} from '../globals';
import {
  GoogleChatCompleteResponseTransform,
  GoogleChatCompleteStreamChunkTransform,
} from '../providers/google/chatComplete';
import {
  GoogleChatCompleteResponseTransform as vertexResponse,
  GoogleChatCompleteStreamChunkTransform as vertexStream,
} from '../providers/google-vertex-ai/chatComplete';
import { PerplexityAIChatCompleteStreamChunkTransform } from '../providers/perplexity-ai/chatComplete';
import { BedrockChatCompleteStreamChunkTransform } from '../providers/bedrock/chatComplete';
import { getStreamModeSplitPattern } from '../utils';
import { handleStreamingMode, readStream } from './streamHandler';

describe.each([
  [
    GOOGLE,
    GoogleChatCompleteResponseTransform,
    GoogleChatCompleteStreamChunkTransform,
  ],
  [GOOGLE_VERTEX_AI, vertexResponse, vertexStream],
] as const)(
  '%s Gemini response metadata',
  (provider, responseTransform, streamTransform) => {
    const transform = (payload: object) =>
      JSON.parse(
        streamTransform(JSON.stringify(payload), 'test', {}, false).slice(6)
      );
    const blocked = {
      modelVersion: 'gemini-3-flash-preview',
      promptFeedback: {
        blockReason: 'SAFETY',
        blockReasonMessage: 'Test prompt blocked',
      },
    };

    it('preserves prompt blocks in JSON and SSE instead of returning an empty completion', () => {
      for (const result of [
        responseTransform(blocked as any, 200, new Headers(), false),
        transform(blocked),
      ]) {
        expect(result).toMatchObject({
          provider,
          error: { code: 'content_policy_violation', type: 'SAFETY' },
        });
        expect((result as any).error.message).toContain('Test prompt blocked');
        expect((result as any).usage).toBeUndefined();
      }
    });

    it.each([0, undefined])(
      'preserves usage when candidatesTokenCount is %s',
      (candidatesTokenCount) => {
        expect(
          transform({
            usageMetadata: {
              promptTokenCount: 10,
              candidatesTokenCount,
              totalTokenCount: 15,
              thoughtsTokenCount: 5,
            },
          }).usage
        ).toMatchObject({
          prompt_tokens: 10,
          completion_tokens: 5,
          total_tokens: 15,
        });
      }
    );

    it('preserves prompt-only usage and never invents a missing usage block', () => {
      expect(
        transform({
          usageMetadata: { promptTokenCount: 10, totalTokenCount: 10 },
        }).usage
      ).toMatchObject({
        prompt_tokens: 10,
        completion_tokens: 0,
        total_tokens: 10,
      });
      expect(transform({ candidates: [] }).usage).toBeUndefined();
    });

    it('rejects Google’s documented unspecified prompt block and retains its usage', () => {
      const payload = {
        promptFeedback: { blockReason: 'BLOCKED_REASON_UNSPECIFIED' },
        usageMetadata: { promptTokenCount: 7, totalTokenCount: 7 },
      };
      for (const result of [
        responseTransform(payload as any, 200, new Headers(), false),
        transform(payload),
      ]) {
        expect(result).toMatchObject({
          provider,
          error: {
            code: 'content_policy_violation',
            type: 'BLOCKED_REASON_UNSPECIFIED',
          },
        });
        expect((result as any).usage).toMatchObject({
          prompt_tokens: 7,
          completion_tokens: 0,
          total_tokens: 7,
        });
      }
    });

    it('retains JSON and SSE upstream errors with their usage', () => {
      const payload = {
        error: {
          code: 429,
          status: 'RESOURCE_EXHAUSTED',
          message: 'Quota exhausted',
        },
        usageMetadata: {
          promptTokenCount: 7,
          candidatesTokenCount: 2,
          totalTokenCount: 9,
        },
      };
      for (const result of [
        responseTransform(payload as any, 429, new Headers(), false),
        transform(payload),
      ]) {
        expect(result).toMatchObject({
          provider,
          error: {
            type: 'RESOURCE_EXHAUSTED',
            message: expect.stringContaining('Quota exhausted'),
          },
          usage: { prompt_tokens: 7, completion_tokens: 2, total_tokens: 9 },
        });
      }
    });

    it('preserves filtered candidates without content parts and their reported usage', () => {
      // Live Vertex safety-filter response to our synthetic test prompt.
      const result = transform({
        candidates: [{ content: { role: 'model' }, finishReason: 'SAFETY' }],
        usageMetadata: {
          promptTokenCount: 12,
          candidatesTokenCount: 3,
          thoughtsTokenCount: 121,
          totalTokenCount: 136,
        },
      });
      expect(result.error).toBeUndefined();
      expect(result.choices[0].finish_reason).toBe('content_filter');
      expect(result.usage).toMatchObject({
        prompt_tokens: 12,
        completion_tokens: 124,
        total_tokens: 136,
      });
    });
  }
);

it('preserves Vertex prompt-block errors across fragmented SSE transport', async () => {
  const bytes = new TextEncoder().encode(
    'data: ' +
      JSON.stringify({
        promptFeedback: { blockReason: 'PROHIBITED_CONTENT' },
        modelVersion: 'gemini-3-flash-preview',
      }) +
      '\r\n\r\n'
  );
  const response = handleStreamingMode(
    new Response(
      new ReadableStream({
        start(controller) {
          for (const byte of bytes) controller.enqueue(Uint8Array.of(byte));
          controller.close();
        },
      }),
      { headers: { 'content-type': 'text/event-stream' } }
    ),
    GOOGLE_VERTEX_AI,
    vertexStream,
    '/publishers/google/models/gemini-3-flash-preview:streamGenerateContent',
    false,
    { model: 'gemini-3-flash-preview' },
    'chatComplete',
    { beforeRequestHooksResult: [], afterRequestHooksResult: [] }
  );
  const events = (await response.text()).trim().split('\n\n');
  expect(events).toHaveLength(1);
  expect(JSON.parse(events[0].slice(6))).toMatchObject({
    error: { code: 'content_policy_violation', type: 'PROHIBITED_CONTENT' },
  });
});

describe('Bedrock Converse stream errors', () => {
  // AWS EventStream frame: prelude, headers, JSON payload, message CRC.
  // The reader ignores headers and CRCs; exercise its actual binary framing.
  function frame(payload: object) {
    const bytes = new TextEncoder().encode(JSON.stringify(payload));
    const chunk = new Uint8Array(16 + bytes.length);
    new DataView(chunk.buffer).setUint32(0, chunk.length);
    chunk.set(bytes, 12);
    return chunk;
  }

  async function stream(payloads: object[], fragmented = false) {
    const response = handleStreamingMode(
      new Response(
        new ReadableStream({
          start(controller) {
            for (const payload of payloads) {
              const bytes =
                payload instanceof Uint8Array ? payload : frame(payload);
              if (fragmented) {
                for (const byte of bytes)
                  controller.enqueue(Uint8Array.of(byte));
              } else controller.enqueue(bytes);
            }
            controller.close();
          },
        })
      ),
      BEDROCK,
      BedrockChatCompleteStreamChunkTransform,
      '/chat/completions',
      false,
      { model: 'us.amazon.nova-2-lite-v1:0' },
      'chatComplete',
      { beforeRequestHooksResult: [], afterRequestHooksResult: [] }
    );
    expect(response.headers.get('content-type')).toBe('text/event-stream');
    const events = (await response.text()).trim().split('\n\n');
    expect(events.pop()).toBe('data: [DONE]');
    return events.map((event) => JSON.parse(event.slice(6)));
  }

  it('preserves the error in a captured Nova invalid-image response', async () => {
    // Synthetic invalid PNG sent directly to Bedrock; no customer input.
    const bytes = Buffer.from(
      'AAABEAAAAGH0K8qADzpleGNlcHRpb24tdHlwZQcAE3ZhbGlkYXRpb25FeGNlcHRpb24NOmNvbnRlbnQtdHlwZQcAEGFwcGxpY2F0aW9uL2pzb24NOm1lc3NhZ2UtdHlwZQcACWV4Y2VwdGlvbnsibWVzc2FnZSI6IlRoZSBtb2RlbCByZXR1cm5lZCB0aGUgZm9sbG93aW5nIGVycm9yczogVGhlIGRldGVjdGVkIGZpbGUgTUlNRSB0eXBlIHRleHQvcGxhaW4gaXMgbm90IHN1cHBvcnRlZCBieSB0aGlzIG1vZGVsLiBUcnkgYWdhaW4gd2l0aCBhIHZhbGlkIGZpbGUgdHlwZS4ifX7Fpzg=',
      'base64'
    );
    expect(await stream([bytes])).toEqual([
      {
        error: {
          message:
            'bedrock error: The model returned the following errors: The detected file MIME type text/plain is not supported by this model. Try again with a valid file type.',
          type: null,
          param: null,
          code: null,
        },
        provider: BEDROCK,
      },
    ]);
  });

  it('keeps partial content and the failure when AWS frames cross network chunks', async () => {
    const events = await stream(
      [
        { delta: { text: 'partial reply' }, contentBlockIndex: 0 },
        { message: 'Service unavailable. Please try again later.' },
      ],
      true
    );
    expect(events).toHaveLength(2);
    expect(events[0].choices[0].delta.content).toBe('partial reply');
    expect(events[1].error.message).toContain('Service unavailable.');
    expect(events[1].choices).toBeUndefined();
    expect(events[1].usage).toBeUndefined();
  });

  it('retains healthy output and provider usage', async () => {
    const events = await stream([
      { delta: { text: 'Hello' }, contentBlockIndex: 0 },
      { stopReason: 'end_turn' },
      { usage: { inputTokens: 10, outputTokens: 2, totalTokens: 12 } },
    ]);
    expect(events[0].choices[0].delta.content).toBe('Hello');
    expect(events[2].choices[0].finish_reason).toBe('stop');
    expect(events[2].usage).toMatchObject({
      prompt_tokens: 10,
      completion_tokens: 2,
      total_tokens: 12,
    });
    expect(events.every((event) => !event.error)).toBe(true);
  });
});

describe('Perplexity SSE framing', () => {
  const usage = {
    prompt_tokens: 4,
    completion_tokens: 1,
    total_tokens: 5,
    cost: { request_cost: 0.005 },
  };
  const events = [
    {
      id: 'test',
      choices: [{ delta: { content: '🌼' }, finish_reason: null }],
    },
    {
      id: 'test',
      choices: [{ delta: { content: '' }, finish_reason: 'stop' }],
      usage,
    },
  ];

  it.each([
    ['LF', '\n\n', false],
    ['CRLF', '\r\n\r\n', false],
    ['LF byte chunks', '\n\n', true],
    ['CRLF byte chunks', '\r\n\r\n', true],
  ])(
    'preserves content and terminal usage with %s',
    async (_, separator, byteChunks) => {
      const input = events
        .map((event) => `data: ${JSON.stringify(event)}${separator}`)
        .join('');
      const bytes = new TextEncoder().encode(input);
      const source = new ReadableStream({
        start(controller) {
          if (byteChunks) {
            for (const byte of bytes) controller.enqueue(Uint8Array.of(byte));
          } else {
            controller.enqueue(bytes);
          }
          controller.close();
        },
      });
      let output = '';
      for await (const chunk of readStream(
        source.getReader(),
        getStreamModeSplitPattern(PERPLEXITY_AI, '/chat/completions'),
        PerplexityAIChatCompleteStreamChunkTransform,
        false,
        'test',
        false,
        {}
      ))
        output += chunk;

      const chunks = output.trim().split('\n\n');
      expect(chunks.pop()).toBe('data: [DONE]');
      const parsed = chunks.map((chunk) => JSON.parse(chunk.slice(6)));
      expect(
        parsed.map((event) => event.choices[0].delta.content).join('')
      ).toBe('🌼');
      expect(parsed[1].usage).toEqual(usage);
      expect(parsed[1].choices[0].finish_reason).toBe('stop');
    }
  );

  it('keeps raw SSE valid when no response transformer is selected', async () => {
    const input = events
      .map((event) => `data: ${JSON.stringify(event)}\r\n\r\n`)
      .join('');
    let output = '';
    for await (const chunk of readStream(
      new Response(input).body!.getReader(),
      getStreamModeSplitPattern(PERPLEXITY_AI, '/chat/completions'),
      undefined,
      false,
      'test',
      false,
      {}
    ))
      output += chunk;
    expect(output).toBe(input.replaceAll('\r\n', '\n'));
  });
});

describe('upstream stream failures', () => {
  it('leaves a healthy stream unchanged', async () => {
    const body =
      'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}\n\n' +
      'data: [DONE]\n\n';
    const response = handleStreamingMode(
      new Response(body),
      OPEN_AI,
      undefined,
      '/chat/completions',
      false,
      {},
      'chatComplete',
      { beforeRequestHooksResult: [], afterRequestHooksResult: [] }
    );
    expect(await response.text()).toBe(body);
  });

  it.each([OPEN_AI, BEDROCK])(
    'preserves the read error for %s',
    async (provider) => {
      const response = handleStreamingMode(
        new Response(
          new ReadableStream({
            pull(controller) {
              controller.error(new Error('upstream connection reset'));
            },
          })
        ),
        provider,
        undefined,
        '/chat/completions',
        false,
        {},
        'chatComplete',
        { beforeRequestHooksResult: [], afterRequestHooksResult: [] }
      );
      const output = await response.text();
      expect(JSON.parse(output.trim().slice(6))).toEqual({
        error: {
          message: 'upstream connection reset',
          type: 'upstream_error',
          code: 'upstream_stream_error',
        },
      });
      expect(output).not.toContain('[DONE]');
    }
  );

  it('preserves already delivered content and emits one complete error event', async () => {
    const delta = 'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n';
    let reads = 0;
    const response = handleStreamingMode(
      new Response(
        new ReadableStream({
          pull(controller) {
            if (reads++ === 0)
              controller.enqueue(new TextEncoder().encode(delta));
            else controller.error(new Error('socket closed'));
          },
        })
      ),
      OPEN_AI,
      undefined,
      '/chat/completions',
      false,
      {},
      'chatComplete',
      { beforeRequestHooksResult: [], afterRequestHooksResult: [] }
    );
    const output = await response.text();
    expect(output.startsWith(delta)).toBe(true);
    expect(output.match(/upstream_stream_error/g)).toHaveLength(1);
    expect(reads).toBe(2);
  });
});
