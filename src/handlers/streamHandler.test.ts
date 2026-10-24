import { BEDROCK, OPEN_AI, PERPLEXITY_AI } from '../globals';
import { PerplexityAIChatCompleteStreamChunkTransform } from '../providers/perplexity-ai/chatComplete';
import { BedrockChatCompleteStreamChunkTransform } from '../providers/bedrock/chatComplete';
import { getStreamModeSplitPattern } from '../utils';
import { handleStreamingMode, readStream } from './streamHandler';

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

  async function stream(
    payloads: object[],
    strict: boolean,
    fragmented = false
  ) {
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
      strict,
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
    const events = await stream([bytes], false);
    expect(events).toHaveLength(1);
    expect(events[0].error.message).toContain(
      'The detected file MIME type text/plain is not supported'
    );
    expect(events[0].choices).toBeUndefined();
    expect(events[0].usage).toBeUndefined();
  });

  it.each([false, true])(
    'preserves the provider error with strict mode %s',
    async (strict) => {
      const events = await stream(
        [{ message: 'The image could not be decoded.' }],
        strict
      );
      expect(events).toEqual([
        {
          error: {
            message: 'bedrock error: The image could not be decoded.',
            type: null,
            param: null,
            code: null,
          },
          provider: BEDROCK,
        },
      ]);
    }
  );

  it('keeps partial content and the failure when AWS frames cross network chunks', async () => {
    const events = await stream(
      [
        { delta: { text: 'partial reply' }, contentBlockIndex: 0 },
        { message: 'Service unavailable. Please try again later.' },
      ],
      false,
      true
    );
    expect(events).toHaveLength(2);
    expect(events[0].choices[0].delta.content).toBe('partial reply');
    expect(events[1].error.message).toContain('Service unavailable.');
    expect(events[1].choices).toBeUndefined();
    expect(events[1].usage).toBeUndefined();
  });

  it.each([false, true])(
    'retains healthy output and provider usage with strict mode %s',
    async (strict) => {
      const events = await stream(
        [
          { delta: { text: 'Hello' }, contentBlockIndex: 0 },
          { stopReason: 'end_turn' },
          { usage: { inputTokens: 10, outputTokens: 2, totalTokens: 12 } },
        ],
        strict
      );
      expect(events[0].choices[0].delta.content).toBe('Hello');
      expect(events[2].choices[0].finish_reason).toBe('stop');
      expect(events[2].usage).toMatchObject({
        prompt_tokens: 10,
        completion_tokens: 2,
        total_tokens: 12,
      });
      expect(events.every((event) => !event.error)).toBe(true);
    }
  );
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
