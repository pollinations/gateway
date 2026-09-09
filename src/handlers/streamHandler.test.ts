import { PERPLEXITY_AI } from '../globals';
import { PerplexityAIChatCompleteStreamChunkTransform } from '../providers/perplexity-ai/chatComplete';
import { getStreamModeSplitPattern } from '../utils';
import { readStream } from './streamHandler';

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
