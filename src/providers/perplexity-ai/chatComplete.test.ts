import {
  PerplexityAIChatCompleteResponse,
  PerplexityAIChatCompleteResponseTransform,
  PerplexityAIChatCompleteStreamChunkTransform,
} from './chatComplete';
import { ChatCompletionResponse } from '../types';

const response = {
  id: 'test-id',
  model: 'sonar',
  object: 'chat.completion',
  created: 1,
  citations: ['https://example.com'],
  choices: [
    {
      message: { role: 'assistant', content: 'ok' },
      delta: { role: 'assistant', content: 'ok' },
      index: 0,
      finish_reason: 'stop',
    },
  ],
  usage: {
    prompt_tokens: 10,
    completion_tokens: 5,
    total_tokens: 15,
    search_context_size: 'low',
    cost: {
      request_cost: 0.005,
      total_cost: 0.00501,
    },
  },
} satisfies PerplexityAIChatCompleteResponse;

describe('PerplexityAIChatCompleteResponseTransform', () => {
  test('preserves provider usage in non-strict mode', () => {
    const transformed = PerplexityAIChatCompleteResponseTransform(
      response,
      200,
      new Headers(),
      false
    );

    expect(transformed).toMatchObject({
      citations: ['https://example.com'],
      usage: {
        prompt_tokens: 10,
        completion_tokens: 5,
        total_tokens: 15,
        search_context_size: 'low',
        cost: {
          request_cost: 0.005,
        },
      },
    });
  });

  test('strips provider usage in strict mode', () => {
    const transformed = PerplexityAIChatCompleteResponseTransform(
      response,
      200,
      new Headers(),
      true
    ) as ChatCompletionResponse;

    expect(transformed).toMatchObject({
      usage: {
        prompt_tokens: 10,
        completion_tokens: 5,
        total_tokens: 15,
      },
    });
    expect(transformed).not.toHaveProperty('citations');
    expect(transformed.usage).not.toHaveProperty('cost');
    expect(transformed.usage).not.toHaveProperty('search_context_size');
  });
});

describe('PerplexityAIChatCompleteStreamChunkTransform', () => {
  test('strips provider usage in strict mode', () => {
    const chunk = `data: ${JSON.stringify({
      ...response,
      object: 'chat.completion.chunk',
      choices: [
        {
          delta: { role: 'assistant', content: '' },
          index: 0,
          finish_reason: 'stop',
        },
      ],
    })}`;

    const transformed = PerplexityAIChatCompleteStreamChunkTransform(
      chunk,
      'fallback-id',
      {},
      true
    );
    const [eventLine] = transformed.split('\n');
    const event = JSON.parse(eventLine.replace(/^data: /, ''));

    expect(event.usage).toEqual({
      prompt_tokens: 10,
      completion_tokens: 5,
      total_tokens: 15,
    });
    expect(event).not.toHaveProperty('citations');
  });
});
