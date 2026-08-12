import { Options, Params } from '../../types/requestBody';
import { ProviderConfig } from '../types';
import { AzureOpenAIChatCompleteConfig } from './chatComplete';
import {
  AzureOpenAIResponsesChatCompleteConfig,
  AzureOpenAIResponsesChatCompleteResponseTransform,
  AzureOpenAIResponsesChatCompleteStreamChunkTransform,
  getAzureResponsesChatEndpoint,
  shouldUseAzureResponsesForChat,
} from './chatCompletionsResponses';

const functionTool = {
  type: 'function',
  function: {
    name: 'get_weather',
    description: 'Get the weather',
    parameters: {
      type: 'object',
      properties: { city: { type: 'string' } },
      required: ['city'],
    },
    strict: true,
  },
};

const providerOptions = {
  provider: 'azure-openai',
  deploymentId: 'gpt-5.6-luna',
  apiVersion: '2025-04-01-preview',
  chatCompletionsApi: 'responses',
} as any;

function transformedValue(
  config: ProviderConfig,
  key: string,
  params: Params,
  options: Options = providerOptions
) {
  const entry = config[key];
  const valueConfig = Array.isArray(entry) ? entry[0] : entry;
  return valueConfig.transform
    ? valueConfig.transform(params, options)
    : (params as any)[key];
}

describe('Azure Chat Completions through Responses', () => {
  it('only bridges opted-in function-tool requests with reasoning enabled', () => {
    expect(
      shouldUseAzureResponsesForChat(
        { tools: [functionTool], reasoning_effort: 'medium' },
        providerOptions
      )
    ).toBe(true);
    expect(
      shouldUseAzureResponsesForChat(
        { tools: [functionTool], reasoning_effort: 'none' },
        providerOptions
      )
    ).toBe(false);
    expect(
      shouldUseAzureResponsesForChat(
        { tools: [functionTool], reasoning_effort: 'medium' },
        { ...providerOptions, chatCompletionsApi: undefined }
      )
    ).toBe(false);
  });

  it('rejects Chat features the bridge cannot preserve', () => {
    expect(() =>
      shouldUseAzureResponsesForChat(
        {
          tools: [functionTool],
          reasoning_effort: 'medium',
          logprobs: false as any,
        },
        providerOptions
      )
    ).not.toThrow();
    expect(() =>
      shouldUseAzureResponsesForChat(
        {
          tools: [functionTool],
          reasoning_effort: 'medium',
          n: 2,
          stop: 'END',
        },
        providerOptions
      )
    ).toThrow('does not support: stop, n');
    expect(() =>
      shouldUseAzureResponsesForChat(
        {
          tools: [functionTool, { type: 'web_search' }],
          reasoning_effort: 'medium',
        },
        providerOptions
      )
    ).toThrow('only supports function tools');
  });

  it('maps messages, tool history, multimodal input, and direct parameters', () => {
    const params = {
      model: 'public-alias',
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: 'Read this image.' },
            {
              type: 'image_url',
              image_url: { url: 'data:image/png;base64,x' },
            },
          ],
        },
        {
          role: 'assistant',
          tool_calls: [
            {
              id: 'call_1',
              type: 'function',
              function: {
                name: 'get_weather',
                arguments: '{"city":"Paris"}',
              },
            },
          ],
        },
        { role: 'tool', tool_call_id: 'call_1', content: '18 C' },
      ],
      tools: [functionTool],
      tool_choice: {
        type: 'function',
        function: { name: 'get_weather' },
      },
      reasoning_effort: 'medium',
      max_completion_tokens: 500,
      response_format: {
        type: 'json_schema',
        json_schema: {
          name: 'answer',
          schema: { type: 'object' },
          strict: true,
        },
      },
      stream: true,
    } as Params;

    expect(
      transformedValue(AzureOpenAIResponsesChatCompleteConfig, 'model', params)
    ).toBe('gpt-5.6-luna');
    expect(
      transformedValue(
        AzureOpenAIResponsesChatCompleteConfig,
        'messages',
        params
      )
    ).toEqual([
      {
        role: 'user',
        content: [
          { type: 'input_text', text: 'Read this image.' },
          { type: 'input_image', image_url: 'data:image/png;base64,x' },
        ],
      },
      {
        type: 'function_call',
        call_id: 'call_1',
        name: 'get_weather',
        arguments: '{"city":"Paris"}',
      },
      {
        type: 'function_call_output',
        call_id: 'call_1',
        output: '18 C',
      },
    ]);
    expect(
      transformedValue(AzureOpenAIResponsesChatCompleteConfig, 'tools', params)
    ).toEqual([
      {
        type: 'function',
        name: 'get_weather',
        description: 'Get the weather',
        parameters: functionTool.function.parameters,
        strict: true,
      },
    ]);
    expect(
      transformedValue(
        AzureOpenAIResponsesChatCompleteConfig,
        'tool_choice',
        params
      )
    ).toEqual({ type: 'function', name: 'get_weather' });
    expect(
      transformedValue(
        AzureOpenAIResponsesChatCompleteConfig,
        'response_format',
        params
      )
    ).toEqual({
      type: 'json_schema',
      name: 'answer',
      schema: { type: 'object' },
      strict: true,
    });
    expect(
      transformedValue(
        AzureOpenAIResponsesChatCompleteConfig,
        'reasoning_effort',
        params
      )
    ).toBe('medium');
  });

  it('uses Azure v1 Responses independently of the Chat API version', () => {
    expect(getAzureResponsesChatEndpoint()).toBe('/v1/responses');
  });

  it('strips explicit none only for opted-in function-tool requests', () => {
    const params = {
      tools: [functionTool],
      reasoning_effort: 'none',
    } as Params;
    expect(
      transformedValue(
        AzureOpenAIChatCompleteConfig,
        'reasoning_effort',
        params
      )
    ).toBeUndefined();
    expect(
      transformedValue(
        AzureOpenAIChatCompleteConfig,
        'reasoning_effort',
        params,
        { ...providerOptions, chatCompletionsApi: undefined }
      )
    ).toBe('none');
  });

  it('maps Responses tool calls and reasoning usage to Chat Completions', () => {
    expect(
      AzureOpenAIResponsesChatCompleteResponseTransform(
        {
          id: 'resp_123',
          object: 'response',
          created_at: 1234,
          status: 'completed',
          model: 'gpt-5.6-luna',
          output: [
            { type: 'reasoning', id: 'rs_1' },
            {
              type: 'function_call',
              call_id: 'call_1',
              name: 'get_weather',
              arguments: '{"city":"Paris"}',
            },
          ],
          usage: {
            input_tokens: 30,
            output_tokens: 20,
            output_tokens_details: { reasoning_tokens: 12 },
            total_tokens: 50,
          },
        },
        200
      )
    ).toMatchObject({
      id: 'resp_123',
      object: 'chat.completion',
      choices: [
        {
          message: {
            role: 'assistant',
            content: null,
            tool_calls: [
              {
                id: 'call_1',
                type: 'function',
                function: {
                  name: 'get_weather',
                  arguments: '{"city":"Paris"}',
                },
              },
            ],
          },
          finish_reason: 'tool_calls',
        },
      ],
      usage: {
        prompt_tokens: 30,
        completion_tokens: 20,
        total_tokens: 50,
        completion_tokens_details: { reasoning_tokens: 12 },
      },
    });
  });

  it('maps Responses SSE tool calls, argument deltas, and usage', () => {
    const state = {};
    const request = { stream_options: { include_usage: true } } as any;
    const chunks = [
      [
        'response.created',
        { response: { id: 'resp_1', created_at: 1, model: 'gpt-5.6-luna' } },
      ],
      [
        'response.output_item.added',
        {
          item: {
            id: 'fc_1',
            call_id: 'call_1',
            type: 'function_call',
            name: 'get_weather',
          },
        },
      ],
      [
        'response.function_call_arguments.delta',
        { item_id: 'fc_1', delta: '{"city":"Paris"}' },
      ],
      [
        'response.completed',
        {
          response: {
            id: 'resp_1',
            created_at: 1,
            status: 'completed',
            model: 'gpt-5.6-luna',
            usage: {
              input_tokens: 30,
              output_tokens: 20,
              total_tokens: 50,
              output_tokens_details: { reasoning_tokens: 12 },
            },
          },
        },
      ],
    ]
      .map(([event, data]) =>
        AzureOpenAIResponsesChatCompleteStreamChunkTransform(
          `event: ${event}\ndata: ${JSON.stringify(data)}`,
          'fallback',
          state,
          true,
          request
        )
      )
      .join('');

    expect(chunks).toContain('"id":"call_1"');
    expect(chunks).toContain('"arguments":"{\\"city\\":\\"Paris\\"}"');
    expect(chunks).toContain('"finish_reason":"tool_calls"');
    expect(chunks).toContain('"reasoning_tokens":12');
    expect(chunks).toContain('data: [DONE]');
  });
});
