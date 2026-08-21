import { Params } from '../../types/requestBody';
import { OpenrouterChatCompleteConfig } from './chatComplete';

describe('OpenrouterChatCompleteConfig', () => {
  it('preserves OpenRouter chat and routing parameters', () => {
    const passthroughParameters = [
      'model',
      'max_tokens',
      'max_completion_tokens',
      'temperature',
      'cache_control',
      'debug',
      'frequency_penalty',
      'image_config',
      'logit_bias',
      'logprobs',
      'metadata',
      'min_p',
      'modalities',
      'models',
      'parallel_tool_calls',
      'plugins',
      'presence_penalty',
      'provider',
      'reasoning',
      'reasoning_effort',
      'repetition_penalty',
      'response_format',
      'route',
      'seed',
      'service_tier',
      'session_id',
      'stop',
      'stop_server_tools_when',
      'stream',
      'stream_options',
      'tool_choice',
      'tools',
      'top_a',
      'top_k',
      'top_logprobs',
      'top_p',
      'trace',
      'transforms',
      'usage',
      'user',
    ];

    for (const parameter of passthroughParameters) {
      expect(OpenrouterChatCompleteConfig[parameter]).toMatchObject({
        param: parameter,
      });
    }
  });

  it('maps developer messages to OpenRouter-compatible system messages', () => {
    const params = {
      model: 'google/gemini-2.5-flash',
      messages: [{ role: 'developer', content: 'Follow this instruction' }],
    } as Params;

    const messageConfig = OpenrouterChatCompleteConfig.messages;
    expect(Array.isArray(messageConfig)).toBe(false);
    const transformed = !Array.isArray(messageConfig)
      ? messageConfig.transform?.(params, { provider: 'openrouter' })
      : undefined;

    expect(transformed).toEqual([
      { role: 'system', content: 'Follow this instruction' },
    ]);
  });
});
