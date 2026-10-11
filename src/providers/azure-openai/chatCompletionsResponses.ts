import { GatewayError } from '../../errors/GatewayError';
import { AZURE_OPEN_AI } from '../../globals';
import { Message, Options, Params, Tool } from '../../types/requestBody';
import { OpenAIErrorResponseTransform } from '../openai/utils';
import {
  ChatCompletionResponse,
  ErrorResponse,
  ProviderConfig,
} from '../types';

type BridgeOptions = Options & { chatCompletionsApi?: string };
type StreamState = {
  id?: string;
  model?: string;
  created?: number;
  toolIndexes?: Record<string, number>;
  hasToolCalls?: boolean;
  done?: boolean;
};

const isFunctionTool = (tool: Tool) =>
  tool?.type === 'function' && Boolean(tool.function?.name);

const allowedChatParameters = new Set([
  'model',
  'messages',
  'max_tokens',
  'max_completion_tokens',
  'temperature',
  'top_p',
  'n',
  'stream',
  'user',
  'tools',
  'tool_choice',
  'response_format',
  'logprobs',
  'top_logprobs',
  'stream_options',
  'service_tier',
  'parallel_tool_calls',
  'store',
  'metadata',
  'modalities',
  'reasoning_effort',
  'prompt_cache_key',
  'safety_identifier',
  'verbosity',
]);

export function hasFunctionTools(params: Params): boolean {
  const request = params as Record<string, any>;
  return Boolean(
    params.tools?.some(isFunctionTool) || request.functions?.length
  );
}

function unsupportedChatParameters(params: Params): string[] {
  const request = params as Record<string, any>;
  const unsupported = [
    'functions',
    'function_call',
    'stop',
    'presence_penalty',
    'frequency_penalty',
    'logit_bias',
    'seed',
    'top_logprobs',
    'audio',
    'prediction',
    'web_search_options',
  ].filter((key) => request[key] !== undefined);

  if (params.n !== undefined && params.n !== 1) unsupported.push('n');
  if (request.logprobs !== undefined && request.logprobs !== false) {
    unsupported.push('logprobs');
  }
  if (params.modalities?.some((modality) => modality !== 'text')) {
    unsupported.push('modalities');
  }

  for (const key of Object.keys(request)) {
    if (
      request[key] !== undefined &&
      !allowedChatParameters.has(key) &&
      !unsupported.includes(key)
    ) {
      unsupported.push(key);
    }
  }

  return unsupported;
}

function assertCompatibleChatRequest(params: Params) {
  if (params.tools?.some((tool) => !isFunctionTool(tool))) {
    throw new GatewayError(
      'Azure Chat-to-Responses compatibility mode only supports function tools',
      400
    );
  }
  if (
    typeof params.tool_choice === 'object' &&
    (params.tool_choice.type !== 'function' ||
      !params.tool_choice.function?.name)
  ) {
    throw new GatewayError(
      'Azure Chat-to-Responses compatibility mode only supports function tool_choice objects',
      400
    );
  }
  const request = params as Record<string, any>;
  if (
    request.reasoning_effort !== undefined &&
    typeof request.reasoning_effort !== 'string'
  ) {
    throw new GatewayError(
      'Azure Chat-to-Responses compatibility mode requires reasoning_effort to be a string',
      400
    );
  }
  const streamOptions = request.stream_options;
  if (
    streamOptions &&
    Object.keys(streamOptions).some((key) => key !== 'include_usage')
  ) {
    throw new GatewayError(
      'Azure Chat-to-Responses compatibility mode only supports stream_options.include_usage',
      400
    );
  }

  for (const message of params.messages || []) {
    const refusal = (message as Record<string, any>).refusal;
    if (
      message.name ||
      message.role === 'function' ||
      message.function_call ||
      message.reasoning_details ||
      message.content_blocks ||
      (message as Record<string, any>).audio ||
      (refusal !== undefined &&
        (message.role !== 'assistant' || typeof refusal !== 'string')) ||
      (['tool', 'function'].includes(message.role) &&
        message.content === undefined) ||
      (message.role === 'tool' && !message.tool_call_id)
    ) {
      throw new GatewayError(
        'Azure Chat-to-Responses compatibility mode cannot preserve named messages, deprecated function messages, audio history, or provider-specific content',
        400
      );
    }

    if (
      Array.isArray(message.tool_calls) &&
      message.tool_calls.some(
        (call: any) =>
          call?.type !== 'function' ||
          !call.id ||
          !call.function?.name ||
          typeof call.function?.arguments !== 'string'
      )
    ) {
      throw new GatewayError(
        'Azure Chat-to-Responses compatibility mode only supports valid function tool call history',
        400
      );
    }

    if (
      Array.isArray(message.content) &&
      message.content.some((part) => {
        if (part.type === 'text') return typeof part.text === 'string';
        if (part.type === 'refusal') {
          return (
            message.role === 'assistant' &&
            typeof (part as Record<string, any>).refusal === 'string'
          );
        }
        if (part.type === 'image_url') {
          return message.role === 'user' && Boolean(part.image_url?.url);
        }
        if (part.type === 'file') {
          const file = part.file as Record<string, any> | undefined;
          return Boolean(
            message.role === 'user' && file && (file.file_id || file.file_data)
          );
        }
        return false;
      })
    ) {
      throw new GatewayError(
        'Azure Chat-to-Responses compatibility mode only supports Chat text, refusal, image_url, and file message content',
        400
      );
    }
  }

  const unsupported = unsupportedChatParameters(params);
  if (unsupported.length) {
    throw new GatewayError(
      `Azure Chat-to-Responses compatibility mode does not support: ${unsupported.join(', ')}`,
      400
    );
  }
}

export function isAzureResponsesChatCompatibilityEnabled(
  providerOptions?: Options
): boolean {
  return (providerOptions as BridgeOptions)?.chatCompletionsApi === 'responses';
}

/** Route only opted-in function-tool requests that Azure Chat cannot serve. */
export function shouldUseAzureResponsesForChat(
  params: Params,
  providerOptions?: Options
): boolean {
  const shouldUse = Boolean(
    isAzureResponsesChatCompatibilityEnabled(providerOptions) &&
      hasFunctionTools(params) &&
      params.reasoning_effort !== 'none'
  );
  if (shouldUse) assertCompatibleChatRequest(params);
  return shouldUse;
}

export function getAzureResponsesChatEndpoint(): string {
  return '/v1/responses';
}

function mapMessageContent(content: Message['content'], role: Message['role']) {
  if (typeof content === 'string') {
    return [
      {
        type: role === 'assistant' ? 'output_text' : 'input_text',
        text: content,
      },
    ];
  }
  if (!Array.isArray(content)) return [];
  return content.map((part) => {
    if (part.type === 'text') {
      return {
        type: role === 'assistant' ? 'output_text' : 'input_text',
        text: part.text || '',
      };
    }
    if (part.type === 'refusal') {
      return {
        type: 'refusal',
        refusal: (part as Record<string, any>).refusal,
      };
    }
    if (part.type === 'file') {
      const file = part.file as Record<string, any>;
      return {
        type: 'input_file',
        ...(file.file_id && { file_id: file.file_id }),
        ...(file.file_data && { file_data: file.file_data }),
        ...((file.filename || file.file_name) && {
          filename: file.filename || file.file_name,
        }),
      };
    }
    return {
      type: 'input_image',
      image_url: part.image_url!.url,
      ...(part.image_url!.detail && { detail: part.image_url!.detail }),
    };
  });
}

function toolOutput(
  content: Message['content']
): string | Record<string, any>[] {
  return typeof content === 'string'
    ? content
    : (mapMessageContent(content, 'user') as any);
}

/** Convert Chat history, including completed tool rounds, to Responses input. */
export function chatMessagesToResponsesInput(messages: Message[] = []) {
  const input: Record<string, any>[] = [];

  for (const message of messages) {
    if (message.role === 'tool') {
      input.push({
        type: 'function_call_output',
        call_id: message.tool_call_id,
        output: toolOutput(message.content),
      });
      continue;
    }
    const calls =
      message.role === 'assistant' && Array.isArray(message.tool_calls)
        ? message.tool_calls
        : [];
    if (
      (message.content !== undefined &&
        message.content !== null &&
        (message.content !== '' || calls.length === 0)) ||
      (message as Record<string, any>).refusal
    ) {
      const content = mapMessageContent(message.content, message.role);
      const refusal = (message as Record<string, any>).refusal;
      if (refusal && Array.isArray(content)) {
        content.push({ type: 'refusal', refusal });
      }
      input.push({
        type: 'message',
        role: message.role,
        content,
      });
    }
    for (const call of calls) {
      input.push({
        type: 'function_call',
        call_id: call.id,
        name: call.function.name,
        arguments: call.function.arguments,
      });
    }
  }

  return input;
}

export function chatToolsToResponsesTools(tools: Tool[] = []) {
  return tools.map(({ function: fn }) => ({
    type: 'function',
    name: fn!.name,
    ...(fn!.description !== undefined && { description: fn!.description }),
    ...(fn!.parameters !== undefined && { parameters: fn!.parameters }),
    ...(fn!.strict !== undefined && { strict: fn!.strict }),
  }));
}

function toolChoice(choice: Params['tool_choice']) {
  if (
    typeof choice === 'object' &&
    choice?.type === 'function' &&
    choice.function?.name
  ) {
    return { type: 'function', name: choice.function.name };
  }
  return choice;
}

function textFormat(responseFormat: Params['response_format']) {
  if (!responseFormat || responseFormat.type !== 'json_schema') {
    return responseFormat;
  }
  const schema = responseFormat.json_schema || {};
  return {
    type: 'json_schema',
    name: schema.name,
    ...(schema.description !== undefined && {
      description: schema.description,
    }),
    schema: schema.schema,
    ...(schema.strict !== undefined && { strict: schema.strict }),
  };
}

export const AzureOpenAIResponsesChatCompleteConfig: ProviderConfig = {
  model: {
    param: 'model',
    transform: (params: Params, options: Options) =>
      options.deploymentId || params.model,
  },
  messages: {
    param: 'input',
    transform: (params: Params) =>
      chatMessagesToResponsesInput(params.messages),
  },
  tools: {
    param: 'tools',
    transform: (params: Params) => chatToolsToResponsesTools(params.tools),
  },
  tool_choice: {
    param: 'tool_choice',
    transform: (params: Params) => toolChoice(params.tool_choice),
  },
  reasoning_effort: { param: 'reasoning.effort' },
  max_tokens: { param: 'max_output_tokens' },
  max_completion_tokens: { param: 'max_output_tokens' },
  response_format: {
    param: 'text.format',
    transform: (params: Params) => textFormat(params.response_format),
  },
  verbosity: { param: 'text.verbosity' },
  stream: { param: 'stream' },
  temperature: { param: 'temperature' },
  top_p: { param: 'top_p' },
  parallel_tool_calls: { param: 'parallel_tool_calls' },
  store: { param: 'store' },
  metadata: { param: 'metadata' },
  user: { param: 'user' },
  prompt_cache_key: { param: 'prompt_cache_key' },
  safety_identifier: { param: 'safety_identifier' },
  service_tier: { param: 'service_tier' },
};

function usage(responsesUsage: any) {
  if (!responsesUsage) return undefined;
  const input = responsesUsage.input_tokens || 0;
  const output = responsesUsage.output_tokens || 0;
  return {
    prompt_tokens: input,
    completion_tokens: output,
    total_tokens: responsesUsage.total_tokens || input + output,
    prompt_tokens_details: {
      cached_tokens: responsesUsage.input_tokens_details?.cached_tokens || 0,
    },
    completion_tokens_details: {
      reasoning_tokens:
        responsesUsage.output_tokens_details?.reasoning_tokens || 0,
    },
  };
}

function finishReason(response: any, hasToolCalls: boolean) {
  if (hasToolCalls) return 'tool_calls';
  if (response.status !== 'incomplete') return 'stop';
  return response.incomplete_details?.reason === 'content_filter'
    ? 'content_filter'
    : 'length';
}

function chatError(error: any, fallbackMessage: string): ErrorResponse {
  return OpenAIErrorResponseTransform(
    {
      error: {
        message: error?.message || fallbackMessage,
        type: error?.type || 'server_error',
        param: error?.param || null,
        code: error?.code || null,
      },
      provider: AZURE_OPEN_AI,
    },
    AZURE_OPEN_AI
  );
}

export const AzureOpenAIResponsesChatCompleteResponseTransform = (
  response: any,
  responseStatus: number
): ChatCompletionResponse | ErrorResponse => {
  if (
    responseStatus !== 200 ||
    response.error ||
    response.status === 'failed'
  ) {
    return chatError(
      response.error,
      `Responses request failed with status ${response.status || responseStatus}`
    );
  }

  if (!['completed', 'incomplete'].includes(response.status)) {
    return chatError(
      undefined,
      `Responses request ended with unsupported status ${response.status}`
    );
  }

  const calls = (response.output || []).filter(
    (item: any) => item.type === 'function_call'
  );
  const parts = (response.output || [])
    .filter((item: any) => ['message', 'output_message'].includes(item.type))
    .flatMap((item: any) => item.content || []);
  const content = parts
    .filter((part: any) => part.type === 'output_text')
    .map((part: any) => part.text || '')
    .join('');
  const refusal = parts
    .filter((part: any) => part.type === 'refusal')
    .map((part: any) => part.refusal || '')
    .join('');

  return {
    id: response.id,
    object: 'chat.completion',
    created: response.created_at,
    model: response.model,
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: content || null,
          ...(refusal && { refusal }),
          ...(calls.length && {
            tool_calls: calls.map((call: any) => ({
              id: call.call_id || call.id,
              type: 'function',
              function: {
                name: call.name,
                arguments: call.arguments || '',
              },
            })),
          }),
        },
        finish_reason: finishReason(response, calls.length > 0),
        logprobs: null,
      },
    ],
    usage: usage(response.usage),
    ...(response.service_tier && { service_tier: response.service_tier }),
  } as ChatCompletionResponse;
};

function parseEvent(chunk: string) {
  const lines = chunk.trim().split('\n');
  return {
    event: lines
      .find((line) => line.startsWith('event:'))
      ?.slice(6)
      .trim(),
    data: lines
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).trimStart())
      .join('\n'),
  };
}

function updateMetadata(state: StreamState, response: any) {
  if (!response) return;
  state.id = response.id || state.id;
  state.model = response.model || state.model;
  state.created = response.created_at || state.created;
}

function streamChunk(
  state: StreamState,
  choices: Record<string, any>[],
  streamUsage?: any
) {
  return `data: ${JSON.stringify({
    id: state.id || `chatcmpl-${Date.now()}`,
    object: 'chat.completion.chunk',
    created: state.created || Math.floor(Date.now() / 1000),
    model: state.model || '',
    choices,
    ...(streamUsage && { usage: streamUsage }),
  })}\n\n`;
}

function toolIndex(state: StreamState, itemId: string) {
  state.toolIndexes ||= {};
  if (state.toolIndexes[itemId] === undefined) {
    state.toolIndexes[itemId] = Object.keys(state.toolIndexes).length;
  }
  return state.toolIndexes[itemId];
}

function streamError(error: any, fallbackMessage: string) {
  return `data: ${JSON.stringify({
    error: {
      message: error?.message || fallbackMessage,
      type: error?.type || 'server_error',
      param: error?.param || null,
      code: error?.code || null,
    },
  })}\n\ndata: [DONE]\n\n`;
}

export const AzureOpenAIResponsesChatCompleteStreamChunkTransform = (
  responseChunk: string,
  _fallbackId: string,
  state: StreamState,
  _strictOpenAiCompliance: boolean,
  request: Params
): string | undefined => {
  const { event: eventName, data } = parseEvent(responseChunk);
  if (!data || state.done) return undefined;
  if (data === '[DONE]') {
    state.done = true;
    return 'data: [DONE]\n\n';
  }
  const parsed = JSON.parse(data);
  const event = eventName || parsed.type;

  if (event === 'response.created') {
    updateMetadata(state, parsed.response);
    return streamChunk(state, [
      {
        index: 0,
        delta: { role: 'assistant', content: '' },
        finish_reason: null,
      },
    ]);
  }
  if (
    event === 'response.output_item.added' &&
    parsed.item?.type === 'function_call'
  ) {
    state.hasToolCalls = true;
    return streamChunk(state, [
      {
        index: 0,
        delta: {
          tool_calls: [
            {
              index: toolIndex(state, parsed.item.id || parsed.item.call_id),
              id: parsed.item.call_id || parsed.item.id,
              type: 'function',
              function: {
                name: parsed.item.name,
                arguments: parsed.item.arguments || '',
              },
            },
          ],
        },
        finish_reason: null,
      },
    ]);
  }
  if (event === 'response.function_call_arguments.delta') {
    state.hasToolCalls = true;
    return streamChunk(state, [
      {
        index: 0,
        delta: {
          tool_calls: [
            {
              index: toolIndex(state, parsed.item_id || parsed.call_id),
              function: { arguments: parsed.delta || '' },
            },
          ],
        },
        finish_reason: null,
      },
    ]);
  }
  if (event === 'response.output_text.delta') {
    return streamChunk(state, [
      {
        index: 0,
        delta: { content: parsed.delta || '' },
        finish_reason: null,
      },
    ]);
  }
  if (event === 'response.refusal.delta') {
    return streamChunk(state, [
      {
        index: 0,
        delta: { refusal: parsed.delta || '' },
        finish_reason: null,
      },
    ]);
  }
  if (event === 'response.completed' || event === 'response.incomplete') {
    updateMetadata(state, parsed.response);
    const finalChunk = streamChunk(state, [
      {
        index: 0,
        delta: {},
        finish_reason: finishReason(
          parsed.response,
          Boolean(state.hasToolCalls)
        ),
      },
    ]);
    const usageChunk = (request as any).stream_options?.include_usage
      ? streamChunk(state, [], usage(parsed.response.usage))
      : '';
    state.done = true;
    return `${finalChunk}${usageChunk}data: [DONE]\n\n`;
  }
  if (
    event === 'response.failed' ||
    event === 'response.cancelled' ||
    event === 'error'
  ) {
    state.done = true;
    const error = parsed.response?.error || parsed.error || parsed;
    return streamError(error, `Responses stream failed with event ${event}`);
  }
  return undefined;
};
