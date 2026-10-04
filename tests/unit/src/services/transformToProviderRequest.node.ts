import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import {
  AZURE_OPEN_AI,
  BEDROCK,
  GOOGLE_VERTEX_AI,
  OPEN_AI,
} from '../../../../src/globals';
import { BedrockConverseAnthropicChatCompletionsParams } from '../../../../src/providers/bedrock/chatComplete';
import { Params } from '../../../../src/types/requestBody';
import { transformToProviderRequest } from '../../../../src/services/transformToProviderRequest';

const chatRequest = (provider: string, params: Params) =>
  JSON.parse(
    JSON.stringify(
      transformToProviderRequest(
        provider,
        params,
        params,
        'chatComplete',
        {},
        {
          provider,
        }
      )
    )
  );

describe('explicit chat parameters', () => {
  for (const top_k of [0, 40]) {
    it(`forwards OpenAI-compatible top_k=${top_k} unchanged`, () => {
      assert.equal(
        chatRequest(OPEN_AI, { model: 'example', top_k }).top_k,
        top_k
      );
    });
  }

  for (const value of [true, false]) {
    it(`forwards Azure parallel_tool_calls=${value}`, () => {
      const params = { model: 'example', parallel_tool_calls: value };
      assert.equal(
        chatRequest(AZURE_OPEN_AI, params).parallel_tool_calls,
        value
      );
    });
  }

  for (const provider of [OPEN_AI, AZURE_OPEN_AI]) {
    it(`does not inject optional ${provider} parameters`, () => {
      const request = chatRequest(provider, { model: 'example' });
      assert.equal('top_k' in request, false);
      assert.equal('parallel_tool_calls' in request, false);
    });
  }

  it('still filters unrelated parameters', () => {
    const params = { model: 'example', unknown_parameter: true };
    assert.equal('unknown_parameter' in chatRequest(OPEN_AI, params), false);
  });
});

describe('Claude Converse parameters', () => {
  const model = 'us.anthropic.claude-sonnet-4-6';
  const json_schema = {
    name: 'answer',
    description: 'A short answer',
    strict: true,
    schema: {
      type: 'object',
      properties: { answer: { type: 'string' } },
      required: ['answer'],
      additionalProperties: false,
    },
  };

  it('maps JSON schema to the Converse output format, not native response_format', () => {
    const params: Params = {
      model,
      response_format: { type: 'json_schema', json_schema },
    };
    const request = chatRequest(BEDROCK, params);
    assert.deepEqual(request.outputConfig, {
      textFormat: {
        type: 'json_schema',
        structure: {
          jsonSchema: {
            name: json_schema.name,
            description: json_schema.description,
            schema: JSON.stringify(json_schema.schema),
          },
        },
      },
    });
    assert.equal('response_format' in request, false);
    assert.equal('additionalModelRequestFields' in request, false);
  });

  for (const type of ['json_object', 'text'] as const) {
    it(`does not invent a schema for ${type}`, () => {
      const request = chatRequest(BEDROCK, {
        model,
        response_format: { type },
      });
      assert.equal('outputConfig' in request, false);
      assert.equal('response_format' in request, false);
    });
  }

  it('preserves effort, thinking and existing native fields alongside structured output', () => {
    const params = {
      model,
      thinking: { type: 'enabled', budget_tokens: 1024 },
      output_config: { effort: 'low' },
      additionalModelRequestFields: { metadata: { user_id: 'example' } },
      response_format: { type: 'json_schema', json_schema },
    } satisfies BedrockConverseAnthropicChatCompletionsParams;
    const request = chatRequest(BEDROCK, params);
    assert.deepEqual(request.additionalModelRequestFields, {
      metadata: { user_id: 'example' },
      thinking: params.thinking,
      output_config: { effort: 'low' },
    });
    assert.equal(
      request.outputConfig.textFormat.structure.jsonSchema.schema,
      JSON.stringify(json_schema.schema)
    );
  });

  it('forwards effort when it is the only native option', () => {
    const params = {
      model,
      output_config: { effort: 'high' },
    } satisfies BedrockConverseAnthropicChatCompletionsParams;
    assert.deepEqual(
      chatRequest(BEDROCK, params).additionalModelRequestFields,
      {
        output_config: { effort: 'high' },
      }
    );
  });

  it('does not inject effort or a schema when omitted', () => {
    const request = chatRequest(BEDROCK, { model, max_tokens: 64 });
    assert.equal('outputConfig' in request, false);
    assert.equal('additionalModelRequestFields' in request, false);
    assert.deepEqual(request.inferenceConfig, { maxTokens: 64 });
  });

  it('does not apply Anthropic output settings to other Bedrock models', () => {
    const params = {
      model: 'amazon.nova-2-lite-v1:0',
      output_config: { effort: 'low' },
      response_format: { type: 'json_schema', json_schema },
    } satisfies BedrockConverseAnthropicChatCompletionsParams;
    const request = chatRequest(BEDROCK, params);
    assert.equal('outputConfig' in request, false);
    assert.equal('additionalModelRequestFields' in request, false);
  });
});

describe('Vertex Gemini system messages', () => {
  it('sends every system message and text part as systemInstruction', () => {
    const request = chatRequest(GOOGLE_VERTEX_AI, {
      model: 'gemini-3-flash-preview',
      messages: [
        { role: 'system', content: 'Answer in German.' },
        {
          role: 'system',
          content: [
            { type: 'text', text: 'Be brief.' },
            { type: 'text', text: 'End with BANANE.' },
          ],
        },
        { role: 'user', content: 'Name a colour.' },
      ],
    });
    assert.deepEqual(request.systemInstruction, {
      parts: [
        { text: 'Answer in German.' },
        { text: 'Be brief.' },
        { text: 'End with BANANE.' },
      ],
      role: 'system',
    });
  });
});
