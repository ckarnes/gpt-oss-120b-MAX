import type { ModelsList, OllamaTagResponse } from './types';
import { generateRequestId, createErrorResponse } from './errors';
import { OLLAMA_HOST, OLLAMA_API_KEY } from './config';

export const handleModelsInternal1 = async (): Promise<ModelsList> => {
  try {
    const response = await fetch(`${OLLAMA_HOST}/api/tags`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${OLLAMA_API_KEY}`,
      },
    });

    if (!response.ok) {
      throw new Error(`Ollama request failed: ${response.status} ${response.statusText}`);
    }

    const ollamaResponse: OllamaTagResponse = await response.json();

    const models = ollamaResponse.models?.map((model) => ({
      name: model.name,
      model: 'model' as const,
      modified_at: Math.floor(new Date(model.modified_at || Date.now()).getTime() / 1000),
      size: '',
      owned_by: 'ollama',
      permission: [],
      root: model.name,
      parent: null
    })) || [];

    return {
      models: ollamaResponse.models
    };
  } catch (error) {
    console.error('Models error:', error);
    throw error;
  }
};

export const handleModels1 = async (req: Request): Promise<Response> => {
  const requestId = generateRequestId();
  const responseHeaders = {
    'Content-Type': 'application/json',
    'x-request-id': requestId,
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization, Organization',
    'x-ratelimit-limit-requests': '10000',
    'x-ratelimit-remaining-requests': '9999',
    'x-ratelimit-reset-requests': new Date(Date.now() + 60000).toISOString(),
  };

  try {
    const modelsData = await handleModelsInternal1();
    return new Response(JSON.stringify(modelsData), {
      headers: responseHeaders
    });
  } catch (error) {
    console.error('Models error:', error);
    return createErrorResponse(
      (error as Error).message || 'Failed to fetch models',
      'internal_server_error',
      500
    );
  }
};
