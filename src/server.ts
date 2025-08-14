import { serve } from 'bun';
import { loadEnvFile, PORT, OLLAMA_API_KEY, KNOWN_ENDPOINTS } from './config';
import { handleChatCompletions } from './chat';
import { handleModels } from './models';
import { handleModels1 } from './models1';
import { createErrorResponse } from './errors';

loadEnvFile();

if (!OLLAMA_API_KEY) {
  console.error('OLLAMA_API_KEY environment variable is required');
  process.exit(1);
}

const server = serve({
  port: PORT,
  idleTimeout: 255,
  async fetch(req) {
    const url = new URL(req.url);

    if (req.method === 'OPTIONS') {
      return new Response(null, {
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type, Authorization',
        },
      });
    }

    console.log(`${new Date().toISOString()} - ${req.method} ${url.pathname}`);

    if (url.pathname === '/v1/chat/completions' && req.method === 'POST') {
      return handleChatCompletions(req);
    }

    if (url.pathname === '/api/chat' && req.method === 'POST') {
      return handleChatCompletions(req);
    }  

    if (url.pathname === '/v1/models' && req.method === 'GET') {
      return handleModels(req);
    }

    if (url.pathname === '/api/tags' && req.method === 'GET') {
      return handleModels1(req);
    }

    if (url.pathname === '/v1/completions' && req.method === 'POST') {
      return createErrorResponse(
        'The Completions API is deprecated. Please use /v1/chat/completions instead.',
        'invalid_request_error',
        400
      );
    }

    if (url.pathname === '/v1/embeddings' && req.method === 'POST') {
      return createErrorResponse(
        'Embeddings are not supported by this proxy.',
        'invalid_request_error',
        400
      );
    }

    if (url.pathname === '/' && req.method === 'GET') {
      return new Response(JSON.stringify({
        message: 'OpenAI-compatible API server',
        version: '1.0.0',
        endpoints: KNOWN_ENDPOINTS.map(e => `${e.method} ${e.path}`)
      }), {
        headers: { 'Content-Type': 'application/json' }
      });
    }

    return new Response(JSON.stringify({
      error: {
        message: 'Not found',
        type: 'invalid_request_error'
      },
      endpoints: KNOWN_ENDPOINTS.map(e => `${e.method} ${e.path}`)
    }), {
      status: 404,
      headers: { 'Content-Type': 'application/json' }
    });
  },
});

console.log('🚀 OpenAI-compatible v1 API server running on http://localhost:3304');
console.log('');
console.log('📡 SUPPORTED ENDPOINTS:');
console.log('  ✅ POST /v1/chat/completions    (Full OpenAI compatibility)');
console.log('  ✅ GET  /v1/models              (List available models)');
console.log('  ❌ POST /v1/completions         (Deprecated - use chat/completions)');
console.log('  ❌ POST /v1/embeddings          (Not supported)');
console.log('');
console.log('⚙️  SUPPORTED FEATURES:');
console.log('  • Streaming & Non-streaming responses');
console.log('  • Tool/Function calling support');
console.log('  • JSON mode & Structured outputs');
console.log('  • All OpenAI parameters (temperature, top_p, max_tokens, etc.)');
console.log('  • Proper error handling with OpenAI-style envelopes');
console.log('  • Rate limit headers & request IDs');
console.log('');
console.log('🔧 IDE CONFIGURATION (RooCode/KiloCode):');
console.log('  Base URL: http://localhost:3304/v1');
console.log('  API Key: any-valid-key (flexible validation)');
console.log('');
console.log('🔗 Proxying to Ollama with full OpenAI v1 API compatibility');
