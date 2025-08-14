/* ==============================================================
   chatHandler.ts
   --------------------------------------------------------------
   * Supports both non‑streaming and streaming (/v1/chat/completions).
   * Streaming now returns **plain NDJSON** – the same format that
     Ollama’s own API returns, which n8n can consume.
   * Extensive console‑debug logging.
   * Optional `debug` section in every JSON response when
     INCLUDE_DEBUG_IN_RESPONSE=true.
   ============================================================== */

import type {
  OpenAIChatRequest,
  OpenAIChatResponse,
  OpenAIStreamChunk,
  OllamaChatRequest,
  OllamaOptions,
} from './types';
import { convertToOllamaMessages } from './utils';
import {
  generateId,
  generateRequestId,
} from './errors';
import { validateAuth } from './auth';
import {
  validateRequest,
  validateParameters,
  validateModel,
} from './validation';
import { handleModelsInternal } from './models';
import { OLLAMA_HOST, OLLAMA_API_KEY } from './config';

/* --------------------------------------------------------------
   Debug helper – toggle with NODE_ENV or a dedicated flag
-------------------------------------------------------------- */
const ENABLE_DEBUG = process.env.NODE_ENV !== 'production';
const INCLUDE_DEBUG_IN_RESPONSE =
  process.env.INCLUDE_DEBUG_IN_RESPONSE === 'true';

const debug = {
  info: (...args: any[]) => ENABLE_DEBUG && console.log('[DEBUG][INFO]', ...args),
  warn: (...args: any[]) => ENABLE_DEBUG && console.warn('[DEBUG][WARN]', ...args),
  error: (...args: any[]) => ENABLE_DEBUG && console.error('[DEBUG][ERROR]', ...args),
};

/* --------------------------------------------------------------
   Helper – safe JSON parsing that also returns the raw request body
-------------------------------------------------------------- */
async function safeParseJson<T>(req: Request): Promise<{
  body?: T;
  raw?: string;
  error?: Error;
}> {
  try {
    const raw = await req.text();            // keep raw payload for debugging
    const body = JSON.parse(raw) as T;
    return { body, raw };
  } catch (e) {
    return { error: e as Error, raw: undefined };
  }
}

/* --------------------------------------------------------------
   Helper – build a JSON error response (adds optional debug info)
-------------------------------------------------------------- */
function buildErrorResponse(
  message: string,
  type: string = 'server_error',
  status: number = 500,
  param?: string,
  debugInfo?: Record<string, any>,
): Response {
  const payload: any = {
    error: {
      message,
      type,
      param,
      code: status,
    },
  };

  if (INCLUDE_DEBUG_IN_RESPONSE && debugInfo) {
    payload.debug = debugInfo;
  }

  return new Response(JSON.stringify(payload), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

/* ==============================================================
   Main entry point – POST /v1/chat/completions
============================================================== */
export const handleChatCompletions = async (req: Request): Promise<Response> => {
  const requestStart = Date.now();

  // ------------------- 1️⃣ Log incoming request -----------------
  debug.info('Incoming HTTP request', {
    method: req.method,
    url: req.url,
    headers: Object.fromEntries(req.headers.entries()),
  });

  // ------------------- 2️⃣ Authentication -----------------------
  const authValidation = validateAuth(req);
  if (!authValidation.valid) {
    debug.warn('Auth validation failed', authValidation);
    return authValidation.error as Response;
  }

  const requestId = generateRequestId();
  debug.info('Auth succeeded', { requestId });

  // ------------------- 3️⃣ Common response headers -------------
  const responseHeaders = {
    'Content-Type': 'application/json',
    'x-request-id': requestId,
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers':
      'Content-Type, Authorization, Organization',
    'x-ratelimit-limit-requests': '10000',
    'x-ratelimit-remaining-requests': '9999',
    'x-ratelimit-reset-requests': new Date(
      Date.now() + 60000,
    ).toISOString(),
  };

  // ------------------- 4️⃣ Safe request parsing -----------------
  const { body, raw: rawClientPayload, error: parseError } =
    await safeParseJson<OpenAIChatRequest>(req);

  if (parseError) {
    debug.error('Failed to parse incoming JSON', {
      rawPayload: rawClientPayload,
      error: parseError,
    });

    return buildErrorResponse(
      'Invalid JSON payload',
      'invalid_request_error',
      400,
      undefined,
      {
        requestId,
        rawPayload: rawClientPayload,
        errorMessage: parseError.message,
      },
    );
  }

  const openAIReq = body!; // safe – parsing succeeded

  // ------------------- 5️⃣ Log parsed request -------------------
  debug.info('Parsed OpenAI request', {
    model: openAIReq.model,
    stream: openAIReq.stream,
    tools: !!openAIReq.tools,
    messagesCount: openAIReq.messages?.length,
  });

  // ------------------- 6️⃣ Generic validation -------------------
  const requestValidation = validateRequest(openAIReq);
  if (requestValidation) {
    debug.warn('Request validation failed', requestValidation);
    return requestValidation; // already a Response
  }

  // ------------------- 7️⃣ Validate model exists ----------------
  const modelsResponse = await handleModelsInternal();
  const availableModels = modelsResponse.data.map((m) => m.id);
  if (!validateModel(openAIReq.model, availableModels)) {
    const msg = `The model '${openAIReq.model}' does not exist`;
    debug.warn(msg, { availableModels });
    return buildErrorResponse(msg, 'invalid_request_error', 404, 'model');
  }

  // ------------------- 8️⃣ Parameter validation -----------------
  const paramValidation = validateParameters(openAIReq);
  if (paramValidation) {
    debug.warn('Parameter validation failed', paramValidation);
    return paramValidation; // already a Response
  }

  // ------------------- 9️⃣ Destructure request -------------------
  const {
    model,
    messages,
    stream = false,
    temperature,
    max_tokens,
    top_p,
    frequency_penalty,
    presence_penalty,
    stop,
    response_format,
  } = openAIReq;

  // ------------------- 10️⃣ Convert messages --------------------
  const ollamaMessages = convertToOllamaMessages(messages);
  debug.info('Converted messages to Ollama format', {
    originalCount: messages.length,
    ollamaCount: ollamaMessages.length,
  });

  // ------------------- 11️⃣ Build Ollama options ----------------
  const options: OllamaOptions = {};
  if (temperature !== undefined) options.temperature = temperature;
  if (max_tokens !== undefined) options.num_predict = max_tokens;
  if (top_p !== undefined) options.top_p = top_p;
  if (frequency_penalty !== undefined) options.frequency_penalty = frequency_penalty;
  if (presence_penalty !== undefined) options.presence_penalty = presence_penalty;
  if (stop !== undefined) options.stop = Array.isArray(stop) ? stop : [stop];

  debug.info('Ollama request options', options);

  // ------------------- 12️⃣ Assemble final Ollama request -------
  const ollamaRequest: OllamaChatRequest = {
    model,
    messages: ollamaMessages,
    stream,               // keep whatever the client asked for
  };
  if (Object.keys(options).length) ollamaRequest.options = options;
  if (response_format?.type === 'json_object') ollamaRequest.format = 'json';

  debug.info('Prepared Ollama request', { ollamaRequest });

  // ------------------- 13️⃣ Dispatch ----------------------------
  if (stream) {
    // ---- STREAMING ----
    return handleStreamingChat(
      ollamaRequest,
      model,
      openAIReq,
      requestId,
      rawClientPayload,
    );
  }

  // ---- NON‑STREAMING ----
  return handleNonStreamingChat(
    ollamaRequest,
    model,
    openAIReq,
    requestId,
    responseHeaders,
    rawClientPayload,
  );
};

/* ==============================================================
   Non‑streaming implementation (unchanged apart from debug)
============================================================== */
export const handleNonStreamingChat = async (
  ollamaRequest: OllamaChatRequest,
  model: string,
  originalRequest: OpenAIChatRequest,
  requestId: string,
  responseHeaders: Record<string, string>,
  rawClientRequestBody?: string,
): Promise<Response> => {
  const start = Date.now();
  debug.info('→ Non‑streaming request start', { requestId, model });

  try {
    const url = `${OLLAMA_HOST}/api/chat`;
    debug.info('Fetching Ollama (non‑stream)', { url });

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${OLLAMA_API_KEY}`,
      },
      body: JSON.stringify(ollamaRequest),
    });

    debug.info('Ollama HTTP response', {
      status: response.status,
      ok: response.ok,
      contentType: response.headers.get('content-type'),
    });

    if (!response.ok) {
      const errBody = await response.text();
      debug.error('Ollama returned non‑2xx', {
        status: response.status,
        body: errBody,
      });

      return buildErrorResponse(
        `Ollama error ${response.status}: ${errBody}`,
        'invalid_request_error',
        response.status,
        'model',
        {
          requestId,
          ollamaRequest,
          ollamaRawError: errBody,
          originalClientRequest: originalRequest,
          rawClientRequestBody,
        },
      );
    }

    const rawOllamaBody = await response.text();
    debug.info('Raw Ollama body (first 500 chars)', {
      body: rawOllamaBody.slice(0, 500),
    });

    // Try to parse as a single JSON object; if that fails, fall back to NDJSON.
    let parsed: any;
    try {
      parsed = JSON.parse(rawOllamaBody);
    } catch {
      const lines = rawOllamaBody
        .split('\n')
        .map((l) => l.trim())
        .filter(Boolean);
      const ndjson: any[] = [];

      for (const line of lines) {
        try {
          ndjson.push(JSON.parse(line));
        } catch (e) {
          debug.error('Failed to parse NDJSON line', { line, error: e });
          return buildErrorResponse(
            'Ollama returned malformed NDJSON – see logs',
            'internal_server_error',
            500,
            undefined,
            { requestId, rawOllamaBody, offendingLine: line },
          );
        }
      }

      // Grab the last content field we can find.
      let finalContent: string | undefined;
      for (let i = ndjson.length - 1; i >= 0; i--) {
        if (ndjson[i].message?.content !== undefined) {
          finalContent = ndjson[i].message.content;
          break;
        }
      }

      parsed = {
        message: { content: finalContent ?? '' },
        prompt_eval_count: ndjson[0]?.prompt_eval_count,
        eval_count: ndjson[ndjson.length - 1]?.eval_count,
      };
    }

    // Build the OpenAI‑compatible payload.
    let content = parsed.message?.content || '';
    if (originalRequest.response_format?.type === 'json_object') {
      try {
        JSON.parse(content);
      } catch {
        content = JSON.stringify({ response: content });
      }
    }
    if (!content && !originalRequest.tools) {
      content = 'Response received from model.';
    }

    const openaiResponse: OpenAIChatResponse = {
      id: generateId(),
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model,
      system_fingerprint: 'fp_ollama_proxy',
      choices: [
        {
          index: 0,
          message: { role: 'assistant', content },
          finish_reason: 'stop',
        },
      ],
      usage: {
        prompt_tokens: parsed.prompt_eval_count || 0,
        completion_tokens: parsed.eval_count || 0,
        total_tokens:
          (parsed.prompt_eval_count || 0) + (parsed.eval_count || 0),
      },
    };

    // Optional debug block.
    if (INCLUDE_DEBUG_IN_RESPONSE) {
      // @ts-ignore – non‑standard field for debugging only
      (openaiResponse as any).debug = {
        requestId,
        originalClientRequest: originalRequest,
        rawClientRequestBody,
        ollamaRequest,
        rawOllamaResponse: rawOllamaBody,
        parsedOllamaResponse: parsed,
      };
    }

    const duration = Date.now() - start;
    debug.info('← Non‑streaming request finished', {
      requestId,
      durationMs: duration,
      usage: openaiResponse.usage,
    });

    return new Response(JSON.stringify(openaiResponse), {
      headers: responseHeaders,
    });
  } catch (error) {
    debug.error('Non‑streaming chat unexpected error', error);
    return buildErrorResponse(
      (error as Error).message || 'Internal server error',
      'internal_server_error',
      500,
      undefined,
      {
        requestId,
        stack: (error as Error).stack,
        originalClientRequest: originalRequest,
        rawClientRequestBody,
        ollamaRequest,
      },
    );
  }
};

/* ==============================================================
   Streaming implementation – **raw NDJSON forwarding**
   --------------------------------------------------------------
   The proxy now simply pipes Ollama’s streaming response straight
   to the client (no `data:` prefix, no extra SSE wrapper). This
   matches the format you posted with `curl` and is accepted by n8n.
============================================================== */
export const handleStreamingChat = async (
  ollamaRequest: OllamaChatRequest,
  model: string,
  originalRequest: OpenAIChatRequest,
  requestId: string,
  rawClientRequestBody?: string,
): Promise<Response> => {
  const start = Date.now();
  debug.info('→ Streaming request start', { requestId, model });

  // Build the URL – note that `OLLAMA_HOST` may be http or https.
  const url = `${OLLAMA_HOST}/api/chat`;
  debug.info('Fetching Ollama (stream)', { url });

  try {
    const ollamaResp = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${OLLAMA_API_KEY}`,
      },
      body: JSON.stringify(ollamaRequest),
    });

    debug.info('Ollama streaming HTTP response', {
      status: ollamaResp.status,
      ok: ollamaResp.ok,
      contentType: ollamaResp.headers.get('content-type'),
    });

    if (!ollamaResp.ok) {
      const errBody = await ollamaResp.text();
      debug.error('Ollama streaming error', {
        status: ollamaResp.status,
        body: errBody,
      });

      return buildErrorResponse(
        `Ollama error ${ollamaResp.status}: ${errBody}`,
        'invalid_request_error',
        ollamaResp.status,
        'model',
        {
          requestId,
          ollamaRequest,
          ollamaRawError: errBody,
          originalClientRequest: originalRequest,
          rawClientRequestBody,
        },
      );
    }

    // -----------------------------------------------------------------
    // At this point `ollamaResp.body` is a ReadableStream that yields
    // the exact NDJSON lines Ollama sends (e.g. `{"model": "..."}\n`).
    // We **do not** wrap them in `data:`; we forward them unchanged.
    // -----------------------------------------------------------------
    const streamingHeaders = {
      // Keep the content‑type that Ollama uses (plain JSON lines)
      // Clients (n8n, curl -N) understand this as a streaming body.
      'Content-Type': 'application/json',
      'x-request-id': requestId,
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
      // Rate‑limit headers – optional but nice to keep consistent
      'x-ratelimit-limit-requests': '10000',
      'x-ratelimit-remaining-requests': '9999',
      'x-ratelimit-reset-requests': new Date(Date.now() + 60000).toISOString(),
    };

    // If you want to add a final `[DONE]` line, you could pipe the
    // stream through a TransformStream that appends it when the
    // upstream closes. Most clients (including n8n) treat the last
    // JSON line that contains `"done": true` as the terminator, so
    // we leave the stream untouched.
    return new Response(ollamaResp.body!, {
      headers: streamingHeaders,
    });
  } catch (error) {
    debug.error('Streaming handler unexpected error', error);
    return buildErrorResponse(
      (error as Error).message || 'Internal streaming error',
      'internal_server_error',
      500,
      undefined,
      {
        requestId,
        stack: (error as Error).stack,
        originalClientRequest: originalRequest,
        rawClientRequestBody,
        ollamaRequest,
      },
    );
  } finally {
    const totalMs = Date.now() - start;
    debug.info('← Streaming request finished (proxy side)', {
      requestId,
      durationMs: totalMs,
    });
  }
};

/* --------------------------------------------------------------
   NOTE
   * The streaming path now forwards **raw NDJSON** exactly as
     Ollama returns it. This matches the curl output you posted
     and works with n8n’s “Continue on response” streaming mode.
   * If you later decide you need the OpenAI‑style SSE
     (prefixed with `data:`), you can re‑introduce the previous
     `handleStreamingChat` implementation and change the client
     accordingly.
-------------------------------------------------------------- */
