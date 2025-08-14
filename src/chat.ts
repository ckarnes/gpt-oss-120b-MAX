/* ==============================================================
   chatHandler.ts
   --------------------------------------------------------------
   * Handles POST /v1/chat/completions (streaming & non‑streaming)
   * Normalises a boolean `tools` field (`tools:true`) → empty array.
   * Keeps a real `tools` array (if provided) and forwards it to Ollama.
   * Adds missing `tool_call_id` on tool‑result messages and missing
     `id` on assistant.tool_calls (so the OpenAI validator passes).
   * **Tool results are turned into normal assistant messages** and
     a short “you can now answer” note is appended – this gives the
     model a clear cue to stop calling tools.
   * **Guard against infinite loops:** after `MAX_TOOL_ITERATIONS`
     tool results the proxy stops forwarding and returns a final
     answer (`Agent stopped due to max iterations.`).  Adjust the
     constant if you need a larger limit.
   * Deep debug logging (raw payload, tools, each message) and optional
     `debug` field in the response (`INCLUDE_DEBUG_IN_RESPONSE=true`).
   * Streaming returns raw NDJSON (exactly the format you see with
     `curl -N …`).
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
   Configurable guard – how many tool results we allow per turn
-------------------------------------------------------------- */
const MAX_TOOL_ITERATIONS = 5;               // change if you need more

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

/* --------------------------------------------------------------
   Helper – deep dump of any tool‑related data in the request
-------------------------------------------------------------- */
function dumpToolInfo(requestId: string, req: OpenAIChatRequest, rawPayload?: string) {
  // 1️⃣ Raw payload (first 500 chars)
  if (rawPayload) {
    debug.info('Raw request payload (first 500 chars)', {
      requestId,
      payload: rawPayload.slice(0, 500),
    });
  }

  // 2️⃣ Tools array (if present)
  if (Array.isArray(req.tools)) {
    debug.info('Tools array (count)', { requestId, count: req.tools.length });
    req.tools.forEach((tool, idx) => {
      debug.info(`Tool #${idx + 1}`, {
        requestId,
        tool,
      });
    });
  } else {
    debug.info('No tools array (or it was a boolean shortcut)', { requestId });
  }

  // 3️⃣ Per‑message inspection
  if (Array.isArray(req.messages)) {
    req.messages.forEach((msg, idx) => {
      const base = {
        requestId,
        index: idx,
        role: msg.role,
        contentSnippet:
          typeof msg.content === 'string' ? msg.content.slice(0, 200) : undefined,
      };

      if (msg.role === 'assistant' && (msg as any).tool_calls) {
        const toolCalls = (msg as any).tool_calls;
        debug.info('Assistant message with tool_calls', {
          ...base,
          toolCalls: toolCalls.map((tc: any) => ({
            id: tc.id,
            functionName: tc.function?.name,
            arguments: tc.function?.arguments,
          })),
        });
      } else if (msg.role === 'tool') {
        debug.info('Tool result message', {
          ...base,
          tool_call_id: (msg as any).tool_call_id,
        });
      } else {
        debug.info('Regular message', base);
      }
    });
  } else {
    debug.warn('Messages field is not an array', { requestId });
  }
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

  // --------------------------------------------------------------
  // 5️⃣ Normalise a boolean shortcut (`tools:true`) – keep real array
  // --------------------------------------------------------------
  if (openAIReq.tools === true) {
    (openAIReq as any).tools = []; // validator wants an array
    debug.info('Normalized tools:true → tools: []');
  }

  // --------------------------------------------------------------
  // 6️⃣ Ensure every tool‑result message has a tool_call_id
  // --------------------------------------------------------------
  if (Array.isArray(openAIReq.messages)) {
    let placeholderIdx = 1;
    for (const msg of openAIReq.messages) {
      // 6.1️⃣ Tool result messages (`role: "tool"`)
      if (msg.role === 'tool' && (msg as any).tool_call_id == null) {
        (msg as any).tool_call_id = `generated-${placeholderIdx++}`;
        debug.info('Added missing tool_call_id to tool message', {
          index: openAIReq.messages.indexOf(msg),
          generatedId: (msg as any).tool_call_id,
        });
      }

      // 6.2️⃣ Assistant messages that contain `tool_calls`
      if (
        msg.role === 'assistant' &&
        (msg as any).tool_calls &&
        Array.isArray((msg as any).tool_calls)
      ) {
        for (const tc of (msg as any).tool_calls) {
          if (tc.id == null) {
            tc.id = `generated-${placeholderIdx++}`;
            debug.info('Added missing id to assistant.tool_calls', {
              generatedId: tc.id,
            });
          }
        }
      }
    }
  }

  // --------------------------------------------------------------
  // 7️⃣ DEEP DEBUG – dump everything that might involve tools
  // --------------------------------------------------------------
  dumpToolInfo(requestId, openAIReq, rawClientPayload);

  // ------------------- 8️⃣ Log parsed request (post‑normalisation) ---
  debug.info('Parsed OpenAI request', {
    model: openAIReq.model,
    stream: openAIReq.stream,
    tools: !!openAIReq.tools,
    messagesCount: openAIReq.messages?.length,
  });

  // ------------------- 9️⃣ Generic validation -------------------
  const requestValidation = validateRequest(openAIReq);
  if (requestValidation) {
    // Log the *body* of the validation error for easier debugging.
    let errBody = '';
    try {
      errBody = await requestValidation.clone().text();
    } catch (_) {
      /* ignore */
    }
    debug.warn('Request validation failed', {
      status: requestValidation.status,
      body: errBody,
    });
    return requestValidation; // already a proper Response
  }

  // ------------------- 🔟 Validate model exists ----------------
  const modelsResponse = await handleModelsInternal();
  const availableModels = modelsResponse.data.map((m) => m.id);
  if (!validateModel(openAIReq.model, availableModels)) {
    const msg = `The model '${openAIReq.model}' does not exist`;
    debug.warn(msg, { availableModels });
    return buildErrorResponse(msg, 'invalid_request_error', 404, 'model');
  }

  // ------------------- 1️⃣1️⃣ Parameter validation -----------------
  const paramValidation = validateParameters(openAIReq);
  if (paramValidation) {
    debug.warn('Parameter validation failed', paramValidation);
    return paramValidation; // already a Response
  }

  // ------------------- 1️⃣2️⃣ Destructure request -----------------
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
    tools,                // keep the (possibly empty) array
  } = openAIReq;

  // --------------------------------------------------------------
  // 13️⃣ Convert tool‑result messages into normal assistant messages
  //     and add a “you can now answer” hint.
  // --------------------------------------------------------------
  // Count how many tool‑result messages we already have.
  const toolResultCount = messages.filter((m) => m.role === 'tool').length;

  // If we have exceeded the safe limit, we abort and return a final answer.
  if (toolResultCount >= MAX_TOOL_ITERATIONS) {
    debug.warn('Maximum tool iterations reached – returning final answer', {
      requestId,
      toolResultCount,
    });

    const finalAnswer: OpenAIChatResponse = {
      id: generateId(),
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model,
      system_fingerprint: 'fp_ollama_proxy',
      choices: [
        {
          index: 0,
          message: {
            role: 'assistant',
            content:
              'Agent stopped due to max iterations. Here is the best answer I can provide based on the information I have.',
          },
          finish_reason: 'stop',
        },
      ],
      usage: {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
      },
    };

    // If you want this to be a streaming response, you could construct
    // the SSE chunks manually, but the non‑streaming path is sufficient
    // for the safety‑guard case.
    return new Response(JSON.stringify(finalAnswer), {
      headers: responseHeaders,
    });
  }

  // Normal case – transform each tool result into an assistant message.
  const transformedMessages = messages.map((msg) => {
    if (msg.role === 'tool') {
      const content = (msg as any).content ?? '';
      debug.info('Converting tool result to assistant message', {
        requestId,
        tool_call_id: (msg as any).tool_call_id,
        contentSnippet: typeof content === 'string' ? content.slice(0, 200) : '',
      });

      // 1️⃣ The raw tool output (so the model can read it)
      const assistantMsg = {
        role: 'assistant' as const,
        content,
      };

      // 2️⃣ A tiny hint that the model now has the answer and can stop.
      //    This line is optional – many LLMs understand the raw JSON,
      //    but adding a clear cue reduces the chance of a loop.
      const hintMsg = {
        role: 'assistant' as const,
        content: 'You have the tool result above. Please answer the user and do not call any more tools unless necessary.',
      };

      // Return both messages in order.
      return [assistantMsg, hintMsg];
    }
    // Regular messages stay as‑is.
    return msg;
  }).flat(); // flatten the array because tool results become two entries

  // ------------------- 1️⃣3️⃣ Convert (transformed) messages --------------------
  const ollamaMessages = convertToOllamaMessages(transformedMessages);
  debug.info('Converted messages to Ollama format', {
    originalCount: transformedMessages.length,
    ollamaCount: ollamaMessages.length,
  });

  // ------------------- 1️⃣4️⃣ Build Ollama options ----------------
  const options: OllamaOptions = {};
  if (temperature !== undefined) options.temperature = temperature;
  if (max_tokens !== undefined) options.num_predict = max_tokens;
  if (top_p !== undefined) options.top_p = top_p;
  if (frequency_penalty !== undefined) options.frequency_penalty = frequency_penalty;
  if (presence_penalty !== undefined) options.presence_penalty = presence_penalty;
  if (stop !== undefined) options.stop = Array.isArray(stop) ? stop : [stop];

  debug.info('Ollama request options', options);

  // ------------------- 1️⃣5️⃣ Assemble final Ollama request -------
  const ollamaRequest: OllamaChatRequest = {
    model,
    messages: ollamaMessages,
    stream,               // keep whatever the client set (true/false)
  };

  // Forward the tools array (if any)
  if (Array.isArray(tools) && tools.length > 0) {
    // @ts-ignore – Ollama accepts the same shape as OpenAI
    (ollamaRequest as any).tools = tools;
    debug.info('Added tools to Ollama request', { count: tools.length });
  }

  if (Object.keys(options).length) ollamaRequest.options = options;
  if (response_format?.type === 'json_object') ollamaRequest.format = 'json';

  // Log the **complete** request that will be sent to Ollama
  debug.info('Prepared Ollama request (final)', { ollamaRequest });

  // ------------------- 1️⃣6️⃣ Dispatch ----------------------------
  if (stream) {
    // Streaming – forward raw NDJSON from Ollama.
    return handleStreamingChat(
      ollamaRequest,
      model,
      openAIReq,
      requestId,
      rawClientPayload,
    );
  }

  // Non‑streaming – single JSON response.
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

    // Try to parse as a single JSON object; fallback to NDJSON if needed.
    let parsed: any;
    try {
      parsed = JSON.parse(rawOllamaBody);
    } catch {
      // NDJSON fallback
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

    // Build OpenAI‑compatible response
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

    // Optional debug block
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
   Streaming implementation – raw NDJSON forwarding
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

    // Forward the raw NDJSON stream unchanged.
    const streamingHeaders = {
      'Content-Type': 'application/json', // raw NDJSON lines
      'x-request-id': requestId,
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
      // Optional rate‑limit headers for consistency
      'x-ratelimit-limit-requests': '10000',
      'x-ratelimit-remaining-requests': '9999',
      'x-ratelimit-reset-requests': new Date(Date.now() + 60000).toISOString(),
    };

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
   * The **tool‑result → assistant** conversion now adds a tiny hint
     that the model may answer and stop.  This alone often prevents
     the model from looping.
   * If you still see repeated calls, the **MAX_TOOL_ITERATIONS**
     guard will cut the conversation after the configured number of
     tool results and return a clear “max iterations” message.
   * All deep‑debug logging remains, and the optional `debug` field
     (enabled with `INCLUDE_DEBUG_IN_RESPONSE=true`) still works.
-------------------------------------------------------------- */
