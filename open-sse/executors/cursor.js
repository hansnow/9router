import { BaseExecutor } from "./base.js";
import { PROVIDERS } from "../config/providers.js";
import { HTTP_STATUS } from "../config/runtimeConfig.js";
import {
  generateCursorBody,
  parseConnectRPCFrame,
  extractTextFromResponse
} from "../utils/cursorProtobuf.js";
import { buildCursorHeaders } from "../utils/cursorChecksum.js";
import { estimateUsage } from "../utils/usageTracking.js";
import { FORMATS } from "../translator/formats.js";
import { proxyAwareFetch } from "../utils/proxyFetch.js";
import zlib from "zlib";

// Detect cloud environment
const isCloudEnv = () => {
  if (typeof caches !== "undefined" && typeof caches === "object") return true;
  if (typeof EdgeRuntime !== "undefined") return true;
  return false;
};

// Lazy import http2 (only in Node.js environment)
let http2 = null;
if (!isCloudEnv()) {
  try {
    http2 = await import("http2");
  } catch {
    // http2 not available
  }
}

const COMPRESS_FLAG = {
  NONE: 0x00,
  GZIP: 0x01,
  TRAILER: 0x02,
  GZIP_TRAILER: 0x03
};

const CURSOR_STREAM_DEBUG = process.env.CURSOR_STREAM_DEBUG === "1";
const debugLog = (...args) => {
  if (CURSOR_STREAM_DEBUG) console.log(...args);
};

const FINAL_THINKING_MARKERS = ["<|final|>", "<\uFF5Cfinal\uFF5C>"];
const PSEUDO_TOOL_MARKERS = {
  callsBegin: "<\uFF5Ctool\u2581calls\u2581begin\uFF5C>",
  callsEnd: "<\uFF5Ctool\u2581calls\u2581end\uFF5C>",
  callBegin: "<\uFF5Ctool\u2581call\u2581begin\uFF5C>",
  callEnd: "<\uFF5Ctool\u2581call\u2581end\uFF5C>",
  sep: "<\uFF5Ctool\u2581sep\uFF5C>"
};
const WHITESPACE_SENSITIVE_PSEUDO_TOOL_ARGS = new Set(["content", "old_string", "new_string"]);

function isThinkingTailVisibleModel(model) {
  const modelId = String(model || "").split("/").pop();
  return modelId === "default" || /^composer(?:-|$)/i.test(modelId);
}

function visibleContentFromThinking(thinking) {
  if (!thinking) return "";
  const endTag = "</think>";
  const endIdx = thinking.lastIndexOf(endTag);
  if (endIdx < 0) return "";

  const visible = thinking.slice(endIdx + endTag.length).trimStart();
  const lowerVisible = visible.toLowerCase();
  for (const marker of FINAL_THINKING_MARKERS) {
    const lowerMarker = marker.toLowerCase();
    if (lowerVisible === "" || (lowerMarker.startsWith(lowerVisible) && lowerVisible.length < lowerMarker.length)) {
      return "";
    }
    if (lowerVisible.startsWith(lowerMarker)) {
      return visible.slice(marker.length).trimStart();
    }
  }
  return visible;
}

function normalizePseudoToolArguments(toolName, args) {
  const normalizedArgs = {};
  for (const [key, value] of Object.entries(args)) {
    normalizedArgs[key] = typeof value === "string" && !WHITESPACE_SENSITIVE_PSEUDO_TOOL_ARGS.has(key)
      ? value.trim()
      : value;
  }

  if (toolName === "Read" && normalizedArgs.path && !normalizedArgs.file_path) {
    const { path, ...rest } = normalizedArgs;
    return { ...rest, file_path: path };
  }

  if (toolName === "Glob") {
    const normalized = { ...normalizedArgs };
    if (normalized.glob_pattern && !normalized.pattern) {
      normalized.pattern = normalized.glob_pattern;
      delete normalized.glob_pattern;
    }
    if (normalized.target_directory && !normalized.path) {
      normalized.path = normalized.target_directory;
      delete normalized.target_directory;
    }
    return normalized;
  }

  return normalizedArgs;
}

function parseCursorPseudoToolCalls(content) {
  if (!content || !content.includes(PSEUDO_TOOL_MARKERS.callsBegin)) {
    return { content, toolCalls: [] };
  }

  const begin = content.indexOf(PSEUDO_TOOL_MARKERS.callsBegin);
  const end = content.indexOf(PSEUDO_TOOL_MARKERS.callsEnd, begin + PSEUDO_TOOL_MARKERS.callsBegin.length);
  if (end < 0) {
    return { content: content.slice(0, begin).trimEnd(), toolCalls: [] };
  }

  const before = content.slice(0, begin).trimEnd();
  const after = content.slice(end + PSEUDO_TOOL_MARKERS.callsEnd.length).trimStart();
  const block = content.slice(begin + PSEUDO_TOOL_MARKERS.callsBegin.length, end);
  const toolCalls = [];

  let offset = 0;
  while (offset < block.length) {
    const callStart = block.indexOf(PSEUDO_TOOL_MARKERS.callBegin, offset);
    if (callStart < 0) break;
    const bodyStart = callStart + PSEUDO_TOOL_MARKERS.callBegin.length;
    const callEnd = block.indexOf(PSEUDO_TOOL_MARKERS.callEnd, bodyStart);
    if (callEnd < 0) break;

    const callBody = block.slice(bodyStart, callEnd);
    const firstSep = callBody.indexOf(PSEUDO_TOOL_MARKERS.sep);
    if (firstSep > 0) {
      const toolName = callBody.slice(0, firstSep).trim();
      const args = {};
      const argBlocks = callBody.slice(firstSep).split(PSEUDO_TOOL_MARKERS.sep).filter(Boolean);

      for (const argBlock of argBlocks) {
        const newline = argBlock.indexOf("\n");
        if (newline < 0) continue;
        const key = argBlock.slice(0, newline).trim();
        const value = argBlock.slice(newline + 1);
        if (key) args[key] = value;
      }

      if (toolName) {
        const index = toolCalls.length;
        const normalizedArgs = normalizePseudoToolArguments(toolName, args);
        toolCalls.push({
          id: `call_cursor_${Date.now()}_${index}`,
          type: "function",
          function: {
            name: toolName,
            arguments: JSON.stringify(normalizedArgs)
          }
        });
      }
    }

    offset = callEnd + PSEUDO_TOOL_MARKERS.callEnd.length;
  }

  const stripped = [before, after].filter(Boolean).join(after && before ? "\n" : "");
  return { content: stripped, toolCalls };
}

function decompressPayload(payload, flags) {
  // Check if payload is JSON error (starts with {"error")
  if (payload.length > 10 && payload[0] === 0x7b && payload[1] === 0x22) {
    try {
      const text = payload.toString("utf-8");
      if (text.startsWith('{"error"')) {
        debugLog(`[DECOMPRESS] Detected JSON error, skipping decompression`);
        return payload;
      }
    } catch {}
  }

  if (
    flags === COMPRESS_FLAG.GZIP ||
    flags === COMPRESS_FLAG.TRAILER ||
    flags === COMPRESS_FLAG.GZIP_TRAILER
  ) {
    // Primary: try gzip decompression (standard gzip header 0x1f 0x8b)
    try {
      return zlib.gunzipSync(payload);
    } catch (gzipErr) {
      // Fallback: TRAILER and GZIP_TRAILER frames sometimes use raw zlib deflate format
      try {
        return zlib.inflateSync(payload);
      } catch (deflateErr) {
        // Last resort: try raw deflate (no zlib header)
        try {
          return zlib.inflateRawSync(payload);
        } catch (rawErr) {
          debugLog(
            `[DECOMPRESS ERROR] flags=${flags}, payloadSize=${payload.length}, gzip=${gzipErr.message}, deflate=${deflateErr.message}, raw=${rawErr.message}`
          );
          debugLog(
            `[DECOMPRESS ERROR] First 50 bytes (hex):`,
            payload.slice(0, 50).toString("hex")
          );
          return payload;
        }
      }
    }
  }
  return payload;
}

function createErrorResponse(jsonError) {
  const errorMsg = jsonError?.error?.details?.[0]?.debug?.details?.title
    || jsonError?.error?.details?.[0]?.debug?.details?.detail
    || jsonError?.error?.message
    || "API Error";
  
  const isRateLimit = jsonError?.error?.code === "resource_exhausted";
  
  return new Response(JSON.stringify({
    error: {
      message: errorMsg,
      type: isRateLimit ? "rate_limit_error" : "api_error",
      code: jsonError?.error?.details?.[0]?.debug?.error || "unknown"
    }
  }), {
    status: isRateLimit ? HTTP_STATUS.RATE_LIMITED : HTTP_STATUS.BAD_REQUEST,
    headers: { "Content-Type": "application/json" }
  });
}

export class CursorExecutor extends BaseExecutor {
  constructor() {
    super("cursor", PROVIDERS.cursor);
  }

  buildUrl() {
    return `${this.config.baseUrl}${this.config.chatPath}`;
  }

  buildHeaders(credentials) {
    const accessToken = credentials.accessToken;
    const machineId = credentials.providerSpecificData?.machineId;
    const ghostMode = credentials.providerSpecificData?.ghostMode !== false;

    if (!machineId) {
      throw new Error("Machine ID is required for Cursor API");
    }

    return buildCursorHeaders(accessToken, machineId, ghostMode);
  }

  transformRequest(model, body, stream, credentials) {
    // Messages are already translated by chatCore (claude→openai→cursor)
    // Do NOT call buildCursorRequest again — double-translation drops tool_results
    const messages = body.messages || [];
    const tools = body.tools || [];
    const reasoningEffort = body.reasoning_effort || null;
    // Detect Claude Code UA to force Agent mode (issue #643)
    const ua = credentials?.rawHeaders?.["user-agent"] || "";
    const forceAgentMode = ua.includes("claude-cli") || ua.includes("claude-code") || ua.includes("Claude Code");
    return generateCursorBody(messages, model, tools, reasoningEffort, forceAgentMode);
  }

  async makeFetchRequest(url, headers, body, signal, proxyOptions = null) {
    const response = await proxyAwareFetch(url, {
      method: "POST",
      headers,
      body,
      signal
    }, proxyOptions);

    return {
      status: response.status,
      headers: Object.fromEntries(response.headers.entries()),
      body: Buffer.from(await response.arrayBuffer())
    };
  }

  makeHttp2Request(url, headers, body, signal) {
    if (!http2) {
      throw new Error("http2 module not available");
    }

    const HTTP2_TIMEOUT_MS = 60000; // 60s max — prevent hung sessions

    return new Promise((resolve, reject) => {
      const urlObj = new URL(url);
      const client = http2.connect(`https://${urlObj.host}`);
      const chunks = [];
      let responseHeaders = {};
      let settled = false;

      // Ensure client is always closed on settle
      const finish = (fn) => (...args) => {
        if (settled) return;
        settled = true;
        clearTimeout(hangTimeout);
        client.close();
        fn(...args);
      };

      // Hard timeout: close session if server never responds
      const hangTimeout = setTimeout(finish(() => {
        reject(new Error("HTTP/2 request timed out"));
      }), HTTP2_TIMEOUT_MS);

      client.on("error", finish(reject));

      const req = client.request({
        ":method": "POST",
        ":path": urlObj.pathname,
        ":authority": urlObj.host,
        ":scheme": "https",
        ...headers
      });

      req.on("response", (hdrs) => { responseHeaders = hdrs; });
      req.on("data", (chunk) => { chunks.push(chunk); });
      req.on("end", finish(() => {
        resolve({
          status: responseHeaders[":status"],
          headers: responseHeaders,
          body: Buffer.concat(chunks)
        });
      }));
      req.on("error", finish(reject));

      if (signal) {
        const onAbort = finish(() => reject(new Error("Request aborted")));
        signal.addEventListener("abort", onAbort, { once: true });
      }

      req.write(body);
      req.end();
    });
  }

  async execute({ model, body, stream, credentials, signal, log, proxyOptions = null }) {
    const url = this.buildUrl();
    const headers = this.buildHeaders(credentials);
    const transformedBody = this.transformRequest(model, body, stream, credentials);

    try {
      const shouldForceFetch = proxyOptions?.enabled === true || proxyOptions?.connectionProxyEnabled === true || !!proxyOptions?.vercelRelayUrl;
      const response = (http2 && !shouldForceFetch)
        ? await this.makeHttp2Request(url, headers, transformedBody, signal)
        : await this.makeFetchRequest(url, headers, transformedBody, signal, proxyOptions);

      if (response.status !== 200) {
        const errorText = response.body?.toString() || "Unknown error";
        const errorResponse = new Response(JSON.stringify({
          error: {
            message: `[${response.status}]: ${errorText}`,
            type: "invalid_request_error",
            code: ""
          }
        }), {
          status: response.status,
          headers: { "Content-Type": "application/json" }
        });
        return { response: errorResponse, url, headers, transformedBody: body };
      }

      const transformedResponse = stream !== false
        ? this.transformProtobufToSSE(response.body, model, body)
        : this.transformProtobufToJSON(response.body, model, body);

      return { response: transformedResponse, url, headers, transformedBody: body };
    } catch (error) {
      const errorResponse = new Response(JSON.stringify({
        error: {
          message: error.message,
          type: "connection_error",
          code: ""
        }
      }), {
        status: HTTP_STATUS.SERVER_ERROR,
        headers: { "Content-Type": "application/json" }
      });
      return { response: errorResponse, url, headers, transformedBody: body };
    }
  }

  transformProtobufToJSON(buffer, model, body) {
    const responseId = `chatcmpl-cursor-${Date.now()}`;
    const created = Math.floor(Date.now() / 1000);

    let offset = 0;
    let totalContent = "";
    let totalThinking = "";
    const toolCalls = [];
    const toolCallsMap = new Map(); // Track streaming tool calls by ID
    const finalizedIds = new Set();
    let frameCount = 0;

    debugLog(`[CURSOR BUFFER] Total length: ${buffer.length} bytes`);

    while (offset < buffer.length) {
      if (offset + 5 > buffer.length) {
        debugLog(
          `[CURSOR BUFFER] Reached end, offset=${offset}, remaining=${buffer.length - offset}`
        );
        break;
      }

      const flags = buffer[offset];
      const length = buffer.readUInt32BE(offset + 1);

      debugLog(
        `[CURSOR BUFFER] Frame ${frameCount + 1}: flags=0x${flags.toString(16).padStart(2, "0")}, length=${length}`
      );

      if (offset + 5 + length > buffer.length) {
        debugLog(
          `[CURSOR BUFFER] Incomplete frame, offset=${offset}, length=${length}, buffer.length=${buffer.length}`
        );
        break;
      }

      let payload = buffer.slice(offset + 5, offset + 5 + length);
      offset += 5 + length;
      frameCount++;

      payload = decompressPayload(payload, flags);
      if (!payload) {
        debugLog(`[CURSOR BUFFER] Frame ${frameCount}: decompression failed, skipping`);
        continue;
      }

      // Check for JSON error frames (byte guard: skip toString on non-JSON frames)
      if (payload.length > 0 && payload[0] === 0x7b) {
        try {
          const text = payload.toString("utf-8");
          if (text.includes('"error"')) {
            const hasContent = totalContent || toolCallsMap.size > 0;
            debugLog(
              `[CURSOR BUFFER] Error frame (hasContent=${hasContent}): ${text.slice(0, 500)}`
            );
            if (hasContent) {
              break;
            }
            return createErrorResponse(JSON.parse(text));
          }
        } catch {}
      }

      const result = extractTextFromResponse(new Uint8Array(payload));
      debugLog(`[CURSOR DECODED] Frame ${frameCount}:`, result);

      if (result.error) {
        const hasContent = totalContent || toolCallsMap.size > 0;
        debugLog(`[CURSOR BUFFER] Decoded error (hasContent=${hasContent}): ${result.error}`);
        if (hasContent) {
          break;
        }
        return new Response(
          JSON.stringify({
            error: {
              message: result.error,
              type: "rate_limit_error",
              code: "rate_limited"
            }
          }),
          {
            status: HTTP_STATUS.RATE_LIMITED,
            headers: { "Content-Type": "application/json" }
          }
        );
      }

      if (result.toolCall) {
        const tc = result.toolCall;

        if (toolCallsMap.has(tc.id)) {
          // Accumulate arguments for existing tool call
          const existing = toolCallsMap.get(tc.id);
          existing.function.arguments += tc.function.arguments;
          existing.isLast = tc.isLast;
        } else {
          // New tool call
          toolCallsMap.set(tc.id, { ...tc });
        }

        // Push to final array when isLast is true
        if (tc.isLast) {
          const finalToolCall = toolCallsMap.get(tc.id);
          finalizedIds.add(tc.id);
          toolCalls.push({
            id: finalToolCall.id,
            type: finalToolCall.type,
            function: {
              name: finalToolCall.function.name,
              arguments: finalToolCall.function.arguments
            }
          });
        }
      }

      if (result.text) totalContent += result.text;
      if (result.thinking) totalThinking += result.thinking;
    }

    let finalContent = totalContent || (isThinkingTailVisibleModel(model) ? visibleContentFromThinking(totalThinking) : "");

    debugLog(
      `[CURSOR BUFFER] Parsed ${frameCount} frames, toolCallsMap size: ${toolCallsMap.size}, finalized toolCalls: ${toolCalls.length}`
    );

    // Finalize all remaining tool calls in map (in case stream ended without isLast=true)
    for (const [id, tc] of toolCallsMap.entries()) {
      // Check if already in final array
      if (!finalizedIds.has(id)) {
        debugLog(`[CURSOR BUFFER] Finalizing incomplete tool call: ${id}, isLast=${tc.isLast}`);
        toolCalls.push({
          id: tc.id,
          type: tc.type,
          function: {
            name: tc.function.name,
            arguments: tc.function.arguments
          }
        });
      }
    }

    const pseudoToolParse = parseCursorPseudoToolCalls(finalContent);
    finalContent = pseudoToolParse.content;
    toolCalls.push(...pseudoToolParse.toolCalls);

    debugLog(`[CURSOR BUFFER] Final toolCalls count: ${toolCalls.length}`);


    const message = {
      role: "assistant",
      content: finalContent || null
    };

    if (toolCalls.length > 0) {
      message.tool_calls = toolCalls;
    }

    const usage = estimateUsage(body, finalContent.length, FORMATS.OPENAI);

    const completion = {
      id: responseId,
      object: "chat.completion",
      created,
      model,
      choices: [{
        index: 0,
        message,
        finish_reason: toolCalls.length > 0 ? "tool_calls" : "stop"
      }],
      usage
    };

    return new Response(JSON.stringify(completion), {
      status: 200,
      headers: { "Content-Type": "application/json" }
    });
  }

  transformProtobufToSSE(buffer, model, body) {
    const responseId = `chatcmpl-cursor-${Date.now()}`;
    const created = Math.floor(Date.now() / 1000);

    const chunks = [];
    let offset = 0;
    let totalContent = "";
    let totalThinking = "";
    let emittedThinkingVisibleContentLength = 0;
    const toolCalls = [];
    const toolCallsMap = new Map(); // Track streaming tool calls by ID
    const finalizedIds = new Set();
    let frameCount = 0;
    const emitRoleChunk = () => {
      chunks.push(
        `data: ${JSON.stringify({
          id: responseId,
          object: "chat.completion.chunk",
          created,
          model,
          choices: [
            {
              index: 0,
              delta: { role: "assistant", content: "" },
              finish_reason: null
            }
          ]
        })}\n\n`
      );
    };
    const emitContentChunk = (content) => {
      if (!content) return;
      chunks.push(
        `data: ${JSON.stringify({
          id: responseId,
          object: "chat.completion.chunk",
          created,
          model,
          choices: [
            {
              index: 0,
              delta:
                chunks.length === 0
                  ? { role: "assistant", content }
                  : { content },
              finish_reason: null
            }
          ]
        })}\n\n`
      );
    };
    const emitToolCallChunk = (tc, index) => {
      chunks.push(
        `data: ${JSON.stringify({
          id: responseId,
          object: "chat.completion.chunk",
          created,
          model,
          choices: [
            {
              index: 0,
              delta: {
                tool_calls: [
                  {
                    index,
                    id: tc.id,
                    type: "function",
                    function: {
                      name: tc.function.name,
                      arguments: tc.function.arguments
                    }
                  }
                ]
              },
              finish_reason: null
            }
          ]
        })}\n\n`
      );
    };

    debugLog(`[CURSOR BUFFER SSE] Total length: ${buffer.length} bytes`);

    while (offset < buffer.length) {
      if (offset + 5 > buffer.length) {
        debugLog(
          `[CURSOR BUFFER SSE] Reached end, offset=${offset}, remaining=${buffer.length - offset}`
        );
        break;
      }

      const flags = buffer[offset];
      const length = buffer.readUInt32BE(offset + 1);

      debugLog(
        `[CURSOR BUFFER SSE] Frame ${frameCount + 1}: flags=0x${flags.toString(16).padStart(2, "0")}, length=${length}`
      );

      if (offset + 5 + length > buffer.length) {
        debugLog(
          `[CURSOR BUFFER SSE] Incomplete frame, offset=${offset}, length=${length}, buffer.length=${buffer.length}`
        );
        break;
      }

      let payload = buffer.slice(offset + 5, offset + 5 + length);
      offset += 5 + length;
      frameCount++;

      payload = decompressPayload(payload, flags);
      if (!payload) {
        debugLog(`[CURSOR BUFFER SSE] Frame ${frameCount}: decompression failed, skipping`);
        continue;
      }

      // Check for JSON error frames (byte-guard: only decode if starts with '{')
      if (payload[0] === 0x7b) {
        try {
          const text = payload.toString("utf-8");
          if (text.includes('"error"')) {
            const hasContent = chunks.length > 0 || totalContent || toolCallsMap.size > 0;
            debugLog(
              `[CURSOR BUFFER SSE] Error frame (hasContent=${hasContent}): ${text.slice(0, 500)}`
            );
            if (hasContent) {
              break;
            }
            return createErrorResponse(JSON.parse(text));
          }
        } catch {}
      }

      const result = extractTextFromResponse(new Uint8Array(payload));
      debugLog(`[CURSOR DECODED SSE] Frame ${frameCount}:`, result);

      if (result.error) {
        const hasContent = chunks.length > 0 || totalContent || toolCallsMap.size > 0;
        debugLog(`[CURSOR BUFFER SSE] Decoded error (hasContent=${hasContent}): ${result.error}`);
        if (hasContent) {
          break;
        }
        return new Response(
          JSON.stringify({
            error: {
              message: result.error,
              type: "rate_limit_error",
              code: "rate_limited"
            }
          }),
          {
            status: HTTP_STATUS.RATE_LIMITED,
            headers: { "Content-Type": "application/json" }
          }
        );
      }

      if (result.toolCall) {
        const tc = result.toolCall;

        if (toolCallsMap.has(tc.id)) {
          // Accumulate arguments for existing tool call
          const existing = toolCallsMap.get(tc.id);
          existing.function.arguments += tc.function.arguments;
          existing.isLast = tc.isLast;
          const emitted = toolCalls[existing.index];
          if (emitted) {
            emitted.function.arguments = existing.function.arguments;
            emitted.isLast = existing.isLast;
          }
        } else {
          // New tool call - assign index and add to map
          const toolCallIndex = toolCalls.length;
          finalizedIds.add(tc.id);
          toolCalls.push({ ...tc, index: toolCallIndex });
          toolCallsMap.set(tc.id, { ...tc, index: toolCallIndex });
        }
      }

      if (result.text) {
        totalContent += result.text;
      }

      if (isThinkingTailVisibleModel(model) && result.thinking) {
        totalThinking += result.thinking;
        const visibleContent = visibleContentFromThinking(totalThinking);
        if (visibleContent.length > emittedThinkingVisibleContentLength) {
          const deltaContent = visibleContent.slice(emittedThinkingVisibleContentLength);
          emittedThinkingVisibleContentLength = visibleContent.length;
          totalContent += deltaContent;
        }
      }
    }

    debugLog(
      `[CURSOR BUFFER SSE] Parsed ${frameCount} frames, toolCallsMap size: ${toolCallsMap.size}, toolCalls array: ${toolCalls.length}`
    );

    // Finalize all remaining tool calls in map (stream may have ended without isLast=true)
    for (const [id, tc] of toolCallsMap.entries()) {
      if (!finalizedIds.has(id)) {
        debugLog(`[CURSOR BUFFER SSE] Finalizing incomplete tool call: ${id}, isLast=${tc.isLast}`);
        const toolCallIndex = toolCalls.length;
        toolCalls.push({
          id: tc.id,
          type: tc.type,
          index: toolCallIndex,
          function: {
            name: tc.function.name,
            arguments: tc.function.arguments
          }
        });
      }
    }

    const pseudoToolParse = parseCursorPseudoToolCalls(totalContent);
    totalContent = pseudoToolParse.content;
    emitContentChunk(totalContent);

    if (chunks.length === 0 && (toolCalls.length > 0 || pseudoToolParse.toolCalls.length > 0)) {
      emitRoleChunk();
    }

    for (const tc of toolCalls) {
      emitToolCallChunk(tc, tc.index ?? 0);
    }

    for (const tc of pseudoToolParse.toolCalls) {
      const toolCallIndex = toolCalls.length;
      toolCalls.push({ ...tc, index: toolCallIndex });
      emitToolCallChunk(tc, toolCallIndex);
    }

    if (chunks.length === 0 && toolCalls.length === 0) {
      emitRoleChunk();
    }

    const usage = estimateUsage(body, totalContent.length, FORMATS.OPENAI);

    chunks.push(
      `data: ${JSON.stringify({
        id: responseId,
        object: "chat.completion.chunk",
        created,
        model,
        choices: [
          {
            index: 0,
            delta: {},
            finish_reason: toolCalls.length > 0 ? "tool_calls" : "stop"
          }
        ],
        usage
      })}\n\n`
    );
    chunks.push("data: [DONE]\n\n");

    return new Response(chunks.join(""), {
      status: 200,
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive"
      }
    });
  }

  async refreshCredentials() {
    return null;
  }
}

export default CursorExecutor;
