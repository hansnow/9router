import { BaseExecutor } from "./base.js";
import { PROVIDERS } from "../config/providers.js";
import { HTTP_STATUS } from "../config/runtimeConfig.js";
import {
  generateCursorBody,
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
const HTTP2_IDLE_TIMEOUT_MS = Number(process.env.CURSOR_HTTP2_IDLE_TIMEOUT_MS || 120000);
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

function createOpenAIErrorResponse(message, status = HTTP_STATUS.SERVER_ERROR, type = "api_error", code = "") {
  return new Response(JSON.stringify({
    error: {
      message,
      type,
      code
    }
  }), {
    status,
    headers: { "Content-Type": "application/json" }
  });
}

function createSSEHeaders() {
  return {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive"
  };
}

export class ConnectRPCFrameBuffer {
  constructor() {
    this.buffer = Buffer.alloc(0);
    this.frameCount = 0;
  }

  push(chunk) {
    if (!chunk || chunk.length === 0) return [];

    this.buffer = this.buffer.length === 0
      ? Buffer.from(chunk)
      : Buffer.concat([this.buffer, Buffer.from(chunk)]);

    const frames = [];
    while (this.buffer.length >= 5) {
      const flags = this.buffer[0];
      const length = this.buffer.readUInt32BE(1);
      if (this.buffer.length < 5 + length) break;

      let payload = this.buffer.slice(5, 5 + length);
      this.buffer = this.buffer.slice(5 + length);
      this.frameCount++;

      debugLog(
        `[CURSOR STREAM FRAME] Frame ${this.frameCount}: flags=0x${flags.toString(16).padStart(2, "0")}, length=${length}`
      );

      payload = decompressPayload(payload, flags);
      if (!payload) {
        debugLog(`[CURSOR STREAM FRAME] Frame ${this.frameCount}: decompression failed, skipping`);
        continue;
      }

      frames.push({ flags, length, payload });
    }

    return frames;
  }

  hasBufferedData() {
    return this.buffer.length > 0;
  }
}

class CursorSSEStreamEncoder {
  constructor(model, body) {
    this.responseId = `chatcmpl-cursor-${Date.now()}`;
    this.created = Math.floor(Date.now() / 1000);
    this.model = model;
    this.body = body;

    this.totalContent = "";
    this.totalThinking = "";
    this.emittedThinkingVisibleContentLength = 0;
    this.pendingContent = "";
    this.insidePseudoToolBlock = false;
    this.pseudoToolBuffer = "";
    this.toolCalls = [];
    this.toolCallsMap = new Map();
    this.hasOutput = false;
    this.hasSignal = false;
    this.finished = false;
    this.frameCount = 0;
  }

  processPayload(payload) {
    if (this.finished) return { events: [] };
    this.frameCount++;

    const jsonErrorResponse = this.handleJSONErrorPayload(payload);
    if (jsonErrorResponse) return jsonErrorResponse;

    const result = extractTextFromResponse(new Uint8Array(payload));
    debugLog(`[CURSOR DECODED STREAM] Frame ${this.frameCount}:`, result);

    if (result.error) {
      return this.handleDecodedError(result.error);
    }

    const events = [];
    if (result.toolCall) {
      this.hasSignal = true;
      events.push(...this.handleNativeToolCall(result.toolCall));
    }

    if (result.text) {
      this.hasSignal = true;
      events.push(...this.appendVisibleContent(result.text));
    }

    if (isThinkingTailVisibleModel(this.model) && result.thinking) {
      this.hasSignal = true;
      this.totalThinking += result.thinking;
      const visibleContent = visibleContentFromThinking(this.totalThinking);
      if (visibleContent.length > this.emittedThinkingVisibleContentLength) {
        const deltaContent = visibleContent.slice(this.emittedThinkingVisibleContentLength);
        this.emittedThinkingVisibleContentLength = visibleContent.length;
        events.push(...this.appendVisibleContent(deltaContent));
      }
    } else if (result.thinking) {
      this.hasSignal = true;
    }

    return { events };
  }

  finish() {
    if (this.finished) return { events: [] };
    this.finished = true;

    const events = [];
    if (!this.insidePseudoToolBlock) {
      events.push(...this.emitPendingContent({ final: true }));
    }

    if (this.hasOpenOutput()) {
      events.push(this.finishChunk());
      events.push("data: [DONE]\n\n");
      return { events };
    }

    if (this.hasSignal) {
      events.push(this.roleChunk());
      events.push(this.finishChunk());
      events.push("data: [DONE]\n\n");
      return { events };
    }

    return {
      errorResponse: createOpenAIErrorResponse(
        "Cursor returned an empty response without text, thinking, or tool calls",
        HTTP_STATUS.SERVER_ERROR,
        "upstream_error",
        "empty_response"
      )
    };
  }

  handleJSONErrorPayload(payload) {
    if (!payload || payload.length === 0 || payload[0] !== 0x7b) return null;

    try {
      const text = payload.toString("utf-8");
      if (!text.includes('"error"')) return null;

      debugLog(
        `[CURSOR STREAM] Error frame (hasOutput=${this.hasOpenOutput()}): ${text.slice(0, 500)}`
      );

      const parsed = JSON.parse(text);
      if (!this.hasOpenOutput()) {
        return { errorResponse: createErrorResponse(parsed), errorPayload: parsed };
      }

      return { events: [this.errorEvent(parsed), "data: [DONE]\n\n"], terminal: true };
    } catch {
      return null;
    }
  }

  handleDecodedError(error) {
    debugLog(`[CURSOR STREAM] Decoded error (hasOutput=${this.hasOpenOutput()}): ${error}`);
    const payload = {
      error: { message: error, type: "rate_limit_error", code: "rate_limited" }
    };
    if (!this.hasOpenOutput()) {
      return {
        errorResponse: createOpenAIErrorResponse(
          error,
          HTTP_STATUS.RATE_LIMITED,
          "rate_limit_error",
          "rate_limited"
        ),
        errorPayload: payload
      };
    }

    return {
      events: [
        this.errorEvent(payload),
        "data: [DONE]\n\n"
      ],
      terminal: true
    };
  }

  handleNativeToolCall(tc) {
    let toolCallIndex;
    let argsDelta = tc.function.arguments || "";
    let isFirstChunk = false;

    if (this.toolCallsMap.has(tc.id)) {
      const existing = this.toolCallsMap.get(tc.id);
      toolCallIndex = existing.index;
      existing.function.arguments += argsDelta;
      existing.isLast = tc.isLast;
    } else {
      toolCallIndex = this.toolCalls.length;
      isFirstChunk = true;
      const stored = { ...tc, index: toolCallIndex, function: { ...tc.function } };
      this.toolCalls.push(stored);
      this.toolCallsMap.set(tc.id, stored);
    }

    return [this.toolCallChunk(tc, toolCallIndex, { argsDelta, isFirstChunk })];
  }

  appendVisibleContent(content) {
    if (!content) return [];

    const events = [];
    if (this.insidePseudoToolBlock) {
      this.pseudoToolBuffer += content;
      events.push(...this.tryFinishPseudoToolBlock());
      return events;
    }

    this.pendingContent += content;
    const markerIndex = this.pendingContent.indexOf(PSEUDO_TOOL_MARKERS.callsBegin);
    if (markerIndex >= 0) {
      const before = this.pendingContent.slice(0, markerIndex).trimEnd();
      const pseudoStart = this.pendingContent.slice(markerIndex);
      this.pendingContent = before;
      events.push(...this.emitPendingContent({ final: true }));
      this.pendingContent = "";
      this.insidePseudoToolBlock = true;
      this.pseudoToolBuffer = pseudoStart;
      events.push(...this.tryFinishPseudoToolBlock());
      return events;
    }

    events.push(...this.emitPendingContent({ final: false }));
    return events;
  }

  tryFinishPseudoToolBlock() {
    const endIndex = this.pseudoToolBuffer.indexOf(PSEUDO_TOOL_MARKERS.callsEnd);
    if (endIndex < 0) return [];

    const blockEnd = endIndex + PSEUDO_TOOL_MARKERS.callsEnd.length;
    const pseudoBlock = this.pseudoToolBuffer.slice(0, blockEnd);
    const after = this.pseudoToolBuffer.slice(blockEnd).trimStart();
    const parsed = parseCursorPseudoToolCalls(pseudoBlock);

    this.insidePseudoToolBlock = false;
    this.pseudoToolBuffer = "";

    const events = [];
    for (const tc of parsed.toolCalls) {
      const toolCallIndex = this.toolCalls.length;
      this.toolCalls.push({ ...tc, index: toolCallIndex });
      events.push(this.toolCallChunk(tc, toolCallIndex, {
        argsDelta: tc.function.arguments,
        isFirstChunk: true
      }));
    }

    if (after) {
      const prefix = this.totalContent ? "\n" : "";
      events.push(...this.appendVisibleContent(prefix + after));
    }

    return events;
  }

  emitPendingContent({ final }) {
    if (!this.pendingContent) return [];

    let emitLength = this.pendingContent.length;
    if (!final) {
      const marker = PSEUDO_TOOL_MARKERS.callsBegin;
      let holdLength = 0;
      const maxHold = Math.min(marker.length - 1, this.pendingContent.length);
      for (let length = 1; length <= maxHold; length++) {
        if (marker.startsWith(this.pendingContent.slice(-length))) {
          holdLength = length;
        }
      }

      const stableContent = this.pendingContent.slice(0, this.pendingContent.length - holdLength);
      const trailingWhitespace = stableContent.match(/\s+$/)?.[0]?.length || 0;
      emitLength -= holdLength + trailingWhitespace;
    }

    if (emitLength <= 0) return [];

    const content = this.pendingContent.slice(0, emitLength);
    this.pendingContent = this.pendingContent.slice(emitLength);
    this.totalContent += content;
    return [this.contentChunk(content)];
  }

  hasOpenOutput() {
    return this.hasOutput || this.totalContent.length > 0 || this.toolCalls.length > 0;
  }

  contentChunk(content) {
    const includeRole = !this.hasOutput;
    this.hasOutput = true;
    return `data: ${JSON.stringify({
      id: this.responseId,
      object: "chat.completion.chunk",
      created: this.created,
      model: this.model,
      choices: [
        {
          index: 0,
          delta: includeRole ? { role: "assistant", content } : { content },
          finish_reason: null
        }
      ]
    })}\n\n`;
  }

  roleChunk() {
    this.hasOutput = true;
    return `data: ${JSON.stringify({
      id: this.responseId,
      object: "chat.completion.chunk",
      created: this.created,
      model: this.model,
      choices: [
        {
          index: 0,
          delta: { role: "assistant", content: "" },
          finish_reason: null
        }
      ]
    })}\n\n`;
  }

  toolCallChunk(tc, index, { argsDelta, isFirstChunk }) {
    const events = [];
    if (!this.hasOutput && this.totalContent.length === 0) {
      events.push(this.roleChunk());
    }

    this.hasOutput = true;
    const toolCall = {
      index,
      function: {
        arguments: argsDelta || ""
      }
    };
    if (isFirstChunk) {
      toolCall.id = tc.id;
      toolCall.type = "function";
      toolCall.function.name = tc.function.name;
    }

    events.push(`data: ${JSON.stringify({
      id: this.responseId,
      object: "chat.completion.chunk",
      created: this.created,
      model: this.model,
      choices: [
        {
          index: 0,
          delta: { tool_calls: [toolCall] },
          finish_reason: null
        }
      ]
    })}\n\n`);

    return events.join("");
  }

  finishChunk() {
    const usage = estimateUsage(this.body, this.totalContent.length, FORMATS.OPENAI);
    return `data: ${JSON.stringify({
      id: this.responseId,
      object: "chat.completion.chunk",
      created: this.created,
      model: this.model,
      choices: [
        {
          index: 0,
          delta: {},
          finish_reason: this.toolCalls.length > 0 ? "tool_calls" : "stop"
        }
      ],
      usage
    })}\n\n`;
  }

  errorEvent(errorPayload) {
    this.finished = true;
    return `event: error\ndata: ${JSON.stringify(errorPayload)}\n\n`;
  }
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

    return new Promise((resolve, reject) => {
      const urlObj = new URL(url);
      const client = http2.connect(`https://${urlObj.host}`);
      const chunks = [];
      let responseHeaders = {};
      let settled = false;
      let idleTimeout = null;

      const resetIdleTimeout = () => {
        clearTimeout(idleTimeout);
        idleTimeout = setTimeout(finish(() => {
          reject(new Error(`HTTP/2 request idle timeout after ${HTTP2_IDLE_TIMEOUT_MS}ms without upstream data`));
        }), HTTP2_IDLE_TIMEOUT_MS);
      };

      // Ensure client is always closed on settle
      const finish = (fn) => (...args) => {
        if (settled) return;
        settled = true;
        clearTimeout(idleTimeout);
        client.close();
        fn(...args);
      };

      resetIdleTimeout();

      client.on("error", finish(reject));

      const req = client.request({
        ":method": "POST",
        ":path": urlObj.pathname,
        ":authority": urlObj.host,
        ":scheme": "https",
        ...headers
      });

      req.on("response", (hdrs) => {
        responseHeaders = hdrs;
        resetIdleTimeout();
      });
      req.on("data", (chunk) => {
        chunks.push(chunk);
        resetIdleTimeout();
      });
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

  makeHttp2StreamingResponse(url, headers, body, model, requestBody, signal) {
    if (!http2) {
      throw new Error("http2 module not available");
    }

    return new Promise((resolve, reject) => {
      const urlObj = new URL(url);
      const client = http2.connect(`https://${urlObj.host}`);
      const frameBuffer = new ConnectRPCFrameBuffer();
      const encoder = new CursorSSEStreamEncoder(model, requestBody);
      const textEncoder = new TextEncoder();
      let controllerRef = null;
      let responseHeaders = {};
      let settled = false;
      let responseResolved = false;
      let streamClosed = false;
      let idleTimeout = null;
      let req = null;
      let upstreamStatus = null;
      const upstreamErrorChunks = [];

      const resolveStream = () => {
        if (responseResolved) return;
        responseResolved = true;
        resolve(new Response(stream, {
          status: 200,
          headers: createSSEHeaders()
        }));
      };

      const closeResources = () => {
        clearTimeout(idleTimeout);
        try { req?.close(); } catch {}
        try { client.close(); } catch {}
      };

      const settleWithError = (error) => {
        if (settled) return;
        settled = true;
        closeResources();
        if (!responseResolved) {
          reject(error);
          return;
        }
        if (!streamClosed) {
          streamClosed = true;
          try {
            controllerRef?.enqueue(textEncoder.encode(
              `event: error\ndata: ${JSON.stringify({ error: { message: error.message, type: "connection_error" } })}\n\n`
            ));
            controllerRef?.enqueue(textEncoder.encode("data: [DONE]\n\n"));
            controllerRef?.close();
          } catch {}
        }
      };

      const resetIdleTimeout = () => {
        clearTimeout(idleTimeout);
        idleTimeout = setTimeout(() => {
          settleWithError(new Error(`HTTP/2 request idle timeout after ${HTTP2_IDLE_TIMEOUT_MS}ms without upstream data`));
        }, HTTP2_IDLE_TIMEOUT_MS);
      };

      const enqueueEvents = (events) => {
        if (!events || events.length === 0) return;
        resolveStream();
        for (const event of events) {
          if (event) controllerRef.enqueue(textEncoder.encode(event));
        }
      };

      const keepAlive = () => {
        resolveStream();
        controllerRef.enqueue(textEncoder.encode(": cursor-upstream\n\n"));
      };

      const finishStream = () => {
        if (settled) return;

        if (upstreamStatus && upstreamStatus !== 200) {
          settled = true;
          closeResources();
          const errorText = upstreamErrorChunks.length > 0
            ? Buffer.concat(upstreamErrorChunks).toString()
            : `Cursor upstream returned HTTP ${upstreamStatus}`;
          resolve(new Response(JSON.stringify({
            error: {
              message: `[${upstreamStatus}]: ${errorText}`,
              type: "invalid_request_error",
              code: ""
            }
          }), {
            status: upstreamStatus,
            headers: { "Content-Type": "application/json" }
          }));
          return;
        }

        if (frameBuffer.hasBufferedData()) {
          closeResources();
          settleWithError(new Error("Cursor upstream ended with an incomplete ConnectRPC frame"));
          return;
        }

        settled = true;
        closeResources();

        const result = encoder.finish();
        if (result.errorResponse && !responseResolved) {
          resolve(result.errorResponse);
          return;
        }

        enqueueEvents(result.events);
        if (!streamClosed) {
          streamClosed = true;
          controllerRef.close();
        }
      };

      const stream = new ReadableStream({
        start(controller) {
          controllerRef = controller;
        },
        cancel() {
          settled = true;
          streamClosed = true;
          closeResources();
        }
      });

      resetIdleTimeout();
      client.on("error", settleWithError);

      req = client.request({
        ":method": "POST",
        ":path": urlObj.pathname,
        ":authority": urlObj.host,
        ":scheme": "https",
        ...headers
      });

      req.on("response", (hdrs) => {
        responseHeaders = hdrs;
        resetIdleTimeout();
        upstreamStatus = responseHeaders[":status"];
      });

      req.on("data", (chunk) => {
        if (settled) return;
        resetIdleTimeout();
        if (upstreamStatus && upstreamStatus !== 200) {
          upstreamErrorChunks.push(Buffer.from(chunk));
          return;
        }
        try {
          const frames = frameBuffer.push(chunk);
          for (const frame of frames) {
            const result = encoder.processPayload(frame.payload);

            if (result.errorResponse) {
              settled = true;
              closeResources();
              if (!responseResolved) {
                resolve(result.errorResponse);
              } else if (!streamClosed) {
                const errorPayload = result.errorPayload || {
                  error: {
                    message: "Cursor upstream error after stream started",
                    type: "upstream_error"
                  }
                };
                streamClosed = true;
                controllerRef.enqueue(textEncoder.encode(
                  `event: error\ndata: ${JSON.stringify(errorPayload)}\n\n`
                ));
                controllerRef.enqueue(textEncoder.encode("data: [DONE]\n\n"));
                controllerRef.close();
              }
              return;
            }

            if (result.events?.length > 0) {
              enqueueEvents(result.events);
            } else if (responseResolved) {
              keepAlive();
            }
            if (result.terminal) {
              settled = true;
              closeResources();
              if (!streamClosed) {
                streamClosed = true;
                controllerRef.close();
              }
              return;
            }
          }
        } catch (error) {
          settleWithError(error);
        }
      });

      req.on("end", finishStream);
      req.on("error", settleWithError);

      if (signal) {
        const onAbort = () => settleWithError(new Error("Request aborted"));
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
      if (stream !== false && http2 && !shouldForceFetch) {
        const response = await this.makeHttp2StreamingResponse(url, headers, transformedBody, model, body, signal);
        return { response, url, headers, transformedBody: body };
      }

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
    return this.transformProtobufChunksToSSE([buffer], model, body);
  }

  transformProtobufChunksToSSE(chunksInput, model, body) {
    const frameBuffer = new ConnectRPCFrameBuffer();
    const encoder = new CursorSSEStreamEncoder(model, body);
    const chunks = [];

    const totalLength = chunksInput.reduce((sum, chunk) => sum + chunk.length, 0);
    debugLog(`[CURSOR BUFFER SSE] Total length: ${totalLength} bytes`);

    let terminal = false;
    for (const chunk of chunksInput) {
      if (terminal) break;
      for (const frame of frameBuffer.push(chunk)) {
        const result = encoder.processPayload(frame.payload);
        if (result.errorResponse && chunks.length === 0) {
          return result.errorResponse;
        }
        if (result.events) chunks.push(...result.events);
        if (result.terminal) {
          terminal = true;
          break;
        }
      }
    }

    if (!terminal && frameBuffer.hasBufferedData()) {
      return createOpenAIErrorResponse(
        "Cursor upstream ended with an incomplete ConnectRPC frame",
        HTTP_STATUS.SERVER_ERROR,
        "upstream_error",
        "incomplete_frame"
      );
    }

    if (!terminal && !encoder.finished) {
      const result = encoder.finish();
      if (result.errorResponse && chunks.length === 0) {
        return result.errorResponse;
      }
      if (result.events) chunks.push(...result.events);
    }

    return new Response(chunks.join(""), {
      status: 200,
      headers: createSSEHeaders()
    });
  }

  async refreshCredentials() {
    return null;
  }
}

export default CursorExecutor;
