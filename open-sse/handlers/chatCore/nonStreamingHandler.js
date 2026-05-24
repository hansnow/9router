import { FORMATS } from "../../translator/formats.js";
import { needsTranslation } from "../../translator/index.js";
import { ollamaBodyToOpenAI } from "../../translator/response/ollama-to-openai.js";
import { addBufferToUsage, filterUsageForFormat } from "../../utils/usageTracking.js";
import { createErrorResult } from "../../utils/error.js";
import { HTTP_STATUS } from "../../config/runtimeConfig.js";
import { parseSSEToOpenAIResponse } from "./sseToJsonHandler.js";
import { buildRequestDetail, extractRequestConfig, extractUsageFromResponse, saveUsageStats } from "./requestDetail.js";
import { appendRequestLog, saveRequestDetail } from "@/lib/usageDb.js";
import { decloakToolNames } from "../../utils/claudeCloaking.js";

/**
 * Translate non-streaming response body from provider format → OpenAI format.
 */
export function translateNonStreamingResponse(responseBody, targetFormat, sourceFormat) {
  if (targetFormat === sourceFormat || targetFormat === FORMATS.OPENAI) return responseBody;

  // Gemini / Antigravity
  if (targetFormat === FORMATS.GEMINI || targetFormat === FORMATS.ANTIGRAVITY || targetFormat === FORMATS.GEMINI_CLI || targetFormat === FORMATS.VERTEX) {
    const response = responseBody.response || responseBody;
    if (!response?.candidates?.[0]) return responseBody;

    const candidate = response.candidates[0];
    const content = candidate.content;
    const usage = response.usageMetadata || responseBody.usageMetadata;
    let textContent = "", reasoningContent = "";
    const toolCalls = [];

    if (content?.parts) {
      for (const part of content.parts) {
        if (part.thought === true && part.text) reasoningContent += part.text;
        else if (part.text !== undefined) textContent += part.text;
        if (part.functionCall) {
          toolCalls.push({
            id: `call_${part.functionCall.name}_${Date.now()}_${toolCalls.length}`,
            type: "function",
            function: { name: part.functionCall.name, arguments: JSON.stringify(part.functionCall.args || {}) }
          });
        }
      }
    }

    const message = { role: "assistant" };
    if (textContent) message.content = textContent;
    if (reasoningContent) message.reasoning_content = reasoningContent;
    if (toolCalls.length > 0) message.tool_calls = toolCalls;
    if (!message.content && !message.tool_calls) message.content = "";

    let finishReason = (candidate.finishReason || "stop").toLowerCase();
    if (finishReason === "stop" && toolCalls.length > 0) finishReason = "tool_calls";

    const result = {
      id: `chatcmpl-${response.responseId || Date.now()}`,
      object: "chat.completion",
      created: Math.floor(new Date(response.createTime || Date.now()).getTime() / 1000),
      model: response.modelVersion || "gemini",
      choices: [{ index: 0, message, finish_reason: finishReason }]
    };

    if (usage) {
      result.usage = {
        prompt_tokens: (usage.promptTokenCount || 0) + (usage.thoughtsTokenCount || 0),
        completion_tokens: usage.candidatesTokenCount || 0,
        total_tokens: usage.totalTokenCount || 0
      };
      if (usage.thoughtsTokenCount > 0) {
        result.usage.completion_tokens_details = { reasoning_tokens: usage.thoughtsTokenCount };
      }
    }
    return result;
  }

  // Claude
  if (targetFormat === FORMATS.CLAUDE) {
    if (!responseBody.content) return responseBody;

    let textContent = "", thinkingContent = "";
    const toolCalls = [];

    for (const block of responseBody.content) {
      if (block.type === "text") {
        // Strip markdown code block markers (e.g. kimi wraps JSON in ```json...```)
        const raw = block.text ?? "";
        const text = raw.replace(/^\s*```\s*json\s*\n?/i, "").replace(/\n?\s*```\s*$/i, "");
        textContent += text;
      } else if (block.type === "thinking") thinkingContent += block.thinking || "";
      else if (block.type === "tool_use") {
        toolCalls.push({ id: block.id, type: "function", function: { name: block.name, arguments: JSON.stringify(block.input || {}) } });
      }
    }

    const message = { role: "assistant" };
    if (textContent) message.content = textContent;
    if (thinkingContent) message.reasoning_content = thinkingContent;
    if (toolCalls.length > 0) message.tool_calls = toolCalls;
    if (!message.content && !message.tool_calls) message.content = "";

    let finishReason = responseBody.stop_reason || "stop";
    if (finishReason === "end_turn") finishReason = "stop";
    if (finishReason === "tool_use") finishReason = "tool_calls";

    const result = {
      id: `chatcmpl-${responseBody.id || Date.now()}`,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model: responseBody.model || "claude",
      choices: [{ index: 0, message, finish_reason: finishReason }]
    };

    if (responseBody.usage) {
      result.usage = {
        prompt_tokens: responseBody.usage.input_tokens || 0,
        completion_tokens: responseBody.usage.output_tokens || 0,
        total_tokens: (responseBody.usage.input_tokens || 0) + (responseBody.usage.output_tokens || 0)
      };
    }
    return result;
  }

  // Ollama
  if (targetFormat === FORMATS.OLLAMA) {
    return ollamaBodyToOpenAI(responseBody);
  }

  return responseBody;
}

function firstTextContent(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";

  return content
    .map((part) => {
      if (!part || typeof part !== "object") return "";
      if (part.type === "text" || part.type === "output_text") return part.text || "";
      return "";
    })
    .join("");
}

function firstReasoningContent(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";

  return content
    .map((part) => {
      if (!part || typeof part !== "object") return "";
      if (part.type === "thinking") return part.thinking || "";
      if (part.type === "reasoning") {
        return Array.isArray(part.summary)
          ? part.summary.map((summaryPart) => summaryPart?.text || "").join("")
          : "";
      }
      return "";
    })
    .join("");
}

function usageToResponses(usage) {
  if (!usage || typeof usage !== "object") return usage;

  const inputTokens = usage.input_tokens ?? usage.prompt_tokens;
  const outputTokens = usage.output_tokens ?? usage.completion_tokens;
  const responseUsage = {};

  if (inputTokens !== undefined) responseUsage.input_tokens = inputTokens;
  if (outputTokens !== undefined) responseUsage.output_tokens = outputTokens;
  if (usage.total_tokens !== undefined) {
    responseUsage.total_tokens = usage.total_tokens;
  } else if (inputTokens !== undefined && outputTokens !== undefined) {
    responseUsage.total_tokens = inputTokens + outputTokens;
  }
  if (usage.input_tokens_details) responseUsage.input_tokens_details = usage.input_tokens_details;
  if (usage.output_tokens_details) responseUsage.output_tokens_details = usage.output_tokens_details;
  if (usage.prompt_tokens_details) responseUsage.input_tokens_details = usage.prompt_tokens_details;
  if (usage.completion_tokens_details) responseUsage.output_tokens_details = usage.completion_tokens_details;
  if (usage.estimated !== undefined) responseUsage.estimated = usage.estimated;

  return responseUsage;
}

function usageToClaude(usage) {
  if (!usage || typeof usage !== "object") return usage;

  const claudeUsage = {};
  const cacheReadTokens = usage.cache_read_input_tokens ?? usage.prompt_tokens_details?.cached_tokens;
  const cacheCreateTokens = usage.cache_creation_input_tokens ?? usage.prompt_tokens_details?.cache_creation_tokens;
  const inputTokens = usage.input_tokens ?? (
    usage.prompt_tokens !== undefined
      ? Math.max(0, usage.prompt_tokens - (cacheReadTokens || 0) - (cacheCreateTokens || 0))
      : undefined
  );
  const outputTokens = usage.output_tokens ?? usage.completion_tokens;

  if (inputTokens !== undefined) claudeUsage.input_tokens = inputTokens;
  if (outputTokens !== undefined) claudeUsage.output_tokens = outputTokens;
  if (cacheReadTokens !== undefined) claudeUsage.cache_read_input_tokens = cacheReadTokens;
  if (cacheCreateTokens !== undefined) claudeUsage.cache_creation_input_tokens = cacheCreateTokens;
  if (usage.estimated !== undefined) claudeUsage.estimated = usage.estimated;

  return claudeUsage;
}

function mapOpenAIStopReasonToClaude(finishReason) {
  if (finishReason === "length" || finishReason === "max_tokens") return "max_tokens";
  if (finishReason === "tool_calls") return "tool_use";
  return "end_turn";
}

function responseIncompleteReason(finishReason) {
  if (finishReason === "length" || finishReason === "max_tokens") return "max_output_tokens";
  if (finishReason === "content_filter") return "content_filter";
  return null;
}

function chatCompletionToResponses(response, model) {
  if (response?.object !== "chat.completion" || !Array.isArray(response.choices)) return response;

  const choice = response.choices[0] || {};
  const message = choice.message || {};
  const output = [];
  const text = firstTextContent(message.content);
  const reasoning = message.reasoning_content || message.reasoning || "";
  const hasToolCalls = Array.isArray(message.tool_calls) && message.tool_calls.length > 0;
  const incompleteReason = responseIncompleteReason(choice.finish_reason);
  const isIncomplete = Boolean(incompleteReason);

  if (reasoning) {
    output.push({
      id: `rs_${response.id || Date.now()}`,
      type: "reasoning",
      summary: [{ type: "summary_text", text: reasoning }]
    });
  }

  if (text || (!reasoning && !hasToolCalls)) {
    output.push({
      id: `msg_${response.id || Date.now()}`,
      type: "message",
      status: isIncomplete ? "incomplete" : "completed",
      role: "assistant",
      content: [{ type: "output_text", text, annotations: [] }]
    });
  }

  if (hasToolCalls) {
    for (const toolCall of message.tool_calls) {
      output.push({
        id: toolCall.id || `fc_${Date.now()}`,
        type: "function_call",
        status: "completed",
        call_id: toolCall.id || `call_${Date.now()}`,
        name: toolCall.function?.name || "",
        arguments: toolCall.function?.arguments || ""
      });
    }
  }

  return {
    id: String(response.id || `resp_${Date.now()}`).replace(/^chatcmpl-/, "resp_"),
    object: "response",
    created_at: response.created || Math.floor(Date.now() / 1000),
    status: isIncomplete ? "incomplete" : "completed",
    background: false,
    error: null,
    incomplete_details: incompleteReason ? { reason: incompleteReason } : null,
    instructions: null,
    max_output_tokens: null,
    model: response.model || model,
    output,
    parallel_tool_calls: true,
    previous_response_id: null,
    reasoning: { effort: null, summary: null },
    store: true,
    temperature: null,
    text: { format: { type: "text" } },
    tool_choice: "auto",
    tools: [],
    top_p: null,
    truncation: "disabled",
    usage: usageToResponses(response.usage)
  };
}

function chatCompletionToClaude(response, model) {
  if (response?.object !== "chat.completion" || !Array.isArray(response.choices)) return response;

  const choice = response.choices[0] || {};
  const message = choice.message || {};
  const content = [];
  const text = firstTextContent(message.content);
  const reasoning = message.reasoning_content || message.reasoning || "";
  const hasToolCalls = Array.isArray(message.tool_calls) && message.tool_calls.length > 0;

  if (reasoning) {
    content.push({ type: "thinking", thinking: reasoning });
  }

  if (text || (!reasoning && !hasToolCalls)) {
    content.push({ type: "text", text });
  }

  if (hasToolCalls) {
    for (const toolCall of message.tool_calls) {
      let input = {};
      try {
        input = JSON.parse(toolCall.function?.arguments || "{}");
      } catch {
        input = {};
      }
      content.push({
        type: "tool_use",
        id: toolCall.id || `toolu_${Date.now()}`,
        name: toolCall.function?.name || "",
        input
      });
    }
  }

  return {
    id: String(response.id || `msg_${Date.now()}`).replace(/^chatcmpl-/, "msg_"),
    type: "message",
    role: "assistant",
    model: response.model || model,
    content,
    stop_reason: mapOpenAIStopReasonToClaude(choice.finish_reason),
    stop_sequence: null,
    usage: usageToClaude(response.usage)
  };
}

function convertNonStreamingResponseForClient(response, sourceFormat, model) {
  if (sourceFormat === FORMATS.OPENAI_RESPONSES || sourceFormat === FORMATS.OPENAI_RESPONSE) {
    return chatCompletionToResponses(response, model);
  }
  if (sourceFormat === FORMATS.CLAUDE) {
    return chatCompletionToClaude(response, model);
  }
  return response;
}

function extractResponseSummary(response) {
  if (response?.choices?.[0]) {
    return {
      content: response.choices[0].message?.content || null,
      thinking: response.choices[0].message?.reasoning_content || null,
      finish_reason: response.choices[0].finish_reason || "unknown"
    };
  }

  if (response?.object === "response") {
    const message = response.output?.find((item) => item.type === "message");
    const reasoning = response.output?.find((item) => item.type === "reasoning");
    return {
      content: firstTextContent(message?.content) || null,
      thinking: firstReasoningContent([reasoning]) || null,
      finish_reason: response.status || "unknown"
    };
  }

  if (response?.type === "message") {
    return {
      content: firstTextContent(response.content) || null,
      thinking: firstReasoningContent(response.content) || null,
      finish_reason: response.stop_reason || "unknown"
    };
  }

  return {
    content: response?.content || null,
    thinking: response?.reasoning_content || null,
    finish_reason: "unknown"
  };
}

/**
 * Handle non-streaming response from provider.
 */
export async function handleNonStreamingResponse({ providerResponse, provider, model, sourceFormat, targetFormat, body, stream, translatedBody, finalBody, requestStartTime, connectionId, apiKey, clientRawRequest, onRequestSuccess, reqLogger, toolNameMap, trackDone, appendLog }) {
  trackDone();
  const contentType = providerResponse.headers.get("content-type") || "";
  let responseBody;

  if (contentType.includes("text/event-stream")) {
    const sseText = await providerResponse.text();
    const parsed = parseSSEToOpenAIResponse(sseText, model);
    if (!parsed) {
      appendLog({ status: `FAILED ${HTTP_STATUS.BAD_GATEWAY}` });
      return createErrorResult(HTTP_STATUS.BAD_GATEWAY, "Invalid SSE response for non-streaming request");
    }
    responseBody = parsed;
  } else {
    try {
      responseBody = await providerResponse.json();
    } catch (err) {
      appendLog({ status: `FAILED ${HTTP_STATUS.BAD_GATEWAY}` });
      console.error(`[ChatCore] Failed to parse JSON from ${provider}:`, err.message);
      return createErrorResult(HTTP_STATUS.BAD_GATEWAY, `Invalid JSON response from ${provider}`);
    }
  }

  reqLogger.logProviderResponse(providerResponse.status, providerResponse.statusText, providerResponse.headers, responseBody);
  if (onRequestSuccess) await onRequestSuccess();

  // Decloak tool_use names once on raw Claude body, before any translation (INPUT side)
  responseBody = decloakToolNames(responseBody, toolNameMap);

  const usage = extractUsageFromResponse(responseBody);
  appendLog({ tokens: usage, status: "200 OK" });
  saveUsageStats({ provider, model, tokens: usage, connectionId, apiKey, endpoint: clientRawRequest?.endpoint });

  const translatedResponse = needsTranslation(targetFormat, sourceFormat)
    ? translateNonStreamingResponse(responseBody, targetFormat, sourceFormat)
    : responseBody;

  // Fix finish_reason for tool_calls: some providers return non-standard values (e.g. "other")
  if (translatedResponse?.choices?.[0]) {
    const choice = translatedResponse.choices[0];
    const msg = choice.message;
    const hasToolCalls = Array.isArray(msg?.tool_calls) && msg.tool_calls.length > 0;
    if (hasToolCalls && choice.finish_reason !== "tool_calls") {
      choice.finish_reason = "tool_calls";
    }
  }

  // Ensure OpenAI-required fields
  if (!translatedResponse.object) translatedResponse.object = "chat.completion";
  if (!translatedResponse.created) translatedResponse.created = Math.floor(Date.now() / 1000);

  // Strip Azure-specific fields
  delete translatedResponse.prompt_filter_results;
  if (translatedResponse?.choices) {
    for (const choice of translatedResponse.choices) delete choice.content_filter_results;
  }

  if (translatedResponse?.usage) {
    translatedResponse.usage = addBufferToUsage(translatedResponse.usage);
  }

  const clientResponse = convertNonStreamingResponseForClient(translatedResponse, sourceFormat, model);
  // Strip reasoning_content only from OpenAI Chat responses. Claude and Responses
  // clients need it converted into native thinking/reasoning blocks above.
  if (sourceFormat === FORMATS.OPENAI && clientResponse?.choices) {
    for (const choice of clientResponse.choices) {
      const hasToolCalls = Array.isArray(choice?.message?.tool_calls) && choice.message.tool_calls.length > 0;
      if (choice?.message?.reasoning_content && (choice.message.content || hasToolCalls)) {
        delete choice.message.reasoning_content;
      }
    }
  }
  if (clientResponse?.usage) {
    clientResponse.usage = filterUsageForFormat(clientResponse.usage, sourceFormat);
  }

  reqLogger.logConvertedResponse(clientResponse);

  const responseSummary = extractResponseSummary(clientResponse);

  const totalLatency = Date.now() - requestStartTime;
  saveRequestDetail(buildRequestDetail({
    provider, model, connectionId,
    latency: { ttft: totalLatency, total: totalLatency },
    tokens: usage || { prompt_tokens: 0, completion_tokens: 0 },
    request: extractRequestConfig(body, stream),
    providerRequest: finalBody || translatedBody || null,
    providerResponse: responseBody || null,
    response: {
      content: responseSummary.content,
      thinking: responseSummary.thinking,
      finish_reason: responseSummary.finish_reason
    },
    status: "success"
  }, { endpoint: clientRawRequest?.endpoint || null })).catch(err => {
    console.error("[RequestDetail] Failed to save:", err.message);
  });

  return {
    success: true,
    response: new Response(JSON.stringify(clientResponse), {
      headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" }
    })
  };
}
