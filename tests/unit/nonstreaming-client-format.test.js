import { beforeAll, describe, expect, it, vi } from "vitest";

import { FORMATS } from "../../open-sse/translator/formats.js";

vi.mock("@/lib/usageDb.js", () => ({
  appendRequestLog: vi.fn(() => Promise.resolve()),
  saveRequestDetail: vi.fn(() => Promise.resolve()),
  saveRequestUsage: vi.fn(() => Promise.resolve()),
}));

let handleNonStreamingResponse;

beforeAll(async () => {
  ({ handleNonStreamingResponse } = await import("../../open-sse/handlers/chatCore/nonStreamingHandler.js"));
});

function buildOptions({ sourceFormat, body }) {
  return {
    providerResponse: new Response(JSON.stringify(body), {
      headers: { "Content-Type": "application/json" },
    }),
    provider: "openai",
    model: "test-model",
    sourceFormat,
    targetFormat: FORMATS.OPENAI,
    body: { messages: [{ role: "user", content: "hi" }] },
    stream: false,
    translatedBody: null,
    finalBody: null,
    requestStartTime: Date.now(),
    connectionId: "conn_test",
    apiKey: "test_key",
    clientRawRequest: { endpoint: "/v1/test" },
    onRequestSuccess: vi.fn(),
    reqLogger: {
      logProviderResponse: vi.fn(),
      logConvertedResponse: vi.fn(),
    },
    toolNameMap: undefined,
    trackDone: vi.fn(),
    appendLog: vi.fn(),
  };
}

describe("non-streaming client format conversion", () => {
  it("preserves reasoning_content as Claude thinking blocks", async () => {
    const result = await handleNonStreamingResponse(buildOptions({
      sourceFormat: FORMATS.CLAUDE,
      body: {
        id: "chatcmpl-test",
        object: "chat.completion",
        created: 1,
        model: "test-model",
        choices: [{
          index: 0,
          message: {
            role: "assistant",
            reasoning_content: "thinking first",
            content: "final answer",
          },
          finish_reason: "stop",
        }],
        usage: { prompt_tokens: 2, completion_tokens: 3, total_tokens: 5 },
      },
    }));

    const payload = await result.response.json();
    expect(payload.type).toBe("message");
    expect(payload.content).toEqual([
      { type: "thinking", thinking: "thinking first" },
      { type: "text", text: "final answer" },
    ]);
    expect(JSON.stringify(payload)).not.toContain("reasoning_content");
  });

  it("maps OpenAI cached prompt tokens into Claude usage fields", async () => {
    const result = await handleNonStreamingResponse(buildOptions({
      sourceFormat: FORMATS.CLAUDE,
      body: {
        id: "chatcmpl-test",
        object: "chat.completion",
        created: 1,
        model: "test-model",
        choices: [{
          index: 0,
          message: { role: "assistant", content: "final answer" },
          finish_reason: "stop",
        }],
        usage: {
          prompt_tokens: 100,
          completion_tokens: 7,
          total_tokens: 107,
          prompt_tokens_details: {
            cached_tokens: 40,
            cache_creation_tokens: 10,
          },
        },
      },
    }));

    const payload = await result.response.json();
    expect(payload.usage).toEqual({
      input_tokens: 2050,
      output_tokens: 7,
      cache_read_input_tokens: 40,
      cache_creation_input_tokens: 10,
    });
  });

  it("preserves reasoning-only output as Responses reasoning items", async () => {
    const result = await handleNonStreamingResponse(buildOptions({
      sourceFormat: FORMATS.OPENAI_RESPONSES,
      body: {
        id: "chatcmpl-test",
        object: "chat.completion",
        created: 1,
        model: "test-model",
        choices: [{
          index: 0,
          message: {
            role: "assistant",
            reasoning_content: "reasoning only",
          },
          finish_reason: "stop",
        }],
        usage: { prompt_tokens: 2, completion_tokens: 3, total_tokens: 5 },
      },
    }));

    const payload = await result.response.json();
    expect(payload.object).toBe("response");
    expect(payload.output).toEqual([{
      id: "rs_chatcmpl-test",
      type: "reasoning",
      summary: [{ type: "summary_text", text: "reasoning only" }],
    }]);
    expect(payload.usage.total_tokens).toBeGreaterThanOrEqual(5);
  });

  it("marks length-limited Responses output as incomplete", async () => {
    const result = await handleNonStreamingResponse(buildOptions({
      sourceFormat: FORMATS.OPENAI_RESPONSES,
      body: {
        id: "chatcmpl-test",
        object: "chat.completion",
        created: 1,
        model: "test-model",
        choices: [{
          index: 0,
          message: { role: "assistant", content: "partial" },
          finish_reason: "length",
        }],
      },
    }));

    const payload = await result.response.json();
    expect(payload.status).toBe("incomplete");
    expect(payload.incomplete_details).toEqual({ reason: "max_output_tokens" });
    expect(payload.output[0].status).toBe("incomplete");
  });

  it("marks max_tokens Responses output as incomplete", async () => {
    const result = await handleNonStreamingResponse(buildOptions({
      sourceFormat: FORMATS.OPENAI_RESPONSES,
      body: {
        id: "chatcmpl-test",
        object: "chat.completion",
        created: 1,
        model: "test-model",
        choices: [{
          index: 0,
          message: { role: "assistant", content: "partial" },
          finish_reason: "max_tokens",
        }],
      },
    }));

    const payload = await result.response.json();
    expect(payload.status).toBe("incomplete");
    expect(payload.incomplete_details).toEqual({ reason: "max_output_tokens" });
    expect(payload.output[0].status).toBe("incomplete");
  });

  it("preserves max_tokens stop reasons for Claude clients", async () => {
    const result = await handleNonStreamingResponse(buildOptions({
      sourceFormat: FORMATS.CLAUDE,
      body: {
        id: "chatcmpl-test",
        object: "chat.completion",
        created: 1,
        model: "test-model",
        choices: [{
          index: 0,
          message: { role: "assistant", content: "partial" },
          finish_reason: "max_tokens",
        }],
      },
    }));

    const payload = await result.response.json();
    expect(payload.stop_reason).toBe("max_tokens");
  });

  it("marks content-filtered Responses output as incomplete", async () => {
    const result = await handleNonStreamingResponse(buildOptions({
      sourceFormat: FORMATS.OPENAI_RESPONSES,
      body: {
        id: "chatcmpl-test",
        object: "chat.completion",
        created: 1,
        model: "test-model",
        choices: [{
          index: 0,
          message: { role: "assistant", content: "" },
          finish_reason: "content_filter",
        }],
      },
    }));

    const payload = await result.response.json();
    expect(payload.status).toBe("incomplete");
    expect(payload.incomplete_details).toEqual({ reason: "content_filter" });
    expect(payload.output[0].status).toBe("incomplete");
  });

  it("still strips reasoning_content for OpenAI Chat clients", async () => {
    const result = await handleNonStreamingResponse(buildOptions({
      sourceFormat: FORMATS.OPENAI,
      body: {
        id: "chatcmpl-test",
        object: "chat.completion",
        created: 1,
        model: "test-model",
        choices: [{
          index: 0,
          message: {
            role: "assistant",
            reasoning_content: "hidden thinking",
            content: "final answer",
          },
          finish_reason: "stop",
        }],
      },
    }));

    const payload = await result.response.json();
    expect(payload.choices[0].message.content).toBe("final answer");
    expect(payload.choices[0].message).not.toHaveProperty("reasoning_content");
  });

  it("strips reasoning_content for OpenAI Chat tool-call responses", async () => {
    const result = await handleNonStreamingResponse(buildOptions({
      sourceFormat: FORMATS.OPENAI,
      body: {
        id: "chatcmpl-test",
        object: "chat.completion",
        created: 1,
        model: "test-model",
        choices: [{
          index: 0,
          message: {
            role: "assistant",
            reasoning_content: "hidden thinking",
            content: null,
            tool_calls: [{
              id: "call_1",
              type: "function",
              function: { name: "lookup", arguments: "{}" },
            }],
          },
          finish_reason: "tool_calls",
        }],
      },
    }));

    const payload = await result.response.json();
    expect(payload.choices[0].message.tool_calls).toHaveLength(1);
    expect(payload.choices[0].message).not.toHaveProperty("reasoning_content");
  });

  it("keeps reasoning-only OpenAI Chat output", async () => {
    const result = await handleNonStreamingResponse(buildOptions({
      sourceFormat: FORMATS.OPENAI,
      body: {
        id: "chatcmpl-test",
        object: "chat.completion",
        created: 1,
        model: "test-model",
        choices: [{
          index: 0,
          message: {
            role: "assistant",
            reasoning_content: "only thinking",
            content: "",
          },
          finish_reason: "stop",
        }],
      },
    }));

    const payload = await result.response.json();
    expect(payload.choices[0].message.reasoning_content).toBe("only thinking");
  });
});
