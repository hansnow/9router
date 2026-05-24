import { describe, it, expect } from "vitest";
import zlib from "zlib";

import { CursorExecutor } from "../../open-sse/executors/cursor.js";
import { encodeField, wrapConnectRPCFrame } from "../../open-sse/utils/cursorProtobuf.js";

const LEN = 2;
const VARINT = 0;

function cursorResponseFrame({ text = "", thinking = "" }) {
  const responseFields = [];

  if (text) {
    responseFields.push(encodeField(1, LEN, text));
  }

  if (thinking) {
    const thinkingMessage = encodeField(1, LEN, thinking);
    responseFields.push(encodeField(25, LEN, thinkingMessage));
  }

  const response = Buffer.concat(responseFields.map((field) => Buffer.from(field)));
  const envelope = encodeField(2, LEN, response);
  return Buffer.from(wrapConnectRPCFrame(envelope));
}

function cursorToolCallFrame({ id, name, args = "{}", isLast = true }) {
  const toolFields = [
    encodeField(3, LEN, id),
    encodeField(9, LEN, name),
    encodeField(10, LEN, args),
    encodeField(11, VARINT, isLast ? 1 : 0),
  ];
  const toolCall = Buffer.concat(toolFields.map((field) => Buffer.from(field)));
  const envelope = encodeField(1, LEN, toolCall);
  return Buffer.from(wrapConnectRPCFrame(envelope));
}

function gzipConnectRPCFrame(frame) {
  const payload = frame.subarray(5);
  const gzippedPayload = zlib.gzipSync(payload);
  const compressed = Buffer.alloc(5 + gzippedPayload.length);
  compressed[0] = 0x01;
  compressed.writeUInt32BE(gzippedPayload.length, 1);
  gzippedPayload.copy(compressed, 5);
  return compressed;
}

function splitBuffer(buffer, sizes) {
  const chunks = [];
  let offset = 0;
  for (const size of sizes) {
    if (offset >= buffer.length) break;
    chunks.push(buffer.subarray(offset, Math.min(offset + size, buffer.length)));
    offset += size;
  }
  if (offset < buffer.length) chunks.push(buffer.subarray(offset));
  return chunks;
}

function seededRandomChunks(buffer) {
  const chunks = [];
  let offset = 0;
  let seed = 17;
  while (offset < buffer.length) {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    const size = (seed % 7) + 1;
    chunks.push(buffer.subarray(offset, Math.min(offset + size, buffer.length)));
    offset += size;
  }
  return chunks;
}

function parseSSE(text) {
  return text
    .split("\n\n")
    .map((chunk) => chunk.split("\n").find((line) => line.startsWith("data: ")))
    .filter(Boolean)
    .map((line) => line.slice("data: ".length))
    .filter((data) => data !== "[DONE]")
    .filter((data) => data.startsWith("{"))
    .map((data) => JSON.parse(data));
}

function contentFromEvents(events) {
  return events
    .map((event) => event.choices?.[0]?.delta?.content || "")
    .join("");
}

describe("CursorExecutor Composer thinking-field responses", () => {
  it("uses visible content after </think> for non-streaming Composer responses", async () => {
    const executor = new CursorExecutor();
    const buffer = cursorResponseFrame({
      thinking: "private reasoning that must not leak</think>OK",
    });

    const response = executor.transformProtobufToJSON(buffer, "cu/composer-2.5", {
      messages: [{ role: "user", content: "reply OK" }],
    });
    const payload = await response.json();

    expect(payload.choices[0].message.content).toBe("OK");
    expect(JSON.stringify(payload)).not.toContain("private reasoning");
    expect(payload.usage.completion_tokens).toBeGreaterThan(0);
  });

  it("streams only visible content after </think> for Composer responses", async () => {
    const executor = new CursorExecutor();
    const buffer = Buffer.concat([
      cursorResponseFrame({ thinking: "private reasoning" }),
      cursorResponseFrame({ thinking: " that must not leak</think>O" }),
      cursorResponseFrame({ thinking: "K" }),
    ]);

    const response = executor.transformProtobufToSSE(buffer, "composer-2.5-fast", {
      messages: [{ role: "user", content: "reply OK" }],
    });
    const events = parseSSE(await response.text());
    const content = contentFromEvents(events);

    expect(content).toBe("OK");
    expect(JSON.stringify(events)).not.toContain("private reasoning");
    expect(events.at(-1).usage.completion_tokens).toBeGreaterThan(0);
  });

  it("uses visible content after </think> for default responses", async () => {
    const executor = new CursorExecutor();
    const buffer = cursorResponseFrame({
      thinking: "private reasoning that must not leak</think><｜final｜>OK",
    });

    const response = executor.transformProtobufToJSON(buffer, "default", {
      messages: [{ role: "user", content: "reply OK" }],
    });
    const payload = await response.json();

    expect(payload.choices[0].message.content).toBe("OK");
    expect(JSON.stringify(payload)).not.toContain("private reasoning");
    expect(JSON.stringify(payload)).not.toContain("final");
  });

  it("waits for split final markers before streaming visible default content", async () => {
    const executor = new CursorExecutor();
    const buffer = Buffer.concat([
      cursorResponseFrame({ thinking: "private reasoning</think><|fi" }),
      cursorResponseFrame({ thinking: "nal|>OK" }),
    ]);

    const response = executor.transformProtobufToSSE(buffer, "default", {
      messages: [{ role: "user", content: "reply OK" }],
    });
    const events = parseSSE(await response.text());
    const content = contentFromEvents(events);

    expect(content).toBe("OK");
    expect(JSON.stringify(events)).not.toContain("private reasoning");
    expect(JSON.stringify(events)).not.toContain("<|fi");
    expect(events.at(-1).usage.completion_tokens).toBeGreaterThan(0);
  });

  it("does not treat thinking as visible output for non-Composer models", async () => {
    const executor = new CursorExecutor();
    const buffer = cursorResponseFrame({
      thinking: "private reasoning</think>SHOULD_NOT_APPEAR",
    });

    const response = executor.transformProtobufToJSON(buffer, "gpt-5.3-codex", {
      messages: [{ role: "user", content: "hi" }],
    });
    const payload = await response.json();

    expect(payload.choices[0].message.content).toBeNull();
    expect(JSON.stringify(payload)).not.toContain("SHOULD_NOT_APPEAR");
  });

  it("streams a normal empty completion for hidden thinking-only non-Composer responses", async () => {
    const executor = new CursorExecutor();
    const buffer = cursorResponseFrame({
      thinking: "private reasoning</think>SHOULD_NOT_APPEAR",
    });

    const response = executor.transformProtobufToSSE(buffer, "gpt-5.3-codex", {
      messages: [{ role: "user", content: "hi" }],
    });
    const text = await response.text();
    const events = parseSSE(text);

    expect(response.status).toBe(200);
    expect(events[0].choices[0].delta).toEqual({ role: "assistant", content: "" });
    expect(events.at(-1).choices[0].finish_reason).toBe("stop");
    expect(JSON.stringify(events)).not.toContain("SHOULD_NOT_APPEAR");
    expect(text).toContain("data: [DONE]");
  });

  it("converts Cursor pseudo tool text to OpenAI tool calls for non-streaming responses", async () => {
    const executor = new CursorExecutor();
    const buffer = cursorResponseFrame({
      text: [
        "先看项目配置。",
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>",
        "Read",
        "<｜tool▁sep｜>path",
        "/tmp/project/package.json",
        "<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
      ].join("\n"),
    });

    const response = executor.transformProtobufToJSON(buffer, "default", {
      messages: [{ role: "user", content: "这个项目是什么技术栈实现的" }],
    });
    const payload = await response.json();

    expect(payload.choices[0].message.content).toBe("先看项目配置。");
    expect(payload.choices[0].message.tool_calls).toHaveLength(1);
    expect(payload.choices[0].message.tool_calls[0].function.name).toBe("Read");
    expect(JSON.parse(payload.choices[0].message.tool_calls[0].function.arguments)).toEqual({
      file_path: "/tmp/project/package.json",
    });
    expect(payload.choices[0].finish_reason).toBe("tool_calls");
    expect(JSON.stringify(payload)).not.toContain("tool▁calls");
  });

  it("preserves whitespace-sensitive pseudo-tool argument values", async () => {
    const executor = new CursorExecutor();
    const buffer = cursorResponseFrame({
      text: [
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>",
        "Edit",
        "<｜tool▁sep｜>old_string\n  const before = 1;\n",
        "<｜tool▁sep｜>new_string\n  const after = 2;\n",
        "<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
      ].join("\n"),
    });

    const response = executor.transformProtobufToJSON(buffer, "default", {
      messages: [{ role: "user", content: "edit" }],
    });
    const payload = await response.json();
    const args = JSON.parse(payload.choices[0].message.tool_calls[0].function.arguments);

    expect(args.old_string).toBe("  const before = 1;\n\n");
    expect(args.new_string).toBe("  const after = 2;\n\n");
  });

  it("trims pseudo-tool enum and selector arguments", async () => {
    const executor = new CursorExecutor();
    const buffer = cursorResponseFrame({
      text: [
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>",
        "Grep",
        "<｜tool▁sep｜>pattern\nsrc\n",
        "<｜tool▁sep｜>glob\n*.js\n",
        "<｜tool▁sep｜>output_mode\nfiles_with_matches\n",
        "<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
      ].join("\n"),
    });

    const response = executor.transformProtobufToJSON(buffer, "default", {
      messages: [{ role: "user", content: "search" }],
    });
    const payload = await response.json();
    const args = JSON.parse(payload.choices[0].message.tool_calls[0].function.arguments);

    expect(args).toEqual({
      pattern: "src",
      glob: "*.js",
      output_mode: "files_with_matches",
    });
  });

  it("converts split Cursor pseudo tool text to OpenAI tool calls for streaming responses", async () => {
    const executor = new CursorExecutor();
    const buffer = Buffer.concat([
      cursorResponseFrame({ text: "先看项目的依赖和配置文件，确认技术栈。\n<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>\nRe" }),
      cursorResponseFrame({ text: "ad\n<｜tool▁sep｜>path\n/tmp/project/package.json\n<｜tool▁call▁end｜><｜tool▁call▁begin｜>\nGlob" }),
      cursorResponseFrame({ text: "\n<｜tool▁sep｜>glob_pattern\n**/*\n<｜tool▁sep｜>target_directory\n/tmp/project\n<｜tool▁call▁end｜><｜tool▁calls▁end｜>" }),
    ]);

    const response = executor.transformProtobufToSSE(buffer, "default", {
      messages: [{ role: "user", content: "这个项目是什么技术栈实现的" }],
    });
    const events = parseSSE(await response.text());
    const content = contentFromEvents(events);
    const toolDeltas = events
      .flatMap((event) => event.choices?.[0]?.delta?.tool_calls || []);

    expect(content).toBe("先看项目的依赖和配置文件，确认技术栈。");
    expect(toolDeltas).toHaveLength(2);
    expect(toolDeltas.map((tc) => tc.function.name)).toEqual(["Read", "Glob"]);
    expect(JSON.parse(toolDeltas[0].function.arguments)).toEqual({
      file_path: "/tmp/project/package.json",
    });
    expect(JSON.parse(toolDeltas[1].function.arguments)).toEqual({
      pattern: "**/*",
      path: "/tmp/project",
    });
    expect(events.at(-1).choices[0].finish_reason).toBe("tool_calls");
    expect(JSON.stringify(events)).not.toContain("tool▁calls");
  });

  it("trims separator whitespace before pseudo-tool markers split across frames", async () => {
    const executor = new CursorExecutor();
    const buffer = Buffer.concat([
      cursorResponseFrame({ text: "先看项目配置。\n" }),
      cursorResponseFrame({ text: "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>" }),
      cursorResponseFrame({ text: "Read<｜tool▁sep｜>path\n/tmp/project/package.json\n<｜tool▁call▁end｜><｜tool▁calls▁end｜>" }),
    ]);

    const response = executor.transformProtobufChunksToSSE(seededRandomChunks(buffer), "default", {
      messages: [{ role: "user", content: "inspect" }],
    });
    const events = parseSSE(await response.text());
    const content = contentFromEvents(events);
    const toolDeltas = events.flatMap((event) => event.choices?.[0]?.delta?.tool_calls || []);

    expect(content).toBe("先看项目配置。");
    expect(toolDeltas).toHaveLength(1);
    expect(JSON.parse(toolDeltas[0].function.arguments)).toEqual({
      file_path: "/tmp/project/package.json",
    });
  });

  it("emits an assistant role chunk before pseudo-tool-only streaming responses", async () => {
    const executor = new CursorExecutor();
    const buffer = cursorResponseFrame({
      text: [
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>",
        "Read",
        "<｜tool▁sep｜>path",
        "/tmp/project/package.json",
        "<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
      ].join("\n"),
    });

    const response = executor.transformProtobufToSSE(buffer, "default", {
      messages: [{ role: "user", content: "inspect" }],
    });
    const events = parseSSE(await response.text());

    expect(events[0].choices[0].delta).toEqual({ role: "assistant", content: "" });
    expect(events[1].choices[0].delta.tool_calls[0].function.name).toBe("Read");
    expect(JSON.parse(events[1].choices[0].delta.tool_calls[0].function.arguments)).toEqual({
      file_path: "/tmp/project/package.json",
    });
    expect(JSON.stringify(events)).not.toContain("tool▁calls");
  });

  it("streams native tool calls before later text when tool frames arrive first", async () => {
    const executor = new CursorExecutor();
    const buffer = Buffer.concat([
      cursorToolCallFrame({
        id: "call_1",
        name: "Read",
        args: JSON.stringify({ file_path: "/tmp/project/package.json" }),
      }),
      cursorResponseFrame({ text: "先看项目配置。" }),
    ]);

    const response = executor.transformProtobufToSSE(buffer, "default", {
      messages: [{ role: "user", content: "inspect" }],
    });
    const events = parseSSE(await response.text());

    expect(events[0].choices[0].delta).toEqual({ role: "assistant", content: "" });
    expect(events[1].choices[0].delta.tool_calls[0].function.name).toBe("Read");
    expect(events[2].choices[0].delta.content).toBe("先看项目配置。");
  });

  it("parses randomly split ConnectRPC chunks the same as a full response", async () => {
    const executor = new CursorExecutor();
    const buffer = Buffer.concat([
      cursorResponseFrame({ text: "Hello " }),
      cursorResponseFrame({ text: "streaming " }),
      cursorResponseFrame({ text: "world" }),
    ]);
    const body = { messages: [{ role: "user", content: "say hello" }] };

    const fullResponse = executor.transformProtobufToSSE(buffer, "default", body);
    const splitResponse = executor.transformProtobufChunksToSSE(seededRandomChunks(buffer), "default", body);
    const fullEvents = parseSSE(await fullResponse.text());
    const splitEvents = parseSSE(await splitResponse.text());

    expect(contentFromEvents(splitEvents)).toBe("Hello streaming world");
    expect(splitEvents.map((event) => event.choices[0].delta)).toEqual(
      fullEvents.map((event) => event.choices[0].delta)
    );
    expect(splitEvents.at(-1).choices[0].finish_reason).toBe("stop");
  });

  it("handles ConnectRPC header and payload split across chunks", async () => {
    const executor = new CursorExecutor();
    const buffer = cursorResponseFrame({ text: "split header and payload" });
    const chunks = splitBuffer(buffer, [2, 3, 1, 4, 2]);

    const response = executor.transformProtobufChunksToSSE(chunks, "default", {
      messages: [{ role: "user", content: "split" }],
    });
    const events = parseSSE(await response.text());

    expect(contentFromEvents(events)).toBe("split header and payload");
  });

  it("decodes gzip ConnectRPC frames while streaming chunks", async () => {
    const executor = new CursorExecutor();
    const buffer = gzipConnectRPCFrame(cursorResponseFrame({ text: "gzipped" }));

    const response = executor.transformProtobufChunksToSSE(splitBuffer(buffer, [1, 4, 2, 3]), "default", {
      messages: [{ role: "user", content: "gzip" }],
    });
    const events = parseSSE(await response.text());

    expect(contentFromEvents(events)).toBe("gzipped");
  });

  it("streams split native tool call argument fragments with stable index", async () => {
    const executor = new CursorExecutor();
    const buffer = Buffer.concat([
      cursorToolCallFrame({
        id: "call_1",
        name: "Edit",
        args: "{\"file_path\":\"/tmp/a",
        isLast: false,
      }),
      cursorToolCallFrame({
        id: "call_1",
        name: "Edit",
        args: ".js\"}",
        isLast: true,
      }),
    ]);

    const response = executor.transformProtobufChunksToSSE(seededRandomChunks(buffer), "default", {
      messages: [{ role: "user", content: "edit" }],
    });
    const events = parseSSE(await response.text());
    const toolDeltas = events.flatMap((event) => event.choices?.[0]?.delta?.tool_calls || []);

    expect(toolDeltas).toHaveLength(2);
    expect(toolDeltas[0].index).toBe(0);
    expect(toolDeltas[0].function.name).toBe("Edit");
    expect(toolDeltas[0].function.arguments).toBe("{\"file_path\":\"/tmp/a");
    expect(toolDeltas[1].index).toBe(0);
    expect(toolDeltas[1].function.arguments).toBe(".js\"}");
    expect(events.at(-1).choices[0].finish_reason).toBe("tool_calls");
  });

  it("returns an error response for upstream error frames before output", async () => {
    const executor = new CursorExecutor();
    const errorPayload = Buffer.from(JSON.stringify({
      error: { code: "resource_exhausted", message: "rate limited" },
    }));
    const buffer = Buffer.from(wrapConnectRPCFrame(errorPayload));

    const response = executor.transformProtobufChunksToSSE(splitBuffer(buffer, [2, 3]), "default", {
      messages: [{ role: "user", content: "hi" }],
    });
    const payload = await response.json();

    expect(response.status).toBe(429);
    expect(payload.error.message).toBe("rate limited");
  });

  it("emits a terminal error event for upstream error frames after output", async () => {
    const executor = new CursorExecutor();
    const errorPayload = Buffer.from(JSON.stringify({
      error: { code: "internal", message: "upstream failed" },
    }));
    const buffer = Buffer.concat([
      cursorResponseFrame({ text: "partial" }),
      Buffer.from(wrapConnectRPCFrame(errorPayload)),
      cursorResponseFrame({ text: "ignored" }),
    ]);

    const response = executor.transformProtobufChunksToSSE(seededRandomChunks(buffer), "default", {
      messages: [{ role: "user", content: "hi" }],
    });
    const text = await response.text();
    const events = parseSSE(text);

    expect(contentFromEvents(events)).toBe("partial");
    expect(text).toContain("event: error");
    expect(text).toContain("upstream failed");
    expect(text).toContain("data: [DONE]");
  });
});
