import { describe, it, expect } from "vitest";

import { buildCursorRequest } from "../../open-sse/translator/request/openai-to-cursor.js";
import { decodeMessage, generateCursorBody } from "../../open-sse/utils/cursorProtobuf.js";

const PNG_1X1_BASE64 =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=";

function firstLenField(message, fieldNum) {
  const fields = message instanceof Map ? message : decodeMessage(message);
  return fields.get(fieldNum)?.[0]?.value;
}

function firstMessageFromBody(body) {
  const wrapper = decodeMessage(body.subarray(5));
  const request = decodeMessage(wrapper.get(1)[0].value);
  return decodeMessage(firstLenField(request, 1));
}

describe("Cursor image protobuf request encoding", () => {
  it("keeps OpenAI image_url parts and encodes them as Cursor ImageProto", () => {
    const cursorBody = buildCursorRequest("default", {
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "describe this image" },
            {
              type: "image_url",
              image_url: { url: `data:image/png;base64,${PNG_1X1_BASE64}` }
            }
          ]
        }
      ]
    });

    expect(cursorBody.messages[0].content).toBe("describe this image");
    expect(cursorBody.messages[0].images).toHaveLength(1);

    const body = Buffer.from(generateCursorBody(cursorBody.messages, "default"));
    const payload = body.subarray(5);
    const message = firstMessageFromBody(body);
    const image = decodeMessage(message.get(10)[0].value);
    const dimension = decodeMessage(image.get(2)[0].value);

    expect(Buffer.from(image.get(1)[0].value).equals(Buffer.from(PNG_1X1_BASE64, "base64"))).toBe(true);
    expect(dimension.get(1)[0].value).toBe(1);
    expect(dimension.get(2)[0].value).toBe(1);
  });

  it("keeps Claude base64 image blocks after Claude to OpenAI translation shape", () => {
    const cursorBody = buildCursorRequest("default", {
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "read image" },
            {
              type: "image",
              source: {
                type: "base64",
                media_type: "image/png",
                data: PNG_1X1_BASE64
              }
            }
          ]
        }
      ]
    });

    expect(cursorBody.messages[0].images).toEqual([
      { url: `data:image/png;base64,${PNG_1X1_BASE64}` }
    ]);

    const body = Buffer.from(generateCursorBody(cursorBody.messages, "default"));
    const message = firstMessageFromBody(body);

    expect(message.has(10)).toBe(true);
  });

  it("keeps remote image URLs as text instead of silently dropping them", () => {
    const cursorBody = buildCursorRequest("default", {
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "look" },
            {
              type: "image_url",
              image_url: { url: "https://example.com/photo.png" }
            }
          ]
        }
      ]
    });

    expect(cursorBody.messages[0].content).toContain("look");
    expect(cursorBody.messages[0].content).toContain("[Image: https://example.com/photo.png]");
    expect(cursorBody.messages[0].images).toBeUndefined();
  });
});
