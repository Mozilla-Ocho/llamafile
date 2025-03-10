/// <reference types="../../jamfile.d.ts"/>

import { test } from "./test.js";
import { assertEquals } from "jamfile:assert";
import { z, zodToJsonSchema } from "jamfile:zod";


function testZod() {
  const Country = z.object({
    name: z.string(),
    capital: z.string(),
    languages: z.array(z.string()),
  });

  assertEquals(
    zodToJsonSchema(Country),
    {
      "type": "object",
      "properties": {
        "name": { "type": "string" },
        "capital": { "type": "string" },
        "languages": { "type": "array", "items": { "type": "string" } },
      },
      "required": ["name", "capital", "languages"],
      "additionalProperties": false,
      "$schema": "http://json-schema.org/draft-07/schema#",
    },
  );
}
export default function () {
  test("jamfile:zod", testZod);
}
