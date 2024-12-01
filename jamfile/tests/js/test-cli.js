import { test } from "./test.js";
import { assertEquals } from "jamfile:assert";

import { parseArgs } from "jamfile:cli";

function testCli() {
  const args = parseArgs(["--foo", "--bar=baz", "./quux.txt"]);
  assertEquals(args, { "_": ["./quux.txt"], "foo": true, "bar": "baz" });
}

export default function () {
  test("jamfile:cli", testCli);
}
