import { test } from "./test.js";

import { formatBytes, formatDuration } from "jamfile:fmt";
import { assertEquals } from "jamfile:assert";

function testFmt() {
  assertEquals(formatBytes(3012), "3.01 kB");
  assertEquals(
    formatDuration(
      (60 * 1000) + (1000 * 32) + 4,
      { ignoreZero: true },
    ),
    "1m 32s 4ms",
  );
}
export default function () {
  test("jamfile:fmt", testFmt);
}
