import { test } from "./test.js";
import { assert, assertEquals } from "jamfile:assert";

function testGlobals() {
  assertEquals(
    Array.from(Object.keys(globalThis)).sort(),
    [
      "Jamfile",
      "argv0",
      "console",
      "performance",
    ],
  );
  assertEquals(
    Object.keys(Jamfile).sort(),
    ["args", "version"],
  );

  assert(typeof Jamfile.version === "string");
  assert(Array.isArray(Jamfile.args));
  assertEquals(Jamfile.args, []);
}
export default function () {
  test("runtime: globals", testGlobals);
}
