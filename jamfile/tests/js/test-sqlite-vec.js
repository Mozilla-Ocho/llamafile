import { test } from "./test.js";
import { assertEquals } from "jamfile:assert";
import { Database } from "jamfile:sqlite";

function testSqliteVec() {
  const db = new Database(":memory:");
  assertEquals(
    db.queryValue("select vec_f32('[1, 2, 3, 4]')"),
    new Float32Array([1, 2, 3, 4])
  );
  assertEquals(
    db.queryValue("select vec_bit(X'F0')"),
    new Uint8Array([0xf0])
  );

  // TODO: wut
  /*
  assertEquals(
    db.queryValue("select vec_int8('[1, -1, -128, 127, 0]')"),
    new Int8Array([1, -1, -128, 127, 0])
  );*/
}

export default function () {
  test("sqlite-vec", testSqliteVec);
}
