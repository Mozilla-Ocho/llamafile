import { test } from "./test.js";
import { assertEquals, assertThrows } from "jamfile:assert";

import { Database, escapeIdentifier, SQLITE_VERSION } from "jamfile:sqlite";

function testSqlite() {
  assertEquals(SQLITE_VERSION, "3.47.1");
  assertEquals(escapeIdentifier("alex"), "alex");
  assertEquals(escapeIdentifier('al"ex'), 'al""ex');
}

function testDb() {
  assertEquals(
    Object.getOwnPropertyNames(Database.prototype).sort(),
    [
      "constructor",
      "execute",
      "executeScript",
      "queryAll",
      "queryRow",
      "queryValue",
      "updateHook",
    ],
  );
}

function testDbQueryAll() {
  const db = new Database(":memory:");
  assertEquals(
    db.queryAll(
      "select fullkey, key, value, X'00' as b, null from json_each('[1,2,3]')",
    ),
    [
      {
        "fullkey": "$[0]",
        "key": 0,
        "value": 1,
        "b": new Uint8Array([0]),
        "null": null,
      },
      {
        "fullkey": "$[1]",
        "key": 1,
        "value": 2,
        "b": new Uint8Array([0]),
        "null": null,
      },
      {
        "fullkey": "$[2]",
        "key": 2,
        "value": 3,
        "b": new Uint8Array([0]),
        "null": null,
      },
    ],
  );
}

function testDbQueryRow() {
  const db = new Database(":memory:");

  const tests = [
    ["select 1 + 1", { "1 + 1": 2 }],
    ["select 1000 as a, 1.111 as b, 'ccc' as c, X'AABBCCDD' as d, null as e", {
      a: 1000,
      b: 1.111,
      c: "ccc",
      d: new Uint8Array([0xaa, 0xbb, 0xcc, 0xdd]),
      e: null,
    }],
  ];

  for (const [sql, expected] of tests) {
    assertEquals(db.queryRow(sql), expected);
  }

  const throwTests = [
    [
      "create table t(a)",
      Error,
      "SQL in queryRow() must be a read-only script",
    ],
  ];

  for (const [sql, errorClass, message] of throwTests) {
    assertThrows(() => db.queryRow(sql), errorClass, message);
  }
}

function testDbQueryValue() {
  const db = new Database(":memory:");

  const tests = [
    {sql:"select 1 + 1", expected: 2},
    {sql:"select 100 + 100", expected: 200},
    {sql:"select upper('mario')", expected: "MARIO"},
    {sql:"select X'00112233'", expected: new Uint8Array([0, 17, 34, 51])},
    {sql:"select sqlite_version()", expected: "3.47.1"},
    {sql:"select vec_version()", expected: "v0.1.6"},
    {sql:"select null", expected: null},

    {sql: "select ?", params: [null], expected: null},
    {sql: "select ?", params: [undefined], expected: null},
    {sql: "select ?", params: [true], expected: 1},
    {sql: "select ?", params: [false], expected: 0},
    {sql: "select ?", params: [1000], expected: 1000},
    {sql: "select ?", params: [123.456], expected: 123.456},
    {sql: "select ?", params: ['alex'], expected: 'alex'},
    {sql: "select ?", params: [new Uint8Array([0xaa, 0xbb])], expected: new Uint8Array([0xaa, 0xbb])},
    {sql: "select ?", params: [[1,2,3,4]], expected: '[1,2,3,4]'},
    {sql: "select ?", params: [{name: 'alex'}], expected: '{"name":"alex"}'},

    // TypedArray
    {sql: "select ?", params: [new Int8Array([2])], expected: new Uint8Array([2])},
    {sql: "select ?", params: [new Float32Array([1.1])], expected: new Uint8Array([205,204,140,63])},

    // Dates?
    {sql: "select ?", params: [new Date('2002')], expected: new Date('2002').toISOString()},

    {sql: "select json(?)", params: [[1,2,3,4]], expected: [1,2,3,4]},

    {sql: "select ? + ?", params: [1, 1], expected: 2},
  ];

  for (const {sql, expected, params} of tests) {
    assertEquals(db.queryValue(sql, params), expected);
  }

  const throwTests = [
    [
      "create table t(a)",
      Error,
      "SQL in queryValue() must be a read-only script",
    ],
  ];

  for (const [sql, errorClass, message] of throwTests) {
    assertThrows(() => db.queryValue(sql), errorClass, message);
  }
}

function testDbUpdateHook() {
  const db = new Database(":memory:");
  const calls = [];

  function onUpdate(op, db, table, rowid) {
    calls.push([op, db, table, rowid]);
  }
  db.updateHook(onUpdate);

  db.executeScript(`
      create table t(a);
      insert into t values (1), (2), (3);
      delete from t where a > 2;
      update t set rowid = 100 where a = 1;
    `);

  assertEquals(calls, [
    ["insert", "main", "t", 1],
    ["insert", "main", "t", 2],
    ["insert", "main", "t", 3],
    ["delete", "main", "t", 3],
    ["update", "main", "t", 100],
  ]);

  // TODO "cleanup" needed to free the duped callback value
  db.updateHook(undefined);
}

export default function () {
  test("jamfile:sqlite", testSqlite);
  test("jamfile:sqlite db", testDb);
  test("jamfile:sqlite db.queryValue", testDbQueryValue);
  test("jamfile:sqlite db.queryRow", testDbQueryRow);
  test("jamfile:sqlite db.queryAll", testDbQueryAll);
  test("jamfile:sqlite db.updateHook", testDbUpdateHook);
}
