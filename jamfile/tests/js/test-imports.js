
import { test } from "./test.js";
import { assertEquals } from "jamfile:assert";

import sql from "./example.sql";
import txt from "./example.txt";
import gbnf from "./example.gbnf";

function testImports() {
  assertEquals(sql, "select 1 + 1;");
  assertEquals(txt, "hello!");
  assertEquals(gbnf, `root  ::= (expr "=" ws term "\n")+
expr  ::= term ([-+*/] term)*
term  ::= ident | num | "(" ws expr ")" ws
ident ::= [a-z] [a-z0-9_]* ws
num   ::= [0-9]+ ws
ws    ::= [ \t\n]*`);
}
export default function () {
  test("runtime: imports", testImports);
}
