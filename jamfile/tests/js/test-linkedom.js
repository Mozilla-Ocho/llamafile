/// <reference types="../../jamfile.d.ts"/>

import { test } from "./test.js";
import { assertEquals } from "jamfile:assert";
import { parseHTML } from "jamfile:linkedom";


function testLinkedom() {
  const {document} = parseHTML('<div id="lol" class="Alex">yo</div>')
  
  assertEquals(
    Array.from(document.querySelector('div').classList), 
    ['Alex']
  );
}

export default function () {
  test("jamfile:linkedom", testLinkedom);
}
