// Copyright 2024 Mozilla Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

const LUA_KEYWORDS = new Set([
  'and',
  'break',
  'do',
  'else',
  'elseif',
  'end',
  'for',
  'function',
  'goto',
  'if',
  'in',
  'local',
  'not',
  'or',
  'repeat',
  'return',
  'then',
  'until',
  'while',
]);

const LUA_BUILTINS = new Set([
  '__add',
  '__band',
  '__bnot',
  '__bor',
  '__bxor',
  '__call',
  '__close',
  '__concat',
  '__div',
  '__eq',
  '__gc',
  '__idiv',
  '__index',
  '__le',
  '__len',
  '__lt',
  '__mod',
  '__mode',
  '__mul',
  '__name',
  '__newindex',
  '__pow',
  '__shl',
  '__shr',
  '__sub',
  '__unm',
  'assert',
  'collectgarbage',
  'coroutine',
  'debug',
  'dofile',
  'error',
  'getmetatable',
  'io',
  'ipairs',
  'load',
  'loadfile',
  'math',
  'next',
  'os',
  'package',
  'pairs',
  'pcall',
  'print',
  'rawequal',
  'rawget',
  'rawlen',
  'rawset',
  'require',
  'select',
  'setmetatable',
  'string',
  'table',
  'tonumber',
  'tostring',
  'type',
  'utf8',
  'warn',
  'xpcall',
]);

const LUA_CONSTANTS = new Set([
  '_G',
  '_VERSION',
  'arg',
  'false',
  'nil',
  'true',
]);

class HighlightLua extends Highlighter {

  constructor(delegate) {
    super(delegate);
  }

  feed(input) {
    this.append(input);
  }

  flush() {
    this.delegate.flush();
  }
}

Highlighter.REGISTRY['lua'] = HighlightLua;
