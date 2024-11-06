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

  static NORMAL = 0;
  static WORD = 1;
  static QUOTE = 2;
  static QUOTE_BACKSLASH = 3;
  static DQUOTE = 4;
  static DQUOTE_BACKSLASH = 5;
  static HYPHEN = 6;
  static HYPHEN_HYPHEN = 7;
  static HYPHEN_HYPHEN_LSB = 8;
  static COMMENT = 9;
  static TICK = 10;
  static LSB = 11;
  static LITERAL = 12;
  static LITERAL_RSB = 13;

  constructor(delegate) {
    super(delegate);
    this.level1 = 0;
    this.level2 = 0;
    this.word = '';
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      let c = input[i];
      switch (this.state) {

      case HighlightLua.NORMAL:
        if (!isascii(c) || isalpha(c) || c == '_') {
          this.state = HighlightLua.WORD;
          this.word += c;
        } else if (c == '-') {
          this.state = HighlightLua.HYPHEN;
        } else if (c == '\'') {
          this.state = HighlightLua.QUOTE;
          this.push("span", "string");
          this.append(c);
        } else if (c == '"') {
          this.state = HighlightLua.DQUOTE;
          this.push("span", "string");
          this.append(c);
        } else if (c == '[') {
          this.state = HighlightLua.LSB;
          this.level1 = 0;
        } else {
          this.append(c);
        }
        break;

      case HighlightLua.WORD:
        if (!isascii(c) || isalnum(c) || c == '_') {
          this.word += c;
        } else {
          if (LUA_KEYWORDS.has(this.word)) {
            this.push("span", "keyword");
            this.append(this.word);
            this.pop();
          } else if (LUA_CONSTANTS.has(this.word)) {
            this.push("span", "constant");
            this.append(this.word);
            this.pop();
          } else if (LUA_BUILTINS.has(this.word)) {
            this.push("span", "builtin");
            this.append(this.word);
            this.pop();
          } else {
            this.append(this.word);
          }
          this.word = '';
          this.epsilon(HighlightLua.NORMAL);
        }
        break;

      case HighlightLua.HYPHEN:
        if (c == '-') {
          this.push("span", "comment");
          this.append("--");
          this.state = HighlightLua.HYPHEN_HYPHEN;
        } else {
          this.append('-');
          this.epsilon(HighlightLua.NORMAL);
        }
        break;

      case HighlightLua.HYPHEN_HYPHEN:
        if (c == '[') {
          this.append('[');
          this.state = HighlightLua.HYPHEN_HYPHEN_LSB;
          this.level1 = 0;
        } else {
          this.epsilon(HighlightLua.COMMENT);
        }
        break;

      case HighlightLua.HYPHEN_HYPHEN_LSB:
        if (c == '=') {
          this.append('=');
          ++this.level1;
        } else if (c == '[') {
          this.append('[');
          this.state = HighlightLua.LITERAL;
        } else {
          this.epsilon(HighlightLua.COMMENT);
        }
        break;

      case HighlightLua.COMMENT:
        this.append(c);
        if (c == '\n') {
          this.pop();
          this.state = HighlightLua.NORMAL;
        }
        break;

      case HighlightLua.QUOTE:
        this.append(c);
        if (c == '\'') {
          this.pop();
          this.state = HighlightLua.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightLua.QUOTE_BACKSLASH;
        }
        break;

      case HighlightLua.QUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightLua.QUOTE;
        break;

      case HighlightLua.DQUOTE:
        this.append(c);
        if (c == '"') {
          this.pop();
          this.state = HighlightLua.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightLua.DQUOTE_BACKSLASH;
        }
        break;

      case HighlightLua.DQUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightLua.DQUOTE;
        break;

      case HighlightLua.LSB:
        if (c == '=') {
          ++this.level1;
        } else if (c == '[') {
          this.push("span", "string");
          this.append('[');
          for (let i = 0; i < this.level1; ++i)
            this.append('=');
          this.append('[');
          this.state = HighlightLua.LITERAL;
        } else {
          this.append('[');
          for (let i = 0; i < this.level1; ++i)
            this.append('=');
          this.epsilon(HighlightLua.NORMAL);
        }
        break;

      case HighlightLua.LITERAL:
        this.append(c);
        if (c == ']') {
          this.state = HighlightLua.LITERAL_RSB;
          this.level2 = 0;
        }
        break;

      case HighlightLua.LITERAL_RSB:
        this.append(c);
        if (c == '=') {
          ++this.level2;
        } else if (c == ']') {
          if (this.level2 == this.level1) {
            this.pop();
            this.state = HighlightLua.NORMAL;
          } else {
            this.level2 = 0;
          }
        } else {
          this.state = HighlightLua.LITERAL;
        }
        break;

      default:
        throw new Error('Invalid state');
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightLua.WORD:
      if (LUA_KEYWORDS.has(this.word)) {
        this.push("span", "keyword");
        this.append(this.word);
        this.pop();
      } else if (LUA_CONSTANTS.has(this.word)) {
        this.push("span", "constant");
        this.append(this.word);
        this.pop();
      } else if (LUA_BUILTINS.has(this.word)) {
        this.push("span", "builtin");
        this.append(this.word);
        this.pop();
      } else {
        this.append(this.word);
      }
      this.word = '';
      break;
    case HighlightLua.LSB:
      this.append('[');
      for (let i = 0; i < this.level1; ++i)
        this.append('=');
      break;
    case HighlightLua.HYPHEN:
      this.append('-');
      break;
    case HighlightLua.QUOTE:
    case HighlightLua.QUOTE_BACKSLASH:
    case HighlightLua.DQUOTE:
    case HighlightLua.DQUOTE_BACKSLASH:
    case HighlightLua.COMMENT:
    case HighlightLua.LITERAL:
    case HighlightLua.LITERAL_RSB:
    case HighlightLua.HYPHEN_HYPHEN:
    case HighlightLua.HYPHEN_HYPHEN_LSB:
      this.pop();
      break;
    default:
      break;
    }
    this.state = HighlightLua.NORMAL;
    this.delegate.flush();
    this.delta = 1;
  }
}

Highlighter.REGISTRY['lua'] = HighlightLua;
