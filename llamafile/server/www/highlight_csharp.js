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

const CSHARP_KEYWORDS = new Set([
  'abstract',
  'add',
  'alias',
  'allows',
  'and',
  'args',
  'as',
  'ascending',
  'async',
  'await',
  'base',
  'bool',
  'break',
  'by',
  'byte',
  'case',
  'catch',
  'char',
  'checked',
  'class',
  'const',
  'continue',
  'decimal',
  'default',
  'delegate',
  'descending',
  'do',
  'double',
  'dynamic',
  'else',
  'enum',
  'equals',
  'event',
  'explicit',
  'extern',
  'file',
  'finally',
  'fixed',
  'float',
  'for',
  'foreach',
  'from',
  'get',
  'global',
  'goto',
  'group',
  'if',
  'implicit',
  'in',
  'init',
  'int',
  'interface',
  'internal',
  'into',
  'is',
  'join',
  'let',
  'lock',
  'long',
  'managed',
  'nameof',
  'namespace',
  'new',
  'nint',
  'not',
  'notnull',
  'nuint',
  'object',
  'on',
  'operator',
  'or',
  'orderby',
  'out',
  'override',
  'params',
  'partial',
  'private',
  'protected',
  'public',
  'readonly',
  'record',
  'ref',
  'remove',
  'required',
  'return',
  'sbyte',
  'scoped',
  'sealed',
  'select',
  'set',
  'short',
  'sizeof',
  'stackalloc',
  'static',
  'string',
  'struct',
  'switch',
  'this',
  'throw',
  'try',
  'typeof',
  'uint',
  'ulong',
  'unchecked',
  'unmanaged',
  'unsafe',
  'ushort',
  'using',
  'value',
  'var',
  'virtual',
  'void',
  'volatile',
  'when',
  'where',
  'while',
  'with',
  'yield',
]);

const CSHARP_CONSTANTS = new Set([
  'false',
  'null',
  'true',
]);

class HighlightCsharp extends Highlighter {

  static NORMAL = 0;
  static WORD = 1;
  static QUOTE = 2;
  static QUOTE_BACKSLASH = 3;
  static SLASH = 4;
  static SLASH_SLASH = 5;
  static SLASH_STAR = 6;
  static SLASH_STAR_STAR = 7;
  static DQUOTE = 8;
  static STR = 9;
  static STR_BACKSLASH = 10;
  static DQUOTE_DQUOTE = 11;
  static DQUOTE_DQUOTE_DQUOTE = 12;
  static TRIPS = 13;
  static TRIPS_DQUOTE = 14;

  constructor(delegate) {
    super(delegate);
    this.trips1 = 0;
    this.trips2 = 0;
    this.word = '';
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      let c = input[i];
      switch (this.state) {

      case HighlightCsharp.NORMAL:
        if (!isascii(c) || isalpha(c) || c == '_' || c == '#') {
          this.state = HighlightCsharp.WORD;
          this.word += c;
        } else if (c == '/') {
          this.state = HighlightCsharp.SLASH;
        } else if (c == '\'') {
          this.state = HighlightCsharp.QUOTE;
          this.push("span", "string");
          this.append(c);
        } else if (c == '"') {
          this.state = HighlightCsharp.DQUOTE;
          this.push("span", "string");
          this.append(c);
        } else {
          this.append(c);
        }
        break;

      case HighlightCsharp.WORD:
        if (!isascii(c) || isalnum(c) || c == '_' || c == '$' || c == '#') {
          this.word += c;
        } else {
          if (CSHARP_KEYWORDS.has(this.word)) {
            this.push("span", "keyword");
            this.append(this.word);
            this.pop();
          } else if (CSHARP_CONSTANTS.has(this.word)) {
            this.push("span", "constant");
            this.append(this.word);
            this.pop();
          } else {
            this.append(this.word);
          }
          this.word = '';
          this.epsilon(HighlightCsharp.NORMAL);
        }
        break;

      case HighlightCsharp.SLASH:
        if (c == '/') {
          this.push("span", "comment");
          this.append("//");
          this.state = HighlightCsharp.SLASH_SLASH;
        } else if (c == '*') {
          this.push("span", "comment");
          this.append("/*");
          this.state = HighlightCsharp.SLASH_STAR;
        } else {
          this.append('/');
          this.epsilon(HighlightCsharp.NORMAL);
        }
        break;

      case HighlightCsharp.SLASH_SLASH:
        this.append(c);
        if (c == '\n') {
          this.pop();
          this.state = HighlightCsharp.NORMAL;
        }
        break;

      case HighlightCsharp.SLASH_STAR:
        this.append(c);
        if (c == '*')
          this.state = HighlightCsharp.SLASH_STAR_STAR;
        break;

      case HighlightCsharp.SLASH_STAR_STAR:
        this.append(c);
        if (c == '/') {
          this.pop();
          this.state = HighlightCsharp.NORMAL;
        } else if (c != '*') {
          this.state = HighlightCsharp.SLASH_STAR;
        }
        break;

      case HighlightCsharp.QUOTE:
        this.append(c);
        if (c == '\'') {
          this.pop();
          this.state = HighlightCsharp.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightCsharp.QUOTE_BACKSLASH;
        }
        break;

      case HighlightCsharp.QUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightCsharp.QUOTE;
        break;

      case HighlightCsharp.DQUOTE:
        this.append(c);
        if (c == '"') {
          this.state = HighlightCsharp.DQUOTE_DQUOTE;
        } else if (c == '\\') {
          this.state = HighlightCsharp.STR_BACKSLASH;
        } else {
          this.state = HighlightCsharp.STR;
        }
        break;

      case HighlightCsharp.STR:
        this.append(c);
        if (c == '"') {
          this.pop();
          this.state = HighlightCsharp.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightCsharp.STR_BACKSLASH;
        }
        break;

      case HighlightCsharp.STR_BACKSLASH:
        this.append(c);
        this.state = HighlightCsharp.STR;
        break;

      case HighlightCsharp.DQUOTE_DQUOTE:
        if (c == '"') {
          this.append(c);
          this.state = HighlightCsharp.DQUOTE_DQUOTE_DQUOTE;
          this.trips1 = 3;
          this.trips2 = 0;
        } else {
          this.pop();
          this.epsilon(HighlightCsharp.NORMAL);
        }
        break;

      case HighlightCsharp.DQUOTE_DQUOTE_DQUOTE:
        if (c == '"') {
          this.append(c);
          ++this.trips1;
          if (++this.trips2 == 3) {
            this.pop();
            this.state = HighlightCsharp.NORMAL;
          }
          break;
        } else {
          this.trips2 = 0;
          this.state = HighlightCsharp.TRIPS;
        }
        // fallthrough

      case HighlightCsharp.TRIPS:
        this.append(c);
        if (c == '"') {
          this.state = HighlightCsharp.TRIPS_DQUOTE;
          this.trips2 = 1;
        }
        break;

      case HighlightCsharp.TRIPS_DQUOTE:
        this.append(c);
        if (c == '"') {
          if (++this.trips2 == this.trips1) {
            this.pop();
            this.state = HighlightCsharp.NORMAL;
          }
        } else {
          this.state = HighlightCsharp.TRIPS;
        }
        break;

      default:
        throw new Error('Invalid state');
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightCsharp.WORD:
      if (CSHARP_KEYWORDS.has(this.word)) {
        this.push("span", "keyword");
        this.append(this.word);
        this.pop();
      } else if (CSHARP_CONSTANTS.has(this.word)) {
        this.push("span", "constant");
        this.append(this.word);
        this.pop();
      } else {
        this.append(this.word);
      }
      this.word = '';
      break;
    case HighlightCsharp.SLASH:
      this.append('/');
      break;
    case HighlightCsharp.QUOTE:
    case HighlightCsharp.QUOTE_BACKSLASH:
    case HighlightCsharp.DQUOTE:
    case HighlightCsharp.STR:
    case HighlightCsharp.STR_BACKSLASH:
    case HighlightCsharp.DQUOTE_DQUOTE:
    case HighlightCsharp.DQUOTE_DQUOTE_DQUOTE:
    case HighlightCsharp.TRIPS:
    case HighlightCsharp.TRIPS_DQUOTE:
    case HighlightCsharp.SLASH_SLASH:
    case HighlightCsharp.SLASH_STAR:
    case HighlightCsharp.SLASH_STAR_STAR:
      this.pop();
      break;
    default:
      break;
    }
    this.state = HighlightCsharp.NORMAL;
    this.delegate.flush();
    this.delta = 1;
  }
}

Highlighter.REGISTRY['csharp'] = HighlightCsharp;
Highlighter.REGISTRY['cs'] = HighlightCsharp;
Highlighter.REGISTRY['c#'] = HighlightCsharp;
