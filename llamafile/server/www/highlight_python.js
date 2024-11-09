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

const PYTHON_KEYWORDS = new Set([
  'and',
  'as',
  'assert',
  'async',
  'await',
  'break',
  'class',
  'continue',
  'def',
  'del',
  'elif',
  'else',
  'except',
  'finally',
  'for',
  'from',
  'global',
  'if',
  'import',
  'in',
  'is',
  'lambda',
  'nonlocal',
  'not',
  'or',
  'pass',
  'raise',
  'return',
  'try',
  'while',
  'with',
  'yield',
]);

const PYTHON_CONSTANTS = new Set([
  'False',
  'None',
  'True',
]);

const PYTHON_BUILTINS = new Set([
  '__import__',
  'abs',
  'aiter',
  'all',
  'anext',
  'any',
  'ascii',
  'bin',
  'bool',
  'breakpoint',
  'bytearray',
  'bytes',
  'callable',
  'chr',
  'classmethod',
  'compile',
  'complex',
  'delattr',
  'dict',
  'dir',
  'divmod',
  'enumerate',
  'eval',
  'exec',
  'filter',
  'float',
  'format',
  'frozenset',
  'getattr',
  'globals',
  'hasattr',
  'hash',
  'help',
  'hex',
  'id',
  'input',
  'int',
  'isinstance',
  'issubclass',
  'iter',
  'len',
  'list',
  'locals',
  'map',
  'max',
  'memoryview',
  'min',
  'next',
  'object',
  'oct',
  'open',
  'ord',
  'pow',
  'print',
  'property',
  'range',
  'repr',
  'reversed',
  'round',
  'set',
  'setattr',
  'slice',
  'sorted',
  'staticmethod',
  'str',
  'sum',
  'super',
  'tuple',
  'type',
  'vars',
  'zip',
]);

class HighlightPython extends Highlighter {

  static NORMAL = 0;
  static WORD = 1;
  static COM = 2;
  static SQUOTE = 3;
  static SQUOTESTR = 4;
  static SQUOTESTR_BACKSLASH = 5;
  static SQUOTE2 = 6;
  static SQUOTE3 = 7;
  static SQUOTE3_BACKSLASH = 8;
  static SQUOTE31 = 9;
  static SQUOTE32 = 10;
  static DQUOTE = 11;
  static DQUOTESTR = 12;
  static DQUOTESTR_BACKSLASH = 13;
  static DQUOTE2 = 14;
  static DQUOTE3 = 15;
  static DQUOTE3_BACKSLASH = 16;
  static DQUOTE31 = 17;
  static DQUOTE32 = 18;

  constructor(delegate) {
    super(delegate);
    this.word = '';
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      let c = input[i];
      switch (this.state) {

      case HighlightPython.NORMAL:
        if (!isascii(c) || isalpha(c) || c == '_') {
          this.epsilon(HighlightPython.WORD);
        } else if (c == '#') {
          this.state = HighlightPython.COM;
          this.push("span", "comment");
          this.append(c);
        } else if (c == '\'') {
          this.state = HighlightPython.SQUOTE;
          this.push("span", "string");
          this.append(c);
        } else if (c == '"') {
          this.state = HighlightPython.DQUOTE;
          this.push("span", "string");
          this.append(c);
        } else {
          this.append(c);
        }
        break;

      case HighlightPython.WORD:
        if (!isascii(c) || isalnum(c) || c == '_') {
          this.word += c;
        } else {
          if (PYTHON_KEYWORDS.has(this.word)) {
            this.push("span", "keyword");
            this.append(this.word);
            this.pop();
          } else if (PYTHON_BUILTINS.has(this.word)) {
            this.push("span", "builtin");
            this.append(this.word);
            this.pop();
          } else if (PYTHON_CONSTANTS.has(this.word)) {
            this.push("span", "constant");
            this.append(this.word);
            this.pop();
          } else {
            this.append(this.word);
          }
          this.word = '';
          this.epsilon(HighlightPython.NORMAL);
        }
        break;

      case HighlightPython.COM:
        this.append(c);
        if (c == '\n') {
          this.pop();
          this.state = HighlightPython.NORMAL;
        }
        break;

      case HighlightPython.SQUOTE:
        this.append(c);
        if (c == '\'') {
          this.state = HighlightPython.SQUOTE2;
        } else if (c == '\\') {
          this.state = HighlightPython.SQUOTESTR_BACKSLASH;
        } else {
          this.state = HighlightPython.SQUOTESTR;
        }
        break;

      case HighlightPython.SQUOTESTR:
        this.append(c);
        if (c == '\'') {
          this.pop();
          this.state = HighlightPython.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightPython.SQUOTESTR_BACKSLASH;
        }
        break;

      case HighlightPython.SQUOTESTR_BACKSLASH:
        this.append(c);
        this.state = HighlightPython.SQUOTESTR;
        break;

      case HighlightPython.SQUOTE2:
        if (c == '\'') {
          this.append(c);
          this.state = HighlightPython.SQUOTE3;
        } else {
          this.pop();
          this.epsilon(HighlightPython.NORMAL);
        }
        break;

      case HighlightPython.SQUOTE3:
        this.append(c);
        if (c == '\'') {
          this.state = HighlightPython.SQUOTE31;
        } else if (c == '\\') {
          this.state = HighlightPython.SQUOTE3_BACKSLASH;
        }
        break;

      case HighlightPython.SQUOTE31:
        this.append(c);
        if (c == '\'') {
          this.state = HighlightPython.SQUOTE32;
        } else if (c == '\\') {
          this.state = HighlightPython.SQUOTE3_BACKSLASH;
        } else {
          this.state = HighlightPython.SQUOTE3;
        }
        break;

      case HighlightPython.SQUOTE32:
        this.append(c);
        if (c == '\'') {
          this.pop();
          this.state = HighlightPython.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightPython.SQUOTE3_BACKSLASH;
        } else {
          this.state = HighlightPython.SQUOTE3;
        }
        break;

      case HighlightPython.SQUOTE3_BACKSLASH:
        this.append(c);
        this.state = HighlightPython.SQUOTE3;
        break;

      case HighlightPython.DQUOTE:
        this.append(c);
        if (c == '"') {
          this.state = HighlightPython.DQUOTE2;
        } else if (c == '\\') {
          this.state = HighlightPython.DQUOTESTR_BACKSLASH;
        } else {
          this.state = HighlightPython.DQUOTESTR;
        }
        break;

      case HighlightPython.DQUOTESTR:
        this.append(c);
        if (c == '"') {
          this.pop();
          this.state = HighlightPython.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightPython.DQUOTESTR_BACKSLASH;
        }
        break;

      case HighlightPython.DQUOTESTR_BACKSLASH:
        this.append(c);
        this.state = HighlightPython.DQUOTESTR;
        break;

      case HighlightPython.DQUOTE2:
        if (c == '"') {
          this.append('"');
          this.state = HighlightPython.DQUOTE3;
        } else {
          this.pop();
          this.epsilon(HighlightPython.NORMAL);
        }
        break;

      case HighlightPython.DQUOTE3:
        this.append(c);
        if (c == '"') {
          this.state = HighlightPython.DQUOTE31;
        } else if (c == '\\') {
          this.state = HighlightPython.DQUOTE3_BACKSLASH;
        }
        break;

      case HighlightPython.DQUOTE31:
        this.append(c);
        if (c == '"') {
          this.state = HighlightPython.DQUOTE32;
        } else if (c == '\\') {
          this.state = HighlightPython.DQUOTE3_BACKSLASH;
        } else {
          this.state = HighlightPython.DQUOTE3;
        }
        break;

      case HighlightPython.DQUOTE32:
        this.append(c);
        if (c == '"') {
          this.pop();
          this.state = HighlightPython.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightPython.DQUOTE3_BACKSLASH;
        } else {
          this.state = HighlightPython.DQUOTE3;
        }
        break;

      case HighlightPython.DQUOTE3_BACKSLASH:
        this.append(c);
        this.state = HighlightPython.DQUOTE3;
        break;

      default:
        throw new Error('Invalid state');
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightPython.WORD:
      if (PYTHON_KEYWORDS.has(this.word)) {
        this.push("span", "keyword");
        this.append(this.word);
        this.pop();
      } else if (PYTHON_BUILTINS.has(this.word)) {
        this.push("span", "builtin");
        this.append(this.word);
        this.pop();
      } else if (PYTHON_CONSTANTS.has(this.word)) {
        this.push("span", "constant");
        this.append(this.word);
        this.pop();
      } else {
        this.append(this.word);
      }
      this.word = '';
      break;
    case HighlightPython.COM:
    case HighlightPython.SQUOTE:
    case HighlightPython.SQUOTESTR:
    case HighlightPython.SQUOTESTR_BACKSLASH:
    case HighlightPython.SQUOTE2:
    case HighlightPython.SQUOTE3:
    case HighlightPython.SQUOTE3_BACKSLASH:
    case HighlightPython.SQUOTE31:
    case HighlightPython.SQUOTE32:
    case HighlightPython.DQUOTE:
    case HighlightPython.DQUOTESTR:
    case HighlightPython.DQUOTESTR_BACKSLASH:
    case HighlightPython.DQUOTE2:
    case HighlightPython.DQUOTE3:
    case HighlightPython.DQUOTE3_BACKSLASH:
    case HighlightPython.DQUOTE31:
    case HighlightPython.DQUOTE32:
      this.pop();
      break;
    default:
      break;
    }
    this.state = HighlightPython.NORMAL;
    this.delegate.flush();
    this.delta = 1;
  }
}

Highlighter.REGISTRY['python'] = HighlightPython;
Highlighter.REGISTRY['py'] = HighlightPython;
