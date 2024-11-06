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

const TCL_KEYWORDS = new Set([
  'body',
  'break',
  'case',
  'chain',
  'class',
  'configbody',
  'constructor',
  'continue',
  'default',
  'destructor',
  'else',
  'elseif',
  'error',
  'eval',
  'exit',
  'for',
  'for_array_keys',
  'for_file',
  'for_recursive_glob',
  'foreach',
  'if',
  'in',
  'itcl_class',
  'loop',
  'method',
  'proc',
  'return',
  'switch',
  'then',
  'uplevel',
  'while',
]);

const TCL_BUILTINS = new Set([
  'after',
  'append',
  'array',
  'bgerror',
  'binary',
  'catch',
  'cd',
  'clock',
  'close',
  'concat',
  'console',
  'dde',
  'encoding',
  'eof',
  'exec',
  'expr',
  'fblocked',
  'fconfigure',
  'fcopy',
  'file',
  'fileevent',
  'flush',
  'format',
  'gets',
  'glob',
  'history',
  'incr',
  'info',
  'interp',
  'join',
  'lappend',
  'lindex',
  'linsert',
  'list',
  'llength',
  'load',
  'lrange',
  'lreplace',
  'lsort',
  'namespace',
  'open',
  'package',
  'pid',
  'puts',
  'pwd',
  'read',
  'regexp',
  'registry',
  'regsub',
  'rename',
  'scan',
  'seek',
  'set',
  'socket',
  'source',
  'split',
  'string',
  'subst',
  'tell',
  'time',
  'trace',
  'unknown',
  'unset',
  'vwait',
]);

const TCL_TYPES = new Set([
  'common',
  'global',
  'inherit',
  'itk_option',
  'private',
  'protected',
  'public',
  'upvar',
  'variable',
]);

class HighlightTcl extends Highlighter {

  static NORMAL = 0;
  static WORD = 1;
  static DQUOTE = 2;
  static DQUOTE_BACKSLASH = 3;
  static VAR = 4;
  static VAR2 = 5;
  static VAR_CURLY = 6;
  static COMMENT = 7;
  static COMMENT_BACKSLASH = 8;
  static BACKSLASH = 9;

  constructor(delegate) {
    super(delegate);
    this.word = '';
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      let c = input[i];
      switch (this.state) {

      case HighlightTcl.NORMAL:
        if (!isascii(c) || isalpha(c) || c == '_') {
          this.state = HighlightTcl.WORD;
          this.word += c;
        } else if (c == '"') {
          this.state = HighlightTcl.DQUOTE;
          this.push("span", "string");
          this.append('"');
        } else if (c == '$') {
          this.append('$');
          this.state = HighlightTcl.VAR;
        } else if (c == '#') {
          this.push("span", "comment");
          this.append('#');
          this.state = HighlightTcl.COMMENT;
        } else if (c == '\\') {
          this.state = HighlightTcl.BACKSLASH;
          this.push("span", "escape");
          this.append('\\');
        } else {
          this.append(c);
        }
        break;

      case HighlightTcl.WORD:
        if (!(isspace(c) || c == ';')) {
          this.word += c;
        } else {
          if (TCL_KEYWORDS.has(this.word)) {
            this.push("span", "keyword");
            this.append(this.word);
            this.pop();
          } else if (TCL_TYPES.has(this.word)) {
            this.push("span", "type");
            this.append(this.word);
            this.pop();
          } else if (TCL_BUILTINS.has(this.word)) {
            this.push("span", "builtin");
            this.append(this.word);
            this.pop();
          } else {
            this.append(this.word);
          }
          this.word = '';
          this.epsilon(HighlightTcl.NORMAL);
        }
        break;

      case HighlightTcl.BACKSLASH:
        this.append(c);
        this.pop();
        this.state = HighlightTcl.NORMAL;
        break;

      case HighlightTcl.VAR:
        if (c == '{') {
          this.append('{');
          this.push("span", "var");
          this.state = HighlightTcl.VAR_CURLY;
          break;
        } else {
          this.push("span", "var");
          this.state = HighlightTcl.VAR2;
        }
        // fallthrough

      case HighlightTcl.VAR2:
        if (!isascii(c) || isalnum(c) || c == '_') {
          this.append(c);
        } else {
          this.pop();
          this.epsilon(HighlightTcl.NORMAL);
        }
        break;

      case HighlightTcl.VAR_CURLY:
        if (c == '}') {
          this.pop();
          this.append('}');
          this.state = HighlightTcl.NORMAL;
        } else {
          this.append(c);
        }
        break;

      case HighlightTcl.COMMENT:
        this.append(c);
        if (c == '\n') {
          this.pop();
          this.state = HighlightTcl.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightTcl.COMMENT_BACKSLASH;
        }
        break;

      case HighlightTcl.COMMENT_BACKSLASH:
        this.append(c);
        this.state = HighlightTcl.COMMENT;
        break;

      case HighlightTcl.DQUOTE:
        this.append(c);
        if (c == '"') {
          this.pop();
          this.state = HighlightTcl.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightTcl.DQUOTE_BACKSLASH;
        }
        break;

      case HighlightTcl.DQUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightTcl.DQUOTE;
        break;

      default:
        throw new Error('Invalid state');
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightTcl.WORD:
      if (TCL_KEYWORDS.has(this.word)) {
        this.push("span", "keyword");
        this.append(this.word);
        this.pop();
      } else if (TCL_TYPES.has(this.word)) {
        this.push("span", "type");
        this.append(this.word);
        this.pop();
      } else if (TCL_BUILTINS.has(this.word)) {
        this.push("span", "builtin");
        this.append(this.word);
        this.pop();
      } else {
        this.append(this.word);
      }
      this.word = '';
      break;
    case HighlightTcl.VAR2:
    case HighlightTcl.VAR_CURLY:
    case HighlightTcl.DQUOTE:
    case HighlightTcl.DQUOTE_BACKSLASH:
    case HighlightTcl.COMMENT:
    case HighlightTcl.COMMENT_BACKSLASH:
    case HighlightTcl.BACKSLASH:
      this.pop();
      break;
    default:
      break;
    }
    this.state = HighlightTcl.NORMAL;
    this.delegate.flush();
    this.delta = 1;
  }
}

Highlighter.REGISTRY['tcl'] = HighlightTcl;
