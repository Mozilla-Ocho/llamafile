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

const SHELL_KEYWORDS = new Set([
  'break',
  'case',
  'coproc',
  'do',
  'done',
  'elif',
  'else',
  'esac',
  'exec',
  'exit',
  'expr',
  'fi',
  'for',
  'function',
  'if',
  'in',
  'return',
  'select',
  'then',
  'time',
  'trap',
  'until',
  'while',
]);

const SHELL_BUILTINS = new Set([
  'alias',
  'bg',
  'bind',
  'builtin',
  'caller',
  'cd',
  'chdir',
  'command',
  'declare',
  'echo',
  'enable',
  'eval',
  'false',
  'fg',
  'getopts',
  'hash',
  'help',
  'jobs',
  'kill',
  'let',
  'local',
  'logout',
  'mapfile',
  'printf',
  'read',
  'readarray',
  'set',
  'shift',
  'source',
  'stop',
  'suspend',
  'test',
  'times',
  'true',
  'type',
  'typeset',
  'ulimit',
  'unalias',
  'unset',
  'wait',
]);

class HighlightShell extends Highlighter {

  static NORMAL = 0;
  static WORD = 1;
  static QUOTE = 2;
  static DQUOTE = 3;
  static DQUOTE_VAR = 4;
  static DQUOTE_VAR2 = 5;
  static DQUOTE_CURL = 6;
  static DQUOTE_CURL_BACKSLASH = 7;
  static DQUOTE_BACKSLASH = 8;
  static TICK = 9;
  static TICK_BACKSLASH = 10;
  static VAR = 11;
  static VAR2 = 12;
  static CURL = 13;
  static CURL_BACKSLASH = 14;
  static COMMENT = 15;
  static LT = 16;
  static LT_LT = 17;
  static LT_LT_NAME = 18;
  static LT_LT_QNAME = 19;
  static HEREDOC_BOL = 20;
  static HEREDOC = 21;
  static HEREDOC_VAR = 22;
  static HEREDOC_VAR2 = 23;
  static HEREDOC_CURL = 24;
  static HEREDOC_CURL_BACKSLASH = 25;
  static BACKSLASH = 26;

  constructor(delegate) {
    super(delegate);
    this.c = 0;
    this.i = 0;
    this.curl = 0;
    this.last = 0;
    this.pending_heredoc = false;
    this.indented_heredoc = false;
    this.no_interpolation = false;
    this.heredoc = '';
    this.word = '';
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      let c = input[i];
      this.last = this.c;
      this.c = c;
      switch (this.state) {

      case HighlightShell.NORMAL:
        if (!isascii(c) || isalpha(c) || c == '_') {
          this.state = HighlightShell.WORD;
          this.word += c;
        } else if (c == '\'') {
          this.state = HighlightShell.QUOTE;
          this.push("span", "string");
          this.append('\'');
        } else if (c == '\\') {
          this.state = HighlightShell.BACKSLASH;
          this.push("span", "escape");
          this.append('\\');
        } else if (c == '"') {
          this.state = HighlightShell.DQUOTE;
          this.push("span", "string");
          this.append('"');
        } else if (c == '`') {
          this.state = HighlightShell.TICK;
          this.push("span", "string");
          this.append('`');
        } else if (c == '$') {
          this.state = HighlightShell.VAR;
          this.append('$');
        } else if (c == '<') {
          this.state = HighlightShell.LT;
          this.append('<');
        } else if (c == '#' && (!this.last || isspace(this.last))) {
          this.push("span", "comment");
          this.append('#');
          this.state = HighlightShell.COMMENT;
        } else if (c == '\n') {
          this.append('\n');
          if (this.pending_heredoc) {
            this.push("span", "string");
            this.pending_heredoc = false;
            this.state = HighlightShell.HEREDOC_BOL;
            this.i = 0;
          }
        } else {
          this.append(c);
        }
        break;

      case HighlightShell.BACKSLASH:
        this.append(c);
        this.pop();
        this.state = HighlightShell.NORMAL;
        break;

      case HighlightShell.WORD:
        if (!isascii(c) || isalnum(c) || c == '_') {
          this.word += c;
        } else {
          if (SHELL_KEYWORDS.has(this.word)) {
            this.push("span", "keyword");
            this.append(this.word);
            this.pop();
          } else if (SHELL_BUILTINS.has(this.word)) {
            this.push("span", "builtin");
            this.append(this.word);
            this.pop();
          } else {
            this.append(this.word);
          }
          this.word = '';
          this.epsilon(HighlightShell.NORMAL);
        }
        break;

      case HighlightShell.VAR:
        if (c == '!' || //
            c == '#' || //
            c == '$' || //
            c == '*' || //
            c == '-' || //
            c == '?' || //
            c == '@' || //
            c == '\\' || //
            c == '^') {
          this.push("span", "var");
          this.append(c);
          this.pop();
          this.state = HighlightShell.NORMAL;
          break;
        } else if (c == '{') {
          this.append('{');
          this.push("span", "var");
          this.state = HighlightShell.CURL;
          this.curl = 1;
          break;
        } else {
          this.push("span", "var");
          this.state = HighlightShell.VAR2;
        }
        // fallthrough

      case HighlightShell.VAR2:
        if (!isascii(c) || isalnum(c) || c == '_') {
          this.append(c);
        } else {
          this.pop();
          this.epsilon(HighlightShell.NORMAL);
        }
        break;

      case HighlightShell.CURL:
        if (c == '\\') {
          this.state = HighlightShell.CURL_BACKSLASH;
          this.pop();
          this.push("span", "escape");
          this.append('\\');
        } else if (c == '{') {
          this.pop();
          this.append('{');
          this.push("span", "var");
          ++this.curl;
        } else if (c == '}') {
          this.pop();
          this.append('}');
          if (!--this.curl) {
            this.state = HighlightShell.NORMAL;
          }
        } else if (ispunct(c)) {
          this.pop();
          this.append(c);
        } else {
          this.append(c);
        }
        break;

      case HighlightShell.CURL_BACKSLASH:
        this.append(c);
        this.pop();
        this.state = HighlightShell.CURL;
        break;

      case HighlightShell.COMMENT:
        this.append(c);
        if (c == '\n') {
          this.pop();
          this.state = HighlightShell.NORMAL;
        }
        break;

      case HighlightShell.QUOTE:
        this.append(c);
        if (c == '\'') {
          this.pop();
          this.state = HighlightShell.NORMAL;
        }
        break;

      case HighlightShell.DQUOTE:
        if (c == '"') {
          this.append(c);
          this.pop();
          this.state = HighlightShell.NORMAL;
        } else if (c == '\\') {
          this.append(c);
          this.state = HighlightShell.DQUOTE_BACKSLASH;
        } else if (c == '$') {
          this.state = HighlightShell.DQUOTE_VAR;
        } else {
          this.append(c);
        }
        break;

      case HighlightShell.DQUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightShell.DQUOTE;
        break;

      case HighlightShell.DQUOTE_VAR:
        if (c == '!' || //
            c == '#' || //
            c == '$' || //
            c == '*' || //
            c == '-' || //
            c == '?' || //
            c == '@' || //
            c == '\\' || //
            c == '^') {
          this.push("span", "bold");
          this.append('$');
          this.append(c);
          this.pop();
          this.state = HighlightShell.DQUOTE;
          break;
        } else if (c == '{') {
          this.push("span", "bold");
          this.append("${");
          this.state = HighlightShell.DQUOTE_CURL;
          this.curl = 1;
          break;
        } else if (c == '(') {
          this.append('$');
          this.state = HighlightShell.DQUOTE_VAR2;
        } else {
          this.push("span", "bold");
          this.append('$');
          this.state = HighlightShell.DQUOTE_VAR2;
        }
        // fallthrough

      case HighlightShell.DQUOTE_VAR2:
        if (!isascii(c) || isalnum(c) || c == '_') {
          this.append(c);
        } else {
          this.pop();
          this.epsilon(HighlightShell.DQUOTE);
        }
        break;

      case HighlightShell.DQUOTE_CURL:
        if (c == '\\') {
          this.state = HighlightShell.DQUOTE_CURL_BACKSLASH;
          this.append('\\');
        } else if (c == '{') {
          this.append('{');
          ++this.curl;
        } else if (c == '}') {
          this.append('}');
          if (!--this.curl) {
            this.pop();
            this.state = HighlightShell.DQUOTE;
          }
        } else if (ispunct(c)) {
          this.append(c);
        } else {
          this.append(c);
        }
        break;

      case HighlightShell.DQUOTE_CURL_BACKSLASH:
        this.append(c);
        this.state = HighlightShell.DQUOTE_CURL;
        break;

      case HighlightShell.TICK:
        this.append(c);
        if (c == '`') {
          this.pop();
          this.state = HighlightShell.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightShell.TICK_BACKSLASH;
        }
        break;

      case HighlightShell.TICK_BACKSLASH:
        this.append(c);
        this.state = HighlightShell.TICK;
        break;

      case HighlightShell.LT:
        if (c == '<') {
          this.append(c);
          this.state = HighlightShell.LT_LT;
          this.heredoc = '';
          this.pending_heredoc = false;
          this.indented_heredoc = false;
          this.no_interpolation = false;
        } else {
          this.epsilon(HighlightShell.NORMAL);
        }
        break;

      case HighlightShell.LT_LT:
        if (c == '-') {
          this.indented_heredoc = true;
          this.append(c);
        } else if (c == '\\') {
          this.append(c);
        } else if (c == '\'') {
          this.state = HighlightShell.LT_LT_QNAME;
          this.push("span", "string");
          this.append(c);
          this.no_interpolation = true;
        } else if (isalpha(c) || c == '_') {
          this.state = HighlightShell.LT_LT_NAME;
          this.heredoc += c;
          this.append(c);
        } else if (isascii(c) && isblank(c)) {
          this.append(c);
        } else {
          this.epsilon(HighlightShell.NORMAL);
        }
        break;

      case HighlightShell.LT_LT_NAME:
        if (isalnum(c) || c == '_') {
          this.state = HighlightShell.LT_LT_NAME;
          this.heredoc += c;
          this.append(c);
        } else if (c == '\n') {
          this.append(c);
          this.push("span", "string");
          this.state = HighlightShell.HEREDOC_BOL;
        } else {
          this.pending_heredoc = true;
          this.epsilon(HighlightShell.NORMAL);
        }
        break;

      case HighlightShell.LT_LT_QNAME:
        this.append(c);
        if (c == '\'') {
          this.pop();
          this.state = HighlightShell.HEREDOC_BOL;
          this.pending_heredoc = true;
          this.state = HighlightShell.NORMAL;
        } else {
          this.heredoc += c;
        }
        break;

      case HighlightShell.HEREDOC_BOL:
        this.append(c);
        if (c == '\n') {
          if (this.i == this.heredoc.length) {
            this.state = HighlightShell.NORMAL;
            this.pop();
          }
          this.i = 0;
        } else if (c == '\t' && this.indented_heredoc) {
          // do nothing
        } else if (this.i < this.heredoc.length && this.heredoc[this.i] == c) {
          this.i++;
        } else {
          this.state = HighlightShell.HEREDOC;
          this.i = 0;
        }
        break;

      case HighlightShell.HEREDOC:
        if (c == '\n') {
          this.append('\n');
          this.state = HighlightShell.HEREDOC_BOL;
        } else if (c == '$' && !this.no_interpolation) {
          this.state = HighlightShell.HEREDOC_VAR;
        } else {
          this.append(c);
        }
        break;

      case HighlightShell.HEREDOC_VAR:
        if (c == '!' || //
            c == '#' || //
            c == '$' || //
            c == '*' || //
            c == '-' || //
            c == '?' || //
            c == '@' || //
            c == '\\' || //
            c == '^') {
          this.push("span", "bold");
          this.append('$');
          this.append(c);
          this.pop();
          this.state = HighlightShell.HEREDOC;
          break;
        } else if (c == '{') {
          this.push("span", "bold");
          this.append("${");
          this.state = HighlightShell.HEREDOC_CURL;
          this.curl = 1;
          break;
        } else if (c == '(') {
          this.append('$');
          this.state = HighlightShell.HEREDOC_VAR2;
        } else {
          this.push("span", "bold");
          this.append('$');
          this.state = HighlightShell.HEREDOC_VAR2;
        }
        // fallthrough

      case HighlightShell.HEREDOC_VAR2:
        if (!isascii(c) || isalnum(c) || c == '_') {
          this.append(c);
        } else {
          this.pop();
          this.epsilon(HighlightShell.HEREDOC);
        }
        break;

      case HighlightShell.HEREDOC_CURL:
        if (c == '\\') {
          this.state = HighlightShell.HEREDOC_CURL_BACKSLASH;
          this.append('\\');
        } else if (c == '{') {
          this.append('{');
          ++this.curl;
        } else if (c == '}') {
          this.append('}');
          if (!--this.curl) {
            this.pop();
            this.state = HighlightShell.HEREDOC;
          }
        } else if (ispunct(c)) {
          this.append(c);
        } else {
          this.append(c);
        }
        break;

      case HighlightShell.HEREDOC_CURL_BACKSLASH:
        this.append(c);
        this.state = HighlightShell.HEREDOC_CURL;
        break;

      default:
        throw new Error('Invalid state');
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightShell.WORD:
      if (SHELL_KEYWORDS.has(this.word)) {
        this.push("span", "keyword");
        this.append(this.word);
        this.pop();
      } else if (SHELL_BUILTINS.has(this.word)) {
        this.push("span", "builtin");
        this.append(this.word);
        this.pop();
      } else {
        this.append(this.word);
      }
      this.word = '';
      break;
    case HighlightShell.DQUOTE_VAR:
      this.append('$');
      this.pop();
      break;
    case HighlightShell.HEREDOC_VAR:
      this.append('$');
      this.pop();
      break;
    case HighlightShell.VAR2:
    case HighlightShell.CURL:
    case HighlightShell.CURL_BACKSLASH:
    case HighlightShell.TICK:
    case HighlightShell.TICK_BACKSLASH:
    case HighlightShell.QUOTE:
    case HighlightShell.DQUOTE:
    case HighlightShell.DQUOTE_VAR2:
    case HighlightShell.DQUOTE_CURL:
    case HighlightShell.DQUOTE_CURL_BACKSLASH:
    case HighlightShell.DQUOTE_BACKSLASH:
    case HighlightShell.COMMENT:
    case HighlightShell.HEREDOC_BOL:
    case HighlightShell.HEREDOC:
    case HighlightShell.HEREDOC_VAR2:
    case HighlightShell.HEREDOC_CURL:
    case HighlightShell.HEREDOC_CURL_BACKSLASH:
    case HighlightShell.LT_LT_QNAME:
    case HighlightShell.BACKSLASH:
      this.pop();
      break;
    default:
      break;
    }
    this.state = HighlightShell.NORMAL;
    this.last = 0;
    this.delegate.flush();
    this.delta = 1;
  }
}

Highlighter.REGISTRY['shell'] = HighlightShell;
Highlighter.REGISTRY['bash'] = HighlightShell;
Highlighter.REGISTRY['ksh'] = HighlightShell;
Highlighter.REGISTRY['sh'] = HighlightShell;
