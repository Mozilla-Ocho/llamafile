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

const D_KEYWORDS = new Set([
  '__FILE_FULL_PATH__',
  '__FILE__',
  '__FUNCTION__',
  '__LINE__',
  '__MODULE__',
  '__PRETTY_FUNCTION__',
  '__gshared',
  '__parameters',
  '__traits',
  '__vector',
  'abstract',
  'alias',
  'align',
  'asm',
  'assert',
  'auto',
  'body',
  'bool',
  'break',
  'byte',
  'case',
  'cast',
  'catch',
  'cdouble',
  'cent',
  'cfloat',
  'char',
  'class',
  'const',
  'continue',
  'creal',
  'dchar',
  'debug',
  'default',
  'delegate',
  'delete',
  'deprecated',
  'do',
  'double',
  'else',
  'enum',
  'export',
  'extern',
  'final',
  'finally',
  'float',
  'for',
  'foreach',
  'foreach_reverse',
  'function',
  'goto',
  'idouble',
  'if',
  'ifloat',
  'immutable',
  'import',
  'in',
  'inout',
  'int',
  'interface',
  'invariant',
  'ireal',
  'is',
  'lazy',
  'long',
  'macro',
  'mixin',
  'module',
  'new',
  'nothrow',
  'out',
  'override',
  'package',
  'pragma',
  'private',
  'protected',
  'public',
  'pure',
  'real',
  'ref',
  'return',
  'scope',
  'shared',
  'short',
  'static',
  'struct',
  'super',
  'switch',
  'synchronized',
  'template',
  'this',
  'throw',
  'try',
  'typeid',
  'typeof',
  'ubyte',
  'ucent',
  'uint',
  'ulong',
  'union',
  'unittest',
  'ushort',
  'version',
  'void',
  'wchar',
  'while',
  'with',
]);

const D_CONSTANTS = new Set([
  'false',
  'null',
  'true',
]);

class HighlightD extends Highlighter {

  static NORMAL = 0;
  static WORD = 1;
  static QUOTE = 2;
  static QUOTE_BACKSLASH = 3;
  static DQUOTE = 4;
  static DQUOTE_BACKSLASH = 5;
  static SLASH = 6;
  static SLASH_SLASH = 7;
  static SLASH_STAR = 8;
  static SLASH_STAR_STAR = 9;
  static SLASH_PLUS = 10;
  static SLASH_PLUS_PLUS = 11;
  static SLASH_PLUS_SLASH = 12;
  static BACKTICK = 13;
  static R = 14;
  static R_DQUOTE = 15;
  static Q = 16;
  static Q_DQUOTE = 17;
  static Q_DQUOTE_STRING = 18;
  static Q_DQUOTE_STRING_END = 19;
  static Q_DQUOTE_IDENT = 20;
  static Q_DQUOTE_HEREDOC = 21;
  static Q_DQUOTE_HEREDOC_BOL = 22;
  static Q_DQUOTE_HEREDOC_END = 23;
  static X = 24;
  static X_DQUOTE = 25;

  static is_line_terminator(c) {
    switch (c) {
    case '\r':
    case '\n':
    case '\u2028': // LINE SEPARATOR
    case '\u2029': // PARAGRAPH SEPARATOR
      return true;
    default:
      return false;
    }
  }

  static mirror(c) {
    switch (c) {
    case '(':
      return ')';
    case '{':
      return '}';
    case '[':
      return ']';
    case '<':
      return '>';
    default:
      return c;
    }
  }

  constructor(delegate) {
    super(delegate);
    this.depth = 0;
    this.opener = '';
    this.closer = '';
    this.word = '';
    this.heredoc = '';
    this.heredoc2 = '';
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      let c = input[i];
      if (c == '\r')
        continue;
      if (c == 0xFEFF)
        continue; // utf-8 bom
      switch (this.state) {

      case HighlightD.NORMAL:
        if (c == 'r') {
          this.state = HighlightD.R;
        } else if (c == 'q') {
          this.state = HighlightD.Q;
        } else if (c == 'x') {
          this.state = HighlightD.X;
        } else if (!isascii(c) || isalpha(c) || c == '_') {
          this.state = HighlightD.WORD;
          this.word += c;
        } else if (c == '`') {
          this.state = HighlightD.BACKTICK;
          this.push("span", "string");
          this.append('`');
        } else if (c == '/') {
          this.state = HighlightD.SLASH;
        } else if (c == '\'') {
          this.state = HighlightD.QUOTE;
          this.push("span", "string");
          this.append('\'');
        } else if (c == '"') {
          this.state = HighlightD.DQUOTE;
          this.push("span", "string");
          this.append('"');
        } else {
          this.append(c);
        }
        break;

      case HighlightD.WORD:
        if (!isascii(c) || isalnum(c) || c == '_') {
          this.word += c;
        } else {
          if (D_KEYWORDS.has(this.word)) {
            this.push("span", "keyword");
            this.append(this.word);
            this.pop();
          } else if (D_CONSTANTS.has(this.word)) {
            this.push("span", "constant");
            this.append(this.word);
            this.pop();
          } else {
            this.append(this.word);
          }
          this.word = '';
          this.epsilon(HighlightD.NORMAL);
        }
        break;

      case HighlightD.SLASH:
        if (c == '/') {
          this.push("span", "comment");
          this.append("//");
          this.state = HighlightD.SLASH_SLASH;
        } else if (c == '*') {
          this.push("span", "comment");
          this.append("/*");
          this.state = HighlightD.SLASH_STAR;
        } else if (c == '+') {
          this.push("span", "comment");
          this.append("/+");
          this.state = HighlightD.SLASH_PLUS;
          this.depth = 1;
        } else {
          this.append('/');
          this.epsilon(HighlightD.NORMAL);
        }
        break;

      case HighlightD.SLASH_SLASH:
        this.append(c);
        if (HighlightD.is_line_terminator(c)) {
          this.pop();
          this.state = HighlightD.NORMAL;
        }
        break;

      case HighlightD.SLASH_STAR:
        this.append(c);
        if (c == '*')
          this.state = HighlightD.SLASH_STAR_STAR;
        break;

      case HighlightD.SLASH_STAR_STAR:
        this.append(c);
        if (c == '/') {
          this.pop();
          this.state = HighlightD.NORMAL;
        } else if (c != '*') {
          this.state = HighlightD.SLASH_STAR;
        }
        break;

      case HighlightD.SLASH_PLUS:
        this.append(c);
        if (c == '+') {
          this.state = HighlightD.SLASH_PLUS_PLUS;
        } else if (c == '/') {
          this.state = HighlightD.SLASH_PLUS_SLASH;
        }
        break;

      case HighlightD.SLASH_PLUS_PLUS:
        this.append(c);
        if (c == '/') {
          if (!--this.depth) {
            this.pop();
            this.state = HighlightD.NORMAL;
          } else {
            this.state = HighlightD.SLASH_PLUS;
          }
        } else if (c != '+') {
          this.state = HighlightD.SLASH_PLUS;
        }
        break;

      case HighlightD.SLASH_PLUS_SLASH:
        this.append(c);
        if (c == '+') {
          ++this.depth;
          this.state = HighlightD.SLASH_PLUS;
        } else if (c != '/') {
          this.state = HighlightD.SLASH_PLUS;
        }
        break;

      case HighlightD.QUOTE:
        this.append(c);
        if (c == '\'') {
          this.pop();
          this.state = HighlightD.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightD.QUOTE_BACKSLASH;
        }
        break;

      case HighlightD.QUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightD.QUOTE;
        break;

      case HighlightD.DQUOTE:
        this.append(c);
        if (c == '"') {
          this.pop();
          this.state = HighlightD.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightD.DQUOTE_BACKSLASH;
        }
        break;

      case HighlightD.DQUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightD.DQUOTE;
        break;

      case HighlightD.R:
        if (c == '"') {
          this.state = HighlightD.R_DQUOTE;
          this.append('r');
          this.push("span", "string");
          this.append('"');
        } else {
          this.word += 'r';
          this.epsilon(HighlightD.WORD);
        }
        break;

      case HighlightD.R_DQUOTE:
        this.append(c);
        if (c == '"') {
          this.pop();
          this.state = HighlightD.NORMAL;
        }
        break;

      case HighlightD.BACKTICK:
        this.append(c);
        if (c == '`') {
          this.pop();
          this.state = HighlightD.NORMAL;
        }
        break;

      case HighlightD.Q:
        if (c == '"') {
          this.state = HighlightD.Q_DQUOTE;
          this.append('q');
          this.push("span", "string");
          this.append('"');
        } else {
          this.word += 'q';
          this.epsilon(HighlightD.WORD);
        }
        break;

      case HighlightD.Q_DQUOTE:
        this.append(c);
        if (!isascii(c) || isalpha(c) || c == '_') {
          this.heredoc = c;
          this.state = HighlightD.Q_DQUOTE_IDENT;
        } else {
          this.opener = c;
          this.closer = HighlightD.mirror(c);
          this.depth = 1;
          this.state = HighlightD.Q_DQUOTE_STRING;
        }
        break;

      case HighlightD.Q_DQUOTE_STRING:
        if (c == this.closer) {
          if (this.closer != this.opener) {
            if (this.depth) {
              --this.depth;
            } else {
              this.pop();
              this.push("span", "warning");
            }
            if (!this.depth) {
              this.state = HighlightD.Q_DQUOTE_STRING_END;
            }
          } else {
            this.state = HighlightD.Q_DQUOTE_STRING_END;
          }
        } else if (c == this.opener && this.closer != this.opener) {
          ++this.depth;
        }
        this.append(c);
        break;

      case HighlightD.Q_DQUOTE_STRING_END:
        if (c == '"') {
          this.append(c);
          this.pop();
          this.state = HighlightD.NORMAL;
        } else {
          this.pop();
          this.push("span", "warning");
          this.epsilon(HighlightD.Q_DQUOTE_STRING);
        }
        break;

      case HighlightD.Q_DQUOTE_IDENT:
        if (HighlightD.is_line_terminator(c)) {
          this.state = HighlightD.Q_DQUOTE_HEREDOC_BOL;
          this.heredoc2 = '';
        } else if (!isascii(c) || isalpha(c) || c == '_') {
          this.heredoc += c;
        } else {
          this.pop();
          this.push("span", "warning");
          this.state = HighlightD.Q_DQUOTE_HEREDOC_BOL;
          this.heredoc2 = '';
        }
        this.append(c);
        break;

      case HighlightD.Q_DQUOTE_HEREDOC:
        this.append(c);
        if (HighlightD.is_line_terminator(c)) {
          this.state = HighlightD.Q_DQUOTE_HEREDOC_BOL;
          this.heredoc2 = '';
        }
        break;

      case HighlightD.Q_DQUOTE_HEREDOC_BOL:
        this.append(c);
        if (HighlightD.is_line_terminator(c)) {
          this.state = HighlightD.Q_DQUOTE_HEREDOC_BOL;
          this.heredoc2 = '';
        } else {
          this.heredoc2 += c;
          if (this.heredoc.startsWith(this.heredoc2)) {
            if (this.heredoc == this.heredoc2) {
              this.state = HighlightD.Q_DQUOTE_HEREDOC_END;
            }
          } else {
            this.state = HighlightD.Q_DQUOTE_HEREDOC;
          }
        }
        break;

      case HighlightD.Q_DQUOTE_HEREDOC_END:
        if (c == '"') {
          this.append(c);
          this.pop();
          this.state = HighlightD.NORMAL;
        } else {
          this.pop();
          this.push("span", "warning");
          this.epsilon(HighlightD.Q_DQUOTE_HEREDOC);
        }
        break;

      case HighlightD.X:
        if (c == '"') {
          this.append('x');
          this.push("span", "string");
          this.append('"');
          this.state = HighlightD.X_DQUOTE;
        } else {
          this.word += 'x';
          this.epsilon(HighlightD.WORD);
        }
        break;

      case HighlightD.X_DQUOTE:
        if (HighlightD.is_line_terminator(c) || isspace(c) || isxdigit(c)) {
          this.append(c);
        } else if (c == '"') {
          this.append('"');
          this.pop();
          this.state = HighlightD.NORMAL;
        } else {
          this.pop();
          this.push("span", "warning");
          this.append(c);
          this.pop();
          this.push("span", "string");
        }
        break;

      default:
        throw new Error('Invalid state');
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightD.WORD:
        if (D_KEYWORDS.has(this.word)) {
            this.push("span", "keyword");
            this.append(this.word);
            this.pop();
        } else if (D_CONSTANTS.has(this.word)) {
            this.push("span", "constant");
            this.append(this.word);
            this.pop();
        } else {
            this.append(this.word);
        }
        this.word = '';
        break;
    case HighlightD.SLASH:
        this.append('/');
        break;
    case HighlightD.R:
        this.append('r');
        break;
    case HighlightD.Q:
        this.append('q');
        break;
    case HighlightD.X:
        this.append('x');
        break;
    case HighlightD.QUOTE:
    case HighlightD.QUOTE_BACKSLASH:
    case HighlightD.DQUOTE:
    case HighlightD.DQUOTE_BACKSLASH:
    case HighlightD.SLASH_SLASH:
    case HighlightD.SLASH_STAR:
    case HighlightD.SLASH_STAR_STAR:
    case HighlightD.SLASH_PLUS:
    case HighlightD.SLASH_PLUS_PLUS:
    case HighlightD.SLASH_PLUS_SLASH:
    case HighlightD.BACKTICK:
    case HighlightD.R_DQUOTE:
    case HighlightD.Q_DQUOTE:
    case HighlightD.Q_DQUOTE_STRING:
    case HighlightD.Q_DQUOTE_STRING_END:
    case HighlightD.Q_DQUOTE_IDENT:
    case HighlightD.Q_DQUOTE_HEREDOC:
    case HighlightD.Q_DQUOTE_HEREDOC_BOL:
    case HighlightD.Q_DQUOTE_HEREDOC_END:
        this.pop();
        break;
    default:
        break;
    }
    this.c = 0;
    this.u = 0;
    this.state = HighlightD.NORMAL;
    this.delegate.flush();
  }
}

Highlighter.REGISTRY['d'] = HighlightD;
