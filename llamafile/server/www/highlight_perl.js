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

const PERL_KEYWORDS = new Set([
  'BEGIN',
  'END',
  'continue',
  'default',
  'die',
  'do',
  'dump',
  'else',
  'elsif',
  'eval',
  'exec',
  'exit',
  'for',
  'foreach',
  'given',
  'goto',
  'has',
  'if',
  'import',
  'last',
  'local',
  'my',
  'next',
  'no',
  'our',
  'package',
  'redo',
  'require',
  'return',
  'state',
  'sub',
  'unless',
  'until',
  'use',
  'when',
  'while',
]);

class HighlightPerl extends Highlighter {

  static NORMAL = 0;
  static WORD = 1;
  static QUOTE = 2;
  static QUOTE_BACKSLASH = 3;
  static DQUOTE = 4;
  static DQUOTE_BACKSLASH = 5;
  static TICK = 6;
  static TICK_BACKSLASH = 7;
  static VAR = 8;
  static VAR2 = 9;
  static COMMENT = 10;
  static LT = 11;
  static LT_LT = 12;
  static LT_LT_NAME = 13;
  static LT_LT_QNAME = 14;
  static HEREDOC_BOL = 15;
  static HEREDOC = 16;
  static REGEX = 17;
  static REGEX_BACKSLASH = 18;
  static S_REGEX = 19;
  static S_REGEX_BACKSLASH = 20;
  static S_REGEX_S = 21;
  static S_REGEX_S_BACKSLASH = 22;
  static EQUAL = 23;
  static BACKSLASH = 24;

  static EXPECT_VALUE = 0;
  static EXPECT_OPERATOR = 1;

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

  static is_magic_var(c) {
    switch (c) {
    case '!':
    case '"':
    case '#':
    case '&':
    case '-':
    case '/':
    case '<':
    case '=':
    case '>':
    case '?':
    case '@':
    case '\'':
    case '\\':
    case '^':
    case '_':
    case '`':
      return true;
    default:
      return false;
    }
  }

  static is_regex_punct(c) {
    switch (c) {
    case '!':
    case '"':
    case '#':
    case '%':
    case '&':
    case '(':
    case '*':
    case ',':
    case '-':
    case '.':
    case '/':
    case ':':
    case ';':
    case '<':
    case '=':
    case '@':
    case '[':
    case '\'':
    case '^':
    case '`':
    case '{':
    case '|':
    case '~':
      return true;
    default:
      return false;
    }
  }

  static is_regex_prefix(s) {
    return s == "m" ||
      s == "s" ||
      s == "y" ||
      s == "q" ||
      s == "tr" ||
      s == "qq" ||
      s == "qw" ||
      s == "qx" ||
      s == "qr";
  }

  static is_double_regex(s) {
    return s == "s" ||
      s == "y" ||
      s == "tr";
  }

  constructor(delegate) {
    super(delegate);
    this.i = 0;
    this.c = 0;
    this.last = 0;
    this.expect = 0;
    this.opener = 0;
    this.closer = 0;
    this.pending_heredoc = false;
    this.indented_heredoc = false;
    this.word = '';
    this.heredoc = '';
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      let c = input[i];
      this.last = this.c;
      this.c = c;
      switch (this.state) {

      case HighlightPerl.NORMAL:
        if (!isascii(c) || isalpha(c) || c == '_') {
          this.state = HighlightPerl.WORD;
          this.word += c;
        } else if (c == '\'') {
          this.state = HighlightPerl.QUOTE;
          this.push("span", "string");
          this.append('\'');
          this.expect = HighlightPerl.EXPECT_OPERATOR;
        } else if (c == '"') {
          this.state = HighlightPerl.DQUOTE;
          this.push("span", "string");
          this.append('"');
          this.expect = HighlightPerl.EXPECT_OPERATOR;
        } else if (c == '=' && (!this.last || this.last == '\n')) {
          this.state = HighlightPerl.EQUAL;
        } else if (c == '\\') {
          this.state = HighlightPerl.BACKSLASH;
          this.push("span", "escape");
          this.append('\\');
        } else if (c == '`') {
          this.state = HighlightPerl.TICK;
          this.push("span", "string");
          this.append('`');
          this.expect = HighlightPerl.EXPECT_OPERATOR;
        } else if (c == '$') {
          this.append('$');
          this.state = HighlightPerl.VAR;
          this.expect = HighlightPerl.EXPECT_OPERATOR;
        } else if (c == '@' || c == '%') {
          this.append(c);
          this.push("span", "var");
          this.state = HighlightPerl.VAR2;
          this.expect = HighlightPerl.EXPECT_OPERATOR;
        } else if (c == '#') {
          this.push("span", "comment");
          this.append('#');
          this.state = HighlightPerl.COMMENT;
        } else if (c == '<') {
          this.append('<');
          this.state = HighlightPerl.LT;
          this.expect = HighlightPerl.EXPECT_VALUE;
        } else if (c == '/' && this.expect == HighlightPerl.EXPECT_VALUE && this.last != '/') {
          this.opener = '/';
          this.closer = '/';
          this.expect = HighlightPerl.EXPECT_OPERATOR;
          this.push("span", "string");
          this.append(c);
          this.state = HighlightPerl.REGEX;
        } else if (c == '\n') {
          this.append('\n');
          if (this.pending_heredoc) {
            this.push("span", "string");
            this.pending_heredoc = false;
            this.state = HighlightPerl.HEREDOC_BOL;
            this.i = 0;
          }
        } else if (c == ')' || c == '}' || c == ']') {
          this.expect = HighlightPerl.EXPECT_OPERATOR;
          this.append(c);
        } else if (ispunct(c)) {
          this.expect = HighlightPerl.EXPECT_VALUE;
          this.append(c);
        } else if (isdigit(c) || c == '.') {
          this.expect = HighlightPerl.EXPECT_OPERATOR;
          this.append(c);
        } else {
          this.append(c);
        }
        break;

      case HighlightPerl.WORD:
        if (!isascii(c) || isalnum(c) || c == '_') {
          this.word += c;
        } else {
          if (PERL_KEYWORDS.has(this.word)) {
            this.push("span", "keyword");
            this.append(this.word);
            this.pop();
            if (this.word == "shift") {
              this.expect = HighlightPerl.EXPECT_OPERATOR;
            } else {
              this.expect = HighlightPerl.EXPECT_VALUE;
            }
          } else {
            this.append(this.word);
            this.expect = HighlightPerl.EXPECT_VALUE;
            if (HighlightPerl.is_regex_punct(c) &&
                HighlightPerl.is_regex_prefix(this.word)) {
              this.opener = c;
              this.closer = HighlightPerl.mirror(c);
              this.push("span", "string");
              this.append(c);
              if (HighlightPerl.is_double_regex(this.word)) {
                this.state = HighlightPerl.S_REGEX;
              } else {
                this.state = HighlightPerl.REGEX;
              }
              this.word = '';
              break;
            }
          }
          this.word = '';
          this.epsilon(HighlightPerl.NORMAL);
        }
        break;

      case HighlightPerl.BACKSLASH:
        this.append(c);
        this.pop();
        this.state = HighlightPerl.NORMAL;
        break;

      case HighlightPerl.VAR:
        if (isdigit(c) || HighlightPerl.is_magic_var(c)) {
          this.push("span", "var");
          this.append(c);
          this.pop();
          this.state = HighlightPerl.NORMAL;
          break;
        } else if (c == '{') {
          this.state = HighlightPerl.VAR2;
          this.append(c);
          this.push("span", "var");
          break;
        } else {
          this.push("span", "var");
          this.state = HighlightPerl.VAR2;
        }
        // fallthrough

      case HighlightPerl.VAR2:
        if (!isascii(c) || isalnum(c) || c == '_') {
          this.append(c);
        } else {
          this.pop();
          this.epsilon(HighlightPerl.NORMAL);
        }
        break;

      case HighlightPerl.COMMENT:
        this.append(c);
        if (c == '\n') {
          this.pop();
          this.state = HighlightPerl.NORMAL;
        }
        break;

      case HighlightPerl.REGEX:
        this.append(c);
        if (c == this.closer) {
          this.pop();
          this.state = HighlightPerl.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightPerl.REGEX_BACKSLASH;
        }
        break;

      case HighlightPerl.REGEX_BACKSLASH:
        this.append(c);
        this.state = HighlightPerl.REGEX;
        break;

      case HighlightPerl.S_REGEX:
        this.append(c);
        if (c == this.opener) {
          this.state = HighlightPerl.S_REGEX_S;
        } else if (c == '\\') {
          this.state = HighlightPerl.S_REGEX_BACKSLASH;
        }
        break;

      case HighlightPerl.S_REGEX_BACKSLASH:
        this.append(c);
        this.state = HighlightPerl.S_REGEX;
        break;

      case HighlightPerl.S_REGEX_S:
        this.append(c);
        if (c == this.closer) {
          this.pop();
          this.state = HighlightPerl.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightPerl.S_REGEX_S_BACKSLASH;
        }
        break;

      case HighlightPerl.S_REGEX_S_BACKSLASH:
        this.append(c);
        this.state = HighlightPerl.S_REGEX_S;
        break;

      case HighlightPerl.QUOTE:
        this.append(c);
        if (c == '\'') {
          this.pop();
          this.state = HighlightPerl.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightPerl.QUOTE_BACKSLASH;
        }
        break;

      case HighlightPerl.QUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightPerl.QUOTE;
        break;

      case HighlightPerl.DQUOTE:
        this.append(c);
        if (c == '"') {
          this.pop();
          this.state = HighlightPerl.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightPerl.DQUOTE_BACKSLASH;
        }
        break;

      case HighlightPerl.DQUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightPerl.DQUOTE;
        break;

      case HighlightPerl.TICK:
        this.append(c);
        if (c == '`') {
          this.pop();
          this.state = HighlightPerl.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightPerl.TICK_BACKSLASH;
        }
        break;

      case HighlightPerl.TICK_BACKSLASH:
        this.append(c);
        this.state = HighlightPerl.TICK;
        break;

      case HighlightPerl.EQUAL:
        if (isalpha(c)) {
          this.push("span", "comment");
          this.append('=');
          this.append(c);
          this.heredoc = "=cut";
          this.state = HighlightPerl.HEREDOC;
          this.i = 0;
        } else {
          this.append('=');
          this.epsilon(HighlightPerl.NORMAL);
        }
        break;

      case HighlightPerl.LT:
        if (c == '<') {
          this.append(c);
          this.state = HighlightPerl.LT_LT;
          this.heredoc = '';
          this.pending_heredoc = false;
          this.indented_heredoc = false;
        } else {
          this.epsilon(HighlightPerl.NORMAL);
        }
        break;

      case HighlightPerl.LT_LT:
        if (c == '-') {
          this.indented_heredoc = true;
          this.append(c);
        } else if (c == '"' || c == '\'') {
          this.closer = c;
          this.state = HighlightPerl.LT_LT_QNAME;
          this.push("span", "string");
          this.append(c);
        } else if (isalpha(c) || c == '_') {
          this.state = HighlightPerl.LT_LT_NAME;
          this.heredoc += c;
          this.append(c);
        } else if (isascii(c) && isblank(c)) {
          this.append(c);
        } else {
          this.epsilon(HighlightPerl.NORMAL);
        }
        break;

      case HighlightPerl.LT_LT_NAME:
        if (isalnum(c) || c == '_') {
          this.state = HighlightPerl.LT_LT_NAME;
          this.heredoc += c;
          this.append(c);
        } else if (c == '\n') {
          this.append(c);
          this.push("span", "string");
          this.state = HighlightPerl.HEREDOC_BOL;
        } else {
          this.pending_heredoc = true;
          this.epsilon(HighlightPerl.NORMAL);
        }
        break;

      case HighlightPerl.LT_LT_QNAME:
        this.append(c);
        if (c == this.closer) {
          this.pop();
          this.state = HighlightPerl.HEREDOC_BOL;
          this.pending_heredoc = true;
          this.state = HighlightPerl.NORMAL;
        } else {
          this.heredoc += c;
        }
        break;

      case HighlightPerl.HEREDOC_BOL:
        this.append(c);
        if (c == '\n') {
          if (this.i == this.heredoc.length) {
            this.state = HighlightPerl.NORMAL;
            this.pop();
          }
          this.i = 0;
        } else if (c == '\t' && this.indented_heredoc) {
          // do nothing
        } else if (this.i < this.heredoc.length && this.heredoc[this.i] == c) {
          this.i++;
        } else {
          this.state = HighlightPerl.HEREDOC;
          this.i = 0;
        }
        break;

      case HighlightPerl.HEREDOC:
        this.append(c);
        if (c == '\n') {
          this.state = HighlightPerl.HEREDOC_BOL;
          this.i = 0;
        }
        break;

      default:
        throw new Error('Invalid state');
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightPerl.WORD:
      if (PERL_KEYWORDS.has(this.word)) {
        this.push("span", "keyword");
        this.append(this.word);
        this.pop();
      } else {
        this.append(this.word);
      }
      this.word = '';
      break;
    case HighlightPerl.EQUAL:
      this.append('=');
      break;
    case HighlightPerl.VAR2:
    case HighlightPerl.TICK:
    case HighlightPerl.TICK_BACKSLASH:
    case HighlightPerl.QUOTE:
    case HighlightPerl.QUOTE_BACKSLASH:
    case HighlightPerl.DQUOTE:
    case HighlightPerl.DQUOTE_BACKSLASH:
    case HighlightPerl.COMMENT:
    case HighlightPerl.HEREDOC_BOL:
    case HighlightPerl.HEREDOC:
    case HighlightPerl.LT_LT_QNAME:
    case HighlightPerl.REGEX:
    case HighlightPerl.REGEX_BACKSLASH:
    case HighlightPerl.S_REGEX:
    case HighlightPerl.S_REGEX_BACKSLASH:
    case HighlightPerl.S_REGEX_S:
    case HighlightPerl.S_REGEX_S_BACKSLASH:
    case HighlightPerl.BACKSLASH:
      this.pop();
      break;
    default:
      break;
    }
    this.state = HighlightPerl.NORMAL;
    this.delegate.flush();
    this.delta = 1;
  }
}

Highlighter.REGISTRY['perl'] = HighlightPerl;
Highlighter.REGISTRY['pl'] = HighlightPerl;
