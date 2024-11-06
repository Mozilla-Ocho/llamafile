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

const RUBY_KEYWORDS = new Set([
  'BEGIN',
  'END',
  'alias',
  'and',
  'begin',
  'break',
  'case',
  'class',
  'def',
  'defined?',
  'do',
  'else',
  'elsif',
  'end',
  'ensure',
  'fail',
  'for',
  'if',
  'in',
  'module',
  'next',
  'not',
  'or',
  'redo',
  'rescue',
  'retry',
  'return',
  'self',
  'super',
  'then',
  'undef',
  'unless',
  'until',
  'when',
  'while',
  'yield',
]);

const RUBY_BUILTINS = new Set([
  '__callee__',
  '__dir__',
  '__method__',
  'abort',
  'alias_method',
  'at_exit',
  'attr',
  'attr_accessor',
  'attr_reader',
  'attr_writer',
  'autoload',
  'autoload?',
  'binding',
  'block_given?',
  'callcc',
  'caller',
  'catch',
  'define_method',
  'eval',
  'exec',
  'exit',
  'exit!',
  'extend',
  'fail',
  'fork',
  'format',
  'global_variables',
  'include',
  'lambda',
  'load',
  'local_variables',
  'loop',
  'module_function',
  'open',
  'p',
  'prepend',
  'print',
  'printf',
  'private',
  'private_class_method',
  'private_constant',
  'proc',
  'protected',
  'public',
  'public_class_method',
  'public_constant',
  'putc',
  'puts',
  'raise',
  'rand',
  'readline',
  'readlines',
  'refine',
  'require',
  'require_relative',
  'sleep',
  'spawn',
  'sprintf',
  'srand',
  'syscall',
  'system',
  'throw',
  'trace_var',
  'trap',
  'untrace_var',
  'using',
  'warn',
]);

const RUBY_CONSTANTS = new Set([
  '__ENCODING__',
  '__FILE__',
  '__LINE__',
  'false',
  'nil',
  'true',
]);

class HighlightRuby extends Highlighter {

  static NORMAL = 0;
  static WORD = 1;
  static EQUAL = 2;
  static EQUAL_WORD = 3;
  static QUOTE = 4;
  static QUOTE_BACKSLASH = 5;
  static DQUOTE = 6;
  static DQUOTE_BACKSLASH = 7;
  static TICK = 8;
  static TICK_BACKSLASH = 9;
  static COMMENT = 10;
  static LT = 11;
  static LT_LT = 12;
  static LT_LT_NAME = 13;
  static LT_LT_QNAME = 14;
  static HEREDOC_BOL = 15;
  static HEREDOC = 16;
  static COLON = 17;
  static COLON_WORD = 18;
  static AT = 19;
  static AT_WORD = 20;
  static DOLLAR = 21;
  static DOLLAR_WORD = 22;
  static PERCENT = 23;
  static PERCENT2 = 24;
  static PERCENT_STRING = 25;
  static MULTICOM = 26;
  static MULTICOM_BOL = 27;
  static REGEX = 28;
  static REGEX_BACKSLASH = 29;

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

  static isident(c) {
    return !isascii(c) ||
      isalnum(c) ||
      c == '!' ||
      c == '$' ||
      c == '%' ||
      c == '&' ||
      c == '-' ||
      c == '/' ||
      c == '=' ||
      c == '?' ||
      c == '@' ||
      c == '^' ||
      c == '_';
  }

  constructor(delegate) {
    super(delegate);
    this.i = 0;
    this.q = 0;
    this.last = 0;
    this.level = 0;
    this.opener = 0;
    this.closer = 0;
    this.pending_heredoc = false;
    this.indented_heredoc = false;
    this.heredoc = '';
    this.word = '';
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      let c = input[i];

      if (!isblank(c) && c != '/' && c != '<')
        this.last = c;

      switch (this.state) {

      case HighlightRuby.NORMAL:
        if (!isascii(c) || isalpha(c) || c == '_') {
          this.state = HighlightRuby.WORD;
          this.word += c;
        } else if (c == ':') {
          this.state = HighlightRuby.COLON;
        } else if (c == '@') {
          this.state = HighlightRuby.AT;
        } else if (c == '=') {
          this.state = HighlightRuby.EQUAL;
        } else if (c == '$') {
          this.state = HighlightRuby.DOLLAR;
        } else if (c == '%') {
          this.state = HighlightRuby.PERCENT;
          this.q = 0;
        } else if (c == '\'') {
          this.state = HighlightRuby.QUOTE;
          this.push("span", "string");
          this.append(c);
        } else if (c == '"') {
          this.state = HighlightRuby.DQUOTE;
          this.push("span", "string");
          this.append(c);
        } else if (c == '`') {
          this.state = HighlightRuby.TICK;
          this.push("span", "string");
          this.append(c);
        } else if (c == '#') {
          this.push("span", "comment");
          this.append(c);
          this.state = HighlightRuby.COMMENT;
        } else if (c == '<' && !isalnum(this.last)) {
          this.append(c);
          this.state = HighlightRuby.LT;
        } else if (c == '/' && !isalnum(this.last)) {
          this.state = HighlightRuby.REGEX;
          this.push("span", "string");
          this.append(c);
        } else if (c == '\n') {
          this.append(c);
          if (this.pending_heredoc) {
            this.push("span", "string");
            this.pending_heredoc = false;
            this.state = HighlightRuby.HEREDOC_BOL;
            this.i = 0;
          }
        } else {
          this.append(c);
        }
        break;

      case HighlightRuby.EQUAL:
        if (!isascii(c) || isalpha(c) || c == '_') {
          this.state = HighlightRuby.EQUAL_WORD;
          this.word += c;
        } else {
          this.append('=');
          this.epsilon(HighlightRuby.NORMAL);
        }
        break;

      case HighlightRuby.EQUAL_WORD:
        if (HighlightRuby.isident(c)) {
          this.word += c;
          break;
        } else if (this.word == "begin") {
          this.push("span", "comment");
          this.append("=begin");
          this.append(c);
          if (c == '\n') {
            this.state = HighlightRuby.MULTICOM_BOL;
            this.i = 0;
          } else {
            this.state = HighlightRuby.MULTICOM;
          }
          this.word = '';
          break;
        } else {
          this.append('=');
          this.state = HighlightRuby.WORD;
        }
        // fallthrough

      case HighlightRuby.WORD:
        if (HighlightRuby.isident(c)) {
          this.word += c;
        } else {
          if (RUBY_KEYWORDS.has(this.word)) {
            this.push("span", "keyword");
            this.append(this.word);
            this.pop();
            this.last = 0;
          } else if (RUBY_BUILTINS.has(this.word)) {
            this.push("span", "builtin");
            this.append(this.word);
            this.pop();
          } else if (RUBY_CONSTANTS.has(this.word)) {
            this.push("span", "constant");
            this.append(this.word);
            this.pop();
          } else if (this.word && this.word[0] == this.word[0].toUpperCase()) {
            this.push("span", "class");
            this.append(this.word);
            this.pop();
          } else {
            this.append(this.word);
          }
          this.word = '';
          this.epsilon(HighlightRuby.NORMAL);
        }
        break;

      case HighlightRuby.REGEX:
        this.append(c);
        if (c == '/') {
          this.pop();
          this.state = HighlightRuby.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightRuby.REGEX_BACKSLASH;
        }
        break;

      case HighlightRuby.REGEX_BACKSLASH:
        this.append(c);
        this.state = HighlightRuby.REGEX;
        break;

      case HighlightRuby.MULTICOM:
        this.append(c);
        if (c == '\n') {
          this.state = HighlightRuby.MULTICOM_BOL;
          this.i = 0;
        }
        break;

      case HighlightRuby.MULTICOM_BOL:
        this.append(c);
        if (c == "=end"[this.i]) {
          if (++this.i == 4) {
            this.state = HighlightRuby.NORMAL;
            this.pop();
          }
        } else {
          this.state = HighlightRuby.MULTICOM;
        }
        break;

      case HighlightRuby.COLON:
        if (HighlightRuby.isident(c)) {
          this.push("span", "lispkw");
          this.append(':');
          this.append(c);
          this.state = HighlightRuby.COLON_WORD;
        } else {
          this.append(':');
          this.epsilon(HighlightRuby.NORMAL);
        }
        break;

      case HighlightRuby.COLON_WORD:
        if (HighlightRuby.isident(c)) {
          this.append(c);
        } else {
          this.pop();
          this.epsilon(HighlightRuby.NORMAL);
        }
        break;

      case HighlightRuby.AT:
        if (HighlightRuby.isident(c)) {
          this.push("span", "var");
          this.append('@');
          this.append(c);
          this.state = HighlightRuby.AT_WORD;
        } else {
          this.append('@');
          this.epsilon(HighlightRuby.NORMAL);
        }
        break;

      case HighlightRuby.AT_WORD:
        if (HighlightRuby.isident(c)) {
          this.append(c);
        } else {
          this.pop();
          this.epsilon(HighlightRuby.NORMAL);
        }
        break;

      case HighlightRuby.PERCENT:
        if (c == 'q' || c == 'Q') {
          this.q = c;
          this.state = HighlightRuby.PERCENT2;
        } else if (ispunct(c)) {
          this.level = 1;
          this.opener = c;
          this.closer = HighlightRuby.mirror(c);
          this.push("span", "string");
          this.append('%');
          this.append(c);
          this.state = HighlightRuby.PERCENT_STRING;
        } else {
          this.append('%');
          this.epsilon(HighlightRuby.NORMAL);
        }
        break;

      case HighlightRuby.PERCENT2:
        if (ispunct(c)) {
          this.level = 1;
          this.opener = c;
          this.closer = HighlightRuby.mirror(c);
          this.push("span", "string");
          this.append('%');
          this.append(this.q);
          this.append(c);
          this.state = HighlightRuby.PERCENT_STRING;
        } else {
          this.append('%');
          this.append(this.q);
          this.epsilon(HighlightRuby.NORMAL);
        }
        break;

      case HighlightRuby.PERCENT_STRING:
        this.append(c);
        if (c == this.opener && this.opener != this.closer) {
          ++this.level;
        } else if (c == this.closer) {
          if (!--this.level) {
            this.pop();
            this.state = HighlightRuby.NORMAL;
          }
        }
        break;

      case HighlightRuby.DOLLAR:
        if (isdigit(c) || //
            c == '!' || //
            c == '"' || //
            c == '#' || //
            c == '$' || //
            c == '&' || //
            c == '-' || //
            c == '/' || //
            c == '<' || //
            c == '=' || //
            c == '>' || //
            c == '@' || //
            c == '\'' || //
            c == '\\' || //
            c == '^' || //
            c == '_' || //
            c == '`') {
          this.push("span", "var");
          this.append('$');
          this.append(c);
          this.pop();
          this.state = HighlightRuby.NORMAL;
        } else if (isalpha(c)) {
          this.push("span", "var");
          this.append('$');
          this.append(c);
          this.state = HighlightRuby.DOLLAR_WORD;
        } else {
          this.append('$');
          this.epsilon(HighlightRuby.NORMAL);
        }
        break;

      case HighlightRuby.DOLLAR_WORD:
        if (HighlightRuby.isident(c)) {
          this.append(c);
        } else {
          this.pop();
          this.epsilon(HighlightRuby.NORMAL);
        }
        break;

      case HighlightRuby.COMMENT:
        this.append(c);
        if (c == '\n') {
          this.pop();
          this.state = HighlightRuby.NORMAL;
        }
        break;

      case HighlightRuby.QUOTE:
        this.append(c);
        if (c == '\'') {
          this.pop();
          this.state = HighlightRuby.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightRuby.QUOTE_BACKSLASH;
        }
        break;

      case HighlightRuby.QUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightRuby.QUOTE;
        break;

      case HighlightRuby.DQUOTE:
        this.append(c);
        if (c == '"') {
          this.pop();
          this.state = HighlightRuby.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightRuby.DQUOTE_BACKSLASH;
        }
        break;

      case HighlightRuby.DQUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightRuby.DQUOTE;
        break;

      case HighlightRuby.TICK:
        this.append(c);
        if (c == '`') {
          this.pop();
          this.state = HighlightRuby.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightRuby.TICK_BACKSLASH;
        }
        break;

      case HighlightRuby.TICK_BACKSLASH:
        this.append(c);
        this.state = HighlightRuby.TICK;
        break;

      case HighlightRuby.LT:
        if (c == '<') {
          this.append(c);
          this.state = HighlightRuby.LT_LT;
          this.heredoc = '';
          this.pending_heredoc = false;
          this.indented_heredoc = false;
        } else {
          this.epsilon(HighlightRuby.NORMAL);
        }
        break;

      case HighlightRuby.LT_LT:
        if (c == '-') {
          this.indented_heredoc = true;
          this.append(c);
        } else if (c == '\'' || c == '`' || c == '"') {
          this.closer = c;
          this.state = HighlightRuby.LT_LT_QNAME;
          this.push("span", "string");
          this.append(c);
        } else if (isalpha(c) || c == '_') {
          this.state = HighlightRuby.LT_LT_NAME;
          this.heredoc += c;
          this.append(c);
        } else {
          this.epsilon(HighlightRuby.NORMAL);
        }
        break;

      case HighlightRuby.LT_LT_NAME:
        if (isalnum(c) || c == '_') {
          this.state = HighlightRuby.LT_LT_NAME;
          this.heredoc += c;
          this.append(c);
        } else if (c == '\n') {
          this.append(c);
          this.push("span", "string");
          this.state = HighlightRuby.HEREDOC_BOL;
        } else {
          this.pending_heredoc = true;
          this.epsilon(HighlightRuby.NORMAL);
        }
        break;

      case HighlightRuby.LT_LT_QNAME:
        this.append(c);
        if (c == this.closer) {
          this.pop();
          this.state = HighlightRuby.HEREDOC_BOL;
          this.pending_heredoc = true;
          this.state = HighlightRuby.NORMAL;
        } else {
          this.heredoc += c;
        }
        break;

      case HighlightRuby.HEREDOC_BOL:
        this.append(c);
        if (c == '\n') {
          if (this.i == this.heredoc.length) {
            this.state = HighlightRuby.NORMAL;
            this.pop();
          }
          this.i = 0;
        } else if (c == '\t' && this.indented_heredoc) {
          // do nothing
        } else if (this.i < this.heredoc.length && this.heredoc[this.i] == c) {
          this.i++;
        } else {
          this.state = HighlightRuby.HEREDOC;
          this.i = 0;
        }
        break;

      case HighlightRuby.HEREDOC:
        this.append(c);
        if (c == '\n')
          this.state = HighlightRuby.HEREDOC_BOL;
        break;

      default:
        throw new Error('Invalid state');
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightRuby.EQUAL_WORD:
      this.append('=');
      // fallthrough
    case HighlightRuby.WORD:
      if (RUBY_KEYWORDS.has(this.word)) {
        this.push("span", "keyword");
        this.append(this.word);
        this.pop();
      } else if (RUBY_BUILTINS.has(this.word)) {
        this.push("span", "builtin");
        this.append(this.word);
        this.pop();
      } else if (RUBY_CONSTANTS.has(this.word)) {
        this.push("span", "constant");
        this.append(this.word);
        this.pop();
      } else if (this.word && this.word[0] == this.word[0].toUpperCase()) {
        this.push("span", "class");
        this.append(this.word);
        this.pop();
      } else {
        this.append(this.word);
      }
      this.word = '';
      break;
    case HighlightRuby.AT:
      this.append('@');
      break;
    case HighlightRuby.EQUAL:
      this.append('=');
      break;
    case HighlightRuby.COLON:
      this.append(':');
      break;
    case HighlightRuby.DOLLAR:
      this.append('$');
      break;
    case HighlightRuby.PERCENT:
      this.append('%');
      break;
    case HighlightRuby.PERCENT2:
      this.append('%');
      this.append(this.q);
      break;
    case HighlightRuby.REGEX:
    case HighlightRuby.REGEX_BACKSLASH:
    case HighlightRuby.PERCENT_STRING:
    case HighlightRuby.AT_WORD:
    case HighlightRuby.DOLLAR_WORD:
    case HighlightRuby.TICK:
    case HighlightRuby.TICK_BACKSLASH:
    case HighlightRuby.QUOTE:
    case HighlightRuby.QUOTE_BACKSLASH:
    case HighlightRuby.DQUOTE:
    case HighlightRuby.DQUOTE_BACKSLASH:
    case HighlightRuby.COMMENT:
    case HighlightRuby.HEREDOC_BOL:
    case HighlightRuby.HEREDOC:
    case HighlightRuby.LT_LT_QNAME:
    case HighlightRuby.COLON_WORD:
    case HighlightRuby.MULTICOM:
    case HighlightRuby.MULTICOM_BOL:
      this.pop();
      break;
    default:
      break;
    }
    this.state = HighlightRuby.NORMAL;
    this.delegate.flush();
    this.delta = 1;
  }
}

Highlighter.REGISTRY['ruby'] = HighlightRuby;
Highlighter.REGISTRY['rb'] = HighlightRuby;
