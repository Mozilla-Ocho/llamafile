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
  static DQUOTE_HASH = 30;
  static QUESTION = 31;
  static QUESTION_BACKSLASH = 32;
  static REGEX_HASH = 33;
  static REGEX_HASH_DOLLAR = 34;
  static REGEX_HASH_DOLLAR_WORD = 35;
  static DQUOTE_HASH = 36;
  static DQUOTE_HASH_DOLLAR = 37;
  static DQUOTE_HASH_DOLLAR_WORD = 38;
  static PERCENT_HASH = 39;
  static PERCENT_HASH_DOLLAR = 40;
  static PERCENT_HASH_DOLLAR_WORD = 41;

  static EXPECT_EXPR = 0;
  static EXPECT_VALUE = 1;
  static EXPECT_OPERATOR = 2;

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

  static ispunct_overridable(c) {
    switch (c) {
    case '%':
    case '&':
    case '*':
    case '+':
    case '-':
    case '/':
    case '<':
    case '>':
    case '^':
    case '_':
    case '`':
    case '|':
    case '~':
      return true;
    default:
      return false;
    }
  }

  static is_dollar_one(c) {
    switch (c) {
    case '!':
    case '"':
    case '#':
    case '$':
    case '&':
    case '-':
    case '/':
    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
    case '<':
    case '=':
    case '>':
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

  static is_percent_literal(c) {
    switch (c) {
    case 'q':
    case 'Q':
    case 'r':
    case 's':
    case 'w':
    case 'W':
    case 'x':
    case 'i':
    case 'I':
      return true;
    default:
      return false;
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
    this.level = 0;
    this.opener = 0;
    this.closer = 0;
    this.pending_heredoc = false;
    this.indented_heredoc = false;
    this.heredoc = '';
    this.word = '';
    this.nest = [];
    this.levels = [];
    this.openers = [];
    this.closers = [];
    this.is_definition = false;
    this.expect = HighlightRuby.EXPECT_EXPR;
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      let c = input[i];
      switch (this.state) {

      case HighlightRuby.NORMAL:
        if (!isascii(c) || isalpha(c) || c == '_' ||
            (this.is_definition && HighlightRuby.ispunct_overridable(c))) {
          this.state = HighlightRuby.WORD;
          this.word += c;
          this.is_definition = false;
        } else if (c == ':') {
          this.state = HighlightRuby.COLON;
          this.is_definition = false;
        } else if (c == '@') {
          this.state = HighlightRuby.AT;
          this.expect = HighlightRuby.EXPECT_VALUE;
          this.is_definition = false;
        } else if (c == '=') {
          this.state = HighlightRuby.EQUAL;
          this.expect = HighlightRuby.EXPECT_VALUE;
          this.is_definition = false;
        } else if (c == '?') {
          this.state = HighlightRuby.QUESTION;
          this.is_definition = false;
        } else if (c == '$') {
          this.state = HighlightRuby.DOLLAR;
          this.expect = HighlightRuby.EXPECT_OPERATOR;
          this.is_definition = false;
        } else if (c == '%') {
          this.state = HighlightRuby.PERCENT;
          this.expect = HighlightRuby.EXPECT_OPERATOR;
          this.q = 0;
        } else if (c == '\'') {
          this.state = HighlightRuby.QUOTE;
          this.push("span", "string");
          this.append(c);
          this.expect = HighlightRuby.EXPECT_OPERATOR;
          this.is_definition = false;
        } else if (c == '"') {
          this.state = HighlightRuby.DQUOTE;
          this.expect = HighlightRuby.EXPECT_OPERATOR;
          this.push("span", "string");
          this.append(c);
          this.is_definition = false;
        } else if (c == '`') {
          this.state = HighlightRuby.TICK;
          this.expect = HighlightRuby.EXPECT_OPERATOR;
          this.push("span", "string");
          this.append(c);
        } else if (c == '#') {
          this.push("span", "comment");
          this.append(c);
          this.state = HighlightRuby.COMMENT;
          this.expect = HighlightRuby.EXPECT_EXPR;
        } else if (c == '<' && (this.expect == HighlightRuby.EXPECT_EXPR ||
                                this.expect == HighlightRuby.EXPECT_VALUE)) {
          this.append(c);
          this.state = HighlightRuby.LT;
        } else if (c == '/' && (this.expect == HighlightRuby.EXPECT_EXPR ||
                                this.expect == HighlightRuby.EXPECT_VALUE)) {
          this.state = HighlightRuby.REGEX;
          this.push("span", "string");
          this.append(c);
        } else if (c == '{' && this.nest.length) {
          this.expect = HighlightRuby.EXPECT_VALUE;
          this.append('{');
          this.nest.push(HighlightRuby.NORMAL);
          this.levels.push(this.level);
          this.openers.push(this.opener);
          this.closers.push(this.closer);
          this.is_definition = false;
        } else if (c == '}' && this.nest.length) {
          this.level = this.levels.pop();
          this.opener = this.openers.pop();
          this.closer = this.closers.pop();
          if ((this.state = this.nest.pop()) != HighlightRuby.NORMAL)
            this.push("span", "string");
          this.append('}');
          this.expect = HighlightRuby.EXPECT_OPERATOR;
          this.is_definition = false;
        } else if (c == '\n') {
          this.expect = HighlightRuby.EXPECT_EXPR;
          this.append(c);
          if (this.pending_heredoc) {
            this.push("span", "string");
            this.pending_heredoc = false;
            this.state = HighlightRuby.HEREDOC_BOL;
            this.i = 0;
          }
        } else if (c == '[' || c == '(') {
          this.expect = HighlightRuby.EXPECT_VALUE;
          this.append(c);
          this.is_definition = false;
        } else if (c == ']' || c == ')') {
          this.expect = HighlightRuby.EXPECT_OPERATOR;
          this.append(c);
          this.is_definition = false;
        } else if (isdigit(c) || c == '.') {
          this.expect = HighlightRuby.EXPECT_OPERATOR;
          this.append(c);
          this.is_definition = false;
        } else if (ispunct(c)) {
          this.expect = HighlightRuby.EXPECT_VALUE;
          this.append(c);
          this.is_definition = false;
        } else if (isspace(c)) {
          this.append(c);
        } else {
          this.append(c);
          this.is_definition = false;
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
            this.expect = HighlightRuby.EXPECT_VALUE;
            if (this.word == "def") {
              this.is_definition = true;
            } else if (this.word == "class" || this.word == "module") {
              this.expect = HighlightRuby.EXPECT_OPERATOR;
            }
          } else if (this.expect == HighlightRuby.EXPECT_EXPR &&
                     RUBY_BUILTINS.has(this.word)) {
            this.push("span", "builtin");
            this.append(this.word);
            this.pop();
            this.expect = HighlightRuby.EXPECT_VALUE;
          } else if (RUBY_CONSTANTS.has(this.word)) {
            this.push("span", "constant");
            this.append(this.word);
            this.pop();
            this.expect = HighlightRuby.EXPECT_OPERATOR;
          } else if (this.word && this.word[0] == this.word[0].toUpperCase()) {
            this.push("span", "class");
            this.append(this.word);
            this.pop();
            this.expect = HighlightRuby.EXPECT_OPERATOR;
          } else {
            this.append(this.word);
            this.expect = HighlightRuby.EXPECT_OPERATOR;
          }
          this.word = '';
          this.epsilon(HighlightRuby.NORMAL);
        }
        break;

      case HighlightRuby.REGEX:
        if (c == '/') {
          this.append(c);
          this.pop();
          this.state = HighlightRuby.NORMAL;
        } else if (c == '#') {
          this.state = HighlightRuby.REGEX_HASH;
        } else if (c == '\\') {
          this.append(c);
          this.state = HighlightRuby.REGEX_BACKSLASH;
        } else {
          this.append(c);
        }
        break;

      case HighlightRuby.REGEX_HASH:
        if (c == '{') {
          this.push("strong", "");
          this.append('#');
          this.pop();
          this.append('{');
          this.pop();
          this.expect = HighlightRuby.EXPECT_VALUE;
          this.nest.push(HighlightRuby.REGEX);
          this.levels.push(this.level);
          this.openers.push(this.opener);
          this.closers.push(this.closer);
          this.state = HighlightRuby.NORMAL;
        } else if (c == '$') {
          this.state = HighlightRuby.REGEX_HASH_DOLLAR;
        } else {
          this.append('#');
          this.epsilon(HighlightRuby.REGEX);
        }
        break;

      case HighlightRuby.REGEX_HASH_DOLLAR:
        if (HighlightRuby.is_dollar_one(c)) {
          this.append('#');
          this.push("strong", "");
          this.append('$');
          this.append(c);
          this.pop();
          this.state = HighlightRuby.REGEX;
        } else if (isalpha(c)) {
          this.append('#');
          this.push("strong", "");
          this.append('$');
          this.append(c);
          this.state = HighlightRuby.REGEX_HASH_DOLLAR_WORD;
        } else {
          this.append('#');
          this.append('$');
          this.epsilon(HighlightRuby.REGEX);
        }
        break;

      case HighlightRuby.REGEX_HASH_DOLLAR_WORD:
        if (HighlightRuby.isident(c)) {
          this.append(c);
        } else {
          this.pop();
          this.epsilon(HighlightRuby.REGEX);
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
          this.expect = HighlightRuby.EXPECT_OPERATOR;
          this.state = HighlightRuby.COLON_WORD;
        } else if (c == ':') {
          this.append("::");
          this.expect = HighlightRuby.EXPECT_VALUE;
          this.state = HighlightRuby.NORMAL;
        } else {
          this.append(':');
          this.expect = HighlightRuby.EXPECT_VALUE;
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
        if (HighlightRuby.is_percent_literal(c)) {
          this.q = c;
          this.state = HighlightRuby.PERCENT2;
        } else if (ispunct(c)) {
          this.level = 1;
          this.opener = c;
          this.closer = HighlightRuby.mirror(c);
          this.push("span", "string");
          this.append('%');
          this.append(c);
          this.expect = HighlightRuby.EXPECT_OPERATOR;
          this.state = HighlightRuby.PERCENT_STRING;
        } else {
          this.append('%');
          this.expect = HighlightRuby.EXPECT_VALUE;
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
          this.expect = HighlightRuby.EXPECT_OPERATOR;
          this.state = HighlightRuby.PERCENT_STRING;
        } else {
          this.word += c;
          this.append('%');
          this.is_definition = false;
          this.epsilon(HighlightRuby.WORD);
        }
        break;

      case HighlightRuby.PERCENT_HASH:
        if (c == '{') {
          this.push("strong", "");
          this.append('#');
          this.pop();
          this.append('{');
          this.pop();
          this.expect = HighlightRuby.EXPECT_VALUE;
          this.nest.push(HighlightRuby.PERCENT_STRING);
          this.levels.push(this.level);
          this.openers.push(this.opener);
          this.closers.push(this.closer);
          this.state = HighlightRuby.NORMAL;
        } else if (c == '$') {
          this.state = HighlightRuby.PERCENT_HASH_DOLLAR;
        } else {
          this.append('#');
          this.epsilon(HighlightRuby.PERCENT_STRING);
        }
        break;

      case HighlightRuby.PERCENT_HASH_DOLLAR:
        if (HighlightRuby.is_dollar_one(c)) {
          this.append('#');
          this.push("strong", "");
          this.append('$');
          this.append(c);
          this.pop();
          this.state = HighlightRuby.PERCENT_STRING;
        } else if (isalpha(c)) {
          this.append('#');
          this.push("strong", "");
          this.append('$');
          this.append(c);
          this.state = HighlightRuby.PERCENT_HASH_DOLLAR_WORD;
        } else {
          this.append('#');
          this.append('$');
          this.epsilon(HighlightRuby.PERCENT_STRING);
        }
        break;

      case HighlightRuby.PERCENT_HASH_DOLLAR_WORD:
        if (HighlightRuby.isident(c)) {
          this.append(c);
        } else {
          this.push("strong", "");
          this.epsilon(HighlightRuby.PERCENT_STRING);
        }
        break;

      case HighlightRuby.PERCENT_STRING:
        if (c == this.opener && this.opener != this.closer) {
          this.append(c);
          ++this.level;
        } else if (c == '#' && this.closer != '#') {
          this.state = HighlightRuby.PERCENT_HASH;
        } else if (c == this.closer) {
          this.append(c);
          if (!--this.level) {
            this.pop();
            this.state = HighlightRuby.NORMAL;
          }
        } else {
          this.append(c);
        }
        break;

      case HighlightRuby.DOLLAR:
        if (HighlightRuby.is_dollar_one(c)) {
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
        if (c == '"') {
          this.append(c);
          this.pop();
          this.state = HighlightRuby.NORMAL;
        } else if (c == '#') {
          this.state = HighlightRuby.DQUOTE_HASH;
        } else if (c == '\\') {
          this.state = HighlightRuby.DQUOTE_BACKSLASH;
          this.append(c);
        } else {
          this.append(c);
        }
        break;

      case HighlightRuby.DQUOTE_HASH:
        if (c == '{') {
          this.push("strong", "");
          this.append('#');
          this.pop();
          this.append('{');
          this.pop();
          this.expect = HighlightRuby.EXPECT_VALUE;
          this.nest.push(HighlightRuby.DQUOTE);
          this.levels.push(this.level);
          this.openers.push(this.opener);
          this.closers.push(this.closer);
          this.state = HighlightRuby.NORMAL;
        } else if (c == '$') {
          this.state = HighlightRuby.DQUOTE_HASH_DOLLAR;
        } else {
          this.append('#');
          this.epsilon(HighlightRuby.DQUOTE);
        }
        break;

      case HighlightRuby.DQUOTE_HASH_DOLLAR:
        if (HighlightRuby.is_dollar_one(c)) {
          this.append('#');
          this.push("strong", "");
          this.append('$');
          this.append(c);
          this.pop();
          this.state = HighlightRuby.DQUOTE;
        } else if (isalpha(c)) {
          this.append('#');
          this.push("strong", "");
          this.append('$');
          this.append(c);
          this.state = HighlightRuby.DQUOTE_HASH_DOLLAR_WORD;
        } else {
          this.append('#');
          this.append('$');
          this.epsilon(HighlightRuby.DQUOTE);
        }
        break;

      case HighlightRuby.DQUOTE_HASH_DOLLAR_WORD:
        if (HighlightRuby.isident(c)) {
          this.append(c);
        } else {
          this.pop();
          this.epsilon(HighlightRuby.DQUOTE);
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
        if (c == '-' || c == '~') {
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
        } else if (isblank(c) && this.indented_heredoc) {
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

      case HighlightRuby.QUESTION:
        if (c == '\\') {
          this.state = HighlightRuby.QUESTION_BACKSLASH;
        } else if (isspace(c)) {
          this.append('?');
          this.expect = HighlightRuby.EXPECT_VALUE;
          this.epsilon(HighlightRuby.NORMAL);
        } else {
          this.push('span', 'escape');
          this.append('?');
          this.append(c);
          this.pop();
          this.state = HighlightRuby.NORMAL;
        }
        break;

      case HighlightRuby.QUESTION_BACKSLASH:
        this.push('span', 'escape');
        this.append("?\\");
        this.append(c);
        this.pop();
        this.state = HighlightRuby.NORMAL;
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
    case HighlightRuby.PERCENT_HASH:
      this.append('#');
      this.pop();
      break;
    case HighlightRuby.PERCENT_HASH_DOLLAR:
      this.append("#$");
      this.pop();
      break;
    case HighlightRuby.QUESTION:
      this.append('?');
      break;
    case HighlightRuby.QUESTION_BACKSLASH:
      this.append("?\\");
      break;
    case HighlightRuby.DQUOTE_HASH:
      this.append('#');
      this.pop();
      break;
    case HighlightRuby.DQUOTE_HASH_DOLLAR:
      this.append("#$");
      this.pop();
      break;
    case HighlightRuby.REGEX_HASH:
      this.append('#');
      this.pop();
      break;
    case HighlightRuby.REGEX_HASH_DOLLAR:
      this.append("#$");
      this.pop();
      break;
    case HighlightRuby.REGEX:
    case HighlightRuby.REGEX_BACKSLASH:
    case HighlightRuby.REGEX_HASH_DOLLAR_WORD:
    case HighlightRuby.PERCENT_STRING:
    case HighlightRuby.PERCENT_HASH_DOLLAR_WORD:
    case HighlightRuby.AT_WORD:
    case HighlightRuby.DOLLAR_WORD:
    case HighlightRuby.TICK:
    case HighlightRuby.TICK_BACKSLASH:
    case HighlightRuby.QUOTE:
    case HighlightRuby.QUOTE_BACKSLASH:
    case HighlightRuby.DQUOTE:
    case HighlightRuby.DQUOTE_BACKSLASH:
    case HighlightRuby.DQUOTE_HASH_DOLLAR_WORD:
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
    this.nest = [];
    this.expect = HighlightRuby.EXPECT_VALUE;
    this.is_definition = false;
    this.state = HighlightRuby.NORMAL;
    this.delegate.flush();
    this.delta = 1;
  }
}

Highlighter.REGISTRY['ruby'] = HighlightRuby;
Highlighter.REGISTRY['rb'] = HighlightRuby;
