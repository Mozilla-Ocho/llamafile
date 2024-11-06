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

const OCAML_KEYWORDS = new Set([
  'and',
  'as',
  'asr',
  'assert',
  'begin',
  'class',
  'constraint',
  'do',
  'done',
  'downto',
  'else',
  'end',
  'exception',
  'external',
  'for',
  'fun',
  'function',
  'functor',
  'if',
  'in',
  'include',
  'inherit',
  'initializer',
  'land',
  'lazy',
  'let',
  'lor',
  'lsl',
  'lsr',
  'lxor',
  'match',
  'method',
  'mod',
  'module',
  'mutable',
  'new',
  'nonrec',
  'object',
  'of',
  'open',
  'or',
  'private',
  'rec',
  'sig',
  'struct',
  'then',
  'to',
  'try',
  'type',
  'val',
  'virtual',
  'when',
  'while',
  'with',
]);

const OCAML_BUILTINS = new Set([
  'Assert_failure',
  'Division_by_zero',
  'End_of_file',
  'Failure',
  'Invalid_argument',
  'Match_failure',
  'Not_found',
  'Out_of_memory',
  'Stack_overflow',
  'Sys_blocked_io',
  'Sys_error',
  'Undefined_recursive_module',
  'at_exit',
  'exit',
  'failwith',
  'failwithf',
  'ignore',
  'invalid_arg',
  'parser',
  'raise',
  'raise_notrace',
  'ref',
]);

const OCAML_CONSTANTS = new Set([
  'false',
  'true',
]);

class HighlightOcaml extends Highlighter {

  static NORMAL = 0;
  static WORD = 1;
  static QUOTE = 2;
  static QUOTE_BACKSLASH = 3;
  static DQUOTE = 4;
  static DQUOTE_BACKSLASH = 5;
  static LPAREN = 6;
  static COMMENT = 7;
  static COMMENT_STAR = 8;
  static COMMENT_LPAREN = 9;
  static LCURLY = 10;
  static RAWSTR = 11;
  static RAWSTR_PIPE = 12;

  constructor(delegate) {
    super(delegate);
    this.nest = 0;
    this.word = '';
    this.word2 = '';
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      let c = input[i];

      switch (this.state) {

      case HighlightOcaml.NORMAL:
        if (!isascii(c) || isalpha(c) || c == '_' || c == '~') {
          this.epsilon(HighlightOcaml.WORD);
        } else if (c == '(') {
          this.state = HighlightOcaml.LPAREN;
        } else if (c == '\'') {
          this.state = HighlightOcaml.QUOTE;
          this.push("span", "string");
          this.append(c);
        } else if (c == '"') {
          this.state = HighlightOcaml.DQUOTE;
          this.push("span", "string");
          this.append(c);
        } else if (c == '{') {
          this.state = HighlightOcaml.LCURLY;
        } else {
          this.append(c);
        }
        break;

      case HighlightOcaml.WORD:
        if (!isascii(c) || isalnum(c) || c == '_' || c == '\'' || c == '~') {
          this.word += c;
        } else {
          if (OCAML_KEYWORDS.has(this.word)) {
            this.push("span", "keyword");
            this.append(this.word);
            this.pop();
          } else if (OCAML_BUILTINS.has(this.word)) {
            this.push("span", "builtin");
            this.append(this.word);
            this.pop();
          } else if (OCAML_CONSTANTS.has(this.word)) {
            this.push("span", "constant");
            this.append(this.word);
            this.pop();
          } else if (this.word.length > 1 && this.word[0] == '~') {
            this.push("span", "property");
            this.append(this.word);
            this.pop();
          } else {
            this.append(this.word);
          }
          this.word = '';
          this.epsilon(HighlightOcaml.NORMAL);
        }
        break;

      case HighlightOcaml.LPAREN:
        if (c == '*') {
          this.push("span", "comment");
          this.append("(*");
          this.state = HighlightOcaml.COMMENT;
          this.nest = 1;
        } else {
          this.append('(');
          this.epsilon(HighlightOcaml.NORMAL);
        }
        break;

      case HighlightOcaml.COMMENT:
        this.append(c);
        if (c == '*') {
          this.state = HighlightOcaml.COMMENT_STAR;
        } else if (c == '(') {
          this.state = HighlightOcaml.COMMENT_LPAREN;
        }
        break;

      case HighlightOcaml.COMMENT_STAR:
        this.append(c);
        if (c == ')') {
          if (!--this.nest) {
            this.pop();
            this.state = HighlightOcaml.NORMAL;
          }
        } else if (c == '(') {
          this.state = HighlightOcaml.COMMENT_LPAREN;
        } else if (c != '*') {
          this.state = HighlightOcaml.COMMENT;
        }
        break;

      case HighlightOcaml.COMMENT_LPAREN:
        this.append(c);
        if (c == '*') {
          ++this.nest;
          this.state = HighlightOcaml.COMMENT;
        } else if (c != '(') {
          this.state = HighlightOcaml.COMMENT;
        }
        break;

      case HighlightOcaml.QUOTE:
        this.append(c);
        if (c == '\'') {
          this.pop();
          this.state = HighlightOcaml.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightOcaml.QUOTE_BACKSLASH;
        }
        break;

      case HighlightOcaml.QUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightOcaml.QUOTE;
        break;

      case HighlightOcaml.DQUOTE:
        this.append(c);
        if (c == '"') {
          this.pop();
          this.state = HighlightOcaml.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightOcaml.DQUOTE_BACKSLASH;
        }
        break;

      case HighlightOcaml.DQUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightOcaml.DQUOTE;
        break;

      case HighlightOcaml.LCURLY:
        if (c == '|') {
          this.push("span", "string");
          this.append('{');
          this.append(this.word);
          this.append('|');
          this.state = HighlightOcaml.RAWSTR;
        } else if (isalpha(c) || c == '_') {
          this.word += c;
        } else {
          this.append('{');
          if (!this.word) {
            this.epsilon(HighlightOcaml.NORMAL);
          } else {
            this.epsilon(HighlightOcaml.WORD);
          }
        }
        break;

      case HighlightOcaml.RAWSTR:
        this.append(c);
        if (c == '|') {
          this.state = HighlightOcaml.RAWSTR_PIPE;
        }
        break;

      case HighlightOcaml.RAWSTR_PIPE:
        this.append(c);
        if (c == '}' && this.word2 == this.word) {
          this.pop();
          this.word2 = '';
          this.word = '';
          this.state = HighlightOcaml.NORMAL;
        } else if (c == '|') {
          this.word2 = '';
        } else if (isalpha(c) || c == '_') {
          this.word2 += c;
        } else {
          this.word2 = '';
          this.state = HighlightOcaml.RAWSTR;
        }
        break;

      default:
        throw new Error('Invalid state');
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightOcaml.WORD:
      if (OCAML_KEYWORDS.has(this.word)) {
        this.push("span", "keyword");
        this.append(this.word);
        this.pop();
      } else if (OCAML_BUILTINS.has(this.word)) {
        this.push("span", "builtin");
        this.append(this.word);
        this.pop();
      } else if (OCAML_CONSTANTS.has(this.word)) {
        this.push("span", "constant");
        this.append(this.word);
        this.pop();
      } else {
        this.append(this.word);
      }
      this.word = '';
      break;
    case HighlightOcaml.LPAREN:
      this.append('(');
      break;
    case HighlightOcaml.LCURLY:
      this.append('{');
      this.append(this.word);
      this.word = '';
      break;
    case HighlightOcaml.RAWSTR:
    case HighlightOcaml.RAWSTR_PIPE:
      this.word = '';
      this.word2 = '';
      this.pop();
      break;
    case HighlightOcaml.QUOTE:
    case HighlightOcaml.QUOTE_BACKSLASH:
    case HighlightOcaml.DQUOTE:
    case HighlightOcaml.DQUOTE_BACKSLASH:
    case HighlightOcaml.COMMENT:
    case HighlightOcaml.COMMENT_STAR:
    case HighlightOcaml.COMMENT_LPAREN:
      this.pop();
      break;
    default:
      break;
    }
    this.state = HighlightOcaml.NORMAL;
    this.delegate.flush();
    this.delta = 1;
  }
}

Highlighter.REGISTRY['ocaml'] = HighlightOcaml;
Highlighter.REGISTRY['ml'] = HighlightOcaml;
Highlighter.REGISTRY['mli'] = HighlightOcaml;
