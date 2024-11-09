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

const LISP_KEYWORDS = new Set([
  'assert',
  'block',
  'break',
  'case',
  'ccase',
  'check-type',
  'cl-defsubst',
  'compiler-let',
  'cond',
  'condition-case',
  'ctypecase',
  'declaim',
  'declare',
  'def',
  'defadvice',
  'defalias',
  'defconst',
  'defconstant',
  'defcustom',
  'defface',
  'defgeneric',
  'defgroup',
  'define',
  'define-advice',
  'define-compiler-macro',
  'define-condition',
  'define-derived-mode',
  'define-generic-mode',
  'define-global-minor-mode',
  'define-globalized-minor-mode',
  'define-inline',
  'define-method-combination',
  'define-minor-mode',
  'define-modify-macro',
  'define-setf-expander',
  'define-skeleton',
  'define-symbol-macro',
  'define-widget',
  'defmacro',
  'defmethod',
  'defn',
  'defpackage',
  'defparameter',
  'defsetf',
  'defstruct',
  'defsubst',
  'deftheme',
  'deftype',
  'defun',
  'defvar-local',
  'defvaralias',
  'destructuring-bind',
  'do',
  'do*',
  'dolist',
  'dotimes',
  'ecase',
  'ert-deftest',
  'etypecase',
  'eval-when',
  'flet',
  'flet*',
  'fn',
  'go',
  'handler-bind',
  'handler-case',
  'if',
  'ignore-errors',
  'in-package',
  'labels',
  'lambda',
  'let',
  'let*',
  'letf',
  'letfn',
  'locally',
  'loop',
  'macrolet',
  'monitor-enter',
  'monitor-exit',
  'multiple-value-bind',
  'multiple-value-prog1',
  'proclaim',
  'prog',
  'prog*',
  'prog1',
  'prog2',
  'progn',
  'progv',
  'quote',
  'recur',
  'restart-bind',
  'restart-case',
  'return',
  'return-from',
  'set!',
  'symbol-macrolet',
  'tagbody',
  'the',
  'throw',
  'try',
  'typecase',
  'unless',
  'unwind-protect',
  'var',
  'when',
  'while',
  'with-accessors',
  'with-compilation-unit',
  'with-condition-restarts',
  'with-hash-table-iterator',
  'with-input-from-string',
  'with-open-file',
  'with-open-stream',
  'with-output-to-string',
  'with-package-iterator',
  'with-simple-restart',
  'with-slots',
  'with-standard-io-syntax',
]);

class HighlightLisp extends Highlighter {

  static NORMAL = 0;
  static SYMBOL = 1;
  static DQUOTE = 2;
  static DQUOTE_BACKSLASH = 3;
  static COMMENT = 4;

  constructor(delegate) {
    super(delegate);
    this.word = '';
    this.is_first = false;
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      let c = input[i];
      switch (this.state) {

      case HighlightLisp.NORMAL:
        if (c == '(') {
          this.append(c);
          this.is_first = true;
        } else if (c == ';') {
          this.push("span", "comment");
          this.append(c);
          this.state = HighlightLisp.COMMENT;
        } else if (c == '[') {
          this.append(c);
          this.state = HighlightLisp.SYMBOL;
          this.is_first = false;
        } else if (c == ')' || c == ']') {
          this.append(c);
          this.is_first = false;
        } else if (c == '\'' || c == '#' || c == '`' || c == ',') {
          this.append(c);
          this.is_first = false;
        } else if (c == '"') {
          this.push("span", "string");
          this.append(c);
          this.state = HighlightLisp.DQUOTE;
          this.is_first = false;
        } else if (isspace(c)) {
          this.append(c);
        } else {
          this.word += c;
          this.state = HighlightLisp.SYMBOL;
        }
        break;

      case HighlightLisp.SYMBOL:
        if (isspace(c) || //
            c == '(' || //
            c == ')' || //
            c == '[' || //
            c == ']' || //
            c == ',' || //
            c == '#' || //
            c == '`' || //
            c == '"' || //
            c == '\'') {
          if (this.is_first && LISP_KEYWORDS.has(this.word)) {
            this.push("span", "keyword");
            this.append(this.word);
            this.pop();
          } else if (this.word.length > 1 && this.word[0] == ':') {
            this.push("span", "lispkw");
            this.append(this.word);
            this.pop();
          } else {
            this.append(this.word);
          }
          this.is_first = false;
          this.word = '';
          this.epsilon(HighlightLisp.NORMAL);
        } else {
          this.word += c;
        }
        break;

      case HighlightLisp.DQUOTE:
        this.append(c);
        if (c == '"') {
          this.pop();
          this.state = HighlightLisp.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightLisp.DQUOTE_BACKSLASH;
        }
        break;

      case HighlightLisp.DQUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightLisp.DQUOTE;
        break;

      case HighlightLisp.COMMENT:
        this.append(c);
        if (c == '\n') {
          this.pop();
          this.state = HighlightLisp.NORMAL;
        }
        break;

      default:
        throw new Error('Invalid state');
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightLisp.SYMBOL:
      if (this.is_first && LISP_KEYWORDS.has(this.word)) {
        this.push("span", "keyword");
        this.append(this.word);
        this.pop();
      } else if (this.word.length > 1 && this.word[0] == ':') {
        this.push("span", "lispkw");
        this.append(this.word);
        this.pop();
      } else {
        this.append(this.word);
      }
      this.word = '';
      break;
    case HighlightLisp.DQUOTE:
    case HighlightLisp.DQUOTE_BACKSLASH:
    case HighlightLisp.COMMENT:
      this.pop();
      break;
    default:
      break;
    }
    this.is_first = false;
    this.state = HighlightLisp.NORMAL;
    this.delegate.flush();
    this.delta = 1;
  }
}

Highlighter.REGISTRY['clojure'] = HighlightLisp;
Highlighter.REGISTRY['racket'] = HighlightLisp;
Highlighter.REGISTRY['scheme'] = HighlightLisp;
Highlighter.REGISTRY['elisp'] = HighlightLisp;
Highlighter.REGISTRY['clisp'] = HighlightLisp;
Highlighter.REGISTRY['lisp'] = HighlightLisp;
Highlighter.REGISTRY['jl'] = HighlightLisp;
Highlighter.REGISTRY['cl'] = HighlightLisp;
