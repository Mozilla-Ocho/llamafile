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
  'locally',
  'loop',
  'macrolet',
  'multiple-value-bind',
  'multiple-value-prog1',
  'proclaim',
  'prog',
  'prog*',
  'prog1',
  'prog2',
  'progn',
  'progv',
  'restart-bind',
  'restart-case',
  'return',
  'return-from',
  'symbol-macrolet',
  'tagbody',
  'the',
  'typecase',
  'unless',
  'unwind-protect',
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

  constructor(delegate) {
    super(delegate);
  }

  feed(input) {
    this.append(input);
  }

  flush() {
    this.delegate.flush();
  }
}

Highlighter.REGISTRY['clojure'] = HighlightLisp;
Highlighter.REGISTRY['scheme'] = HighlightLisp;
Highlighter.REGISTRY['elisp'] = HighlightLisp;
Highlighter.REGISTRY['clisp'] = HighlightLisp;
Highlighter.REGISTRY['lisp'] = HighlightLisp;
Highlighter.REGISTRY['jl'] = HighlightLisp;
Highlighter.REGISTRY['cl'] = HighlightLisp;
