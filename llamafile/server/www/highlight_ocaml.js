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

Highlighter.REGISTRY['ocaml'] = HighlightOcaml;
Highlighter.REGISTRY['ml'] = HighlightOcaml;
Highlighter.REGISTRY['mli'] = HighlightOcaml;
