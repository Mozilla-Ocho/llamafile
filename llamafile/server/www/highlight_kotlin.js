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

const KOTLIN_KEYWORDS = new Set([
  'abstract',
  'actual',
  'annotation',
  'as',
  'break',
  'by',
  'catch',
  'class',
  'companion',
  'const',
  'constructor',
  'continue',
  'crossinline',
  'data',
  'delegate',
  'do',
  'dynamic',
  'else',
  'enum',
  'expect',
  'external',
  'field',
  'file',
  'final',
  'finally',
  'for',
  'fun',
  'get',
  'if',
  'import',
  'in',
  'infix',
  'init',
  'inline',
  'inner',
  'interface',
  'internal',
  'is',
  'it',
  'lateinit',
  'noinline',
  'object',
  'open',
  'operator',
  'out',
  'override',
  'package',
  'param',
  'private',
  'property',
  'protected',
  'public',
  'receiver',
  'reified',
  'return',
  'sealed',
  'set',
  'setparam',
  'super',
  'suspend',
  'tailrec',
  'this',
  'throw',
  'try',
  'typealias',
  'typeof',
  'val',
  'var',
  'vararg',
  'when',
  'where',
  'while',
]);

class HighlightKotlin extends Highlighter {

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

Highlighter.REGISTRY['kotlin'] = HighlightKotlin;
Highlighter.REGISTRY['kts'] = HighlightKotlin;
Highlighter.REGISTRY['kt'] = HighlightKotlin;
