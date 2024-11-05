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

class HighlightD extends HighlightC {
  constructor(delegate) {
    super(delegate);
    this.keywords = D_KEYWORDS;
    this.builtins = null;
    this.constants = D_CONSTANTS;
    this.types = null;
  }
}

Highlighter.REGISTRY['d'] = HighlightD;
