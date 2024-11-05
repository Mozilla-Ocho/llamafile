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

const CSHARP_KEYWORDS = new Set([
  'abstract',
  'add',
  'alias',
  'allows',
  'and',
  'args',
  'as',
  'ascending',
  'async',
  'await',
  'base',
  'bool',
  'break',
  'by',
  'byte',
  'case',
  'catch',
  'char',
  'checked',
  'class',
  'const',
  'continue',
  'decimal',
  'default',
  'delegate',
  'descending',
  'do',
  'double',
  'dynamic',
  'else',
  'enum',
  'equals',
  'event',
  'explicit',
  'extern',
  'file',
  'finally',
  'fixed',
  'float',
  'for',
  'foreach',
  'from',
  'get',
  'global',
  'goto',
  'group',
  'if',
  'implicit',
  'in',
  'init',
  'int',
  'interface',
  'internal',
  'into',
  'is',
  'join',
  'let',
  'lock',
  'long',
  'managed',
  'nameof',
  'namespace',
  'new',
  'nint',
  'not',
  'notnull',
  'nuint',
  'object',
  'on',
  'operator',
  'or',
  'orderby',
  'out',
  'override',
  'params',
  'partial',
  'private',
  'protected',
  'public',
  'readonly',
  'record',
  'ref',
  'remove',
  'required',
  'return',
  'sbyte',
  'scoped',
  'sealed',
  'select',
  'set',
  'short',
  'sizeof',
  'stackalloc',
  'static',
  'string',
  'struct',
  'switch',
  'this',
  'throw',
  'try',
  'typeof',
  'uint',
  'ulong',
  'unchecked',
  'unmanaged',
  'unsafe',
  'ushort',
  'using',
  'value',
  'var',
  'virtual',
  'void',
  'volatile',
  'when',
  'where',
  'while',
  'with',
  'yield',
]);

const CSHARP_CONSTANTS = new Set([
  'false',
  'null',
  'true',
]);

class HighlightCsharp extends Highlighter {

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

Highlighter.REGISTRY['csharp'] = HighlightCsharp;
Highlighter.REGISTRY['cs'] = HighlightCsharp;
Highlighter.REGISTRY['c#'] = HighlightCsharp;
