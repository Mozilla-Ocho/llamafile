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

const RUST_KEYWORDS = new Set([
  'Self',
  'abstract',
  'as',
  'async',
  'await',
  'become',
  'box',
  'break',
  'const',
  'continue',
  'crate',
  'do',
  'dyn',
  'else',
  'enum',
  'extern',
  'final',
  'fn',
  'for',
  'if',
  'impl',
  'in',
  'let',
  'loop',
  'macro',
  'match',
  'mod',
  'move',
  'mut',
  'override',
  'priv',
  'pub',
  'ref',
  'return',
  'self',
  'static',
  'struct',
  'super',
  'trait',
  'try',
  'type',
  'typeof',
  'union',
  'unsafe',
  'unsized',
  'use',
  'virtual',
  'where',
  'while',
  'yield',
]);

const RUST_CONSTANTS = new Set([
  'false',
  'true',
]);

const RUST_TYPES = new Set([
  '!',
  'bool',
  'char',
  'f32',
  'f64',
  'i128',
  'i16',
  'i32',
  'i64',
  'i8',
  'isize',
  'str',
  'u128',
  'u16',
  'u32',
  'u64',
  'u8',
  'usize',
]);

class HighlightRust extends Highlighter {

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

Highlighter.REGISTRY['rust'] = HighlightRust;
Highlighter.REGISTRY['rs'] = HighlightRust;
