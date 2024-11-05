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

const ASM_PREFIXS = new Set([
  'addr32',
  'cs',
  'data16',
  'ds',
  'es',
  'fs',
  'gs',
  'lock',
  'rep',
  'repe',
  'repne',
  'repnz',
  'repz',
  'rex',
  'rex.b',
  'rex.r',
  'rex.rb',
  'rex.rx',
  'rex.rxb',
  'rex.w',
  'rex.wb',
  'rex.wr',
  'rex.wrb',
  'rex.wrx',
  'rex.wrxb',
  'rex.wx',
  'rex.wxb',
  'rex.x',
  'rex.xb',
  'ss',
]);

const ASM_QUALIFIERS = new Set([
  ':req',
  ':vararg',
  '@dtpmod',
  '@dtpoff',
  '@fini_array',
  '@function',
  '@gnu_indirect_function',
  '@got',
  '@gotoff',
  '@gotpcrel',
  '@gottpoff',
  '@init_array',
  '@nobits',
  '@note',
  '@notype',
  '@object',
  '@plt',
  '@pltoff',
  '@progbits',
  '@size',
  '@tlsgd',
  '@tlsld',
  '@tpoff',
  '@unwind',
  'comdat',
]);

class HighlightAsm extends Highlighter {

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

Highlighter.REGISTRY['asm'] = HighlightAsm;
