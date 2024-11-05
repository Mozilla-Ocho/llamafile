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

const FORTH_KEYWORDS = new Set([
  '+loop',
  '2literal',
  ':noname',
  ';',
  '?do',
  '?of',
  'again',
  'begin',
  'begin-structure',
  'case',
  'do',
  'does>',
  'else',
  'end-structure',
  'endcase',
  'endof',
  'exit',
  'fliteral',
  'if',
  'immediate',
  'is',
  'leave',
  'literal',
  'loop',
  'of',
  'postpone',
  'repeat',
  'sliteral',
  'then',
  'to',
  'unloop',
  'until',
  'while',
]);

const FORTH_DEFS = new Set([
  '+field',
  '2constant',
  '2value',
  '2variable',
  ':',
  'cfield:',
  'code',
  'constant',
  'create',
  'defer',
  'dffield:',
  'fconstant',
  'ffield:',
  'field:',
  'fvalue',
  'fvariable',
  'sffield:',
  'synonym',
  'value',
  'variable',
]);

class HighlightForth extends Highlighter {

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

Highlighter.REGISTRY['forth'] = HighlightForth;
Highlighter.REGISTRY['fth'] = HighlightForth;
Highlighter.REGISTRY['frt'] = HighlightForth;
Highlighter.REGISTRY['4th'] = HighlightForth;
Highlighter.REGISTRY['fs'] = HighlightForth;
