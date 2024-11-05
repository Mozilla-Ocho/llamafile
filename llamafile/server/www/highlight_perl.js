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

const PERL_KEYWORDS = new Set([
  'BEGIN',
  'END',
  'continue',
  'default',
  'die',
  'do',
  'dump',
  'else',
  'elsif',
  'eval',
  'exec',
  'exit',
  'for',
  'foreach',
  'given',
  'goto',
  'has',
  'if',
  'import',
  'last',
  'local',
  'my',
  'next',
  'no',
  'our',
  'package',
  'redo',
  'require',
  'return',
  'state',
  'sub',
  'unless',
  'until',
  'use',
  'when',
  'while',
]);

class HighlightPerl extends Highlighter {

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

Highlighter.REGISTRY['perl'] = HighlightPerl;
Highlighter.REGISTRY['pl'] = HighlightPerl;
