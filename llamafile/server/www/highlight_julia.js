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

const JULIA_KEYWORDS = new Set([
  'baremodule',
  'begin',
  'break',
  'catch',
  'const',
  'continue',
  'do',
  'else',
  'elseif',
  'end',
  'export',
  'false',
  'finally',
  'for',
  'function',
  'global',
  'if',
  'import',
  'let',
  'local',
  'macro',
  'module',
  'quote',
  'return',
  'struct',
  'true',
  'try',
  'using',
  'while',
]);

class HighlightJulia extends Highlighter {

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

Highlighter.REGISTRY['julia'] = HighlightJulia;
