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

const MAKE_KEYWORDS = new Set([
  '-include',
  '-sinclude',
  'define',
  'else',
  'endef',
  'endif',
  'export',
  'if',
  'ifdef',
  'ifeq',
  'ifndef',
  'ifneq',
  'include',
  'override',
  'private',
  'sinclude',
  'undefine',
  'unexport',
  'vpath',
]);

const MAKE_BUILTINS = new Set([
  'abspath',
  'addprefix',
  'addsuffix',
  'and',
  'basename',
  'call',
  'dir',
  'error',
  'eval',
  'file',
  'filter',
  'filter-out',
  'findstring',
  'firstword',
  'flavor',
  'foreach',
  'if',
  'join',
  'lastword',
  'notdir',
  'or',
  'origin',
  'patsubst',
  'realpath',
  'shell',
  'sort',
  'strip',
  'subst',
  'suffix',
  'value',
  'warning',
  'wildcard',
  'word',
  'wordlist',
  'words',
]);

class HighlightMake extends Highlighter {

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

Highlighter.REGISTRY['make'] = HighlightMake;
Highlighter.REGISTRY['mk'] = HighlightMake;
Highlighter.REGISTRY['gmake'] = HighlightMake;
Highlighter.REGISTRY['makefile'] = HighlightMake;
Highlighter.REGISTRY['gmakefile'] = HighlightMake;
