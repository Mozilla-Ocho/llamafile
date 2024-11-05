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

const LD_KEYWORDS = new Set([
  'AFTER',
  'ASSERT',
  'AS_NEEDED',
  'AT',
  'BEFORE',
  'BYTE',
  'COMMON',
  'CONSTRUCTORS',
  'COPY',
  'CREATE_OBJECT_SYMBOLS',
  'DSECT',
  'ENTRY',
  'EXTERN',
  'FILEHDR',
  'FILL',
  'FLAGS',
  'FORCE_COMMON_ALLOCATION',
  'GROUP',
  'HIDDEN',
  'INCLUDE',
  'INFO',
  'INHIBIT_COMMON_ALLOCATION',
  'INPUT',
  'INSERT',
  'KEEP',
  'LD_FEATURE',
  'LONG',
  'MEMORY',
  'NOCROSSREFS',
  'NOCROSSREFS_TO',
  'NOLOAD',
  'ONLY_IF_RO',
  'ONLY_IF_RW',
  'OUTPUT',
  'OUTPUT_ARCH',
  'OUTPUT_FORMAT',
  'OVERLAY',
  'PHDR',
  'PHDRS',
  'PROVIDE',
  'PROVIDE_HIDDEN',
  'PT_DYNAMIC',
  'PT_GNU_STACK',
  'PT_INTERP',
  'PT_LOAD',
  'PT_NOTE',
  'PT_NULL',
  'PT_PHDR',
  'PT_SHLIB',
  'PT_TLS',
  'QUAD',
  'REGION_ALIAS',
  'SEARCH_DIR',
  'SECTIONS',
  'SHORT',
  'SORT',
  'SORT_BY_ALIGNMENT',
  'SORT_BY_INIT_PRIORITY',
  'SORT_BY_NAME',
  'SORT_NONE',
  'SQUAD',
  'STARTUP',
  'SUBALIGN',
  'TARGET',
  'VERSION',
  '__CTOR_END__',
  '__CTOR_LIST__',
  '__DTOR_END__',
  '__DTOR_LIST__',
]);

const LD_BUILTINS = new Set([
  'ABSOLUTE',
  'ADDR',
  'ALIGN',
  'ALIGNOF',
  'BLOCK',
  'COMMONPAGESIZE',
  'CONSTANT',
  'DATA_SEGMENT_ALIGN',
  'DATA_SEGMENT_END',
  'DATA_SEGMENT_RELRO_END',
  'DEFINED',
  'LENGTH',
  'LOADADDR',
  'LOG2CEIL',
  'MAX',
  'MAXPAGESIZE',
  'MIN',
  'NEXT',
  'ORIGIN',
  'SEGMENT_START',
  'SIZEOF',
  'SIZEOF_HEADERS',
  'l',
  'len',
  'o',
  'org',
  'sizeof_headers',
]);

const LD_WARNINGS = new Set([
  '/DISCARD/',
  ':NONE',
  'EXCLUDE_FILE',
]);

class HighlightLd extends Highlighter {

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

Highlighter.REGISTRY['ld-script'] = HighlightLd;
Highlighter.REGISTRY['lds'] = HighlightLd;
Highlighter.REGISTRY['ld'] = HighlightLd;
