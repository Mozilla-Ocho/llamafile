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

const M4_KEYWORDS = new Set([
  '__file__',
  '__gnu__',
  '__line__',
  '__os2__',
  '__program__',
  '__unix__',
  '__windows__',
  'argn',
  'array',
  'array_set',
  'builtin',
  'capitalize',
  'changecom',
  'changequote',
  'changeword',
  'cleardivert',
  'cond',
  'copy',
  'curry',
  'debugfile',
  'debugmode',
  'decr',
  'define',
  'define_blind',
  'defn',
  'divert',
  'divnum',
  'downcase',
  'dquote',
  'dquote_elt',
  'dumpdef',
  'errprint',
  'esyscmd',
  'eval',
  'example',
  'exch',
  'fatal_error',
  'file',
  'foreach',
  'foreachq',
  'forloop',
  'format',
  'gnu',
  'ifdef',
  'ifelse',
  'include',
  'incr',
  'index',
  'indir',
  'join',
  'joinall',
  'len',
  'line',
  'm4___file__',
  'm4___gnu__',
  'm4___line__',
  'm4___os2__',
  'm4___program__',
  'm4___unix__',
  'm4___windows__',
  'm4_argn',
  'm4_array',
  'm4_array_set',
  'm4_builtin',
  'm4_capitalize',
  'm4_changecom',
  'm4_changequote',
  'm4_changeword',
  'm4_cleardivert',
  'm4_cond',
  'm4_copy',
  'm4_curry',
  'm4_debugfile',
  'm4_debugmode',
  'm4_decr',
  'm4_define',
  'm4_define_blind',
  'm4_defn',
  'm4_divert',
  'm4_divnum',
  'm4_downcase',
  'm4_dquote',
  'm4_dquote_elt',
  'm4_dumpdef',
  'm4_errprint',
  'm4_esyscmd',
  'm4_eval',
  'm4_example',
  'm4_exch',
  'm4_fatal_error',
  'm4_file',
  'm4_foreach',
  'm4_foreachq',
  'm4_forloop',
  'm4_format',
  'm4_gnu',
  'm4_ifdef',
  'm4_ifelse',
  'm4_include',
  'm4_incr',
  'm4_index',
  'm4_indir',
  'm4_join',
  'm4_joinall',
  'm4_len',
  'm4_line',
  'm4_m4exit',
  'm4_m4wrap',
  'm4_maketemp',
  'm4_mkstemp',
  'm4_nargs',
  'm4_os2',
  'm4_patsubst',
  'm4_popdef',
  'm4_pushdef',
  'm4_quote',
  'm4_regexp',
  'm4_rename',
  'm4_reverse',
  'm4_shift',
  'm4_sinclude',
  'm4_stack_foreach',
  'm4_stack_foreach_lifo',
  'm4_stack_foreach_sep',
  'm4_stack_foreach_sep_lifo',
  'm4_substr',
  'm4_syscmd',
  'm4_sysval',
  'm4_traceoff',
  'm4_traceon',
  'm4_translit',
  'm4_undefine',
  'm4_undivert',
  'm4_unix',
  'm4_upcase',
  'm4_windows',
  'm4exit',
  'm4wrap',
  'maketemp',
  'mkstemp',
  'nargs',
  'os2',
  'patsubst',
  'popdef',
  'pushdef',
  'quote',
  'regexp',
  'rename',
  'reverse',
  'shift',
  'sinclude',
  'stack_foreach',
  'stack_foreach_lifo',
  'stack_foreach_sep',
  'stack_foreach_sep_lifo',
  'substr',
  'syscmd',
  'sysval',
  'traceoff',
  'traceon',
  'translit',
  'undefine',
  'undivert',
  'unix',
  'upcase',
  'windows',
]);

class HighlightM4 extends Highlighter {

  static NORMAL = 0;
  static WORD = 1;
  static COMMENT = 2;
  static DOLLAR = 3;

  constructor(delegate) {
    super(delegate);
    this.word = '';
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      let c = input[i];
      switch (this.state) {

      case HighlightM4.NORMAL:
        if (!isascii(c) || isalpha(c) || c == '_') {
          this.state = HighlightM4.WORD;
          this.word += c;
        } else if (c == '#') {
          this.state = HighlightM4.COMMENT;
          this.push("span", "comment");
          this.append('#');
        } else if (c == '$') {
          this.state = HighlightM4.DOLLAR;
        } else {
          this.append(c);
        }
        break;

      case HighlightM4.WORD:
        if (!isascii(c) || isalnum(c) || c == '_') {
          this.word += c;
        } else {
          if (M4_KEYWORDS.has(this.word)) {
            this.push("span", "keyword");
            this.append(this.word);
            this.pop();
          } else if (this.word == "dnl" || this.word == "m4_dnl") {
            this.push("span", "comment");
            this.append(this.word);
            this.word = '';
            this.epsilon(HighlightM4.COMMENT);
          } else {
            this.append(this.word);
          }
          this.word = '';
          this.epsilon(HighlightM4.NORMAL);
        }
        break;

      case HighlightM4.DOLLAR:
        if (isdigit(c) || c == '*' || c == '#' || c == '@') {
          this.append('$');
          this.push("span", "var");
          this.append(c);
          this.pop();
          this.state = HighlightM4.NORMAL;
        } else {
          this.append('$');
          this.epsilon(HighlightM4.NORMAL);
        }
        break;

      case HighlightM4.COMMENT:
        this.append(c);
        if (c == '\n') {
          this.pop();
          this.state = HighlightM4.NORMAL;
        }
        break;

      default:
        throw new Error('Invalid state');
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightM4.WORD:
      if (M4_KEYWORDS.has(this.word)) {
        this.push("span", "keyword");
        this.append(this.word);
        this.pop();
      } else {
        this.append(this.word);
      }
      this.word = '';
      break;
    case HighlightM4.DOLLAR:
      this.append('$');
      break;
    case HighlightM4.COMMENT:
      this.pop();
      break;
    default:
      break;
    }
    this.state = HighlightM4.NORMAL;
    this.delegate.flush();
    this.delta = 1;
  }
}

Highlighter.REGISTRY['m4'] = HighlightM4;
Highlighter.REGISTRY['ac'] = HighlightM4;
