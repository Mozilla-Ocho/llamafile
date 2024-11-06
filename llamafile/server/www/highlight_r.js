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

const R_KEYWORDS = new Set([
  'NA',
  'NA_character_',
  'NA_complex_',
  'NA_integer_',
  'NA_real_',
  'break',
  'else',
  'for',
  'function',
  'if',
  'in',
  'next',
  'repeat',
  'while',
]);

const R_BUILTINS = new Set([
  'Arg',
  'Conj',
  'Im',
  'Mod',
  'Re',
  'acos',
  'aggregate',
  'anova',
  'aov',
  'apply',
  'as.array',
  'as.character',
  'as.complex',
  'as.data.frame',
  'as.logical',
  'as.numeric',
  'asin',
  'atan',
  'atan2',
  'attr',
  'attributes',
  'barplot',
  'binom.test',
  'boxplot',
  'by',
  'cbind',
  'choose',
  'class',
  'colMeans',
  'colSums',
  'colsum',
  'convolve',
  'coplot',
  'cor',
  'cos',
  'cov',
  'cummax',
  'cummin',
  'cumprod',
  'cumsum',
  'cut',
  'density',
  'diag',
  'diff',
  'dim',
  'dimnames',
  'do.call',
  'dotchart',
  'exp',
  'fft',
  'filter',
  'grep',
  'gsub',
  'hist',
  'ifelse',
  'interaction.plot',
  'is.array',
  'is.character',
  'is.complex',
  'is.data.frame',
  'is.element',
  'is.na',
  'is.null',
  'is.numeric',
  'lapply',
  'length',
  'list',
  'log',
  'log10',
  'match',
  'matrix',
  'max',
  'mean',
  'median',
  'merge',
  'min',
  'mvfft',
  'na.fail',
  'na.omit',
  'ncol',
  'nrow',
  'optim',
  'pairwise.t.test',
  'paste',
  'pie',
  'plot',
  'pmatch',
  'pmax',
  'pmin',
  'power.t.test',
  'print',
  'prod',
  'prop.table',
  'prop.test',
  'quantile',
  'range',
  'rank',
  'rbeta',
  'rbind',
  'rbinom',
  'rcauchy',
  'rchisq',
  'reshape',
  'return',
  'rev',
  'rexp',
  'rf',
  'rgamma',
  'rgeom',
  'rhyper',
  'rlnorm',
  'rlogis',
  'rnbinom',
  'rnorm',
  'round',
  'rowMeans',
  'rowsum',
  'rpois',
  'rt',
  'runif',
  'rweibull',
  'rwilcox',
  'sample',
  'scale',
  'sd',
  'seq',
  'sin',
  'solve',
  'sort',
  'stack',
  'stripplot',
  'strsplit',
  'subset',
  'substr',
  'sum',
  'sunflowerplot',
  't',
  'table',
  'tan',
  'tapply',
  'tolower',
  'toupper',
  'typeof',
  'unclass',
  'union',
  'unique',
  'unstack',
  'var',
  'weighted.mean',
  'which',
  'which.max',
  'which.min',
  'xtabs',
]);

const R_CONSTANTS = new Set([
  'FALSE',
  'Inf',
  'NULL',
  'NaN',
  'TRUE',
]);

class HighlightR extends Highlighter {

  static NORMAL = 0;
  static WORD = 1;
  static COMMENT = 2;
  static QUOTE = 3;
  static QUOTE_BACKSLASH = 4;
  static DQUOTE = 5;
  static DQUOTE_BACKSLASH = 6;
  static HYPHEN = 7;
  static HYPHEN_GT = 8;
  static LT = 9;
  static LT_LT = 10;
  static COLON = 11;

  constructor(delegate) {
    super(delegate);
    this.word = '';
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      let c = input[i];
      switch (this.state) {

      case HighlightR.NORMAL:
        if (!isascii(c) || isalpha(c)) {
          this.state = HighlightR.WORD;
          this.word += c;
        } else if (c == '#') {
          this.push("span", "comment");
          this.append('#');
          this.state = HighlightR.COMMENT;
        } else if (c == '\'') {
          this.state = HighlightR.QUOTE;
          this.push("span", "string");
          this.append(c);
        } else if (c == '"') {
          this.state = HighlightR.DQUOTE;
          this.push("span", "string");
          this.append(c);
        } else if (c == '-') {
          this.state = HighlightR.HYPHEN;
        } else if (c == '<') {
          this.state = HighlightR.LT;
        } else if (c == ':') {
          this.state = HighlightR.COLON;
        } else if (c == '$' || c == '@') {
          this.push("span", "operator");
          this.append(c);
          this.pop();
        } else {
          this.append(c);
        }
        break;

      case HighlightR.WORD:
        if (!isascii(c) || isalnum(c) || c == '_' || c == '.') {
          this.word += c;
        } else {
          if (R_KEYWORDS.has(this.word)) {
            this.push("span", "keyword");
            this.append(this.word);
            this.pop();
          } else if (R_BUILTINS.has(this.word)) {
            this.push("span", "builtin");
            this.append(this.word);
            this.pop();
          } else if (R_CONSTANTS.has(this.word)) {
            this.push("span", "constant");
            this.append(this.word);
            this.pop();
          } else {
            this.append(this.word);
          }
          this.word = '';
          this.epsilon(HighlightR.NORMAL);
        }
        break;

      case HighlightR.COMMENT:
        this.append(c);
        if (c == '\n') {
          this.pop();
          this.state = HighlightR.NORMAL;
        }
        break;

      case HighlightR.QUOTE:
        this.append(c);
        if (c == '\'') {
          this.pop();
          this.state = HighlightR.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightR.QUOTE_BACKSLASH;
        }
        break;

      case HighlightR.QUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightR.QUOTE;
        break;

      case HighlightR.DQUOTE:
        this.append(c);
        if (c == '"') {
          this.pop();
          this.state = HighlightR.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightR.DQUOTE_BACKSLASH;
        }
        break;

      case HighlightR.DQUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightR.DQUOTE;
        break;

      case HighlightR.COLON:
        if (c == ':') {
          this.push("span", "operator");
          this.append("::");
          this.pop();
          this.state = HighlightR.NORMAL;
        } else {
          this.append(':');
          this.epsilon(HighlightR.NORMAL);
        }
        break;

      case HighlightR.LT:
        if (c == '<') {
          this.state = HighlightR.LT_LT;
        } else if (c == '-') {
          this.push("span", "operator");
          this.append("<-");
          this.pop();
          this.state = HighlightR.NORMAL;
        } else {
          this.append('<');
          this.epsilon(HighlightR.NORMAL);
        }
        break;

      case HighlightR.HYPHEN:
        if (c == '>') {
          this.state = HighlightR.HYPHEN_GT;
        } else {
          this.append('-');
          this.epsilon(HighlightR.NORMAL);
        }
        break;

      case HighlightR.LT_LT:
        if (c == '-') {
          this.push("span", "operator");
          this.append("<<-");
          this.pop();
          this.state = HighlightR.NORMAL;
        } else {
          this.append("<<");
          this.epsilon(HighlightR.NORMAL);
        }
        break;

      case HighlightR.HYPHEN_GT:
        if (c == '>') {
          this.push("span", "operator");
          this.append("->>");
          this.pop();
          this.state = HighlightR.NORMAL;
        } else {
          this.push("span", "operator");
          this.append("->");
          this.pop();
          this.epsilon(HighlightR.NORMAL);
        }
        break;

      default:
        throw new Error('Invalid state');
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightR.WORD:
      if (R_KEYWORDS.has(this.word)) {
        this.push("span", "keyword");
        this.append(this.word);
        this.pop();
      } else if (R_BUILTINS.has(this.word)) {
        this.push("span", "builtin");
        this.append(this.word);
        this.pop();
      } else if (R_CONSTANTS.has(this.word)) {
        this.push("span", "constant");
        this.append(this.word);
        this.pop();
      } else {
        this.append(this.word);
      }
      this.word = '';
      break;
    case HighlightR.HYPHEN:
      this.append('-');
      break;
    case HighlightR.HYPHEN_GT:
      this.append("->");
      break;
    case HighlightR.LT:
      this.append('<');
      break;
    case HighlightR.LT_LT:
      this.append("<<");
      break;
    case HighlightR.COLON:
      this.append(':');
      break;
    case HighlightR.QUOTE:
    case HighlightR.QUOTE_BACKSLASH:
    case HighlightR.DQUOTE:
    case HighlightR.DQUOTE_BACKSLASH:
    case HighlightR.COMMENT:
      this.pop();
      break;
    default:
      break;
    }
    this.state = HighlightR.NORMAL;
    this.delegate.flush();
    this.delta = 1;
  }
}

Highlighter.REGISTRY['r'] = HighlightR;
