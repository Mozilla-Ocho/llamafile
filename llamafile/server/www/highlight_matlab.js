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

const MATLAB_KEYWORDS = new Set([
  'break',
  'case',
  'catch',
  'classdef',
  'continue',
  'else',
  'elseif',
  'end',
  'for',
  'function',
  'global',
  'if',
  'methods',
  'otherwise',
  'parfor',
  'pause',
  'persistent',
  'properties',
  'return',
  'switch',
  'try',
  'while',
]);

const MATLAB_BUILTINS = new Set([
  'abs',
  'acos',
  'acosd',
  'acosh',
  'acot',
  'acotd',
  'acoth',
  'acsc',
  'acscd',
  'acsch',
  'addedge',
  'addnode',
  'adjacency',
  'airy',
  'allcycles',
  'allfinite',
  'allpaths',
  'amd',
  'angle',
  'anymissing',
  'anynan',
  'asec',
  'asecd',
  'asech',
  'asin',
  'asind',
  'asinh',
  'atan',
  'atan2',
  'atan2d',
  'atand',
  'atanh',
  'balance',
  'bandwidth',
  'bctree',
  'besselh',
  'besseli',
  'besselj',
  'besselk',
  'bessely',
  'beta',
  'betainc',
  'betaincinv',
  'betaln',
  'bfsearch',
  'bicg',
  'bicgstab',
  'bicgstabl',
  'biconncomp',
  'blkdiag',
  'bsxfun',
  'cart2pol',
  'cart2sph',
  'cat',
  'cdf2rdf',
  'ceil',
  'centrality',
  'cgs',
  'chol',
  'cholupdate',
  'circshift',
  'clip',
  'colamd',
  'colon',
  'colperm',
  'combinations',
  'compan',
  'complex',
  'cond',
  'condeig',
  'condensation',
  'condest',
  'conj',
  'conncomp',
  'conv',
  'conv2',
  'convn',
  'cos',
  'cosd',
  'cosh',
  'cospi',
  'cot',
  'cotd',
  'coth',
  'cplxpair',
  'cross',
  'csc',
  'cscd',
  'csch',
  'ctranspose',
  'cumprod',
  'cumsum',
  'cyclebasis',
  'decomposition',
  'deconv',
  'deg2rad',
  'degree',
  'det',
  'detrend',
  'dfsearch',
  'diag',
  'diff',
  'digraph',
  'disp',
  'dissect',
  'distances',
  'dmperm',
  'dot',
  'edgecount',
  'eig',
  'eigs',
  'ellipj',
  'ellipke',
  'equilibrate',
  'erf',
  'erfc',
  'erfcinv',
  'erfcx',
  'erfinv',
  'etree',
  'etreeplot',
  'eval',
  'exp',
  'expint',
  'expm',
  'expm1',
  'expmv',
  'eye',
  'factor',
  'factorial',
  'fft',
  'fft2',
  'fftn',
  'fftshift',
  'fftw',
  'fillmissing',
  'fillmissing2',
  'filloutliers',
  'filter',
  'filter2',
  'find',
  'findedge',
  'findnode',
  'fix',
  'flip',
  'flipedge',
  'fliplr',
  'flipud',
  'floor',
  'fminbnd',
  'fminsearch',
  'freqspace',
  'full',
  'funm',
  'fzero',
  'gallery',
  'gamma',
  'gammainc',
  'gammaincinv',
  'gammaln',
  'gcd',
  'gmres',
  'gplot',
  'graph',
  'griddata',
  'griddatan',
  'griddedInterpolant',
  'gsvd',
  'hadamard',
  'hankel',
  'hascycles',
  'head',
  'hess',
  'highlight',
  'hilb',
  'horzcat',
  'hypot',
  'ichol',
  'idivide',
  'ifft',
  'ifft2',
  'ifftn',
  'ifftshift',
  'ilu',
  'imag',
  'incidence',
  'ind2sub',
  'indegree',
  'inedges',
  'interp1',
  'interp2',
  'interp3',
  'interpft',
  'interpn',
  'inv',
  'invhilb',
  'ipermute',
  'isbanded',
  'isbetween',
  'ischange',
  'iscolumn',
  'isdag',
  'isdiag',
  'isempty',
  'isfinite',
  'ishermitian',
  'isinf',
  'isisomorphic',
  'islocalmax',
  'islocalmax2',
  'islocalmin',
  'islocalmin2',
  'ismatrix',
  'ismember',
  'ismembertol',
  'ismissing',
  'ismultigraph',
  'isnan',
  'isomorphism',
  'isoutlier',
  'isprime',
  'isreal',
  'isregular',
  'isrow',
  'isscalar',
  'issorted',
  'issortedrows',
  'issparse',
  'issymmetric',
  'istril',
  'istriu',
  'isuniform',
  'isvector',
  'kron',
  'labeledge',
  'labelnode',
  'laplacian',
  'layout',
  'layoutcoords',
  'lcm',
  'ldl',
  'legendre',
  'length',
  'linsolve',
  'linspace',
  'log',
  'log10',
  'log1p',
  'log2',
  'logical',
  'logm',
  'logspace',
  'lscov',
  'lsqminnorm',
  'lsqnonneg',
  'lsqr',
  'lu',
  'magic',
  'makima',
  'matchpairs',
  'maxflow',
  'meshgrid',
  'minres',
  'minspantree',
  'mkpp',
  'mldivide',
  'mod',
  'movmean',
  'movmedian',
  'movsum',
  'mpower',
  'mrdivide',
  'mtimes',
  'nchoosek',
  'ndgrid',
  'ndims',
  'nearest',
  'neighbors',
  'nextpow2',
  'nnz',
  'nonzeros',
  'norm',
  'normalize',
  'normest',
  'nthroot',
  'nufft',
  'nufftn',
  'null',
  'numedges',
  'numel',
  'numnodes',
  'nzmax',
  'ones',
  'optimget',
  'optimset',
  'ordeig',
  'ordqz',
  'ordschur',
  'orth',
  'outdegree',
  'outedges',
  'paddata',
  'padecoef',
  'pagectranspose',
  'pageeig',
  'pageinv',
  'pagelsqminnorm',
  'pagemldivide',
  'pagemrdivide',
  'pagemtimes',
  'pagenorm',
  'pagepinv',
  'pagesvd',
  'pagetranspose',
  'pascal',
  'pcg',
  'pchip',
  'perms',
  'permute',
  'pinv',
  'planerot',
  'plot',
  'pol2cart',
  'poly',
  'polyder',
  'polydiv',
  'polyeig',
  'polyfit',
  'polyint',
  'polyval',
  'polyvalm',
  'pow2',
  'ppval',
  'predecessors',
  'primes',
  'prod',
  'psi',
  'qmr',
  'qr',
  'qrdelete',
  'qrinsert',
  'qrupdate',
  'qz',
  'rad2deg',
  'rand',
  'randi',
  'randn',
  'randperm',
  'rank',
  'rat',
  'rats',
  'rcond',
  'real',
  'reallog',
  'realpow',
  'realsqrt',
  'rem',
  'reordernodes',
  'repelem',
  'repmat',
  'rescale',
  'reshape',
  'residue',
  'resize',
  'retime',
  'rmedge',
  'rmmissing',
  'rmnode',
  'rmoutliers',
  'rng',
  'roots',
  'rosser',
  'rot90',
  'round',
  'rows2vars',
  'rref',
  'rsf2csf',
  'scatteredInterpolant',
  'schur',
  'sec',
  'secd',
  'sech',
  'shiftdim',
  'shortestpath',
  'shortestpathtree',
  'sign',
  'simplify',
  'sin',
  'sind',
  'sinh',
  'sinpi',
  'size',
  'smoothdata',
  'smoothdata2',
  'sort',
  'sortrows',
  'spalloc',
  'sparse',
  'spaugment',
  'spconvert',
  'spdiags',
  'speye',
  'spfun',
  'sph2cart',
  'spline',
  'spones',
  'spparms',
  'sprand',
  'sprandn',
  'sprandsym',
  'sprank',
  'spy',
  'sqrt',
  'sqrtm',
  'squeeze',
  'ss2tf',
  'stack',
  'standardizeMissing',
  'sub2ind',
  'subgraph',
  'subspace',
  'successors',
  'sum',
  'svd',
  'svdappend',
  'svds',
  'svdsketch',
  'sylvester',
  'symamd',
  'symbfact',
  'symmlq',
  'symrcm',
  'table',
  'tail',
  'tan',
  'tand',
  'tanh',
  'tensorprod',
  'tfqmr',
  'toeplitz',
  'toposort',
  'trace',
  'transclosure',
  'transpose',
  'transreduction',
  'treelayout',
  'treeplot',
  'trenddecomp',
  'tril',
  'trimdata',
  'triu',
  'uminus',
  'unique',
  'uniquetol',
  'unmesh',
  'unmkpp',
  'unstack',
  'unwrap',
  'uplus',
  'vander',
  'vecnorm',
  'vertcat',
  'wilkinson',
  'zeros',
]);

const MATLAB_CONSTANTS = new Set([
  'Inf',
  'NaN',
  'eps',
  'false',
  'flintmax',
  'pi',
  'true',
]);

class HighlightMatlab extends Highlighter {

  static NORMAL = 0;
  static WORD = 1;
  static COMMENT = 2;
  static QUOTE = 3;
  static QUOTE_BACKSLASH = 4;
  static DQUOTE = 5;
  static DQUOTE_BACKSLASH = 6;

  constructor(delegate) {
    super(delegate);
    this.word = '';
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      let c = input[i];
      switch (this.state) {

      case HighlightMatlab.NORMAL:
        if (!isascii(c) || isalpha(c) || c == '_') {
          this.state = HighlightMatlab.WORD;
          this.word += c;
        } else if (c == '%') {
          this.state = HighlightMatlab.COMMENT;
          this.push("span", "comment");
          this.append('%');
        } else if (c == '\'') {
          this.state = HighlightMatlab.QUOTE;
          this.push("span", "string");
          this.append('\'');
        } else if (c == '"') {
          this.state = HighlightMatlab.DQUOTE;
          this.push("span", "string");
          this.append('"');
        } else {
          this.append(c);
        }
        break;

      case HighlightMatlab.WORD:
        if (!isascii(c) || isalnum(c) || c == '_') {
          this.word += c;
        } else {
          if (MATLAB_KEYWORDS.has(this.word)) {
            this.push("span", "keyword");
            this.append(this.word);
            this.pop();
          } else if (MATLAB_BUILTINS.has(this.word)) {
            this.push("span", "builtin");
            this.append(this.word);
            this.pop();
          } else if (MATLAB_CONSTANTS.has(this.word)) {
            this.push("span", "constant");
            this.append(this.word);
            this.pop();
          } else {
            this.append(this.word);
          }
          this.word = '';
          this.epsilon(HighlightMatlab.NORMAL);
        }
        break;

      case HighlightMatlab.COMMENT:
        this.append(c);
        if (c == '\n') {
          this.pop();
          this.state = HighlightMatlab.NORMAL;
        }
        break;

      case HighlightMatlab.QUOTE:
        this.append(c);
        if (c == '\'') {
          this.pop();
          this.state = HighlightMatlab.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightMatlab.QUOTE_BACKSLASH;
        }
        break;

      case HighlightMatlab.QUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightMatlab.QUOTE;
        break;

      case HighlightMatlab.DQUOTE:
        this.append(c);
        if (c == '"') {
          this.pop();
          this.state = HighlightMatlab.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightMatlab.DQUOTE_BACKSLASH;
        }
        break;

      case HighlightMatlab.DQUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightMatlab.DQUOTE;
        break;

      default:
        throw new Error('Invalid state');
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightMatlab.WORD:
      if (MATLAB_KEYWORDS.has(this.word)) {
        this.push("span", "keyword");
        this.append(this.word);
        this.pop();
      } else if (MATLAB_BUILTINS.has(this.word)) {
        this.push("span", "builtin");
        this.append(this.word);
        this.pop();
      } else if (MATLAB_CONSTANTS.has(this.word)) {
        this.push("span", "constant");
        this.append(this.word);
        this.pop();
      } else {
        this.append(this.word);
      }
      this.word = '';
      break;
    case HighlightMatlab.QUOTE:
    case HighlightMatlab.QUOTE_BACKSLASH:
    case HighlightMatlab.DQUOTE:
    case HighlightMatlab.DQUOTE_BACKSLASH:
    case HighlightMatlab.COMMENT:
      this.pop();
      break;
    default:
      break;
    }
    this.state = HighlightMatlab.NORMAL;
    this.delegate.flush();
    this.delta = 1;
  }
}

Highlighter.REGISTRY['matlab'] = HighlightMatlab;
