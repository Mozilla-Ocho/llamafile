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

const ZIG_KEYWORDS = new Set([
  'addrspace',
  'align',
  'allowzero',
  'and',
  'anyframe',
  'anytype',
  'asm',
  'async',
  'await',
  'break',
  'callconv',
  'catch',
  'comptime',
  'const',
  'continue',
  'defer',
  'else',
  'enum',
  'errdefer',
  'error',
  'export',
  'extern',
  'fn',
  'for',
  'if',
  'inline',
  'linksection',
  'noalias',
  'noinline',
  'nosuspend',
  'opaque',
  'or',
  'orelse',
  'packed',
  'pub',
  'resume',
  'return',
  'struct',
  'suspend',
  'switch',
  'test',
  'threadlocal',
  'try',
  'union',
  'unreachable',
  'usingnamespace',
  'var',
  'volatile',
  'while',
]);

const ZIG_BUILTINS = new Set([
  '@abs',
  '@addrSpaceCast',
  '@addWithOverflow',
  '@alignCast',
  '@alignOf',
  '@ArgType',
  '@as',
  '@atomicLoad',
  '@atomicRmw',
  '@atomicStore',
  '@bitCast',
  '@bitOffsetOf',
  '@bitreverse',
  '@bitReverse',
  '@bitSizeOf',
  '@boolToInt',
  '@branchHint',
  '@breakpoint',
  '@bswap',
  '@byteOffsetOf',
  '@bytesToSlice',
  '@byteSwap',
  '@call',
  '@cDefine',
  '@ceil',
  '@cImport',
  '@cInclude',
  '@clz',
  '@cmpxchgStrong',
  '@cmpxchgWeak',
  '@compileError',
  '@compileLog',
  '@constCast',
  '@cos',
  '@ctz',
  '@cUndef',
  '@cVaArg',
  '@cVaCopy',
  '@cVaEnd',
  '@cVaStart',
  '@divExact',
  '@divFloor',
  '@divTrunc',
  '@embedFile',
  '@enumFromInt',
  '@enumToInt',
  '@errorCast',
  '@errorFromInt',
  '@errorName',
  '@errorReturnTrace',
  '@errorToInt',
  '@errSetCast',
  '@exp',
  '@exp2',
  '@export',
  '@extern',
  '@fence',
  '@field',
  '@fieldParentPtr',
  '@FieldType',
  '@floatCast',
  '@floatFromInt',
  '@floatToInt',
  '@floor',
  '@frameAddress',
  '@handle',
  '@hasDecl',
  '@hasField',
  '@import',
  '@inComptime',
  '@inlineCall',
  '@intCast',
  '@intFromBool',
  '@intFromEnum',
  '@intFromError',
  '@intFromFloat',
  '@intFromPtr',
  '@intToEnum',
  '@intToError',
  '@intToFloat',
  '@intToPtr',
  '@IntType',
  '@log',
  '@log10',
  '@log2',
  '@max',
  '@memberCount',
  '@memberName',
  '@memberType',
  '@memcpy',
  '@memset',
  '@min',
  '@mod',
  '@mulAdd',
  '@mulWithOverflow',
  '@newStackCall',
  '@noInlineCall',
  '@offsetOf',
  '@OpaqueType',
  '@panic',
  '@popCount',
  '@prefetch',
  '@ptrCast',
  '@ptrFromInt',
  '@ptrToInt',
  '@reduce',
  '@rem',
  '@returnAddress',
  '@round',
  '@select',
  '@setAlignStack',
  '@setCold',
  '@setEvalBranchQuota',
  '@setFloatMode',
  '@setGlobalLinkage',
  '@setRuntimeSafety',
  '@shlExact',
  '@shlWithOverflow',
  '@shrExact',
  '@shuffle',
  '@sin',
  '@sizeOf',
  '@sliceToBytes',
  '@splat',
  '@sqrt',
  '@src',
  '@subWithOverflow',
  '@tagName',
  '@TagType',
  '@tan',
  '@This',
  '@trap',
  '@trunc',
  '@truncate',
  '@Type',
  '@typeId',
  '@typeInfo',
  '@typeName',
  '@typeOf',
  '@TypeOf',
  '@unionInit',
  '@Vector',
  '@volatileCast',
  '@wasmMemoryGrow',
  '@wasmMemorySize',
  '@workGroupId',
  '@workGroupSize',
  '@workItemId',
]);

const ZIG_CONSTANTS = new Set([
  'false',
  'null',
  'true',
  'undefined',
]);

const ZIG_TYPES = new Set([
  'anyerror',
  'anyframe',
  'anyopaque',
  'anytype',
  'bool',
  'c_char',
  'c_int',
  'c_long',
  'c_longdouble',
  'c_longlong',
  'c_short',
  'c_uint',
  'c_ulong',
  'c_ulonglong',
  'c_ushort',
  'comptime_float',
  'comptime_int',
  'error',
  'f128',
  'f16',
  'f32',
  'f64',
  'f80',
  'i128',
  'i16',
  'i2',
  'i29',
  'i3',
  'i32',
  'i4',
  'i5',
  'i6',
  'i64',
  'i7',
  'i8',
  'isize',
  'noreturn',
  'type',
  'u128',
  'u16',
  'u2',
  'u29',
  'u3',
  'u32',
  'u4',
  'u5',
  'u6',
  'u64',
  'u7',
  'u8',
  'usize',
  'void',
]);

class HighlightZig extends Highlighter {

  static NORMAL = 0;
  static WORD = 1;
  static QUOTE = 2;
  static QUOTE_BACKSLASH = 3;
  static DQUOTE = 4;
  static DQUOTE_BACKSLASH = 5;
  static SLASH = 6;
  static SLASH_SLASH = 7;
  static BACKSLASH = 8;
  static BACKSLASH_BACKSLASH = 9;

  constructor(delegate) {
    super(delegate);
    this.word = '';
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      let c = input[i];
      switch (this.state) {

      case HighlightZig.NORMAL:
        if (!isascii(c) || isalpha(c) || c == '_' || c == '@') {
          this.state = HighlightZig.WORD;
          this.word += c;
        } else if (c == '/') {
          this.state = HighlightZig.SLASH;
        } else if (c == '\\') {
          this.state = HighlightZig.BACKSLASH;
        } else if (c == '\'') {
          this.state = HighlightZig.QUOTE;
          this.push("span", "string");
          this.append('\'');
        } else if (c == '"') {
          this.state = HighlightZig.DQUOTE;
          this.push("span", "string");
          this.append('"');
        } else {
          this.append(c);
        }
        break;

      case HighlightZig.WORD:
        if (!isascii(c) || isalnum(c) || c == '_') {
          this.word += c;
        } else {
          if (ZIG_KEYWORDS.has(this.word)) {
            this.push("span", "keyword");
            this.append(this.word);
            this.pop();
          } else if (ZIG_TYPES.has(this.word)) {
            this.push("span", "type");
            this.append(this.word);
            this.pop();
          } else if (ZIG_BUILTINS.has(this.word)) {
            this.push("span", "builtin");
            this.append(this.word);
            this.pop();
          } else if (ZIG_CONSTANTS.has(this.word)) {
            this.push("span", "constant");
            this.append(this.word);
            this.pop();
          } else {
            this.append(this.word);
          }
          this.word = '';
          this.epsilon(HighlightZig.NORMAL);
        }
        break;

      case HighlightZig.BACKSLASH:
        if (c == '\\') {
          this.push("span", "string");
          this.append("\\\\");
          this.state = HighlightZig.BACKSLASH_BACKSLASH;
        } else {
          this.append('\\');
          this.epsilon(HighlightZig.NORMAL);
        }
        break;

      case HighlightZig.SLASH_SLASH:
      case HighlightZig.BACKSLASH_BACKSLASH:
        this.append(c);
        if (c == '\n') {
          this.pop();
          this.state = HighlightZig.NORMAL;
        }
        break;

      case HighlightZig.SLASH:
        if (c == '/') {
          this.push("span", "comment");
          this.append("//");
          this.state = HighlightZig.SLASH_SLASH;
        } else {
          this.append('/');
          this.epsilon(HighlightZig.NORMAL);
        }
        break;

      case HighlightZig.QUOTE:
        this.append(c);
        if (c == '\'') {
          this.pop();
          this.state = HighlightZig.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightZig.QUOTE_BACKSLASH;
        }
        break;

      case HighlightZig.QUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightZig.QUOTE;
        break;

      case HighlightZig.DQUOTE:
        this.append(c);
        if (c == '"') {
          this.pop();
          this.state = HighlightZig.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightZig.DQUOTE_BACKSLASH;
        }
        break;

      case HighlightZig.DQUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightZig.DQUOTE;
        break;

      default:
        throw new Error('Invalid state');
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightZig.WORD:
      if (ZIG_KEYWORDS.has(this.word)) {
        this.push("span", "keyword");
        this.append(this.word);
        this.pop();
      } else if (ZIG_TYPES.has(this.word)) {
        this.push("span", "type");
        this.append(this.word);
        this.pop();
      } else if (ZIG_BUILTINS.has(this.word)) {
        this.push("span", "builtin");
        this.append(this.word);
        this.pop();
      } else if (ZIG_CONSTANTS.has(this.word)) {
        this.push("span", "constant");
        this.append(this.word);
        this.pop();
      } else {
        this.append(this.word);
      }
      this.word = '';
      break;
    case HighlightZig.SLASH:
      this.append('/');
      break;
    case HighlightZig.BACKSLASH:
      this.append('\\');
      break;
    case HighlightZig.QUOTE:
    case HighlightZig.QUOTE_BACKSLASH:
    case HighlightZig.DQUOTE:
    case HighlightZig.DQUOTE_BACKSLASH:
    case HighlightZig.SLASH_SLASH:
    case HighlightZig.BACKSLASH_BACKSLASH:
      this.pop();
      break;
    default:
      break;
    }
    this.state = HighlightZig.NORMAL;
    this.delegate.flush();
    this.delta = 1;
  }
}

Highlighter.REGISTRY['zig'] = HighlightZig;
