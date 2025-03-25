/**
 * Original source: https://jsr.io/@std/fmt/1.0.3/colors.ts
 *
 * Generated from jsr:@std/fmt@1.03 using esbuild --bundle.
 *
 */


// node_modules/@std/fmt/colors.js
var { Deno } = globalThis;
var noColor = typeof Deno?.noColor === "boolean" ? Deno.noColor : false;
var enabled = !noColor;
function setColorEnabled(value) {
  if (Deno?.noColor) {
    return;
  }
  enabled = value;
}
function getColorEnabled() {
  return enabled;
}
function code(open, close) {
  return {
    open: `\x1B[${open.join(";")}m`,
    close: `\x1B[${close}m`,
    regexp: new RegExp(`\\x1b\\[${close}m`, "g")
  };
}
function run(str, code2) {
  return enabled ? `${code2.open}${str.replace(code2.regexp, code2.open)}${code2.close}` : str;
}
function reset(str) {
  return run(str, code([
    0
  ], 0));
}
function bold(str) {
  return run(str, code([
    1
  ], 22));
}
function dim(str) {
  return run(str, code([
    2
  ], 22));
}
function italic(str) {
  return run(str, code([
    3
  ], 23));
}
function underline(str) {
  return run(str, code([
    4
  ], 24));
}
function inverse(str) {
  return run(str, code([
    7
  ], 27));
}
function hidden(str) {
  return run(str, code([
    8
  ], 28));
}
function strikethrough(str) {
  return run(str, code([
    9
  ], 29));
}
function black(str) {
  return run(str, code([
    30
  ], 39));
}
function red(str) {
  return run(str, code([
    31
  ], 39));
}
function green(str) {
  return run(str, code([
    32
  ], 39));
}
function yellow(str) {
  return run(str, code([
    33
  ], 39));
}
function blue(str) {
  return run(str, code([
    34
  ], 39));
}
function magenta(str) {
  return run(str, code([
    35
  ], 39));
}
function cyan(str) {
  return run(str, code([
    36
  ], 39));
}
function white(str) {
  return run(str, code([
    37
  ], 39));
}
function gray(str) {
  return brightBlack(str);
}
function brightBlack(str) {
  return run(str, code([
    90
  ], 39));
}
function brightRed(str) {
  return run(str, code([
    91
  ], 39));
}
function brightGreen(str) {
  return run(str, code([
    92
  ], 39));
}
function brightYellow(str) {
  return run(str, code([
    93
  ], 39));
}
function brightBlue(str) {
  return run(str, code([
    94
  ], 39));
}
function brightMagenta(str) {
  return run(str, code([
    95
  ], 39));
}
function brightCyan(str) {
  return run(str, code([
    96
  ], 39));
}
function brightWhite(str) {
  return run(str, code([
    97
  ], 39));
}
function bgBlack(str) {
  return run(str, code([
    40
  ], 49));
}
function bgRed(str) {
  return run(str, code([
    41
  ], 49));
}
function bgGreen(str) {
  return run(str, code([
    42
  ], 49));
}
function bgYellow(str) {
  return run(str, code([
    43
  ], 49));
}
function bgBlue(str) {
  return run(str, code([
    44
  ], 49));
}
function bgMagenta(str) {
  return run(str, code([
    45
  ], 49));
}
function bgCyan(str) {
  return run(str, code([
    46
  ], 49));
}
function bgWhite(str) {
  return run(str, code([
    47
  ], 49));
}
function bgBrightBlack(str) {
  return run(str, code([
    100
  ], 49));
}
function bgBrightRed(str) {
  return run(str, code([
    101
  ], 49));
}
function bgBrightGreen(str) {
  return run(str, code([
    102
  ], 49));
}
function bgBrightYellow(str) {
  return run(str, code([
    103
  ], 49));
}
function bgBrightBlue(str) {
  return run(str, code([
    104
  ], 49));
}
function bgBrightMagenta(str) {
  return run(str, code([
    105
  ], 49));
}
function bgBrightCyan(str) {
  return run(str, code([
    106
  ], 49));
}
function bgBrightWhite(str) {
  return run(str, code([
    107
  ], 49));
}
function clampAndTruncate(n, max = 255, min = 0) {
  return Math.trunc(Math.max(Math.min(n, max), min));
}
function rgb8(str, color) {
  return run(str, code([
    38,
    5,
    clampAndTruncate(color)
  ], 39));
}
function bgRgb8(str, color) {
  return run(str, code([
    48,
    5,
    clampAndTruncate(color)
  ], 49));
}
function rgb24(str, color) {
  if (typeof color === "number") {
    return run(str, code([
      38,
      2,
      color >> 16 & 255,
      color >> 8 & 255,
      color & 255
    ], 39));
  }
  return run(str, code([
    38,
    2,
    clampAndTruncate(color.r),
    clampAndTruncate(color.g),
    clampAndTruncate(color.b)
  ], 39));
}
function bgRgb24(str, color) {
  if (typeof color === "number") {
    return run(str, code([
      48,
      2,
      color >> 16 & 255,
      color >> 8 & 255,
      color & 255
    ], 49));
  }
  return run(str, code([
    48,
    2,
    clampAndTruncate(color.r),
    clampAndTruncate(color.g),
    clampAndTruncate(color.b)
  ], 49));
}
var ANSI_PATTERN = new RegExp([
  "[\\u001B\\u009B][[\\]()#;?]*(?:(?:(?:(?:;[-a-zA-Z\\d\\/#&.:=?%@~_]+)*|[a-zA-Z\\d]+(?:;[-a-zA-Z\\d\\/#&.:=?%@~_]*)*)?\\u0007)",
  "(?:(?:\\d{1,4}(?:;\\d{0,4})*)?[\\dA-PR-TXZcf-nq-uy=><~]))"
].join("|"), "g");
function stripAnsiCode(string) {
  return string.replace(ANSI_PATTERN, "");
}
export {
  bgBlack,
  bgBlue,
  bgBrightBlack,
  bgBrightBlue,
  bgBrightCyan,
  bgBrightGreen,
  bgBrightMagenta,
  bgBrightRed,
  bgBrightWhite,
  bgBrightYellow,
  bgCyan,
  bgGreen,
  bgMagenta,
  bgRed,
  bgRgb24,
  bgRgb8,
  bgWhite,
  bgYellow,
  black,
  blue,
  bold,
  brightBlack,
  brightBlue,
  brightCyan,
  brightGreen,
  brightMagenta,
  brightRed,
  brightWhite,
  brightYellow,
  cyan,
  dim,
  getColorEnabled,
  gray,
  green,
  hidden,
  inverse,
  italic,
  magenta,
  red,
  reset,
  rgb24,
  rgb8,
  setColorEnabled,
  strikethrough,
  stripAnsiCode,
  underline,
  white,
  yellow
};
