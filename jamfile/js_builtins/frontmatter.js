/**
 * Original source: https://jsr.io/@std/front-matter
 *
 * Generated from jsr:@std/front-matter@1.05 with:
 * 
 * ```bash
 *  echo 'export * from "@std/front-matter"' \
 *    |  ./node_modules/.bin/esbuild --bundle --format=esm \
 *    '--external:@jsr/std__toml*' '--external:@jsr/std__yaml*' \
 *    > ../jamfile/js_builtins/frontmatter.js
 * ```
 *
 */

// node_modules/@std/front-matter/_formats.js
var BOM = "\\ufeff?";
var YAML_DELIMITER = "= yaml =|---";
var YAML_HEADER = `(---yaml|${YAML_DELIMITER})\\s*`;
var YAML_FOOTER = `(?:---|${YAML_DELIMITER})`;
var TOML_DELIMITER = "\\+\\+\\+|= toml =";
var TOML_HEADER = `(---toml|${TOML_DELIMITER})\\s*`;
var TOML_FOOTER = `(?:---|${TOML_DELIMITER})`;
var JSON_DELIMITER = `= json =`;
var JSON_HEADER = `(---json|${JSON_DELIMITER})\\s*`;
var JSON_FOOTER = `(?:---|${JSON_DELIMITER})`;
var DATA = "([\\s\\S]+?)";
var NEWLINE = "\\r?\\n?";
var RECOGNIZE_YAML_REGEXP = new RegExp(`^${YAML_HEADER}$`, "im");
var RECOGNIZE_TOML_REGEXP = new RegExp(`^${TOML_HEADER}$`, "im");
var RECOGNIZE_JSON_REGEXP = new RegExp(`^${JSON_HEADER}$`, "im");
var EXTRACT_YAML_REGEXP = new RegExp(`^(${BOM}${YAML_HEADER}$${DATA}^${YAML_FOOTER}\\s*$${NEWLINE})`, "im");
var EXTRACT_TOML_REGEXP = new RegExp(`^(${BOM}${TOML_HEADER}$${DATA}^${TOML_FOOTER}\\s*$${NEWLINE})`, "im");
var EXTRACT_JSON_REGEXP = new RegExp(`^(${BOM}${JSON_HEADER}$${DATA}^${JSON_FOOTER}\\s*$${NEWLINE})`, "im");
var EXTRACT_REGEXP_MAP = /* @__PURE__ */ new Map([
  [
    "yaml",
    EXTRACT_YAML_REGEXP
  ],
  [
    "toml",
    EXTRACT_TOML_REGEXP
  ],
  [
    "json",
    EXTRACT_JSON_REGEXP
  ]
]);

// node_modules/@std/front-matter/_shared.js
function extractAndParse(input, extractRegExp, parse3) {
  const match = extractRegExp.exec(input);
  if (!match || match.index !== 0) {
    throw new TypeError("Unexpected end of input");
  }
  const frontMatter = match.at(-1)?.replace(/^\s+|\s+$/g, "") ?? "";
  const attrs = parse3(frontMatter);
  const body = input.replace(match[0], "");
  return {
    frontMatter,
    body,
    attrs
  };
}

// node_modules/@std/front-matter/json.js
function extract(text) {
  return extractAndParse(text, EXTRACT_JSON_REGEXP, JSON.parse);
}

// node_modules/@std/front-matter/toml.js
import { parse } from "jamfile:toml";
function extract2(text) {
  return extractAndParse(text, EXTRACT_TOML_REGEXP, parse);
}

// node_modules/@std/front-matter/yaml.js
import { parse as parse2 } from "jamfile:yaml";
function extract3(text) {
  return extractAndParse(text, EXTRACT_YAML_REGEXP, (s) => parse2(s));
}

// node_modules/@std/front-matter/test.js
function test(str, formats) {
  if (!formats) formats = [
    ...EXTRACT_REGEXP_MAP.keys()
  ];
  for (const format of formats) {
    const regexp = EXTRACT_REGEXP_MAP.get(format);
    if (!regexp) {
      throw new TypeError(`Unable to test for ${format} front matter format`);
    }
    const match = regexp.exec(str);
    if (match?.index === 0) {
      return true;
    }
  }
  return false;
}
export {
  extract as extractJson,
  extract2 as extractToml,
  extract3 as extractYaml,
  test
};
