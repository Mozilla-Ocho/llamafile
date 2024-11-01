// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
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

#include "json.h"

#include <cstdio>
#include <cstdlib>
#include <time.h>

#define ARRAYLEN(A) \
    ((sizeof(A) / sizeof(*(A))) / ((unsigned)!(sizeof(A) % sizeof(*(A)))))

#define STRING(sl) std::string(sl, sizeof(sl) - 1)

static const char kHuge[] = R"([
    "JSON Test Pattern pass1",
    {"object with 1 member":["array with 1 element"]},
    {},
    [],
    -42,
    true,
    false,
    null,
    {
        "integer": 1234567890,
        "real": -9876.543210,
        "e": 0.123456789e-12,
        "E": 1.234567890E+34,
        "":  23456789012E66,
        "zero": 0,
        "one": 1,
        "space": " ",
        "quote": "\"",
        "backslash": "\\",
        "controls": "\b\f\n\r\t",
        "slash": "/ & \/",
        "alpha": "abcdefghijklmnopqrstuvwyz",
        "ALPHA": "ABCDEFGHIJKLMNOPQRSTUVWYZ",
        "digit": "0123456789",
        "0123456789": "digit",
        "special": "`1~!@#$%^&*()_+-={':[,]}|;.</>?",
        "hex": "\u0123\u4567\u89AB\uCDEF\uabcd\uef4A",
        "true": true,
        "false": false,
        "null": null,
        "array":[  ],
        "object":{  },
        "address": "50 St. James Street",
        "url": "http://www.JSON.org/",
        "comment": "// /* <!-- --",
        "# -- --> */": " ",
        " s p a c e d " :[1,2 , 3

,

4 , 5        ,          6           ,7        ],"compact":[1,2,3,4,5,6,7],
        "jsontext": "{\"object with 1 member\":[\"array with 1 element\"]}",
        "quotes": "&#34; \u0022 %22 0x22 034 &#x22;",
        "\/\\\"\uCAFE\uBABE\uAB98\uFCDE\ubcda\uef4A\b\f\n\r\t`1~!@#$%^&*()_+-=[]{}|;:',./<>?"
: "A key can be any string"
    },
    0.5 ,98.6
,
99.44
,

1066,
1e1,
0.1e1,
1e-1,
1e00,2e+00,2e-00
,"rosebud"])";

#define BENCH(ITERATIONS, WORK_PER_RUN, CODE) \
    do { \
        struct timespec start = now(); \
        for (int __i = 0; __i < ITERATIONS; ++__i) { \
            asm volatile("" ::: "memory"); \
            CODE; \
        } \
        long long work = (WORK_PER_RUN) * (ITERATIONS); \
        double nanos = (tonanos(tub(now(), start)) + work - 1) / (double)work; \
        printf("%10g ns %2dx %s\n", nanos, (ITERATIONS), #CODE); \
    } while (0)

struct timespec
now(void)
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ts;
}

struct timespec
tub(struct timespec a, struct timespec b)
{
    a.tv_sec -= b.tv_sec;
    if (a.tv_nsec < b.tv_nsec) {
        a.tv_nsec += 1000000000;
        a.tv_sec--;
    }
    a.tv_nsec -= b.tv_nsec;
    return a;
}

int64_t
tonanos(struct timespec x)
{
    return x.tv_sec * 1000000000ull + x.tv_nsec;
}

void
object_test()
{
    Json obj;
    obj["content"] = "hello";
    if (obj.toString() != "{\"content\":\"hello\"}")
        exit(1);
}

void
deep_test()
{
    Json A1;
    A1[0] = 0;
    A1[1] = 10;
    A1[2] = 20;
    A1[3] = 3.14;
    A1[4] = 40;
    Json A2;
    A2[0] = std::move(A1);
    Json A3;
    A3[0] = std::move(A2);
    Json obj;
    obj["content"] = std::move(A3);
    if (obj.toString() != "{\"content\":[[[0,10,20,3.14,40]]]}")
        exit(2);
}

void
parse_test()
{
    std::pair<Json::Status, Json> res =
      Json::parse("{ \"content\":[[[0,10,20,3.14,40]]]}");
    if (res.first != Json::success)
        exit(3);
    if (res.second.toString() != "{\"content\":[[[0,10,20,3.14,40]]]}")
        exit(4);
    if (res.second.toStringPretty() !=
        R"({"content": [[[0, 10, 20, 3.14, 40]]]})")
        exit(5);
    res = Json::parse("{ \"a\": 1, \"b\": [2,   3]}");
    if (res.second.toString() != R"({"a":1,"b":[2,3]})")
        exit(6);
    if (res.second.toStringPretty() !=
        R"({
  "a": 1,
  "b": [2, 3]
})")
        exit(7);
}

static const struct
{
    std::string before;
    std::string after;
} kRoundTrip[] = {

    // valid utf16 sequences
    { " [\"\\u0020\"] ", "[\" \"]" },
    { " [\"\\u00A0\"] ", "[\"\\u00a0\"]" },

    // when we encounter invalid utf16 sequences
    // we turn them into ascii
    { "[\"\\uDFAA\"]", "[\"\\\\uDFAA\"]" },
    { " [\"\\uDd1e\\uD834\"] ", "[\"\\\\uDd1e\\\\uD834\"]" },
    { " [\"\\ud800abc\"] ", "[\"\\\\ud800abc\"]" },
    { " [\"\\ud800\"] ", "[\"\\\\ud800\"]" },
    { " [\"\\uD800\\uD800\\n\"] ", "[\"\\\\uD800\\\\uD800\\n\"]" },
    { " [\"\\uDd1ea\"] ", "[\"\\\\uDd1ea\"]" },
    { " [\"\\uD800\\n\"] ", "[\"\\\\uD800\\n\"]" },

    // underflow and overflow
    { " [123.456e-789] ", "[0]" },
    { " [0."
      "4e0066999999999999999999999999999999999999999999999999999999999999999999"
      "9999999999999999999999999999999999999999999999999969999999006] ",
      "[1e5000]" },
    { " [1.5e+9999] ", "[1e5000]" },
    { " [-1.5e+9999] ", "[-1e5000]" },
    { " [-123123123123123123123123123123] ", "[-1.2312312312312312e+29]" },
};

// https://github.com/nst/JSONTestSuite/
static const struct
{
    Json::Status error;
    std::string json;
} kJsonTestSuite[] = {
    { Json::absent_value, "" },
    { Json::trailing_content, "[] []" },
    { Json::illegal_character, "[nan]" },
    { Json::bad_negative, "[-nan]" },
    { Json::illegal_character, "[+NaN]" },
    { Json::trailing_content,
      "{\"Extra value after close\": true} \"misplaced quoted value\"" },
    { Json::illegal_character, "{\"Illegal expression\": 1 + 2}" },
    { Json::illegal_character, "{\"Illegal invocation\": alert()}" },
    { Json::unexpected_octal, "{\"Numbers cannot have leading zeroes\": 013}" },
    { Json::illegal_character, "{\"Numbers cannot be hex\": 0x14}" },
    { Json::hex_escape_not_printable, "[\"Illegal backslash escape: \\x15\"]" },
    { Json::illegal_character, "[\\naked]" },
    { Json::invalid_escape_character, "[\"Illegal backslash escape: \\017\"]" },
    { Json::depth_exceeded,
      "[[[[[[[[[[[[[[[[[[[[\"Too deep\"]]]]]]]]]]]]]]]]]]]]" },
    { Json::missing_colon, "{\"Missing colon\" null}" },
    { Json::unexpected_colon, "{\"Double colon\":: null}" },
    { Json::unexpected_comma, "{\"Comma instead of colon\", null}" },
    { Json::unexpected_colon, "[\"Colon instead of comma\": false]" },
    { Json::illegal_character, "[\"Bad value\", truth]" },
    { Json::illegal_character, "[\'single quote\']" },
    { Json::non_del_c0_control_code_in_string,
      "[\"\ttab\tcharacter\tin\tstring\t\"]" },
    { Json::invalid_escape_character,
      "[\"tab\\   character\\   in\\  string\\  \"]" },
    { Json::non_del_c0_control_code_in_string, "[\"line\nbreak\"]" },
    { Json::invalid_escape_character, "[\"line\\\nbreak\"]" },
    { Json::bad_exponent, "[0e]" },
    { Json::unexpected_eof, "[\"Unclosed array\"" },
    { Json::bad_exponent, "[0e+]" },
    { Json::bad_exponent, "[0e+-1]" },
    { Json::unexpected_eof, "{\"Comma instead if closing brace\": true," },
    { Json::unexpected_end_of_object, "[\"mismatch\"}" },
    { Json::illegal_character, "{unquoted_key: \"keys must be quoted\"}" },
    { Json::unexpected_end_of_array, "[\"extra comma\",]" },
    { Json::unexpected_comma, "[\"double extra comma\",,]" },
    { Json::unexpected_comma, "[   , \"<-- missing value\"]" },
    { Json::trailing_content, "[\"Comma after the close\"]," },
    { Json::trailing_content, "[\"Extra close\"]]" },
    { Json::unexpected_end_of_object, "{\"Extra comma\": true,}" },
    { Json::unexpected_eof, " {\"a\" " },
    { Json::unexpected_eof, " {\"a\": " },
    { Json::unexpected_colon, " {:\"b\" " },
    { Json::illegal_character, " {\"a\" b} " },
    { Json::illegal_character, " {key: 'value'} " },
    { Json::object_key_must_be_string, " {\"a\":\"a\" 123} " },
    { Json::illegal_character, " \x7b\xf0\x9f\x87\xa8\xf0\x9f\x87\xad\x7d " },
    { Json::object_key_must_be_string, " {[: \"x\"} " },
    { Json::illegal_character, " [1.8011670033376514H-308] " },
    { Json::illegal_character, " [1.2a-3] " },
    { Json::illegal_character, " [.123] " },
    { Json::bad_exponent, " [1e\xe5] " },
    { Json::bad_exponent, " [1ea] " },
    { Json::illegal_character, " [-1x] " },
    { Json::bad_negative, " [-.123] " },
    { Json::bad_negative, " [-foo] " },
    { Json::bad_negative, " [-Infinity] " },
    { Json::illegal_character, " \x5b\x30\xe5\x5d " },
    { Json::illegal_character, " \x5b\x31\x65\x31\xe5\x5d " },
    { Json::illegal_character, " \x5b\x31\x32\x33\xe5\x5d " },
    { Json::missing_comma,
      " \x5b\x2d\x31\x32\x33\x2e\x31\x32\x33\x66\x6f\x6f\x5d " },
    { Json::bad_exponent, " [0e+-1] " },
    { Json::illegal_character, " [Infinity] " },
    { Json::illegal_character, " [0x42] " },
    { Json::illegal_character, " [0x1] " },
    { Json::illegal_character, " [1+2] " },
    { Json::illegal_character, " \x5b\xef\xbc\x91\x5d " },
    { Json::illegal_character, " [NaN] " },
    { Json::illegal_character, " [Inf] " },
    { Json::bad_double, " [9.e+] " },
    { Json::bad_exponent, " [1eE2] " },
    { Json::bad_exponent, " [1e0e] " },
    { Json::bad_exponent, " [1.0e-] " },
    { Json::bad_exponent, " [1.0e+] " },
    { Json::bad_exponent, " [0e] " },
    { Json::bad_exponent, " [0e+] " },
    { Json::bad_exponent, " [0E] " },
    { Json::bad_exponent, " [0E+] " },
    { Json::bad_exponent, " [0.3e] " },
    { Json::bad_exponent, " [0.3e+] " },
    { Json::illegal_character, " [0.1.2] " },
    { Json::illegal_character, " [.2e-3] " },
    { Json::illegal_character, " [.-1] " },
    { Json::bad_negative, " [-NaN] " },
    { Json::illegal_character, " [+Inf] " },
    { Json::illegal_character, " [+1] " },
    { Json::illegal_character, " [++1234] " },
    { Json::illegal_character, " [tru] " },
    { Json::illegal_character, " [nul] " },
    { Json::illegal_character, " [fals] " },
    { Json::unexpected_eof, " [{} " },
    { Json::unexpected_eof, "\n[1,\n1\n,1  " },
    { Json::unexpected_eof, " [1, " },
    { Json::unexpected_eof, " [\"\" " },
    { Json::illegal_character, " [* " },
    { Json::non_del_c0_control_code_in_string,
      " \x5b\x22\x0b\x61\x22\x5c\x66\x5d " },
    { Json::unexpected_eof, "[\"a\",\n4\n,1,1  " },
    { Json::unexpected_colon, " [1:2] " },
    { Json::illegal_character, " \x5b\xff\x5d " },
    { Json::illegal_character, " \x5b\x78 " },
    { Json::unexpected_eof, " [\"x\" " },
    { Json::unexpected_colon, " [\"\": 1] " },
    { Json::illegal_character, " [a\xe5] " },
    { Json::unexpected_comma, " {\"x\", null} " },
    { Json::illegal_character, " [\"x\", truth] " },
    { Json::illegal_character, STRING("\x00") },
    { Json::trailing_content, "\n[\"x\"]]" },
    { Json::unexpected_octal, " [012] " },
    { Json::unexpected_octal, " [-012] " },
    { Json::missing_comma, " [1 000.0] " },
    { Json::unexpected_octal, " [-01] " },
    { Json::bad_negative, " [- 1] " },
    { Json::bad_negative, " [-] " },
    { Json::illegal_utf8_character, " {\"\xb9\":\"0\",} " },
    { Json::unexpected_colon, " {\"x\"::\"b\"} " },
    { Json::unexpected_comma, " [1,,] " },
    { Json::unexpected_end_of_array, " [1,] " },
    { Json::unexpected_comma, " [1,,2] " },
    { Json::unexpected_comma, " [,1] " },
    { Json::missing_comma, " [ 3[ 4]] " },
    { Json::missing_comma, " [1 true] " },
    { Json::missing_comma, " [\"a\" \"b\"] " },
    { Json::bad_negative, " [--2.] " },
    { Json::bad_double, " [1.] " },
    { Json::bad_double, " [2.e3] " },
    { Json::bad_double, " [2.e-3] " },
    { Json::bad_double, " [2.e+3] " },
    { Json::bad_double, " [0.e1] " },
    { Json::bad_double, " [-2.] " },
    { Json::illegal_character, " \xef\xbb\xbf{} " },
    { Json::illegal_character, STRING(" [\x00\"\x00\xe9\x00\"\x00]\x00 ") },
    { Json::illegal_character, STRING(" \x00[\x00\"\x00\xe9\x00\"\x00] ") },
    { Json::malformed_utf8, " [\"\xe0\xff\"] " },
    { Json::illegal_utf8_character, " [\"\xfc\x80\x80\x80\x80\x80\"] " },
    { Json::illegal_utf8_character, " [\"\xfc\x83\xbf\xbf\xbf\xbf\"] " },
    { Json::overlong_ascii, " [\"\xc0\xaf\"] " },
    { Json::utf8_exceeds_utf16_range, " [\"\xf4\xbf\xbf\xbf\"] " },
    { Json::c1_control_code_in_string, " [\"\x81\"] " },
    { Json::malformed_utf8, " [\"\xe9\"] " },
    { Json::illegal_utf8_character, " [\"\xff\"] " },
    { Json::success, kHuge },
    { Json::success,
      R"([[[[[[[[[[[[[[[[[[["Not too deep"]]]]]]]]]]]]]]]]]]])" },
    { Json::success, R"({
    "JSON Test Pattern pass3": {
        "The outermost value": "must be an object or array.",
        "In this test": "It is an object."
    }
}
)" },
};

void
round_trip_test()
{
    for (size_t i = 0; i < ARRAYLEN(kRoundTrip); ++i) {
        std::pair<Json::Status, Json> res = Json::parse(kRoundTrip[i].before);
        if (res.first != Json::success) {
            printf(
              "error: Json::parse returned Json::%s but wanted Json::%s: %s\n",
              Json::StatusToString(res.first),
              Json::StatusToString(Json::success),
              kRoundTrip[i].before.c_str());
            exit(10);
        }
        if (res.second.toString() != kRoundTrip[i].after) {
            printf("error: Json::parse(%s).toString() was %s but should have "
                   "been %s\n",
                   kRoundTrip[i].before.c_str(),
                   res.second.toString().c_str(),
                   kRoundTrip[i].after.c_str());
            exit(11);
        }
    }
}

void
json_test_suite()
{
    for (size_t i = 0; i < ARRAYLEN(kJsonTestSuite); ++i) {
        std::pair<Json::Status, Json> res = Json::parse(kJsonTestSuite[i].json);
        if (res.first != kJsonTestSuite[i].error) {
            printf(
              "error: Json::parse returned Json::%s but wanted Json::%s: %s\n",
              Json::StatusToString(res.first),
              Json::StatusToString(kJsonTestSuite[i].error),
              kJsonTestSuite[i].json.c_str());
            exit(12);
        }
    }
}

int
main()
{
    object_test();
    deep_test();
    parse_test();
    round_trip_test();

    BENCH(2000, 1, object_test());
    BENCH(2000, 1, deep_test());
    BENCH(2000, 1, parse_test());
    BENCH(2000, 1, round_trip_test());
    BENCH(2000, 1, json_test_suite());
}
