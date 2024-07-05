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
#include "utils.h"

#include <cosmo.h>
#include <stdckdint.h>

#include "double-conversion/double-to-string.h"
#include "double-conversion/string-to-double.h"

#define KEY 1
#define COMMA 2
#define COLON 4
#define ARRAY 8
#define OBJECT 16
#define DEPTH 20

#define ASCII 0
#define C0 1
#define DQUOTE 2
#define BACKSLASH 3
#define UTF8_2 4
#define UTF8_3 5
#define UTF8_4 6
#define C1 7
#define UTF8_3_E0 8
#define UTF8_3_ED 9
#define UTF8_4_F0 10
#define BADUTF8 11
#define EVILUTF8 12

static const char kJsonStr[256] = {
    1,  1,  1,  1,  1,  1,  1,  1, // 0000 ascii (0)
    1,  1,  1,  1,  1,  1,  1,  1, // 0010
    1,  1,  1,  1,  1,  1,  1,  1, // 0020 c0 (1)
    1,  1,  1,  1,  1,  1,  1,  1, // 0030
    0,  0,  2,  0,  0,  0,  0,  0, // 0040 dquote (2)
    0,  0,  0,  0,  0,  0,  0,  0, // 0050
    0,  0,  0,  0,  0,  0,  0,  0, // 0060
    0,  0,  0,  0,  0,  0,  0,  0, // 0070
    0,  0,  0,  0,  0,  0,  0,  0, // 0100
    0,  0,  0,  0,  0,  0,  0,  0, // 0110
    0,  0,  0,  0,  0,  0,  0,  0, // 0120
    0,  0,  0,  0,  3,  0,  0,  0, // 0130 backslash (3)
    0,  0,  0,  0,  0,  0,  0,  0, // 0140
    0,  0,  0,  0,  0,  0,  0,  0, // 0150
    0,  0,  0,  0,  0,  0,  0,  0, // 0160
    0,  0,  0,  0,  0,  0,  0,  0, // 0170
    7,  7,  7,  7,  7,  7,  7,  7, // 0200 c1 (8)
    7,  7,  7,  7,  7,  7,  7,  7, // 0210
    7,  7,  7,  7,  7,  7,  7,  7, // 0220
    7,  7,  7,  7,  7,  7,  7,  7, // 0230
    11, 11, 11, 11, 11, 11, 11, 11, // 0240 latin1 (4)
    11, 11, 11, 11, 11, 11, 11, 11, // 0250
    11, 11, 11, 11, 11, 11, 11, 11, // 0260
    11, 11, 11, 11, 11, 11, 11, 11, // 0270
    12, 12, 4,  4,  4,  4,  4,  4, // 0300 utf8-2 (5)
    4,  4,  4,  4,  4,  4,  4,  4, // 0310
    4,  4,  4,  4,  4,  4,  4,  4, // 0320 utf8-2
    4,  4,  4,  4,  4,  4,  4,  4, // 0330
    8,  5,  5,  5,  5,  5,  5,  5, // 0340 utf8-3 (6)
    5,  5,  5,  5,  5,  9,  5,  5, // 0350
    10, 6,  6,  6,  6,  11, 11, 11, // 0360 utf8-4 (7)
    11, 11, 11, 11, 11, 11, 11, 11, // 0370
};

static const char kEscapeLiteral[128] = {
    9, 9, 9, 9, 9, 9, 9, 9, 9, 1, 2, 9, 4, 3, 9, 9, // 0x00
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, // 0x10
    0, 0, 7, 0, 0, 0, 9, 9, 0, 0, 0, 0, 0, 0, 0, 6, // 0x20
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 9, 0, // 0x30
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 0x40
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, // 0x50
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 0x60
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, // 0x70
};

static const double_conversion::DoubleToStringConverter kDoubleToJson(
  double_conversion::DoubleToStringConverter::UNIQUE_ZERO |
    double_conversion::DoubleToStringConverter::EMIT_POSITIVE_EXPONENT_SIGN,
  "1e5000",
  "null",
  'e',
  -6,
  21,
  6,
  0);

static const double_conversion::StringToDoubleConverter kJsonToDouble(
  double_conversion::StringToDoubleConverter::ALLOW_CASE_INSENSITIVITY |
    double_conversion::StringToDoubleConverter::ALLOW_LEADING_SPACES |
    double_conversion::StringToDoubleConverter::ALLOW_TRAILING_JUNK |
    double_conversion::StringToDoubleConverter::ALLOW_TRAILING_SPACES,
  0.0,
  1.0,
  "Infinity",
  "NaN");

static double
StringToDouble(const char* s, size_t n, int* out_processed)
{
    if (n == -1ull)
        n = strlen(s);
    int processed;
    double res = kJsonToDouble.StringToDouble(s, n, &processed);
    if (out_processed)
        *out_processed = processed;
    return res;
}

void
Json::clear()
{
    switch (type_) {
        case String:
            string_value.~string();
            break;
        case Array:
            array_value.~vector();
            break;
        case Object:
            object_value.~map();
            break;
        default:
            break;
    }
    type_ = Null;
}

Json::Json(const Json& other) : type_(other.type_)
{
    switch (type_) {
        case Null:
            break;
        case Bool:
            bool_value = other.bool_value;
            break;
        case Long:
            long_value = other.long_value;
            break;
        case Ulong:
            ulong_value = other.ulong_value;
            break;
        case Float:
            float_value = other.float_value;
            break;
        case Double:
            double_value = other.double_value;
            break;
        case String:
            new (&string_value) ctl::string(other.string_value);
            break;
        case Array:
            new (&array_value) ctl::vector<Json>(other.array_value);
            break;
        case Object:
            new (&object_value) ctl::map<ctl::string, Json>(other.object_value);
            break;
        default:
            __builtin_trap();
    }
}

Json&
Json::operator=(const Json& other)
{
    if (this != &other) {
        clear();
        type_ = other.type_;
        switch (type_) {
            case Null:
                break;
            case Bool:
                bool_value = other.bool_value;
                break;
            case Long:
                long_value = other.long_value;
                break;
            case Ulong:
                ulong_value = other.ulong_value;
                break;
            case Float:
                float_value = other.float_value;
                break;
            case Double:
                double_value = other.double_value;
                break;
            case String:
                new (&string_value) ctl::string(other.string_value);
                break;
            case Array:
                new (&array_value) ctl::vector<Json>(other.array_value);
                break;
            case Object:
                new (&object_value)
                  ctl::map<ctl::string, Json>(other.object_value);
                break;
            default:
                __builtin_trap();
        }
    }
    return *this;
}

Json::Json(Json&& other) noexcept : type_(other.type_)
{
    switch (type_) {
        case Null:
            break;
        case Bool:
            bool_value = other.bool_value;
            break;
        case Long:
            long_value = other.long_value;
            break;
        case Ulong:
            ulong_value = other.ulong_value;
            break;
        case Float:
            float_value = other.float_value;
            break;
        case Double:
            double_value = other.double_value;
            break;
        case String:
            new (&string_value) ctl::string(ctl::move(other.string_value));
            break;
        case Array:
            new (&array_value) ctl::vector<Json>(ctl::move(other.array_value));
            break;
        case Object:
            new (&object_value)
              ctl::map<ctl::string, Json>(ctl::move(other.object_value));
            break;
        default:
            __builtin_trap();
    }
    other.type_ = Null;
}

Json&
Json::operator=(Json&& other) noexcept
{
    if (this != &other) {
        clear();
        type_ = other.type_;
        switch (type_) {
            case Null:
                break;
            case Bool:
                bool_value = other.bool_value;
                break;
            case Long:
                long_value = other.long_value;
                break;
            case Ulong:
                ulong_value = other.ulong_value;
                break;
            case Float:
                float_value = other.float_value;
                break;
            case Double:
                double_value = other.double_value;
                break;
            case String:
                new (&string_value) ctl::string(ctl::move(other.string_value));
                break;
            case Array:
                new (&array_value)
                  ctl::vector<Json>(ctl::move(other.array_value));
                break;
            case Object:
                new (&object_value)
                  ctl::map<ctl::string, Json>(ctl::move(other.object_value));
                break;
            default:
                __builtin_trap();
        }
        other.type_ = Null;
    }
    return *this;
}

double
Json::getNumber() const
{
    switch (type_) {
        case Null:
            return 0;
        case Bool:
            return bool_value;
        case Long:
            return long_value;
        case Ulong:
            return ulong_value;
        case Float:
            return float_value;
        case Double:
            return double_value;
        default:
            __builtin_trap();
    }
}

long
Json::getLong() const
{
    if (!isLong())
        __builtin_trap();
    return long_value;
}

unsigned long
Json::getUlong() const
{
    if (!isUlong())
        __builtin_trap();
    return ulong_value;
}

bool
Json::getBool() const
{
    if (!isBool())
        __builtin_trap();
    return bool_value;
}

float
Json::getFloat() const
{
    if (!isFloat())
        __builtin_trap();
    return float_value;
}

double
Json::getDouble() const
{
    if (!isDouble())
        __builtin_trap();
    return double_value;
}

ctl::string&
Json::getString()
{
    if (!isString())
        __builtin_trap();
    return string_value;
}

ctl::vector<Json>&
Json::getArray()
{
    if (!isArray())
        __builtin_trap();
    return array_value;
}

ctl::map<ctl::string, Json>&
Json::getObject()
{
    if (!isObject())
        __builtin_trap();
    return object_value;
}

void
Json::setNull()
{
    clear();
    type_ = Null;
}

void
Json::setBool(bool value)
{
    clear();
    type_ = Bool;
    bool_value = value;
}

void
Json::setFloat(float value)
{
    clear();
    type_ = Float;
    float_value = value;
}

void
Json::setLong(long value)
{
    clear();
    type_ = Long;
    long_value = value;
}

void
Json::setUlong(unsigned long value)
{
    clear();
    type_ = Ulong;
    ulong_value = value;
}

void
Json::setDouble(double value)
{
    clear();
    type_ = Double;
    double_value = value;
}

void
Json::setString(const char* value)
{
    clear();
    type_ = String;
    new (&string_value) ctl::string(value);
}

void
Json::setString(ctl::string&& value)
{
    clear();
    type_ = String;
    new (&string_value) ctl::string(ctl::move(value));
}

void
Json::setString(const ctl::string& value)
{
    clear();
    type_ = String;
    new (&string_value) ctl::string(value);
}

void
Json::setString(const ctl::string_view& value)
{
    clear();
    type_ = String;
    new (&string_value) ctl::string_view(value);
}

void
Json::setArray()
{
    clear();
    type_ = Array;
    new (&array_value) ctl::vector<Json>();
}

void
Json::setObject()
{
    clear();
    type_ = Object;
    new (&object_value) ctl::map<ctl::string, Json>();
}

Json&
Json::operator[](size_t index) noexcept
{
    if (type_ != Array) {
        clear();
        setArray();
    }
    if (index >= array_value.size()) {
        array_value.resize(index + 1);
    }
    return array_value[index];
}

Json&
Json::operator[](const ctl::string& key) noexcept
{
    if (type_ != Object) {
        clear();
        setObject();
    }
    return object_value[key];
}

ctl::string
Json::toString(bool pretty) const noexcept
{
    ctl::string b;
    marshal(b, pretty, 0);
    return b;
}

void
Json::marshal(ctl::string& b, bool pretty, int indent) const noexcept
{
    switch (type_) {
        case Null:
            b += "null";
            break;
        case String:
            stringify(b, string_value);
            break;
        case Bool:
            b += bool_value ? "true" : "false";
            break;
        case Long: {
            char buf[21];
            b.append(buf, FormatInt64(buf, long_value) - buf);
            break;
        }
        case Ulong: {
            char buf[21];
            b.append(buf, FormatUint64(buf, ulong_value) - buf);
            break;
        }
        case Float: {
            char buf[128];
            double_conversion::StringBuilder db(buf, 128);
            kDoubleToJson.ToShortestSingle(float_value, &db);
            db.Finalize();
            b += buf;
            break;
        }
        case Double: {
            char buf[128];
            double_conversion::StringBuilder db(buf, 128);
            kDoubleToJson.ToShortest(double_value, &db);
            db.Finalize();
            b += buf;
            break;
        }
        case Array: {
            bool once = false;
            b += '[';
            for (auto i = array_value.begin(); i != array_value.end(); ++i) {
                if (once) {
                    b += ',';
                    if (pretty)
                        b += ' ';
                } else {
                    once = true;
                }
                i->marshal(b, pretty, indent);
            }
            b += ']';
            break;
        }
        case Object: {
            bool once = false;
            b += '{';
            for (auto i = object_value.begin(); i != object_value.end(); ++i) {
                if (once) {
                    b += ',';
                } else {
                    once = true;
                }
                if (pretty && object_value.size() > 1) {
                    b += '\n';
                    ++indent;
                    for (int j = 0; j < indent; ++j)
                        b += "  ";
                }
                stringify(b, i->first);
                b += ':';
                if (pretty)
                    b += ' ';
                i->second.marshal(b, pretty, indent);
                if (pretty && object_value.size() > 1)
                    --indent;
            }
            if (pretty && object_value.size() > 1) {
                b += '\n';
                for (int j = 0; j < indent; ++j)
                    b += "  ";
                ++indent;
            }
            b += '}';
            break;
        }
        default:
            __builtin_trap();
    }
}

void
Json::stringify(ctl::string& b, const ctl::string_view& s) noexcept
{
    b += '"';
    serialize(b, s);
    b += '"';
}

void
Json::serialize(ctl::string& sb, const ctl::string_view& s) noexcept
{
    uint64_t w;
    size_t i, j, m;
    wint_t x, a, b;
    for (i = 0; i < s.size();) {
        x = s[i++] & 255;
        if (x >= 0300) {
            a = ThomPikeByte(x);
            m = ThomPikeLen(x) - 1;
            if (i + m <= s.size()) {
                for (j = 0;;) {
                    b = s[i + j] & 0xff;
                    if (!ThomPikeCont(b))
                        break;
                    a = ThomPikeMerge(a, b);
                    if (++j == m) {
                        x = a;
                        i += j;
                        break;
                    }
                }
            }
        }
        switch (0 <= x && x <= 127 ? kEscapeLiteral[x] : 9) {
            case 0:
                sb += x;
                break;
            case 1:
                sb += "\\t";
                break;
            case 2:
                sb += "\\n";
                break;
            case 3:
                sb += "\\r";
                break;
            case 4:
                sb += "\\f";
                break;
            case 5:
                sb += "\\\\";
                break;
            case 6:
                sb += "\\/";
                break;
            case 7:
                sb += "\\\"";
                break;
            case 9:
                w = EncodeUtf16(x);
                do {
                    char esc[6];
                    esc[0] = '\\';
                    esc[1] = 'u';
                    esc[2] = "0123456789abcdef"[(w & 0xF000) >> 014];
                    esc[3] = "0123456789abcdef"[(w & 0x0F00) >> 010];
                    esc[4] = "0123456789abcdef"[(w & 0x00F0) >> 004];
                    esc[5] = "0123456789abcdef"[(w & 0x000F) >> 000];
                    sb.append(esc, 6);
                } while ((w >>= 16));
                break;
            default:
                __builtin_trap();
        }
    }
}

Json::Status
Json::parse(Json& json, const char*& p, const char* e, int context, int depth)
{
    long x;
    char w[4];
    const char* a;
    int A, B, C, D, c, d, i, u;
    if (!depth)
        return depth_exceeded;
    for (a = p, d = +1; p < e;) {
        switch ((c = *p++ & 255)) {
            case ' ': // spaces
            case '\n':
            case '\r':
            case '\t':
                a = p;
                break;

            case ',': // present in list and object
                if (context & COMMA) {
                    context = 0;
                    a = p;
                    break;
                } else {
                    return unexpected_comma;
                }

            case ':': // present only in object after key
                if (context & COLON) {
                    context = 0;
                    a = p;
                    break;
                } else {
                    return unexpected_colon;
                }

            case 'n': // null
                if (context & (KEY | COLON | COMMA))
                    goto OnColonCommaKey;
                if (p + 3 <= e && READ32LE(p - 1) == READ32LE("null")) {
                    json.setNull();
                    p += 3;
                    return success;
                } else {
                    return illegal_character;
                }

            case 'f': // false
                if (context & (KEY | COLON | COMMA))
                    goto OnColonCommaKey;
                if (p + 4 <= e && READ32LE(p) == READ32LE("alse")) {
                    json.setBool(true);
                    p += 4;
                    return success;
                } else {
                    return illegal_character;
                }

            case 't': // true
                if (context & (KEY | COLON | COMMA))
                    goto OnColonCommaKey;
                if (p + 3 <= e && READ32LE(p - 1) == READ32LE("true")) {
                    json.setBool(true);
                    p += 3;
                    return success;
                } else {
                    return illegal_character;
                }

            default:
                return illegal_character;

            OnColonCommaKey:
                if (context & KEY)
                    return object_key_must_be_string;
            OnColonComma:
                if (context & COLON)
                    return missing_colon;
                return missing_comma;

            case '-': // negative
                if (context & (COLON | COMMA | KEY))
                    goto OnColonCommaKey;
                if (p < e && isdigit(*p)) {
                    d = -1;
                    break;
                } else {
                    return bad_negative;
                }

            case '0': // zero or number
                if (context & (COLON | COMMA | KEY))
                    goto OnColonCommaKey;
                if (p < e) {
                    if (*p == '.') {
                        if (p + 1 == e || !isdigit(p[1]))
                            return bad_double;
                        goto UseDubble;
                    } else if (*p == 'e' || *p == 'E') {
                        goto UseDubble;
                    } else if (isdigit(*p)) {
                        return unexpected_octal;
                    }
                }
                json.setLong(0);
                return success;

            case '1' ... '9': // integer
                if (context & (COLON | COMMA | KEY))
                    goto OnColonCommaKey;
                for (x = (c - '0') * d; p < e; ++p) {
                    c = *p & 255;
                    if (isdigit(c)) {
                        if (ckd_mul(&x, x, 10) ||
                            ckd_add(&x, x, (c - '0') * d)) {
                            goto UseDubble;
                        }
                    } else if (c == '.') {
                        if (p + 1 == e || !isdigit(p[1]))
                            return bad_double;
                        goto UseDubble;
                    } else if (c == 'e' || c == 'E') {
                        goto UseDubble;
                    } else {
                        break;
                    }
                }
                json.setLong(x);
                return success;

            UseDubble: // number
                json.setDouble(StringToDouble(a, e - a, &c));
                if (c <= 0)
                    return bad_double;
                if (a + c < e && (a[c] == 'e' || a[c] == 'E'))
                    return bad_exponent;
                p = a + c;
                return success;

            case '[': { // Array
                if (context & (COLON | COMMA | KEY))
                    goto OnColonCommaKey;
                json.setArray();
                Json value;
                for (context = ARRAY, i = 0;;) {
                    Status status = parse(value, p, e, context, depth - 1);
                    if (status == absent_value)
                        return success;
                    if (status != success)
                        return status;
                    json.getArray().emplace_back(ctl::move(value));
                    context = ARRAY | COMMA;
                }
            }

            case ']':
                if (context & ARRAY)
                    return absent_value;
                return unexpected_end_of_array;

            case '}':
                if (context & OBJECT)
                    return absent_value;
                return unexpected_end_of_object;

            case '{': { // Object
                if (context & (COLON | COMMA | KEY))
                    goto OnColonCommaKey;
                json.setObject();
                context = KEY | OBJECT;
                Json key, value;
                for (;;) {
                    Status status = parse(key, p, e, context, depth - 1);
                    if (status == absent_value)
                        return success;
                    if (status != success)
                        return status;
                    status = parse(value, p, e, COLON, depth - 1);
                    if (status == absent_value)
                        return object_missing_value;
                    if (status != success)
                        return status;
                    json[key.getString()] = ctl::move(value);
                    context = KEY | COMMA | OBJECT;
                }
            }

            case '"': { // string
                ctl::string b;
                if (context & (COLON | COMMA))
                    goto OnColonComma;
                for (;;) {
                    if (p >= e)
                        return unexpected_end_of_string;
                    switch (kJsonStr[(c = *p++ & 255)]) {

                        case ASCII:
                            b += c;
                            break;

                        case DQUOTE:
                            json.setString(ctl::move(b));
                            return success;

                        case BACKSLASH:
                            if (p >= e)
                                return unexpected_end_of_string;
                            switch ((c = *p++ & 255)) {
                                case '"':
                                case '/':
                                case '\\':
                                    b += c;
                                    break;
                                case 'b':
                                    b += '\b';
                                    break;
                                case 'f':
                                    b += '\f';
                                    break;
                                case 'n':
                                    b += '\n';
                                    break;
                                case 'r':
                                    b += '\r';
                                    break;
                                case 't':
                                    b += '\t';
                                    break;
                                case 'x':
                                    if (p + 2 <= e && //
                                        (A = kHexToInt[p[0] & 255]) !=
                                          -1 && // HEX
                                        (B = kHexToInt[p[1] & 255]) != -1) { //
                                        c = A << 4 | B;
                                        if (!(0x20 <= c && c <= 0x7E))
                                            return hex_escape_not_printable;
                                        p += 2;
                                        b += c;
                                        break;
                                    } else {
                                        return invalid_hex_escape;
                                    }
                                case 'u':
                                    if (p + 4 <= e && //
                                        (A = kHexToInt[p[0] & 255]) != -1 && //
                                        (B = kHexToInt[p[1] & 255]) !=
                                          -1 && // UCS-2
                                        (C = kHexToInt[p[2] & 255]) != -1 && //
                                        (D = kHexToInt[p[3] & 255]) != -1) { //
                                        c = A << 12 | B << 8 | C << 4 | D;
                                        if (!IsSurrogate(c)) {
                                            p += 4;
                                        } else if (IsHighSurrogate(c)) {
                                            if (p + 4 + 6 <= e && //
                                                p[4] == '\\' && //
                                                p[5] == 'u' && //
                                                (A = kHexToInt[p[6] & 255]) !=
                                                  -1 && // UTF-16
                                                (B = kHexToInt[p[7] & 255]) !=
                                                  -1 && //
                                                (C = kHexToInt[p[8] & 255]) !=
                                                  -1 && //
                                                (D = kHexToInt[p[9] & 255]) !=
                                                  -1) { //
                                                u =
                                                  A << 12 | B << 8 | C << 4 | D;
                                                if (IsLowSurrogate(u)) {
                                                    p += 4 + 6;
                                                    c = MergeUtf16(c, u);
                                                } else {
                                                    goto BadUnicode;
                                                }
                                            } else {
                                                goto BadUnicode;
                                            }
                                        } else {
                                            goto BadUnicode;
                                        }
                                        // UTF-8
                                    EncodeUtf8:
                                        if (c <= 0x7f) {
                                            w[0] = c;
                                            i = 1;
                                        } else if (c <= 0x7ff) {
                                            w[0] = 0300 | (c >> 6);
                                            w[1] = 0200 | (c & 077);
                                            i = 2;
                                        } else if (c <= 0xffff) {
                                            if (IsSurrogate(c)) {
                                            ReplacementCharacter:
                                                c = 0xfffd;
                                            }
                                            w[0] = 0340 | (c >> 12);
                                            w[1] = 0200 | ((c >> 6) & 077);
                                            w[2] = 0200 | (c & 077);
                                            i = 3;
                                        } else if (~(c >> 18) & 007) {
                                            w[0] = 0360 | (c >> 18);
                                            w[1] = 0200 | ((c >> 12) & 077);
                                            w[2] = 0200 | ((c >> 6) & 077);
                                            w[3] = 0200 | (c & 077);
                                            i = 4;
                                        } else {
                                            goto ReplacementCharacter;
                                        }
                                        b.append(w, i);
                                    } else {
                                        return invalid_unicode_escape;
                                    BadUnicode:
                                        // Echo invalid \uXXXX sequences
                                        // Rather than corrupting UTF-8!
                                        b += "\\u";
                                    }
                                    break;
                                default:
                                    return invalid_escape_character;
                            }
                            break;

                        case UTF8_2:
                            if (p < e && //
                                (p[0] & 0300) == 0200) { //
                                c = (c & 037) << 6 | //
                                    (p[0] & 077); //
                                p += 1;
                                goto EncodeUtf8;
                            } else {
                                return malformed_utf8;
                            }

                        case UTF8_3_E0:
                            if (p + 2 <= e && //
                                (p[0] & 0377) < 0240 && //
                                (p[0] & 0300) == 0200 && //
                                (p[1] & 0300) == 0200) {
                                return overlong_utf8_0x7ff;
                            }
                            // fallthrough

                        case UTF8_3:
                        ThreeUtf8:
                            if (p + 2 <= e && //
                                (p[0] & 0300) == 0200 && //
                                (p[1] & 0300) == 0200) { //
                                c = (c & 017) << 12 | //
                                    (p[0] & 077) << 6 | //
                                    (p[1] & 077); //
                                p += 2;
                                goto EncodeUtf8;
                            } else {
                                return malformed_utf8;
                            }

                        case UTF8_3_ED:
                            if (p + 2 <= e && //
                                (p[0] & 0377) >= 0240) { //
                                if (p + 5 <= e && //
                                    (p[0] & 0377) >= 0256 && //
                                    (p[1] & 0300) == 0200 && //
                                    (p[2] & 0377) == 0355 && //
                                    (p[3] & 0377) >= 0260 && //
                                    (p[4] & 0300) == 0200) { //
                                    A = (0355 & 017) << 12 | // CESU-8
                                        (p[0] & 077) << 6 | //
                                        (p[1] & 077); //
                                    B = (0355 & 017) << 12 | //
                                        (p[3] & 077) << 6 | //
                                        (p[4] & 077); //
                                    c = ((A - 0xDB80) << 10) + //
                                        ((B - 0xDC00) + 0x10000); //
                                    goto EncodeUtf8;
                                } else if ((p[0] & 0300) == 0200 && //
                                           (p[1] & 0300) == 0200) { //
                                    return utf16_surrogate_in_utf8;
                                } else {
                                    return malformed_utf8;
                                }
                            }
                            goto ThreeUtf8;

                        case UTF8_4_F0:
                            if (p + 3 <= e && (p[0] & 0377) < 0220 &&
                                (((uint32_t)(p[+2] & 0377) << 030 |
                                  (uint32_t)(p[+1] & 0377) << 020 |
                                  (uint32_t)(p[+0] & 0377) << 010 |
                                  (uint32_t)(p[-1] & 0377) << 000) &
                                 0xC0C0C000) == 0x80808000) {
                                return overlong_utf8_0xffff;
                            }
                            // fallthrough
                        case UTF8_4:
                            if (p + 3 <= e && //
                                ((A = ((uint32_t)(p[+2] & 0377) << 030 | //
                                       (uint32_t)(p[+1] & 0377) << 020 | //
                                       (uint32_t)(p[+0] & 0377) << 010 | //
                                       (uint32_t)(p[-1] & 0377) << 000)) & //
                                 0xC0C0C000) == 0x80808000) { //
                                A = (A & 7) << 18 | //
                                    (A & (077 << 010)) << (12 - 010) | //
                                    (A & (077 << 020)) >> -(6 - 020) | //
                                    (A & (077 << 030)) >> 030; //
                                if (A <= 0x10FFFF) {
                                    c = A;
                                    p += 3;
                                    goto EncodeUtf8;
                                } else {
                                    return utf8_exceeds_utf16_range;
                                }
                            } else {
                                return malformed_utf8;
                            }

                        case EVILUTF8:
                            if (p < e && (p[0] & 0300) == 0200)
                                return overlong_ascii;
                            // fallthrough
                        case BADUTF8:
                            return illegal_utf8_character;
                        case C0:
                            return non_del_c0_control_code_in_string;
                        case C1:
                            return c1_control_code_in_string;
                        default:
                            __builtin_unreachable();
                    }
                }
                __builtin_unreachable();
            }
        }
    }
    if (depth == DEPTH)
        return absent_value;
    return unexpected_eof;
}

ctl::pair<Json::Status, Json>
Json::parse(const ctl::string_view& s)
{
    Json::Status s2;
    ctl::pair<Json::Status, Json> res;
    const char* p = s.data();
    const char* e = s.data() + s.size();
    res.first = parse(res.second, p, e, 0, DEPTH);
    if (res.first == Json::success) {
        if (!res.second.isObject() && !res.second.isArray()) {
            res.first = json_payload_should_be_object_or_array;
        } else {
            s2 = parse(res.second, p, e, 0, DEPTH);
            if (s2 != absent_value)
                res.first = trailing_content;
        }
    }
    return res;
}

const char*
Json::StatusToString(Json::Status status)
{
    switch (status) {
        case success:
            return "success";
        case bad_double:
            return "bad_double";
        case absent_value:
            return "absent_value";
        case bad_negative:
            return "bad_negative";
        case bad_exponent:
            return "bad_exponent";
        case missing_comma:
            return "missing_comma";
        case missing_colon:
            return "missing_colon";
        case malformed_utf8:
            return "malformed_utf8";
        case depth_exceeded:
            return "depth_exceeded";
        case stack_overflow:
            return "stack_overflow";
        case unexpected_eof:
            return "unexpected_eof";
        case overlong_ascii:
            return "overlong_ascii";
        case unexpected_comma:
            return "unexpected_comma";
        case unexpected_colon:
            return "unexpected_colon";
        case unexpected_octal:
            return "unexpected_octal";
        case trailing_content:
            return "trailing_content";
        case illegal_character:
            return "illegal_character";
        case invalid_hex_escape:
            return "invalid_hex_escape";
        case overlong_utf8_0x7ff:
            return "overlong_utf8_0x7ff";
        case overlong_utf8_0xffff:
            return "overlong_utf8_0xffff";
        case object_missing_value:
            return "object_missing_value";
        case illegal_utf8_character:
            return "illegal_utf8_character";
        case invalid_unicode_escape:
            return "invalid_unicode_escape";
        case utf16_surrogate_in_utf8:
            return "utf16_surrogate_in_utf8";
        case unexpected_end_of_array:
            return "unexpected_end_of_array";
        case hex_escape_not_printable:
            return "hex_escape_not_printable";
        case invalid_escape_character:
            return "invalid_escape_character";
        case utf8_exceeds_utf16_range:
            return "utf8_exceeds_utf16_range";
        case unexpected_end_of_string:
            return "unexpected_end_of_string";
        case unexpected_end_of_object:
            return "unexpected_end_of_object";
        case object_key_must_be_string:
            return "object_key_must_be_string";
        case c1_control_code_in_string:
            return "c1_control_code_in_string";
        case non_del_c0_control_code_in_string:
            return "non_del_c0_control_code_in_string";
        case json_payload_should_be_object_or_array:
            return "json_payload_should_be_object_or_array";
        default:
            __builtin_trap();
    }
}
