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

#pragma once
#include <map>
#include <string>
#include <vector>

#if __cplusplus >= 201703L
#define JTJSON_STRING_VIEW std::string_view
#else
#define JTJSON_STRING_VIEW std::string
#endif

namespace jt {

class Json
{
  public:
    enum Type
    {
        Null,
        Bool,
        Long,
        Float,
        Double,
        String,
        Array,
        Object
    };

    enum Status
    {
        success,
        bad_double,
        absent_value,
        bad_negative,
        bad_exponent,
        missing_comma,
        missing_colon,
        malformed_utf8,
        depth_exceeded,
        stack_overflow,
        unexpected_eof,
        overlong_ascii,
        unexpected_comma,
        unexpected_colon,
        unexpected_octal,
        trailing_content,
        illegal_character,
        invalid_hex_escape,
        overlong_utf8_0x7ff,
        overlong_utf8_0xffff,
        object_missing_value,
        illegal_utf8_character,
        invalid_unicode_escape,
        utf16_surrogate_in_utf8,
        unexpected_end_of_array,
        hex_escape_not_printable,
        invalid_escape_character,
        utf8_exceeds_utf16_range,
        unexpected_end_of_string,
        unexpected_end_of_object,
        object_key_must_be_string,
        c1_control_code_in_string,
        non_del_c0_control_code_in_string,
    };

  private:
    Type type_;
    union
    {
        bool bool_value;
        float float_value;
        double double_value;
        long long long_value;
        std::string string_value;
        std::vector<Json> array_value;
        std::map<std::string, Json> object_value;
    };

  public:
    static const char* StatusToString(Status);
    static std::pair<Status, Json> parse(const JTJSON_STRING_VIEW&);

    Json(const Json&);
    Json(Json&&) noexcept;
    Json(unsigned long long);
    Json(const char*);
    Json(const JTJSON_STRING_VIEW&);
    ~Json();

    Json(const std::nullptr_t = nullptr) : type_(Null)
    {
    }

    Json(bool value) : type_(Bool), bool_value(value)
    {
    }

    Json(int value) : type_(Long), long_value(value)
    {
    }

    Json(float value) : type_(Float), float_value(value)
    {
    }

    Json(unsigned value) : type_(Long), long_value(value)
    {
    }

    Json(long long value) : type_(Long), long_value(value)
    {
    }

    Json(double value) : type_(Double), double_value(value)
    {
    }

    Json(std::string&& value) : type_(String), string_value(std::move(value))
    {
    }

    Type getType() const
    {
        return type_;
    }

    bool isNull() const
    {
        return type_ == Null;
    }

    bool isBool() const
    {
        return type_ == Bool;
    }

    bool isNumber() const
    {
        return isFloat() || isDouble() || isLong();
    }

    bool isLong() const
    {
        return type_ == Long;
    }

    bool isFloat() const
    {
        return type_ == Float;
    }

    bool isDouble() const
    {
        return type_ == Double;
    }

    bool isString() const
    {
        return type_ == String;
    }

    bool isArray() const
    {
        return type_ == Array;
    }

    bool isObject() const
    {
        return type_ == Object;
    }

    bool getBool() const;
    float getFloat() const;
    double getDouble() const;
    double getNumber() const;
    long long getLong() const;
    std::string& getString();
    std::vector<Json>& getArray();
    std::map<std::string, Json>& getObject();

    void setNull();
    void setBool(bool);
    void setFloat(float);
    void setDouble(double);
    void setLong(long long);
    void setString(const char*);
    void setString(std::string&&);
    void setString(const JTJSON_STRING_VIEW&);
    void setArray();
    void setObject();

    std::string toString() const;
    std::string toStringPretty() const;

    Json& operator=(const Json&);
    Json& operator=(Json&&) noexcept;

    Json& operator[](size_t);
    Json& operator[](const std::string&);

    operator std::string() const
    {
        return toString();
    }

  private:
    void clear();
    void marshal(std::string&, bool, int) const;
    static void stringify(std::string&, const JTJSON_STRING_VIEW&);
    static void serialize(std::string&, const JTJSON_STRING_VIEW&);
    static Status parse(Json&, const char*&, const char*, int, int);
};

} // namespace jt
