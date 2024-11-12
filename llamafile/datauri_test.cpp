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

#include "datauri.h"
#include "image.h"
#include "string.h"
#include <string>
#include <vector>

namespace {

void rfc2397_example1() {
    DataUri uri;
    std::string_view s = ",A%20brief%20note";
    if (uri.parse(s) != s.size())
        exit(1);
    if (uri.mime != "text/plain")
        exit(2);
    if (lf::strcasecmp(uri.get_param("charset"), "US-ASCII"))
        exit(3);
    if (uri.decode() != "A brief note")
        exit(4);
}

void rfc2397_example2() {
    DataUri uri;
    std::string_view s = "image/gif;base64,R0lGODdhMAAwAPAAAAAAAP///ywAAAAAMAAw"
                         "AAAC8IyPqcvt3wCcDkiLc7C0qwyGHhSWpjQu5yqmCYsapyuvUUlvONmOZtfzgFz"
                         "ByTB10QgxOR0TqBQejhRNzOfkVJ+5YiUqrXF5Y5lKh/DeuNcP5yLWGsEbtLiOSp"
                         "a/TPg7JpJHxyendzWTBfX0cxOnKPjgBzi4diinWGdkF8kjdfnycQZXZeYGejmJl"
                         "ZeGl9i2icVqaNVailT6F5iJ90m6mvuTS4OK05M0vDk0Q4XUtwvKOzrcd3iq9uis"
                         "F81M1OIcR7lEewwcLp7tuNNkM3uNna3F2JQFo97Vriy/Xl4/f1cf5VWzXyym7PH"
                         "hhx4dbgYKAAA7\" ALT=\"Larry\">";
    size_t pos = uri.parse(s);
    if (pos == std::string_view::npos)
        exit(5);
    if (s.substr(pos) != "\" ALT=\"Larry\">")
        exit(6);
    if (uri.mime != "image/gif")
        exit(7);
    std::string image = uri.decode();
    if (lf::get_image_type(image) != lf::IMAGE_TYPE_GIF)
        exit(8);
    if (!lf::is_image(image))
        exit(9);
}

void rfc2397_example3() {
    DataUri uri;
    std::string_view s = "text/plain;charset=iso-8859-7,%be%fg%be";
    size_t pos = uri.parse(s);
    if (pos != s.size())
        exit(10);
    if (uri.mime != "text/plain")
        exit(11);
    if (uri.get_param("charset") != "iso-8859-7")
        exit(12);
    if (uri.decode() != "\xbe%fg\xbe")
        exit(13);
}

void mime_missing_slash() {
    DataUri uri;
    if (uri.parse("a,") != std::string_view::npos)
        exit(14);
}

void empty_string() {
    DataUri uri;
    if (uri.parse("") != std::string_view::npos)
        exit(15);
}

void empty_data() {
    DataUri uri;
    if (uri.parse(",") != 1)
        exit(16);
}

void bad_token() {
    DataUri uri;
    if (uri.parse("\1/b;c,") != std::string_view::npos)
        exit(17);
    if (uri.parse("a/\1;c,") != std::string_view::npos)
        exit(18);
    if (uri.parse("a/b;\1,") != std::string_view::npos)
        exit(19);
    if (uri.parse("a/b;c,") == std::string_view::npos)
        exit(20);
}

void empty_components() {
    DataUri uri;
    if (uri.parse("/b;c,") != std::string_view::npos)
        exit(21);
    if (uri.parse("a/;c,") != std::string_view::npos)
        exit(22);
    if (uri.parse("a/b;,") != std::string_view::npos)
        exit(23);
}

void bad_percent() {
    DataUri uri;
    uri.data = "%";
    if (uri.decode() != "%")
        exit(24);
    uri.data = "%%";
    if (uri.decode() != "%%")
        exit(25);
    uri.data = "%a%";
    if (uri.decode() != "%a%")
        exit(26);
    uri.data = "%a%a";
    if (uri.decode() != "%a%a")
        exit(27);
    uri.data = "%!%";
    if (uri.decode() != "%!%")
        exit(28);
    uri.data = "%!%!";
    if (uri.decode() != "%!%!")
        exit(29);
    uri.data = "%a!%a";
    if (uri.decode() != "%a!%a")
        exit(30);
    uri.data = "%a!%a!";
    if (uri.decode() != "%a!%a!")
        exit(31);
    uri.data = "%a%%a%";
    if (uri.decode() != "%a%%a%")
        exit(32);
}

} // namespace

int main(int argc, char *argv[]) {
    rfc2397_example1();
    rfc2397_example2();
    rfc2397_example3();
    mime_missing_slash();
    empty_string();
    empty_data();
    bad_token();
    empty_components();
    bad_percent();
}
