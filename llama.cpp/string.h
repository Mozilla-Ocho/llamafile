// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;tab-width:8;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi

#pragma once
#include <string>

std::string string_strip(const std::string & str);
std::string string_get_sortable_timestamp();
std::string replace_all(const std::string & s, const std::string & search, const std::string & replace);

void string_process_escapes(std::string & input);

bool fs_validate_filename(const std::string & filename);
bool fs_create_directory_with_parents(const std::string & path);

std::string fs_get_cache_directory();
std::string fs_get_cache_file(const std::string & filename);
