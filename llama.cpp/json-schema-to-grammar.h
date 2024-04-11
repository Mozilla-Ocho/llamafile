#pragma once
#include "json.h"

std::string json_schema_to_grammar(const nlohmann::ordered_json& schema);
