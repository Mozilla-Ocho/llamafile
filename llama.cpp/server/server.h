#pragma once
#include "llama.cpp/llama.h"

extern bool server_log_json;
extern bool g_server_background_mode;
extern llama_model *g_server_force_llama_model;
extern void (*g_server_on_listening)(const char *host, int port);

int server_cli(int, char **);
