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

#include <assert.h>
#include <cosmo.h>

#include "llama.cpp/llama.h"
#include "llamafile/llamafile.h"
#include "llamafile/pool.h"
#include "llamafile/version.h"

#include "log.h"
#include "model.h"
#include "server.h"
#include "signals.h"
#include "time.h"
#include "tokenbucket.h"
#include "utils.h"

Server* g_server;
llama_model* g_model;
ctl::string g_url_prefix;

int
main(int argc, char* argv[])
{
    mallopt(M_GRANULARITY, 2 * 1024 * 1024);
    mallopt(M_MMAP_THRESHOLD, 16 * 1024 * 1024);
    mallopt(M_TRIM_THRESHOLD, 128 * 1024 * 1024);
    FLAG_gpu = LLAMAFILE_GPU_DISABLE;
    llamafile_check_cpu();
    ShowCrashReports();

    if (llamafile_has(argv, "--version")) {
        puts("llamafiler v" LLAMAFILE_VERSION_STRING);
        exit(0);
    }

    if (llamafile_has(argv, "-h") || llamafile_has(argv, "-help") ||
        llamafile_has(argv, "--help")) {
        llamafile_help("/zip/llamafile/server/main.1.asc");
        __builtin_unreachable();
    }

    // get config
    LoadZipArgs(&argc, &argv);
    llamafile_get_flags(argc, argv);

    // normalize URL prefix
    g_url_prefix = normalize_url_prefix(FLAG_url_prefix);

    // initialize subsystems
    time_init();
    tokenbucket_init();

    // we must disable the llama.cpp logger
    // otherwise pthread_cancel() will cause deadlocks
    FLAG_log_disable = true;

    // load model
    llama_model_params mparams = {
        .n_gpu_layers = FLAG_n_gpu_layers,
        .split_mode = (enum llama_split_mode)FLAG_split_mode,
        .main_gpu = FLAG_main_gpu,
        .tensor_split = nullptr,
        .rpc_servers = nullptr,
        .progress_callback = nullptr,
        .progress_callback_user_data = nullptr,
        .kv_overrides = nullptr,
        .vocab_only = false,
        .use_mmap = true,
        .use_mlock = false,
        .check_tensors = false,
    };
    g_model = llama_load_model_from_file(FLAG_model, mparams);

    // create server
    if (FLAG_workers <= 0)
        FLAG_workers = __get_cpu_count() + 4;
    if (FLAG_workers <= 0)
        FLAG_workers = 16;
    set_thread_name("server");
    g_server = new Server(create_listening_socket(FLAG_listen));
    for (int i = 0; i < FLAG_workers; ++i)
        npassert(!g_server->spawn());

    // install security
    if (!FLAG_unsecure) {
        if (pledge(0, 0)) {
            SLOG("warning: this OS doesn't support pledge() security\n");
        } else if (pledge("stdio anet", 0)) {
            perror("pledge");
            exit(1);
        }
    }

    // run server
    signals_init();
    llama_backend_init();
    g_server->run();
    llama_backend_free();
    signals_destroy();

    // shutdown server
    SLOG("shutdown");
    g_server->shutdown();
    g_server->close();
    delete g_server;
    llama_free_model(g_model);
    tokenbucket_destroy();
    time_destroy();
    SLOG("exit");

    // quality assurance
    llamafile_task_shutdown();
    while (!pthread_orphan_np())
        pthread_decimate_np();
    CheckForMemoryLeaks();
}