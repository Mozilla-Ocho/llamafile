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

#include "client.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <string.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

#include "llama.cpp/llama.h"
#include "llamafile/llamafile.h"
#include "llamafile/threadlocal.h"
#include "llamafile/trust.h"
#include "llamafile/version.h"

#include "log.h"
#include "time.h"
#include "tokenbucket.h"
#include "worker.h"

#define STANDARD_RESPONSE_HEADERS \
    "Server: llamafile/" LLAMAFILE_VERSION_STRING "\r\n" \
    "Referrer-Policy: origin\r\n" \
    "Cache-Control: private; max-age=0\r\n"

using namespace ctl;

static void
on_http_cancel(Client* client)
{
    if (client->should_send_error_if_canceled) {
        fcntl(client->fd, F_SETFL, fcntl(client->fd, F_GETFL) | O_NONBLOCK);
        client->send_error(503);
    }
}

static ThreadLocal<Client> g_http_cancel(on_http_cancel);

Client::Client()
  : cleanups(nullptr), ibuf(FLAG_http_ibuf_size), obuf(FLAG_http_obuf_size)
{
    InitHttpMessage(&msg, 0);
    url.params.p = nullptr;
}

int
Client::close()
{
    int rc = 0;
    clear();
    DestroyHttpMessage(&msg);
    if (fd != -1) {
        if (FLAG_verbose >= 2)
            SLOG("close");
        rc = ::close(fd);
        fd = -1;
    }
    return rc;
}

void
Client::cleanup()
{
    Cleanup* clean;
    while ((clean = cleanups)) {
        cleanups = clean->next;
        clean->func(clean->arg);
        delete clean;
    }
}

void
Client::clear()
{
    cleanup();
    free(url_memory);
    url_memory = nullptr;
    free(params_memory);
    params_memory = nullptr;
    free(url.params.p);
    url.params.p = nullptr;
    close_connection = false;
    payload = "";
    unread = 0;
}

void
Client::defer_cleanup(void (*func)(void*), void* arg)
{
    Cleanup* clean = new Cleanup;
    clean->next = cleanups;
    clean->func = func;
    clean->arg = arg;
    cleanups = clean;
}

void
Client::run()
{
    ibuf.n = 0;
    for (;;) {

        // read headers
        clear();
        if (!read_request())
            break;

        // process message
        if (!transport())
            break;

        // synchronize message stream
        if (close_connection)
            break;
        if (!read_payload())
            break;

        // move pipelined bytes back to beginning
        if (ibuf.n == ibuf.i) {
            ibuf.n = 0;
        } else {
            memmove(ibuf.p, ibuf.p + ibuf.i, ibuf.n - ibuf.i);
            ibuf.n -= ibuf.i;
        }
    }
}

bool
Client::read_request()
{
    int inmsglen;
    ResetHttpMessage(&msg, kHttpRequest);
    for (;;) {
        inmsglen = ParseHttpMessage(&msg, ibuf.p, ibuf.n, ibuf.c);
        if (inmsglen > 0) {
            message_started = timespec_real();
            ibuf.i = inmsglen;
            return true;
        }
        if (inmsglen == -1) {
            SLOG("bad message %m");
            return false;
        }
        if (ibuf.n)
            SLOG("fragmented message with %zu bytes", ibuf.n);
        ssize_t got;
        got = read(fd, ibuf.p + ibuf.n, ibuf.c - ibuf.n);
        if (!got && ibuf.n)
            SLOG("unexpected eof after %zu bytes", ibuf.n);
        if (got == -1 && (ibuf.n || (errno != EAGAIN && errno != ECONNRESET)))
            SLOG("read failed %m");
        if (got <= 0)
            return false;
        ibuf.n += got;
    }
}

bool
Client::transport()
{
    if (msg.version > 11) {
        close_connection = true;
        return send_error(505);
    }

    if (msg.method == kHttpConnect) {
        close_connection = true;
        return send_error(501);
    }

    if (!has_at_most_this_element(kHttpExpect, "100-continue")) {
        close_connection = true;
        return send_error(417);
    }

    effective_ip = client_ip;
    effective_ip_trusted = client_ip_trusted;
    if (FLAG_ip_header) {
        if (is_loopback_ip(client_ip) || client_ip_trusted) {
            ctl::string_view ip_header = get_header(FLAG_ip_header);
            if (!ip_header.empty()) {
                long ip;
                if ((ip = parse_ip(ip_header)) == -1) {
                    effective_ip = ip;
                    effective_ip_trusted = is_trusted_ip(ip);
                } else {
                    SLOG("client's --ip-header wasn't a single ipv4 address");
                    effective_ip_trusted = false;
                }
            }
        } else {
            SLOG("received direct connect from untrusted ip");
            effective_ip_trusted = false;
        }
    }

    if (get_header("X-Priority") == "batch") {
        worker->deprioritize();
    } else if (!effective_ip_trusted) {
        if (tokenbucket_acquire(client_ip) > FLAG_token_burst) {
            SLOG("deprioritizing");
            worker->deprioritize();
        }
    }

    if (HasHeader(kHttpTransferEncoding))
        if (!HeaderEqualCase(kHttpTransferEncoding, "identity")) {
            close_connection = true;
            return send_error(501, "Transfer-Encoding Not Implemented");
        }

    if (HasHeader(kHttpContentLength)) {
        long cl;
        cl = ParseContentLength(HeaderData(kHttpContentLength),
                                HeaderLength(kHttpContentLength));
        if (cl == -1) {
            close_connection = true;
            return send_error(400, "Bad Content-Length");
        }
        if (cl > ibuf.c - ibuf.i) {
            close_connection = true;
            return send_error(413);
        }
        unread = cl;
    } else if (msg.method == kHttpPost || msg.method == kHttpPut) {
        close_connection = true;
        return send_error(411);
    }

    if (FLAG_verbose >= 1)
        SLOG("get %#.*s", msg.uri.b - msg.uri.a, ibuf.p + msg.uri.a);

    if (msg.version >= 11)
        if (HeaderEqualCase(kHttpExpect, "100-continue"))
            if (!send("HTTP/1.1 100 Continue\r\n\r\n"))
                return false;

    url_memory = ParseUrl(
      ibuf.p + msg.uri.a, msg.uri.b - msg.uri.a, &url, kUrlPlus | kUrlLatin1);
    if (!url_memory)
        __builtin_trap();

    return dispatch();
}

// sends http error response message and body in one shot
//
// after this function is called, the handler must return control.
//
// @param code must be a number between 400 and 999
// @param reason must be a sanitized http token, or null for default
bool
Client::send_error(int code, const char* reason)
{
    if (!reason)
        reason = GetHttpReason(code);
    SLOG("error %d %s", code, reason);
    char* p = append_http_response_message(obuf.p, code, reason);
    return send_response(obuf.p, p, string(reason) + "\r\n");
}

// appends start of http response message to `p`
//
// after this function is called, more header lines may be appended.
// afterwards, either send_response() or send_response_start() should be
// called to transmit the message over the wire.
//
// @param p is a page guarded buffer
// @param code must be a number between 200 and 999
// @param reason must be a sanitized http token, or null for default
// @return p + len(appended content)
char*
Client::append_http_response_message(char* p, int code, const char* reason)
{
    // generate http message starting line
    *p++ = 'H';
    *p++ = 'T';
    *p++ = 'T';
    *p++ = 'P';
    *p++ = '/';
    *p++ = '1';
    *p++ = '.';
    *p++ = '0' + (msg.version & 1);
    *p++ = ' ';
    *p++ = '0' + code / 100;
    *p++ = '0' + code / 10 % 10;
    *p++ = '0' + code % 10;
    *p++ = ' ';
    if (!reason)
        reason = GetHttpReason(code);
    p = stpcpy(p, reason);
    *p++ = '\r';
    *p++ = '\n';

    // append standard headers
    p = stpcpy(p, STANDARD_RESPONSE_HEADERS);

    // append date header
    tm tm;
    p = stpcpy(p, "Date: ");
    gmtime_lockless(message_started.tv_sec, &tm);
    p = FormatHttpDateTime(p, &tm);
    *p++ = '\r';
    *p++ = '\n';

    // inform client of close() intent
    if (msg.version < 11)
        close_connection = true;
    if (HeaderEqualCase(kHttpConnection, "close"))
        close_connection = true;
    if (close_connection)
        p = stpcpy(p, "Connection: close\r\n");

    return p;
}

// sends http response message and body in one shot
//
// after this function is called, the handler must return control.
//
// @param p0 points to start of http response message
// @param p points to end of http response message headers; we assume
//     that we can keep appending to `p` which must be page guarded
bool
Client::send_response(char* p0, char* p, string_view content)
{
    cleanup();
    pthread_testcancel();
    should_send_error_if_canceled = false;

    // append content length
    p = stpcpy(p, "Content-Length: ");
    p = FormatInt64(p, content.size());
    *p++ = '\r';
    *p++ = '\n';

    // finish message
    *p++ = '\r';
    *p++ = '\n';

    return send2(string_view(p0, p - p0), content);
}

// sends http response message, but not its body.
//
// after this function is called, send_response_chunk() may be called to
// stream individual pieces of the response body. when you're done, you
// must call send_response_finish() to complete the response.
//
// here's an example handler that uses this:
//
//     bool
//     Client::fun()
//     {
//         char* message = obuf.p;
//         char* p = append_http_response_message(message, 200);
//         if (!send_response_start(message, p))
//             return false;
//         sleep(1);
//         if (!send_response_chunk("hello\r\n"))
//             return false;
//         sleep(1);
//         if (!send_response_chunk("world\r\n"))
//             return false;
//         sleep(1);
//         return send_response_finish();
//     }
//
// @param p0 points to start of http response message
// @param p points to end of http response message headers; we assume
//     that we can keep appending to `p` which must be page guarded
bool
Client::send_response_start(char* p0, char* p)
{
    // use chunked transfer encoding if http/1.1
    if (msg.version >= 11)
        p = stpcpy(p, "Transfer-Encoding: chunked\r\n");

    // finish message
    *p++ = '\r';
    *p++ = '\n';

    return send(string_view(p0, p - p0));
}

// finishes sending chunked http response body.
//
// once you are finished sending chunks, call send_response_finish().
bool
Client::send_response_chunk(const string_view content)
{
    // don't encode chunk boundaries for simple http client
    // it will need to rely on read() doing the right thing
    if (msg.version < 11)
        return send(content);

    // sent in three pieces
    iovec iov[3];
    size_t bytes = 0;

    // 1. send "%zx\r\n" % (len(content))
    char start[32];
    char* p = start;
    p = FormatHex64(p, content.size(), 0);
    *p++ = '\r';
    *p++ = '\n';
    iov[0].iov_base = start;
    iov[0].iov_len = p - start;
    bytes += iov[0].iov_len;

    // 2. send content
    iov[1].iov_base = (void*)content.data();
    iov[1].iov_len = content.size();
    bytes += iov[1].iov_len;

    // 3. send newline
    iov[2].iov_base = (void*)"\r\n";
    iov[2].iov_len = 2;
    bytes += iov[2].iov_len;

    // perform send system call
    ssize_t sent;
    if ((sent = writev(fd, iov, 3)) != bytes) {
        if (sent == -1 && errno != EAGAIN && errno != ECONNRESET)
            SLOG("writev failed %m");
        return false;
    }
    return true;
}

// finishes sending chunked http response body.
//
// after this function is called, the handler must return control.
bool
Client::send_response_finish()
{
    cleanup();

    // don't encode chunk boundaries for simple http client
    // it will need to rely on read() doing the right thing
    if (msg.version < 11)
        return true;

    // send terminating chunk
    return send("0\r\n\r\n");
}

// writes raw data to socket
//
// consider using the higher level methods like send_error(),
// send_response(), send_response_start(), etc.
bool
Client::send(const string_view s)
{
    ssize_t sent;
    if ((sent = write(fd, s.data(), s.size())) != s.size()) {
        if (sent == -1 && errno != EAGAIN && errno != ECONNRESET)
            SLOG("write failed %m");
        return false;
    }
    return true;
}

// writes two pieces of raw data to socket in single system call
//
// consider using the higher level methods like send_error(),
// send_response(), send_response_start(), etc.
bool
Client::send2(const string_view s1, const string_view s2)
{
    iovec iov[2];
    ssize_t sent;
    iov[0].iov_base = (void*)s1.data();
    iov[0].iov_len = s1.size();
    iov[1].iov_base = (void*)s2.data();
    iov[1].iov_len = s2.size();
    if ((sent = writev(fd, iov, 2)) != s1.size() + s2.size()) {
        if (sent == -1 && errno != EAGAIN && errno != ECONNRESET)
            SLOG("writev failed %m");
        return false;
    }
    return true;
}

bool
Client::has_at_most_this_element(int h, const string_view s)
{
    if (!HasHeader(h))
        return true;
    if (!SlicesEqualCase(s.data(), s.size(), HeaderData(h), HeaderLength(h)))
        return false;
    struct HttpHeader* x;
    for (unsigned i = 0; i < msg.xheaders.n; ++i) {
        x = msg.xheaders.p + i;
        if (GetHttpHeader(ibuf.p + x->k.a, x->k.b - x->k.a) == h &&
            !SlicesEqualCase(
              ibuf.p + x->v.a, x->v.b - x->v.a, s.data(), s.size())) {
            return false;
        }
    }
    return true;
}

ctl::string_view
Client::get_header(const ctl::string_view& key)
{
    int h;
    size_t i, keylen;
    if ((h = GetHttpHeader(key.data(), key.size())) != -1) {
        if (msg.headers[h].a)
            return ctl::string_view(ibuf.p + msg.headers[h].a,
                                    msg.headers[h].b - msg.headers[h].a);
    } else {
        for (i = 0; i < msg.xheaders.n; ++i)
            if (SlicesEqualCase(key.data(),
                                key.size(),
                                ibuf.p + msg.xheaders.p[i].k.a,
                                msg.xheaders.p[i].k.b - msg.xheaders.p[i].k.a))
                return ctl::string_view(ibuf.p + msg.xheaders.p[i].v.a,
                                        msg.xheaders.p[i].v.b -
                                          msg.xheaders.p[i].v.a);
    }
    return ctl::string_view();
}

bool
Client::read_payload()
{
    while (ibuf.n - ibuf.i < unread) {
        ssize_t got;
        if ((got = read(fd, ibuf.p + ibuf.n, ibuf.c - ibuf.n)) <= 0) {
            if (!got)
                SLOG("unexpected eof");
            if (got == -1)
                SLOG("read failed %m");
            return false;
        }
        ibuf.n += got;
    }
    payload = string_view(ibuf.p + ibuf.i, unread);
    ibuf.i += unread;
    unread = 0;
    if (msg.method == kHttpPost && //
        HasHeader(kHttpContentType) &&
        IsMimeType(HeaderData(kHttpContentType),
                   HeaderLength(kHttpContentType),
                   "application/x-www-form-urlencoded")) {
        params_memory =
          ParseParams(payload.data(), payload.size(), &url.params);
    }
    return true;
}

bool
Client::dispatch()
{
    bool res;
    should_send_error_if_canceled = true;
    g_http_cancel.set(this);
    res = dispatcher();
    g_http_cancel.set(nullptr);
    return res;
}

bool
Client::dispatcher()
{
    ctl::string_view p = path();

    if (!g_url_prefix.empty()) {
        if (FLAG_verbose >= 2) {
             SLOG("request path %.*s", (int)p.size(), p.data());
        }

        size_t prefix_len = g_url_prefix.size();
        if (p.size() < prefix_len ||
            memcmp(p.data(), g_url_prefix.c_str(), prefix_len) != 0) {
            SLOG("path prefix mismatch");
            return send_error(404);
        }

        // Adjust path view to exclude prefix
        p = ctl::string_view(p.data() + prefix_len,
                           p.size() - prefix_len);
    }

    if (p == "/tokenize")
        return tokenize();
    if (p == "/embedding")
        return embedding();
    if (p == "/v1/embeddings")
        return embedding();
    if (p == "/completion")
        return completion();
    return send_error(404);
}

string_view
Client::path()
{
    if (!url.path.n)
        return "/";
    return { url.path.p, url.path.n };
}

optional<string_view>
Client::param(string_view key)
{
    for (size_t i = 0; i < url.params.n; ++i)
        if (key.size() == url.params.p[i].key.n)
            if (!memcmp(key.data(), url.params.p[i].key.p, key.size()))
                return optional(
                  string_view(url.params.p[i].val.p, url.params.p[i].val.n));
    return {};
}
