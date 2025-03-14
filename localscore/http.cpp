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

#include "llamafile/macros.h"
#include "llamafile/net.h"
#include "llamafile/string.h"
#include "third_party/mbedtls/ctr_drbg.h"
#include "third_party/mbedtls/debug.h"
#include "third_party/mbedtls/error.h"
#include "third_party/mbedtls/iana.h"
#include "third_party/mbedtls/net_sockets.h"
#include "third_party/mbedtls/ssl.h"
#include "third_party/mbedtls/x509.h"
#include <cosmo.h>
#include <errno.h>
#include <net/http/http.h>
#include <net/http/url.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>
#include <format>

#include "http.h"

/**
 * @fileoverview Downloads HTTP URL to stdout.
 */

static const char *prog;

static wontreturn void PrintUsage(int fd, int rc) {
    tinyprint(fd, "usage: ", prog, " [-iksvV] URL\n", NULL);
    exit(rc);
}

static const char *DescribeErrno(void) {
    return strerror(errno);
}

static int GetSslEntropy(void *c, unsigned char *p, size_t n) {
    if (getrandom(p, n, 0) != n) {
        perror("getrandom");
        exit(1);
    }
    return 0;
}

static void OnSslDebug(void *ctx, int level, const char *file, int line, const char *message) {
    char sline[12];
    char slevel[12];
    FormatInt32(sline, line);
    FormatInt32(slevel, level);
    tinyprint(2, file, ":", sline, ": (", slevel, ") ", message, "\n", NULL);
}

static int TlsSend(void *c, const unsigned char *p, size_t n) {
    int rc;
    if ((rc = write(*(int *)c, p, n)) == -1) {
        perror("TlsSend");
        exit(1);
    }
    return rc;
}

static int TlsRecv(void *c, unsigned char *p, size_t n, uint32_t o) {
    int r;
    struct iovec v[2];
    static unsigned a, b;
    static unsigned char t[4096];
    if (a < b) {
        r = MIN(n, b - a);
        memcpy(p, t + a, r);
        if ((a += r) == b) {
            a = b = 0;
        }
        return r;
    }
    v[0].iov_base = p;
    v[0].iov_len = n;
    v[1].iov_base = t;
    v[1].iov_len = sizeof(t);
    if ((r = readv(*(int *)c, v, 2)) == -1) {
        perror("TlsRecv");
        exit(1);
    }
    if (r > n) {
        b = r - n;
    }
    return MIN(n, r);
}

struct ParsedUrl {
    std::string host;
    std::string port;
    std::string path;
};

struct SSLContext {
    mbedtls_ssl_context ssl;
    mbedtls_ssl_config conf;
    mbedtls_ctr_drbg_context drbg;

    SSLContext() {
        mbedtls_ssl_init(&ssl);
        mbedtls_ssl_config_init(&conf);
        mbedtls_ctr_drbg_init(&drbg);
    }

    ~SSLContext() {
        mbedtls_ssl_free(&ssl);
        mbedtls_ssl_config_free(&conf);
        mbedtls_ctr_drbg_free(&drbg);
    }
};

static ParsedUrl ExtractUrlComponents(const std::string& url_str, bool* usessl) {
    ParsedUrl result;
    struct Url url;

    /*
     * Parse URL.
     */
    gc(ParseUrl(url_str.c_str(), -1, &url, kUrlPlus));
    gc(url.params.p);

    // Check scheme
    if (url.scheme.n) {
        if (url.scheme.n == 5 && !memcasecmp(url.scheme.p, "https", 5)) {
            *usessl = true;
        } else if (!(url.scheme.n == 4 && !memcasecmp(url.scheme.p, "http", 4))) {
            printf("not an http/https url: %s\n", url_str.c_str());
            exit(1);
        }
    }

    // Set host and port
    if (url.host.n) {
        // Copy the host data into the string
        result.host = std::string(url.host.p, url.host.n);
        if (url.port.n) {
            result.port = std::string(url.port.p, url.port.n);
        } else {
            result.port = *usessl ? "443" : "80";
        }
    } else {
        result.host = "127.0.0.1";
        result.port = *usessl ? "443" : "80";
    }

    // Validate host
    if (!IsAcceptableHost(result.host.c_str(), -1)) {
        printf("invalid host: %s\n", url_str.c_str());
        exit(1);
    }

    // Handle path
    if (!url.path.n || url.path.p[0] != '/') {
        // If path is empty or doesn't start with '/', prepend '/'
        result.path = "/" + std::string(url.path.p, url.path.n);
    } else {
        result.path = std::string(url.path.p, url.path.n);
    }

    return result;
}

static std::string BuildHTTPRequest(const ParsedUrl url, const Headers& headers, const std::string& body = "") {
    std::string request;
    request += std::format("POST {} HTTP/1.1\r\n"
                      "Host: {}\r\n"
                      "Connection: close\r\n",
                      url.path, url.host);

    // write all the headers. iterate through the map and write them to request
    for (auto const& [key, val] : headers) {
        request += key;
        request += ": ";
        request += val;
        request += "\r\n";
    }

    if (!body.empty()) {
        // Add Content-Length header
        char length_str[21];
        FormatUint64(length_str, body.size());
        request += "Content-Length: ";
        request += length_str;
        request += "\r\n";

        // End headers and add payload
        request += "\r\n";
        request += body;
    }

    return request;
}

static int ConnectToServer(const ParsedUrl url) {
    int sock = -1;
    struct addrinfo *addr = nullptr;
    struct addrinfo hints = {};
    
    // Initialize hints structure
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    hints.ai_flags = AI_NUMERICSERV;

    // Perform DNS lookup
    int gai_result = getaddrinfo(url.host.c_str(), url.port.c_str(), &hints, &addr);
    if (gai_result != 0) {
        tinyprint(2, prog, ": could not resolve host: ", url.host.c_str(), "\n", NULL);
        return -1;
    }

    // cleanup using a unique_ptr with custom deleter
    std::unique_ptr<struct addrinfo, decltype(&freeaddrinfo)> addr_guard(addr, freeaddrinfo);

    // Connect to server
    struct timeval tt = {-60};
    sock = lf::socket(addr->ai_family, addr->ai_socktype, addr->ai_protocol, false, &tt);
    if (sock == -1) {
        perror("socket");
        return -1;
    }

    // Try to connect
    if (connect(sock, addr->ai_addr, addr->ai_addrlen)) {
        tinyprint(2, prog, ": failed to connect to ", url.host.c_str(), 
                 " port ", url.port.c_str(), ": ", DescribeErrno(), "\n", NULL);
        close(sock);  // Clean up the socket if connect fails
        return -1;
    }

    return sock;
}

static std::unique_ptr<SSLContext> SetupSSL(int& sock, const std::string& hostname) {
    auto ctx = std::make_unique<SSLContext>();
    
    // Setup SSL configuration
    unassert(!mbedtls_ctr_drbg_seed(&ctx->drbg, GetSslEntropy, 0, "justine", 7));
    unassert(!mbedtls_ssl_config_defaults(&ctx->conf, 
                                        MBEDTLS_SSL_IS_CLIENT,
                                        MBEDTLS_SSL_TRANSPORT_STREAM, 
                                        MBEDTLS_SSL_PRESET_SUITEC));

    mbedtls_ssl_conf_authmode(&ctx->conf, MBEDTLS_SSL_VERIFY_REQUIRED);
    mbedtls_ssl_conf_ca_chain(&ctx->conf, lf::sslroots(), 0);
    mbedtls_ssl_conf_rng(&ctx->conf, mbedtls_ctr_drbg_random, &ctx->drbg);
    #ifndef NDEBUG
    mbedtls_ssl_conf_dbg(&ctx->conf, OnSslDebug, 0);
    #endif

    // Setup SSL context
    unassert(!mbedtls_ssl_setup(&ctx->ssl, &ctx->conf));
    unassert(!mbedtls_ssl_set_hostname(&ctx->ssl, hostname.c_str()));
    mbedtls_ssl_set_bio(&ctx->ssl, &sock, TlsSend, 0, TlsRecv);

    // Perform handshake
    int ret = mbedtls_ssl_handshake(&ctx->ssl);
    if (ret != 0) {
        throw std::runtime_error("SSL handshake failed: " + 
                               std::string(DescribeSslClientHandshakeError(&ctx->ssl, ret)));
    }

    return ctx;
}

static void SendRequest(const std::string& request, int sock, SSLContext* ssl_ctx) {
    size_t sent = 0;
    while (sent < request.size()) {
        ssize_t rc;
        if (ssl_ctx) {
            rc = mbedtls_ssl_write(&ssl_ctx->ssl, 
                                 reinterpret_cast<const unsigned char*>(request.data() + sent), 
                                 request.size() - sent);
        } else {
            rc = write(sock, request.data() + sent, request.size() - sent);
        }

        if (rc <= 0) {
            throw std::runtime_error("Failed to send request");
        }
        sent += rc;
    }
}

bool hasHeader(const HttpMessage& msg, int header) {
    return msg.headers[header].a != 0;
}

const char* getHeaderData(const HttpMessage& msg, int header, const char* base) {
    return base + msg.headers[header].a;
}

size_t getHeaderLength(const HttpMessage& msg, int header) {
    return msg.headers[header].b - msg.headers[header].a;
}

bool headerEqualCase(const HttpMessage& msg, int header, const char* str, const char* base) {
    return SlicesEqualCase(str, strlen(str), 
                          getHeaderData(msg, header, base), 
                          getHeaderLength(msg, header));
}

static Response DecodeHttpResponse(int sock, SSLContext* ssl_ctx, size_t initial_buffer_size = 1000) {
    std::vector<char> buffer;
    buffer.reserve(initial_buffer_size);
    
    size_t current_pos = 0;
    Response response;
    struct HttpMessage msg;
    struct HttpUnchunker chunker;
    
    InitHttpMessage(&msg, kHttpResponse);
    
    enum class ParseState {
        Headers = 0,
        Body,
        BodyLengthed,
        BodyChunked
    } state = ParseState::Headers;
    
    auto read_chunk = [sock, ssl_ctx](char* buf, size_t len) -> ssize_t {
        if (ssl_ctx) {
            return mbedtls_ssl_read(&ssl_ctx->ssl, 
                                  reinterpret_cast<unsigned char*>(buf), 
                                  len);
        } else {
            return read(sock, buf, len);
        }
    };

    size_t header_length = 0;
    size_t payload_length = 0;
    bool headers_parsed = false;
    
    while (true) {
        if (current_pos == buffer.size()) {
            buffer.resize(buffer.size() + initial_buffer_size);
        }
        
        ssize_t bytes_read = read_chunk(&buffer[current_pos], 
                                      buffer.size() - current_pos);
        
        if (bytes_read < 0) {
            if (ssl_ctx && bytes_read == MBEDTLS_ERR_SSL_PEER_CLOSE_NOTIFY) {
                break;
            }
            throw std::runtime_error("Failed to read response");
        }
        
        if (bytes_read == 0 && state != ParseState::Headers) {
            break;
        }
        
        current_pos += bytes_read;
        
        if (!headers_parsed) {
            int rc = ParseHttpMessage(&msg, buffer.data(), current_pos, buffer.size());
            if (rc == -1) {
                throw std::runtime_error("Invalid HTTP message");
            }
            if (rc > 0) {
                header_length = rc;
                response.status = msg.status;
                response.raw_headers = std::string(buffer.data(), header_length);
                headers_parsed = true;
                
                if (hasHeader(msg, kHttpTransferEncoding) && 
                    !headerEqualCase(msg, kHttpTransferEncoding, "identity", buffer.data())) {
                    if (!headerEqualCase(msg, kHttpTransferEncoding, "chunked", buffer.data())) {
                        throw std::runtime_error("Unsupported transfer encoding");
                    }
                    state = ParseState::BodyChunked;
                    memset(&chunker, 0, sizeof(chunker));
                } else if (hasHeader(msg, kHttpContentLength)) {
                    int content_length = ParseContentLength(
                        getHeaderData(msg, kHttpContentLength, buffer.data()),
                        getHeaderLength(msg, kHttpContentLength));
                    if (content_length == -1) {
                        throw std::runtime_error("Invalid content length");
                    }
                    response.content_length = content_length;
                    state = ParseState::BodyLengthed;
                } else {
                    state = ParseState::Body;
                }
                
                // Process any body data that came with the headers
                if (current_pos > header_length) {
                    goto process_body;
                }
                continue;
            }
        } else {
process_body:
            switch (state) {
                case ParseState::Body: {
                    response.body.append(buffer.data() + header_length, 
                                       current_pos - header_length);
                    break;
                }
                
                case ParseState::BodyLengthed: {
                    size_t remaining = response.content_length - response.body.size();
                    size_t to_copy = std::min(remaining, 
                                            current_pos - header_length - response.body.size());
                    response.body.append(buffer.data() + header_length + response.body.size(), 
                                       to_copy);
                    if (response.body.size() >= response.content_length) {
                        return response;
                    }
                    break;
                }
                
                case ParseState::BodyChunked: {
                    size_t chunk_length;
                    int rc = Unchunk(&chunker, 
                                   buffer.data() + header_length,
                                   current_pos - header_length, 
                                   &chunk_length);
                    if (rc == -1) {
                        throw std::runtime_error("Invalid chunk encoding");
                    }
                    if (rc > 0) {
                        response.body.append(buffer.data() + header_length, chunk_length);
                        return response;
                    }
                    break;
                }
                
                default:
                    break;
            }
        }
    }
    
    return response;
}

Response SendHttpRequest(const std::string& url_str, uint64_t method, 
                        const Headers& headers, const std::string& body = "") {
    const char *agent = "hurl/1.o (https://github.com/jart/cosmopolitan)";
    bool usessl = false;

    ParsedUrl url = ExtractUrlComponents(url_str, &usessl);
    
    std::string request = (method == kHttpGet) 
        ? BuildHTTPRequest(url, headers)
        : BuildHTTPRequest(url, headers, body);

    int sock = ConnectToServer(url);

    std::unique_ptr<SSLContext> ssl_ctx;
    if (usessl) {
        try {
            ssl_ctx = SetupSSL(sock, url.host);
        } catch (const std::exception& e) {
            printf("Error setting up SSL: %s\n", e.what());
            close(sock);
            return Response();
        }
    }

    SendRequest(request, sock, ssl_ctx.get());
    Response resp = DecodeHttpResponse(sock, ssl_ctx.get());
    close(sock);
    return resp;
}

Response GET(const std::string& url_str, const Headers& headers) {
    return SendHttpRequest(url_str, kHttpGet, headers);
}

Response POST(const std::string& url_str, const std::string& body, const Headers& headers) {
    return SendHttpRequest(url_str, kHttpPost, headers, body);
}