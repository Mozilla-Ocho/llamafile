# LLaMAfiler Technical Details

LLaMAfiler is designed to be a high-performance server for generating
embeddings from various language models. This document outlines key
technical aspects of LLaMAfiler.

## Performance

This server can serve embeddings an order of a magnitude faster than the
llama.cpp example server. We measured this on Threadripper PRO 7995WX w/
all-MiniLM-L6-v2 Q6\_K. Using a prompt with 50 tokens this server served
2,427 `/embedding` requests per second, whereas
llama.cpp/build/bin/server built with glibc served 771 per second.

The following wrk script was used:

```lua
wrk.method = "POST"
wrk.body = "{\"content\": \"hello world how are you today what is up what is up what is up what is up is up what is up what is up is up what is up what is up is up what is up what is up is up what is up what is up\"}"
wrk.headers["Content-Type"] = "application/json"
```

With the following commands:

```
llamafiler -m /weights/all-MiniLM-L6-v2.Q6_K.gguf --trust 127.0.0.1/32
wrk --latency -t 32 -c 32 -s embedding.lua 'http://127.0.0.1:8080/embedding'

~/llama.cpp/llama-server -m /weights/all-MiniLM-L6-v2.Q6_K.gguf --embeddings
wrk --latency -t 32 -c 32 -s embedding.lua 'http://127.0.0.1:8080/embedding'
```

The results for llamafiler were:

```
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency    13.12ms    2.95ms  32.75ms   87.75%
    Req/Sec    75.85      8.86   101.00     76.56%
  Latency Distribution
     50%   12.66ms
     75%   13.54ms
     90%   15.29ms
     99%   25.43ms
  2427 requests in 1.00s, 12.58MB read
Requests/sec:   2420.96
Transfer/sec:     12.55MB
```

The results for llama.cpp in CPU mode were:

```
  32 threads and 32 connections
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency    41.16ms   20.70ms 107.46ms   59.20%
    Req/Sec    24.14      8.80    40.00     73.44%
  Latency Distribution
     50%   51.99ms
     75%   56.98ms
     90%   60.19ms
     99%   87.11ms
  2163 requests in 2.80s, 16.96MB read
Requests/sec:    771.50
Transfer/sec:      6.05MB
```

GPU mode was also attempted with a pair of RTX 4090s. However it was
only able to do 450 requests per second. CPU goes faster here probably
because we're using a model that's small enough to fit entirely inside
Threadripper's L3 cache.

For less computationally intensive endpoints, such as `/tokenize`,
llamafiler on Threadripper PRO 7995WX is able to serve 3.3 million
requests per second.

```
wrk --latency -t $(nproc) -c $(nproc) 'http://127.0.0.1:8080/tokenize?prompt=hello+world+how+are+you+today+what+is+up+what+is+up+what+is+up+what+is+up+is+up+what+is+up+what+is+up+is+up+what+is+up+what+is+up+is+up+what+is+up+what+is+up+is+up+what+is+up+what+is+up'
Running 10s test @ http://127.0.0.1:8080/tokenize?prompt=hello+world+how+are+you+today+what+is+up+what+is+up+what+is+up+what+is+up+is+up+what+is+up+what+is+up+is+up+what+is+up+what+is+up+is+up+what+is+up+what+is+up+is+up+what+is+up+what+is+up
  192 threads and 192 connections
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency    56.46us   11.03us   3.69ms   97.53%
    Req/Sec    17.36k     1.05k   25.48k    95.69%
  Latency Distribution
     50%   56.00us
     75%   58.00us
     90%   59.00us
     99%   66.00us
  4971957 requests in 1.50s, 4.32GB read
Requests/sec: 3316068.33
Transfer/sec:      2.88GB
```

## Cancelation

LLaMAfiler uses `pthread_cancel()` to asynchronously cancel requests
when appropriate. The server is designed this way so that high-priority
or latency-sensitive requests can preempt batch background jobs.

Clients can voluntarily send an `X-Priority: batch` HTTP header that'll
make a request first in line for preemption. For low-priority requests,
if a higher priority job happens to come in, the low-priority job can be
canceled in less than a millisecond, even if it's inside a deep,
long-running matrix multiplication operation. This will not leak memory.
It ensures that resources are freed up immediately for the other job.

Clients can also be deprioritized involuntarily, if they create more
connections or send more requests than the token bucket burst limit.
This provides a last line of defense against DDOS. Even without tokens,
clients can still do whatever they want, because no action is taken by
the server until resource pressure happens. The `-w N` flag allows you
to specify a fixed number of HTTP workers. By default, this will be set
to the number of CPU cores multiplied by two. HTTP clients are permitted
to idle (via the HTTP keep-alive mechanism) or spam requests as long and
as much as they wish. However, if the number of available workers runs
out when a new request comes in, then the most recently deprioritized
worker will be canceled to make room.

When a cancelation happens, HTTP connections in the idle state will
simply be closed. If an HTTP connection is in the middle of serving a
request, then a 503 Service Unavailable response will be sent to the
client. In such cases, it is recommended that client software retry with
exponential backoff. If multiple instances of LLaMAfiler are running on
multiple servers, then failover strategies can also be used.

## Crash Proofing

One of the issues with the upstream llama.cpp server is that it's very
easy to make it crash. LLaMAfiler is designed to be impossible to crash.
Let's say there's a bug in llama.cpp or LLaMAfiler that causes a SIGSEGV
to happen while processing a request. In that case, the worker thread
will be canceled and a backtrace will be logged. A fresh new worker
thread will then be spawned to take its place.

Each thread is given a stack size of 81KB, which includes the guard
page. Stack overflows are reported and recoverable with a backtrace,
thanks to `sigaltstack()`.

Recovering from a crash should not leak memory. For this reason, certain
kinds of crashes aren't recoverable. If the heap becomes corrupted, then
the entire process will crash.
