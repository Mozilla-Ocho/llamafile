# LLaMAfiler Technical Details

LLaMAfiler is designed to be a high-performance server for generating
embeddings from various language models. This document outlines key
technical aspects of LLaMAfiler.

## Performance

This server can serve embeddings an order of a magnitude faster than the
llama.cpp example server. We measured this on Threadripper PRO 7995WX w/
all-MiniLM-L6-v2 Q6\_K. Using a prompt with 50 tokens this server served
930 `/embedding` requests per second, whereas llama.cpp/build/bin/server
built with glibc served 100 per second.

The following wrk script was used:

```lua
wrk.method = "POST"
wrk.body = "{\"content\": \"hello world how are you today what is up what is up what is up what is up is up what is up what is up is up what is up what is up is up what is up what is up is up what is up what is up\"}"
wrk.headers["Content-Type"] = "application/json"
```

With the following commands:

```
llamafiler -m /weights/all-MiniLM-L6-v2.Q6_K.gguf
wrk --latency -t 32 -c 32 -s embedding.lua 'http://127.0.0.1:8080/embedding'

~/llama.cpp/build/bin/server -m /weights/all-MiniLM-L6-v2.Q6_K.gguf --embeddings
wrk --latency -t 32 -c 32 -s embedding.lua 'http://127.0.0.1:8080/embedding'
```

The results for llamafiler were:

```
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency    33.89ms    7.48ms  61.55ms   66.90%
    Req/Sec    29.26      6.52    50.00     59.74%
  Latency Distribution
     50%   33.31ms
     75%   38.76ms
     90%   43.76ms
     99%   53.31ms
  1592 requests in 1.70s, 8.26MB read
Requests/sec:    934.80
Transfer/sec:      4.85MB
```

The results for llama.cpp were:

```
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency   262.34ms   75.10ms 322.91ms   82.80%
    Req/Sec     3.73      1.88    10.00     93.55%
  Latency Distribution
     50%  281.00ms
     75%  321.60ms
     90%  322.32ms
     99%  322.91ms
  93 requests in 901.53ms, 748.74KB read
Requests/sec:    103.16
Transfer/sec:    830.53KB
```

It's possible to make the llama.cpp server go faster, by passing the
`-np 10` flag. This will preallocate ten server slots which allow
requests to be handled in parallel. However doing this boosted the
performance to no more than 200 requests per second. Specifying higher
values, e.g. `-np 50` causes the llama.cpp to crash.

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

For example, we intend to implement an HTTP header that will allow a
client to elect the priority of its request. For low-priority requests,
if a higher priority job happens to come in, the low-priority job can be
canceled in less than a millisecond, even if it's inside a deep,
long-running matrix multiplication operation. This will not leak memory.
It ensures that resources are freed up immediately for the other job.

Cancelation is also useful for DDoS mitigation. The `-w N` flag allows
you to specify a fixed number of HTTP workers. By default, this will be
set to the number of CPU cores multiplied by two. HTTP clients are
permitted to idle (via the HTTP keep-alive mechanism) as long as they
wish. However, if the number of available workers runs out when a new
request comes in, the oldest worker will be canceled to make room.

When a cancelation happens, HTTP connections in the idle state will
simply be closed. If an HTTP connection is in the middle of serving a
request, then a 503 Service Unavailable response will be sent to the
client. In such cases, it is recommended that client software retry with
exponential backoff. If multiple instances of LLaMAfiler are running on
multiple servers, then failover strategies can also be used.

## Crash Proofing

LLaMAfiler is designed to be impossible to crash. Let's say there's a
bug in llama.cpp or LLaMAfiler that causes a SIGSEGV to happen while
processing a request. In that case, the worker thread will be canceled
and a backtrace will be logged. A fresh new worker thread will then be
spawned to take its place.

Each thread is given a stack size of 81KB, which includes the guard
page. Stack overflows are reported and recoverable with a backtrace,
thanks to `sigaltstack()`.

Recovering from a crash should not leak memory. For this reason, certain
kinds of crashes aren't recoverable. If the heap becomes corrupted, then
the entire process will crash.
