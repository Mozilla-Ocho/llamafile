#include <string>
#include <map>

typedef std::map<std::string, std::string> Headers;

struct Response {
    std::string raw_headers;
    std::string body;
    int status;
    size_t content_length;
    bool is_chunked;
};

Response GET(const std::string& url, const Headers& headers = {});

Response POST(const std::string& url, const std::string& body, const Headers& headers = {});