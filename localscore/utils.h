#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <chrono>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <memory>
#include <array>
#include <cstdio>
#include <stdexcept>

namespace utils {

    inline uint64_t get_time_ns() {
        using clock = std::chrono::high_resolution_clock;
        return std::chrono::nanoseconds(clock::now().time_since_epoch()).count();
    }

    template<class T>
    inline std::string join(const std::vector<T>& values, const std::string& delim) {
        std::ostringstream str;
        for (size_t i = 0; i < values.size(); i++) {
            str << values[i];
            if (i < values.size() - 1) {
                str << delim;
            }
        }
        return str.str();
    }

    template<class T>
    inline std::vector<T> split(const std::string& str, char delim) {
        std::vector<T> values;
        std::istringstream str_stream(str);
        std::string token;
        while (std::getline(str_stream, token, delim)) {
            T value;
            std::istringstream token_stream(token);
            token_stream >> value;
            values.push_back(value);
        }
        return values;
    }

    template<typename T, typename F>
    inline std::vector<std::string> transform_to_str(const std::vector<T>& values, F f) {
        std::vector<std::string> str_values;
        std::transform(values.begin(), values.end(), std::back_inserter(str_values), f);
        return str_values;
    }

    template<typename T>
    inline T avg(const std::vector<T>& v) {
        if (v.empty()) {
            return 0;
        }
        T sum = std::accumulate(v.begin(), v.end(), T(0));
        return sum / static_cast<T>(v.size());
    }

    template<typename T>
    inline T stdev(const std::vector<T>& v) {
        if (v.size() <= 1) {
            return 0;
        }
        T mean = avg(v);
        T sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), T(0));
        T stdev = std::sqrt(sq_sum / static_cast<T>(v.size() - 1) - 
                        mean * mean * static_cast<T>(v.size()) / static_cast<T>(v.size() - 1));
        return stdev;
    }

    inline std::string exec(const char* cmd) {
        std::array<char, 128> buffer;
        std::string result;
        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
        if (!pipe) {
            throw std::runtime_error("popen() failed!");
        }
        while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
            result += buffer.data();
        }
        return result;
    }

} // namespace utils

#endif // UTILS_H
