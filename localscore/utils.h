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
#include <cstring>
#include <stdexcept>
#include <string.h>
#include <stdarg.h>
#include <math.h>

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

    inline void print_centered(FILE *stream, int width, char decoration_char, const char *format, ...) {
        char buffer[1024]; // Adjust size as needed
        
        // Handle variable arguments
        va_list args;
        va_start(args, format);
        vsnprintf(buffer, sizeof(buffer), format, args);
        va_end(args);
        
        const char *text = buffer;
        
        // Calculate the visible length (excluding ANSI escape codes)
        int visible_length = 0;
        const char *ptr = text;
        while (*ptr) {
            if (*ptr == '\033') {
                // Skip the escape sequence
                ptr++;
                if (*ptr == '[') {
                    ptr++;
                    while (*ptr && !isalpha(*ptr)) {
                        ptr++;
                    }
                    if (*ptr) ptr++; // Skip the final character of the sequence
                    continue;
                }
            }
            visible_length++;
            ptr++;
        }
        
        // Need at least 2 spaces to separate text from decorations
        int remaining_width = width - visible_length - 2;
        
        // Not enough space for decorations and spaces
        if (remaining_width < 2) {
            fprintf(stream, "%s\n", text);
            return;
        }
        
        // Calculate how many decoration characters on each side
        int decoration_count = remaining_width / 2;
        
        // Calculate padding to truly center everything
        int total_decorated_length = visible_length + 2 + (decoration_count * 2);
        int padding = (width - total_decorated_length) / 2;
        
        // Print left padding
        for (int i = 0; i < padding; i++) {
            fprintf(stream, " ");
        }
        
        // Print left decorations
        for (int i = 0; i < decoration_count; i++) {
            fprintf(stream, "%c", decoration_char);
        }
        
        // Print text with spaces
        fprintf(stream, " %s ", text);
        
        // Print right decorations
        for (int i = 0; i < decoration_count; i++) {
            fprintf(stream, "%c", decoration_char);
        }
        
        // Add extra padding on right if needed
        int right_padding = width - total_decorated_length - padding;
        for (int i = 0; i < right_padding; i++) {
            fprintf(stream, " ");
        }
        
        fprintf(stream, "\n");
    }

    inline void sanitize_string(char *dest, const char *src, size_t max_len) {
        size_t i = 0;
        for (; i < max_len - 1 && src[i]; i++) {
            // Only allow printable ASCII
            dest[i] = (src[i] >= 32 && src[i] < 127) ? src[i] : ' ';
        }
        dest[i] = '\0';
    }

    /**
    * Rounds a double value to the specified number of decimal places.
    * 
    * @param value The value to round
    * @param decimal_places The number of decimal places to round to
    * @return The rounded value
    */
    inline double round_to_decimal(double value, int decimal_places) {
        double multiplier = pow(10.0, decimal_places);
        return round(value * multiplier) / multiplier;
    }

    inline bool should_use_color() {
        const char* no_color = getenv("NO_COLOR");
        if (no_color != NULL && no_color[0] != '0' && no_color[0] != '\0') {
            return false;
        }
        return true;
    }

    inline const char* color_str(const char* color_code) {
        if (should_use_color()) {
            return color_code;
        }
        return "";
    }

} // namespace utils

#endif // UTILS_H
