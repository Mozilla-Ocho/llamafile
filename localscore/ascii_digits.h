#ifndef ASCII_DIGITS_H
#define ASCII_DIGITS_H

#include <string>
#include <iostream>

namespace ascii_display {

// Array of ASCII art digits
extern const char* digits[10];

/**
 * @brief Prints a number using ASCII art characters
 * @param number The number to print
 */
void printLargeNumber(int number);

inline void printLargeNumber(int number) {
    // Convert number to string
    std::string num_str = std::to_string(number);
    
    // Handle negative numbers
    bool is_negative = number < 0;
    if (is_negative) {
        num_str = num_str.substr(1); // Remove the minus sign for processing
    }

    // Print each row
    for (int row = 0; row < 5; row++) {
        for (char digit : num_str) {
            int d = digit - '0';  // Convert char to int
            
            // Find the start and end of the current row in the digit's string
            const char* ptr = digits[d];
            int line_start = 0;
            for (int i = 0; i < row; i++) {
                while (ptr[line_start] != '\n') line_start++;
                line_start++; // Skip the newline
            }
            
            int line_end = line_start;
            while (ptr[line_end] && ptr[line_end] != '\n') line_end++;
            
            // Print the current row
            for (int i = line_start; i < line_end; i++) {
                std::cout << ptr[i];
            }
            std::cout << " "; // Space between digits
        }
        std::cout << "\n";
    }
}

inline void print_logo() {
    std::cout << "    __    ____  _________    __   _____ __________  ____  ______\n"
    << "   / /   / __ \\/ ____/   |  / /  / ___// ____/ __ \\/ __ \\/ ____/\n"
    << "  / /   / / / / /   / /| | / /   \\__ \\/ /   / / / / /_/ / __/   \n"
    << " / /___/ /_/ / /___/ ___ |/ /______/ / /___/ /_/ / _, _/ /___   \n"
    << "/_____/\\____/\\____/_/  |_/_____/____/\\____/\\____/_/ |_/_____/   \n";
}

// Definition of the digits array
inline const char* digits[10] = {
    // 0
    " ██████ \n"
    "██    ██\n"
    "██    ██\n"
    "██    ██\n"
    " ██████ \n",
    
    // 1
    " ██ \n"
    "███ \n"
    " ██ \n"
    " ██ \n"
    " ██ \n",
    
    // 2
    "██████  \n"
    "     ██ \n"
    " █████  \n"
    "██      \n"
    "███████ \n",
    
    // 3
    "██████  \n"
    "     ██ \n"
    " █████  \n"
    "     ██ \n"
    "██████  \n",
    
    // 4
    "██   ██ \n"
    "██   ██ \n"
    "███████ \n"
    "     ██ \n"
    "     ██ \n",
    
    // 5
    "███████ \n"
    "██      \n"
    "██████  \n"
    "     ██ \n"
    "██████  \n",
    
    // 6
    " ██████ \n"
    "██      \n"
    "███████ \n"
    "██    ██\n"
    " ██████ \n",
    
    // 7
    "███████ \n"
    "     ██ \n"
    "    ██  \n"
    "   ██   \n"
    "  ██    \n",
    
    // 8
    " █████  \n"
    "██   ██ \n"
    " █████  \n"
    "██   ██ \n"
    " █████  \n",
    
    // 9
    " ██████ \n"
    "██    ██\n"
    " ███████\n"
    "      ██\n"
    " ██████ \n"
};

} // namespace ascii_display

#endif // ASCII_DIGITS_H
