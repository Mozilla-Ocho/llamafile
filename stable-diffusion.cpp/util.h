#ifndef __UTIL_H__
#define __UTIL_H__

#include <cstdint>
#include <string>
#include <vector>

#include "stable-diffusion.h"

bool ends_with(const std::string& str, const std::string& ending);
bool starts_with(const std::string& str, const std::string& start);
bool contains(const std::string& str, const std::string& substr);

std::string format(const char* fmt, ...);

void replace_all_chars(std::string& str, char target, char replacement);

bool file_exists(const std::string& filename);
bool is_directory(const std::string& path);
std::string get_full_path(const std::string& dir, const std::string& filename);

std::vector<std::string> get_files_from_dir(const std::string& dir);

std::u32string utf8_to_utf32(const std::string& utf8_str);
std::string utf32_to_utf8(const std::u32string& utf32_str);
std::u32string unicode_value_to_utf32(int unicode_value);

sd_image_t* preprocess_id_image(sd_image_t* img);

// std::string sd_basename(const std::string& path);

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t channel;
    float* data;
} sd_image_f32_t;

void normalize_sd_image_f32_t(sd_image_f32_t image, float means[3], float stds[3]);

sd_image_f32_t sd_image_t_to_sd_image_f32_t(sd_image_t image);

sd_image_f32_t resize_sd_image_f32_t(sd_image_f32_t image, int target_width, int target_height);

sd_image_f32_t clip_preprocess(sd_image_f32_t image, int size);

std::string path_join(const std::string& p1, const std::string& p2);

void pretty_progress(int step, int steps, float time);

void log_printf(sd_log_level_t level, const char* file, int line, const char* format, ...);

std::string trim(const std::string& s);

std::vector<std::pair<std::string, float>> parse_prompt_attention(const std::string& text);

#define LOG_DEBUG(format, ...) log_printf(SD_LOG_DEBUG, __FILE__, __LINE__, format, ##__VA_ARGS__)
#define LOG_INFO(format, ...) log_printf(SD_LOG_INFO, __FILE__, __LINE__, format, ##__VA_ARGS__)
#define LOG_WARN(format, ...) log_printf(SD_LOG_WARN, __FILE__, __LINE__, format, ##__VA_ARGS__)
#define LOG_ERROR(format, ...) log_printf(SD_LOG_ERROR, __FILE__, __LINE__, format, ##__VA_ARGS__)
#endif  // __UTIL_H__
