#pragma once
#include <string>
#include <vector>
#include <filesystem>
#include <unordered_map>

struct TimePoint { int Y, m, d, H, M, S; };

struct TimeRanges {
    std::vector<std::pair<TimePoint, TimePoint>> ranges; // 保留（离线模式不使用）
};

struct Args {
    std::filesystem::path work;

    // 本地输入/输出
    std::filesystem::path input_dir;
    std::filesystem::path output_dir;

    // 以下下载相关在离线模式不使用（保留，为以后扩展）
    std::string stations_data_url;
    std::string stations_data_json;
    std::filesystem::path station_list_path;
    std::vector<std::string> stations;

    TimeRanges ranges;
    std::string step = "daily";
    std::vector<std::string> url_templates;
    std::vector<std::string> download_suffixes;
    std::filesystem::path rtkget;
    std::filesystem::path convrnx;

    // 期望信号（字符串，来自 config.ini），例如：S1C,S1W,S2W,S2I,S6I,S7I
    std::vector<std::string> expect_signals;
};

bool load_config_ini(const std::filesystem::path& cfg, Args& A);

// 将 config.ini 的 signals=... 转成：
//  1) RTKLIB 的观测码列表（unsigned char）
//  2) code → "Sxx" 字符串映射（用于输出与日志）
bool build_expected_codes(
    const Args& A,
    std::vector<unsigned char>& out_codes,
    std::unordered_map<unsigned char, std::string>& out_code2label);

// 可复用工具（解析日期/区间，离线模式不强制使用）
bool parse_date(const std::string& s, TimePoint& t);
bool parse_ranges(const std::string& s, TimeRanges& r);
