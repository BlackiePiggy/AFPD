#pragma once
#include <filesystem>
#include <string>
#include <vector>
#include <unordered_map>
#include "config.h"

// 读取 RNX 并按 年/卫星/信号/测站 输出 CSV；
// 仅处理 expected_codes 中的码型；输出标签来自 code2label。
void extract_rnx_to_csvs(const std::filesystem::path& rnx_path,
                         const std::filesystem::path& out_root,
                         const std::string& station_tag,
                         const std::vector<unsigned char>& expected_codes,
                         const std::unordered_map<unsigned char, std::string>& code2label);
