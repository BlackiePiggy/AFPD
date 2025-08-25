#pragma once
#include <string>
#include <vector>
#include <filesystem>
#include "config.h"

namespace fs = std::filesystem;

void ensure_dir(const fs::path& p);
void rm_quiet(const fs::path& p);
std::string basename_no_ext(const fs::path& p);

int run_cmd(const std::string& cmdline);

int yday(int Y, int m, int d);
std::string two(int x);
std::string three(int x);
std::string four(int x);

// 递归列出目录内所有 *.rnx / *.obs
std::vector<fs::path> list_rnx_files_recursive(const fs::path& root);
