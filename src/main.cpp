#include "config.h"
#include "log.h"
#include "utils.h"
#include "rnx_extract.h"
#include <iostream>
#include <vector>
#include <unordered_map>


int main() {
    // 记录开始时间
    auto start_time = std::chrono::high_resolution_clock::now();

    // 1. 初始日志（当前目录）
    log_open("snr_extract.log");

    Args A;
    auto cfg = std::filesystem::current_path() / "config.ini";
    if (!load_config_ini(cfg, A)) {
        std::cerr << "Load config failed: " << cfg << std::endl;
        return 1;
    }
    if (A.input_dir.empty()) {
        LOG_ERROR("input_dir not set in config.ini");
        std::cerr << "input_dir not set\n";
        return 2;
    }

    // 2. 切换日志到 <work>/snr_extract.log
    auto work_dir = A.work.empty() ? std::filesystem::current_path() : A.work;
    ensure_dir(work_dir);
    log_close();
    log_open((work_dir / "snr_extract.log").string());
    LOG_INFO("=== snr_extract start ===");

    // 3. 解析 signals=... → codes & code2label（必须非空）
    std::vector<unsigned char> expected_codes;
    std::unordered_map<unsigned char, std::string> code2label;
    if (!build_expected_codes(A, expected_codes, code2label) || expected_codes.empty()) {
        LOG_ERROR("No valid signals in config.ini; abort.");
        std::cerr << "No valid signals in config.ini; abort.\n";
        log_close();
        return 3;
    }

    // 4. 确定输入/输出目录
    auto in_root  = A.input_dir;
    auto out_root = A.output_dir.empty() ? (work_dir / "out") : A.output_dir;
    ensure_dir(out_root);

    // 5. 递归扫描 RNX 文件
    auto files = list_rnx_files_recursive(in_root);
    if (files.empty()) {
        LOG_WARN("No RNX files found under: %s", in_root.string().c_str());
        std::cout << "No RNX files found.\n";
        log_close();
        return 0;
    }
    LOG_INFO("Found %zu RNX files in %s", files.size(), in_root.string().c_str());

    // 6. 逐个处理 RNX
    for (size_t idx=0; idx<files.size(); ++idx) {
        auto& rnx = files[idx];
        auto station_tag = basename_no_ext(rnx.filename());

        std::cout << "[INFO] (" << (idx+1) << "/" << files.size()
                  << ") Processing " << rnx.string() << " ..." << std::endl;

        try {
            extract_rnx_to_csvs(rnx, out_root, station_tag, expected_codes, code2label);
            LOG_INFO("OK: %s", rnx.string().c_str());
        } catch (...) {
            LOG_ERROR("EXTRACT CRASHED: %s", rnx.string().c_str());
        }
    }

    LOG_INFO("=== snr_extract done ===");
    log_close();

    // 记录结束时间
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "All done in " << duration.count() << " seconds.\n";
    return 0;
}
