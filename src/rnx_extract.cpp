// src/rnx_extract.cpp
#include "rnx_extract.h"
#include "utils.h"
#include "log.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <iostream>

extern "C" {
#include "rtklib.h"
}

// 兼容不同 RTKLIB 的 SNR 字段名
#ifndef SNR_FIELD
#define SNR_FIELD SNR
#endif

// 统一的输出键：year/signal/station
struct Key {
    int year;
    std::string sig;     // e.g. "S1C","S2I"
    std::string station; // station tag

    bool operator==(const Key& o) const noexcept {
        return year==o.year && sig==o.sig && station==o.station;
    }
};
struct KeyHash {
    size_t operator()(const Key& k) const noexcept {
        std::hash<std::string> hs; std::hash<int> hi;
        size_t h = hi(k.year);
        h ^= (hs(k.sig)     + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2));
        h ^= (hs(k.station) + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2));
        return h;
    }
};

static void print_csv_header(FILE *fo) {
    // 保留卫星字段 sat
    std::fprintf(fo, "time_utc,sat,signal_code,station,CN0_dBHz\n");
}

static void time_to_str(gtime_t t, char *buf, size_t n) {
    double ep[6]; time2epoch(t, ep);
    std::snprintf(buf, n, "%04d-%02d-%02d %02d:%02d:%06.3f",
                  (int)ep[0], (int)ep[1], (int)ep[2],
                  (int)ep[3], (int)ep[4], ep[5]);
}

static int read_one_rinex(const char *path, obs_t *obs, nav_t *nav, sta_t *sta) {
    gtime_t ts=(gtime_t){0}, te=(gtime_t){0};
    double  tint=0.0;
    int     rcv=0;
    const char *opt="";
    return readrnxt(path, rcv, ts, te, tint, opt, obs, nav, sta);
}

void extract_rnx_to_csvs(const std::filesystem::path& rnx_path,
                         const std::filesystem::path& out_root,
                         const std::string& station_tag,
                         const std::vector<unsigned char>& expected_codes_vec,
                         const std::unordered_map<unsigned char, std::string>& code2label) {
    if (expected_codes_vec.empty()) {
        LOG_WARN("expected_codes is empty; nothing to do. file=%s", rnx_path.string().c_str());
        return;
    }
    std::unordered_set<unsigned char> expected_codes(expected_codes_vec.begin(), expected_codes_vec.end());

    obs_t obs={0}; nav_t nav={0}; sta_t sta={0};
    if (!read_one_rinex(rnx_path.string().c_str(), &obs, &nav, &sta)) {
        LOG_ERROR("READ RNX FAILED: %s", rnx_path.string().c_str());
        return;
    }

    std::map<std::string, std::set<unsigned char>> seen_by_sat;
    std::unordered_set<std::string> created_dirs;
    std::unordered_map<Key, FILE*, KeyHash> sinks;

    auto get_sink = [&](int year, const std::string& sig, const std::string& station) -> FILE* {
        Key k{year, sig, station};
        auto it = sinks.find(k);
        if (it != sinks.end()) return it->second;

        // 路径：<out>/<year>/<sig>/<station>.csv
        auto dir = out_root / std::to_string(year) / sig;
        const std::string dir_str = dir.string();
        if (!created_dirs.count(dir_str)) {
            ensure_dir(dir);
            created_dirs.insert(dir_str);
        }
        auto fname = dir / (station + std::string(".csv"));
        const std::string fn_str = fname.string();

        const bool new_file = !std::filesystem::exists(fname);
        FILE* fo = std::fopen(fn_str.c_str(), new_file ? "wb" : "ab");
        if (!fo) {
            LOG_ERROR("OPEN CSV FAILED: %s", fn_str.c_str());
            return nullptr;
        }
        if (new_file) print_csv_header(fo);

        static const size_t BUFSZ = 1<<20;
        setvbuf(fo, nullptr, _IOFBF, BUFSZ);

        sinks.emplace(k, fo);
        return fo;
    };

    for (int i=0;i<obs.n;++i){
//        if (i % (obs.n/20 + 1) == 0) { // 大约每5%
//            double pct = (100.0 * i) / obs.n;
//            std::cout << "   ... progress " << (int)pct << "% ("
//                      << i << "/" << obs.n << " records)\r" << std::flush;
//        }

        obsd_t *o = &obs.data[i];
        char tbuf[64]; time_to_str(o->time, tbuf, sizeof tbuf);
        char satid[8]; satno2id(o->sat, satid);

        for (int j=0;j<NFREQ+NEXOBS;++j){
            if (o->SNR_FIELD[j] <= 0 || o->code[j] == CODE_NONE) continue;
            const unsigned char code = o->code[j];
            if (!expected_codes.count(code)) continue;

            seen_by_sat[satid].insert(code);

            std::string sig = "S??";
            if (auto it = code2label.find(code); it != code2label.end()) sig = it->second;
            double cn0 = o->SNR_FIELD[j];
            double ep[6]; time2epoch(o->time, ep);
            const int year = (int)ep[0];

            FILE* fo = get_sink(year, sig, station_tag);
            if (!fo) continue;
            std::fprintf(fo, "%s,%s,%s,%s,%.2f\n",
                         tbuf, satid, sig.c_str(), station_tag.c_str(), cn0);
        }
    }
    //std::cout << "   ... progress 100% (" << obs.n << "/" << obs.n << " records)" << std::endl;

    for (const auto& kv : seen_by_sat){
        const auto& sat = kv.first;
        for (auto code_needed : expected_codes){
            if (!kv.second.count(code_needed)){
                auto it = code2label.find(code_needed);
                const std::string sig = (it==code2label.end()) ? "S??" : it->second;
                LOG_WARN("MISSING SIGNAL: sat=%s signal=%s file=%s",
                         sat.c_str(), sig.c_str(), rnx_path.string().c_str());
            }
        }
    }

    for (auto& kv : sinks) {
        if (kv.second) std::fclose(kv.second);
    }

    std::free(obs.data);
    std::free(nav.eph);
    std::free(nav.geph);
    std::free(nav.seph);
}
