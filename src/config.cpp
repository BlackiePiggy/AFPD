#include "config.h"
#include "log.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <cstdio>

static std::string trim(const std::string& s){
    size_t l=0,r=s.size();
    while(l<r && std::isspace((unsigned char)s[l])) ++l;
    while(r>l && std::isspace((unsigned char)s[r-1])) --r;
    return s.substr(l,r-l);
}
static std::vector<std::string> split(const std::string& s, char sep){
    std::vector<std::string> v; std::stringstream ss(s); std::string x;
    while(std::getline(ss,x,sep)){ x=trim(x); if(!x.empty()) v.push_back(x); }
    return v;
}

bool parse_date(const std::string& s, TimePoint& t){
    int n = std::sscanf(s.c_str(), "%d-%d-%dT%d:%d:%d",&t.Y,&t.m,&t.d,&t.H,&t.M,&t.S);
    if (n==3){ t.H=0; t.M=0; t.S=0; return true; }
    if (n==6) return true;
    return false;
}

bool parse_ranges(const std::string& s, TimeRanges& r){
    auto items = split(s, ';');
    for (auto& it : items){
        auto pos = it.find("..");
        if (pos==std::string::npos){
            TimePoint a{}; if(!parse_date(it,a)) return false;
            TimePoint b=a; b.d+=1;
            r.ranges.push_back({a,b});
        } else {
            std::string a = it.substr(0,pos), b = it.substr(pos+2);
            TimePoint ta{}, tb{}; if(!parse_date(a,ta)||!parse_date(b,tb)) return false;
            r.ranges.push_back({ta,tb});
        }
    }
    return !r.ranges.empty();
}

bool load_config_ini(const std::filesystem::path& cfg, Args& A){
    std::ifstream in(cfg);
    if(!in){ LOG_ERROR("open config failed: %s", cfg.string().c_str()); return false; }
    std::string line;
    while(std::getline(in,line)){
        line = trim(line);
        if(line.empty() || line[0]=='#' || line[0]==';') continue;
        auto pos = line.find('=');
        if(pos==std::string::npos) continue;
        std::string k=trim(line.substr(0,pos)), v=trim(line.substr(pos+1));
        if(k=="work")            A.work=v;
        else if(k=="input_dir")  A.input_dir=v;
        else if(k=="output_dir") A.output_dir=v;

        else if(k=="stations_data_url")  A.stations_data_url=v;
        else if(k=="stations_data_json") A.stations_data_json=v;
        else if(k=="station_list_path")  A.station_list_path=v;
        else if(k=="stations")           A.stations=split(v,',');

        else if(k=="ranges")            { parse_ranges(v,A.ranges); }
        else if(k=="step")               A.step=v;
        else if(k=="url_template")      { A.url_templates=split(v,';'); }
        else if(k=="download_suffixes")  A.download_suffixes=split(v,',');
        else if(k=="rtkget")             A.rtkget=v;
        else if(k=="convrnx")            A.convrnx=v;

        else if(k=="signals")            A.expect_signals=split(v,',');
    }

    if (A.work.empty()) {
        LOG_WARN("work not set; fallback to current directory");
        A.work = std::filesystem::current_path();
    }
    if (A.input_dir.empty()) {
        LOG_ERROR("input_dir is required in config.ini");
        return false;
    }
    return true;
}

// ===== 将 config.ini 的 "Sxx" → RTKLIB 码型，并建立 code→label 的映射 =====
extern "C" {
#include "rtklib.h"
}

static bool str_to_code(const std::string& s, unsigned char& code) {
    std::string u=s; std::transform(u.begin(),u.end(),u.begin(),::toupper);
    // GPS（示例）
    if (u=="S1C") { code = CODE_L1C; return true; }
    if (u=="S1W") { code = CODE_L1W; return true; }
    if (u=="S2W") { code = CODE_L2W; return true; }

    // BDS（常用映射；按你的 RTKLIB 版本补充即可）
    if (u=="S2I") { code = CODE_L1I; return true; }
    if (u=="S6I") { code = CODE_L6I; return true; }
    if (u=="S7I") { code = CODE_L7I; return true; }

    // TODO: 需要其他系统/信号时继续在此补充：
    //  E (Galileo), R (GLONASS), J (QZSS), I (IRNSS), S (SBAS) 等
    return false;
}

bool build_expected_codes(
    const Args& A,
    std::vector<unsigned char>& out_codes,
    std::unordered_map<unsigned char, std::string>& out_code2label)
{
    out_codes.clear();
    out_code2label.clear();

    for (auto s : A.expect_signals) {
        // 规范化为大写
        std::transform(s.begin(), s.end(), s.begin(), ::toupper);

        unsigned char c=0;
        if (str_to_code(s, c)) {
            if (std::find(out_codes.begin(), out_codes.end(), c) == out_codes.end()) {
                out_codes.push_back(c);
            }
            out_code2label[c] = s; // 原样用 "Sxx" 作为输出标签
        } else {
            LOG_WARN("Unknown signal in config: %s", s.c_str());
        }
    }

    if (out_codes.empty()) {
        LOG_ERROR("No valid signals parsed from config.ini 'signals='");
        return false;
    }
    return true;
}
