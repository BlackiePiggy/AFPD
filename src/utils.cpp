#include "utils.h"
#include "log.h"
#include <system_error>
#include <sstream>
#include <algorithm>
#include <cctype>

void ensure_dir(const fs::path& p){ std::error_code ec; fs::create_directories(p, ec); }
void rm_quiet(const fs::path& p){ std::error_code ec; fs::remove(p, ec); }
std::string basename_no_ext(const fs::path& p){
    auto s = p.filename().string(); auto pos = s.find_last_of('.');
    return pos==std::string::npos ? s : s.substr(0,pos);
}

int run_cmd(const std::string& cmdline){
    LOG_INFO("RUN: %s", cmdline.c_str());
    return std::system(cmdline.c_str());
}

int yday(int Y, int m, int d){
    static const int mdays[] = {31,28,31,30,31,30,31,31,30,31,30,31};
    auto leap = [](int y){ return (y%4==0 && y%100!=0) || (y%400==0); };
    int doy = d;
    for (int i=1;i<m;i++) doy += mdays[i-1] + (i==2 && leap(Y) ? 1:0);
    return doy;
}
std::string two(int x){ char b[8]; std::snprintf(b,sizeof b,"%02d",x); return b; }
std::string three(int x){ char b[8]; std::snprintf(b,sizeof b,"%03d",x); return b; }
std::string four(int x){ char b[8]; std::snprintf(b,sizeof b,"%04d",x); return b; }

std::vector<fs::path> list_rnx_files_recursive(const fs::path& root){
    std::vector<fs::path> v;
    if (!fs::exists(root)) return v;
    for (auto& e : fs::recursive_directory_iterator(root)){
        if (!e.is_regular_file()) continue;
        auto p = e.path(); auto ext = p.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".rnx" || ext == ".obs" ||
            (ext.size() == 4 && ext[3] == 'o')) {
                    v.push_back(p);
            }
        }
    std::sort(v.begin(), v.end());
    return v;
}
