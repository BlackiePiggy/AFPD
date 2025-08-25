#include "log.h"
#include <cstdarg>
#include <chrono>
#include <ctime>

FILE* g_log = nullptr;

static std::string now_str() {
    auto tp = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(tp);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d %02d:%02d:%02d",
                  tm.tm_year+1900, tm.tm_mon+1, tm.tm_mday,
                  tm.tm_hour, tm.tm_min, tm.tm_sec);
    return buf;
}

void log_open(const std::string& path) {
    if (g_log) std::fclose(g_log);
    g_log = std::fopen(path.c_str(), "a");
}

void log_close() {
    if (g_log) std::fclose(g_log);
    g_log = nullptr;
}

void logf(const char* level, const char* fmt, ...) {
    if (!g_log) return;
    std::fprintf(g_log, "[%s] [%s] ", now_str().c_str(), level);
    va_list ap; va_start(ap, fmt);
    std::vfprintf(g_log, fmt, ap);
    va_end(ap);
    std::fprintf(g_log, "\n");
    std::fflush(g_log);
}
