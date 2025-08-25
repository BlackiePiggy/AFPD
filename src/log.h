#pragma once
#include <cstdio>
#include <string>

extern FILE* g_log;

void log_open(const std::string& path);
void log_close();

void logf(const char* level, const char* fmt, ...);

#define LOG_INFO(...)  logf("INFO",  __VA_ARGS__)
#define LOG_WARN(...)  logf("WARN",  __VA_ARGS__)
#define LOG_ERROR(...) logf("ERROR", __VA_ARGS__)
