// snr_extract.c
// build: 见下方 CMake 或 Makefile
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rtklib.h"
#include "signals.h"

static void print_csv_header(void) {
    printf("time_utc,sat,signal_code,CN0_dBHz\n");
}

static void time_to_str(gtime_t t, char *buf, size_t n) {
    double ep[6];
    time2epoch(t, ep);
    snprintf(buf, n, "%04d-%02d-%02d %02d:%02d:%06.3f",
             (int)ep[0], (int)ep[1], (int)ep[2], (int)ep[3], (int)ep[4], ep[5]);
}

// 尝试把 RTKLIB 的 code 常量转成类似 "S1W/S2I/S6I" 字符串（不同版本函数名可能略有差异）
static void code_to_str(int sys, unsigned char code, char *out, size_t n) {
    // RTKLIB 提供的 code->字符串函数在不同版本名字可能不同：
    // 有的叫 code2obs(sys, code, type) 组合；有的有 code2str(code, s)。
    // 这里做一个兜底：优先用 code2str；否则简易映射常用几个码。
#ifdef code2str
    code2str(code, out); // 某些版本有效，返回如 "1C","1W","1P" 等
    // 转成 RINEX Sxx 风格不统一，这里不强转，直接输出该字符串
#else
    const char *s = "UNK";
    switch (code) {
        case CODE_L1W: s = "S1W"; break;
        case CODE_L2W: s = "S2W"; break;
        case CODE_L1C: s = "S1C"; break;
        default:       s = "S??"; break;
    }
    snprintf(out, n, "%s", s);
#endif
}

static int read_one_rinex(const char *path, obs_t *obs, nav_t *nav, sta_t *sta) {
    gtime_t ts = {0}, te = {0};
    double tint = 0.0;
    int rcv = 0;      // 0 = all receivers
    char *opt[] = {""};
    int flag = 1;     // 1=append to obs

    return readrnxt(path, rcv, ts, te, tint, opt, flag, obs, nav, sta);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input1.obs> [input2.obs ...] > out.csv\n", argv[0]);
        return 1;
    }

    obs_t obs = {0};
    nav_t nav = {0};
    sta_t sta = {0};

    // 读取所有文件，累计到 obs/nav
    for (int i = 1; i < argc; i++) {
        if (!read_one_rinex(argv[i], &obs, &nav, &sta)) {
            fprintf(stderr, "Failed to read RINEX: %s\n", argv[i]);
            // 不中断，继续读下一个
        }
    }

    print_csv_header();

    for (int i = 0; i < obs.n; i++) {
        obsd_t *o = &obs.data[i];

        char tbuf[64];
        time_to_str(o->time, tbuf, sizeof tbuf);

        char satid[8];
        satno2id(o->sat, satid); // 例如 G12、C06 等

        // 遍历该历元该卫星的所有频点/扩展观测
        for (int j = 0; j < NFREQ + NEXOBS; j++) {
            if (o->SNR[j] <= 0 || o->code[j] == CODE_NONE) continue;
            if (!is_target_code(o->code[j])) continue;

            double cn0 = 0.25 * o->SNR[j]; // RTKLIB：单位步长 0.25 dB-Hz

            char sig[16];
            int sys = satsys(o->sat, NULL); // SYS_GPS/SYS_BDS/...
            code_to_str(sys, o->code[j], sig, sizeof sig);

            printf("%s,%s,%s,%.2f\n", tbuf, satid, sig, cn0);
        }
    }

    // 资源释放（不同版本 nav 成员不同，尽量逐个 free）
    free(obs.data);
    free(nav.eph);   free(nav.geph);  free(nav.seph);
#if 0 // 不同版本还有 nav.peph/pclk 等，根据编译器告警再补
    free(nav.peph);  free(nav.pclk);
#endif
    return 0;
}
