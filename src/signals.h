// signals.h
#ifndef SIGNALS_H
#define SIGNALS_H

#include "rtklib.h"

// 返回 true 则输出该码型的 C/N0
static inline int is_target_code(unsigned char code) {
    // GPS L1W -> S1W
    if (code == CODE_L1W) return 1;

    // BeiDou B1I/B3I（RINEX 中常写作 S2I / S6I）
    //if (code == CODE_B1I) return 1; // S2I
    //if (code == CODE_B3I) return 1; // S6I

    // 你也可以按需打开更多：
    //if (code == CODE_B2I) return 1; // S7I
    if (code == CODE_L2W) return 1; // GPS L2W -> S2W
    if (code == CODE_L1C) return 1; // GPS L1C -> S1C
    return 0;
}

#endif // SIGNALS_H
