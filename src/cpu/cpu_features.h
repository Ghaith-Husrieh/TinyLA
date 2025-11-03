#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

bool has_sse42(void);
bool has_avx2(void);

#ifdef __cplusplus
}
#endif
