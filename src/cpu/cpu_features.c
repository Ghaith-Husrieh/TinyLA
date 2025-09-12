#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

#include "cpu_features.h"
#include <stdbool.h>

bool has_avx2(void) {
#ifdef _MSC_VER
    int info[4];
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) != 0; // EBX bit 5 = AVX2
#else
    unsigned int eax, ebx, ecx, edx;
    __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);
    return (ebx & (1 << 5)) != 0; // EBX bit 5 = AVX2
#endif
}

bool has_sse42(void) {
#if _MSC_VER
    int info[4];
    __cpuid(info, 1);
    return (info[2] & (1 << 20)) != 0; // ECX bit 20 = SSE4.2
#else
    unsigned int eax, ebx, ecx, edx;
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    return (ecx & (1 << 20)) != 0; // ECX bit 20 = SSE4.2
#endif
}
