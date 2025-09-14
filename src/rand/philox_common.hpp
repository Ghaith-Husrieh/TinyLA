#pragma once
#include <cmath>
#include <cstdint>

constexpr uint32_t PHILOX_M0 = 0xD2511F53UL;
constexpr uint32_t PHILOX_M1 = 0xCD9E8D57UL;
constexpr uint32_t PHILOX_W0 = 0x9E3779B9UL;
constexpr uint32_t PHILOX_W1 = 0xBB67AE85UL;

constexpr double UINT32_TO_DOUBLE = 1.0 / 4294967296.0;

#ifdef __CUDACC__
#define TLA_INLINE __device__ inline
#else
#define TLA_INLINE inline
#endif

struct PhiloxState {
    uint64_t seed;
    uint64_t offset;
};

struct PhiloxCtr {
    uint32_t ctr[4];
};

struct PhiloxKey {
    uint32_t key[2];
};

TLA_INLINE void philox_round(PhiloxCtr& ctr, PhiloxKey& key) {
    uint64_t prod0 = uint64_t(ctr.ctr[0]) * PHILOX_M0;
    uint64_t prod1 = uint64_t(ctr.ctr[2]) * PHILOX_M1;

    uint32_t hi0 = uint32_t(prod0 >> 32), lo0 = uint32_t(prod0);
    uint32_t hi1 = uint32_t(prod1 >> 32), lo1 = uint32_t(prod1);

    ctr.ctr[0] = hi1 ^ ctr.ctr[1] ^ key.key[0];
    ctr.ctr[1] = lo1;
    ctr.ctr[2] = hi0 ^ ctr.ctr[3] ^ key.key[1];
    ctr.ctr[3] = lo0;

    key.key[0] += PHILOX_W0;
    key.key[1] += PHILOX_W1;
}

TLA_INLINE PhiloxCtr philox_4x32_10(PhiloxCtr ctr, PhiloxKey key) {
    for (int i = 0; i < 10; ++i)
        philox_round(ctr, key);

    return ctr;
}

template <typename Derived> struct PhiloxGenerator {
    TLA_INLINE void generate(PhiloxState& st, double out[4], size_t thread_offset = 0) {
        Derived::generate(st, out, thread_offset);
    }
};

struct PhiloxUniform : public PhiloxGenerator<PhiloxUniform> {
    TLA_INLINE static void generate(PhiloxState& st, double out[4], size_t thread_offset = 0) {
        PhiloxCtr ctr;
        PhiloxKey key;

#ifdef __CUDA_ARCH__
        uint64_t idx = thread_offset;
#else
        uint64_t idx = st.offset++;
#endif

        ctr.ctr[0] = uint32_t(idx >> 32);
        ctr.ctr[1] = uint32_t(idx);
        ctr.ctr[2] = 0;
        ctr.ctr[3] = 0;

        key.key[0] = uint32_t(st.seed >> 32);
        key.key[1] = uint32_t(st.seed);

        PhiloxCtr res = philox_4x32_10(ctr, key);

        for (int i = 0; i < 4; ++i)
            out[i] = res.ctr[i] * UINT32_TO_DOUBLE;
    }
};

struct PhiloxNormal : public PhiloxGenerator<PhiloxNormal> {
    TLA_INLINE static void generate(PhiloxState& st, double out[4], size_t thread_offset = 0) {
        double u[4];
        PhiloxUniform::generate(st, u, thread_offset);

        for (int i = 0; i < 2; ++i) {
            double r = std::sqrt(-2.0 * std::log(u[2 * i]));
            double theta = 2.0 * 3.141592653589793 * u[2 * i + 1];
            out[2 * i] = r * std::cos(theta);
            out[2 * i + 1] = r * std::sin(theta);
        }
    }
};
