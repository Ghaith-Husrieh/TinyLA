#include "alignment.h"
#include <stdlib.h>

#ifdef _MSC_VER
#include <malloc.h>
#endif

bool is_aligned(void* ptr, size_t alignment) { return (size_t)ptr % alignment == 0; }

void* tla_aligned_malloc(size_t bytes, size_t alignment) {
    void* ptr = NULL;

#ifdef _MSC_VER
    ptr = _aligned_malloc(bytes, alignment);
#elif defined(HAVE_POSIX_MEMALIGN)
    if (posix_memalign(&ptr, alignment, bytes) != 0) {
        ptr = NULL;
    }
#else
    // NOTE: C11 fallback, aligned_alloc requires bytes to be a multiple of alignment
    if (bytes % alignment != 0) {
        bytes += alignment - (bytes % alignment);
    }
    ptr = aligned_alloc(alignment, bytes);
#endif

    return ptr;
}

void tla_aligned_free(void* ptr) {
#ifdef _MSC_VER
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}
