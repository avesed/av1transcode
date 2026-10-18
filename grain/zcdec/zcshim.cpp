// zcshim: the native half of grain_apply's zero-copy reads. Host code only, built with the image's g++ against the
// SYCL headers of torch's own runtime (/opt/cvvdp/include), so the context below is the one torch allocates in.
//   zc_native   torch's sycl::queue* (torch.xpu.current_stream().sycl_queue) -> its Level Zero context and device
//   zc_import   a DMA-BUF fd -> device memory in that context (Level Zero external memory import, no copy)
//   zc_free     releases an import
//   zc_dlpack   a DLPack tensor (kDLOneAPI, uint8, 1-D) over an import, for torch.from_dlpack; it does not own the
//               memory (zc_free does), its deleter only frees the descriptor
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

extern "C" {

typedef struct { int32_t device_type; int32_t device_id; } DLDevice;
typedef struct { uint8_t code; uint8_t bits; uint16_t lanes; } DLDataType;
typedef struct {
    void *data;
    DLDevice device;
    int32_t ndim;
    DLDataType dtype;
    int64_t *shape;
    int64_t *strides;
    uint64_t byte_offset;
} DLTensor;
typedef struct DLManagedTensor {
    DLTensor dl_tensor;
    void *manager_ctx;
    void (*deleter)(struct DLManagedTensor *self);
} DLManagedTensor;

struct zc_dl {
    DLManagedTensor t;
    int64_t shape[1];
};

static void zc_dl_delete(DLManagedTensor *self) { free(self); }

int zc_native(void *queue, void **ctx, void **dev) {
    try {
        auto *q = static_cast<sycl::queue *>(queue);
        *ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q->get_context());
        *dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q->get_device());
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "zc_native: %s\n", e.what());
        return -1;
    }
}

int zc_import(void *ctx, void *dev, int fd, size_t size, void **ptr) {
    ze_external_memory_import_fd_t imp = {};
    imp.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD;
    imp.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF;
    imp.fd = fd;
    ze_device_mem_alloc_desc_t d = {};
    d.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    d.pNext = &imp;
    *ptr = nullptr;
    return (int)zeMemAllocDevice((ze_context_handle_t)ctx, &d, size, 0, (ze_device_handle_t)dev, ptr);
}

int zc_free(void *ctx, void *ptr) { return (int)zeMemFree((ze_context_handle_t)ctx, ptr); }

void *zc_dlpack(void *ptr, int64_t nbytes, int32_t device_id) {
    auto *m = static_cast<zc_dl *>(calloc(1, sizeof(zc_dl)));
    if (!m)
        return nullptr;
    m->shape[0] = nbytes;
    m->t.dl_tensor.data = ptr;
    m->t.dl_tensor.device = {14 /* kDLOneAPI */, device_id};
    m->t.dl_tensor.ndim = 1;
    m->t.dl_tensor.dtype = {1 /* kDLUInt */, 8, 1};
    m->t.dl_tensor.shape = m->shape;
    m->t.deleter = zc_dl_delete;
    return &m->t;
}
}
