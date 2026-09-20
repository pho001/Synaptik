#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define SYNAPTIK_EXPORT __attribute__((visibility("default")))

enum {
    SYNAPTIK_METAL_STATUS_OK = 0,
    SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT = 1,
    SYNAPTIK_METAL_STATUS_NO_DEVICE = 2,
    SYNAPTIK_METAL_STATUS_NO_COMMAND_QUEUE = 3,
    SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED = 4,
    SYNAPTIK_METAL_STATUS_RANGE_OUT_OF_BOUNDS = 5,
    SYNAPTIK_METAL_STATUS_COPY_FAILED = 6,
    SYNAPTIK_METAL_STATUS_INTERNAL_ERROR = 7
};

@interface SynaptikMetalContextBox : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> commandQueue;
@end

@implementation SynaptikMetalContextBox
@end

@interface SynaptikMetalBufferBox : NSObject
@property(nonatomic, strong) id<MTLBuffer> buffer;
@property(nonatomic) uint64_t logicalByteSize;
@end

@implementation SynaptikMetalBufferBox
@end

SYNAPTIK_EXPORT uint32_t synaptik_metal_foundation_abi_version(void) {
    return 1U;
}

SYNAPTIK_EXPORT int32_t synaptik_metal_context_create(void **out_context) {
    if (out_context == NULL) {
        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    }
    *out_context = NULL;
    @try {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return SYNAPTIK_METAL_STATUS_NO_DEVICE;
        }
        id<MTLCommandQueue> command_queue = [device newCommandQueue];
        if (command_queue == nil) {
            return SYNAPTIK_METAL_STATUS_NO_COMMAND_QUEUE;
        }
        SynaptikMetalContextBox *box = [SynaptikMetalContextBox new];
        if (box == nil) {
            return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
        }
        box.device = device;
        box.commandQueue = command_queue;
        *out_context = (__bridge_retained void *)box;
        return SYNAPTIK_METAL_STATUS_OK;
    } @catch (__unused NSException *exception) {
        return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
    }
}

SYNAPTIK_EXPORT int32_t synaptik_metal_context_release(void *context) {
    if (context == NULL) {
        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    }
    @try {
        __unused id consumed_context = (__bridge_transfer id)context;
        return SYNAPTIK_METAL_STATUS_OK;
    } @catch (__unused NSException *exception) {
        return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
    }
}

SYNAPTIK_EXPORT int32_t synaptik_metal_buffer_create(
        void *context,
        uint64_t logical_byte_size,
        void **out_buffer) {
    if (out_buffer == NULL) {
        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    }
    *out_buffer = NULL;
    if (context == NULL) {
        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    }
    @try {
        SynaptikMetalContextBox *context_box =
                (__bridge SynaptikMetalContextBox *)context;
        uint64_t physical_byte_size = logical_byte_size == 0U ? 1U : logical_byte_size;
        if (physical_byte_size > (uint64_t)NSUIntegerMax) {
            return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
        }
        id<MTLBuffer> buffer = [context_box.device
                newBufferWithLength:(NSUInteger)physical_byte_size
                            options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
        }
        SynaptikMetalBufferBox *box = [SynaptikMetalBufferBox new];
        if (box == nil) {
            return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
        }
        box.buffer = buffer;
        box.logicalByteSize = logical_byte_size;
        *out_buffer = (__bridge_retained void *)box;
        return SYNAPTIK_METAL_STATUS_OK;
    } @catch (__unused NSException *exception) {
        return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
    }
}

SYNAPTIK_EXPORT int32_t synaptik_metal_buffer_release(void *buffer) {
    if (buffer == NULL) {
        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    }
    @try {
        __unused id consumed_buffer = (__bridge_transfer id)buffer;
        return SYNAPTIK_METAL_STATUS_OK;
    } @catch (__unused NSException *exception) {
        return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
    }
}

static int32_t synaptik_metal_validate_copy(
        SynaptikMetalBufferBox *box,
        uint64_t buffer_offset,
        const void *bytes,
        uint64_t byte_count) {
    if (box == nil || (byte_count != 0U && bytes == NULL)) {
        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    }
    uint64_t logical_size = box.logicalByteSize;
    if (buffer_offset > logical_size || byte_count > logical_size - buffer_offset) {
        return SYNAPTIK_METAL_STATUS_RANGE_OUT_OF_BOUNDS;
    }
    return SYNAPTIK_METAL_STATUS_OK;
}

SYNAPTIK_EXPORT int32_t synaptik_metal_buffer_upload(
        void *buffer,
        uint64_t buffer_offset,
        const void *source,
        uint64_t byte_count) {
    if (buffer == NULL) {
        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    }
    @try {
        SynaptikMetalBufferBox *box = (__bridge SynaptikMetalBufferBox *)buffer;
        int32_t validation =
                synaptik_metal_validate_copy(box, buffer_offset, source, byte_count);
        if (validation != SYNAPTIK_METAL_STATUS_OK || byte_count == 0U) {
            return validation;
        }
        void *contents = box.buffer.contents;
        if (contents == NULL) {
            return SYNAPTIK_METAL_STATUS_COPY_FAILED;
        }
        memcpy((uint8_t *)contents + (size_t)buffer_offset, source, (size_t)byte_count);
        return SYNAPTIK_METAL_STATUS_OK;
    } @catch (__unused NSException *exception) {
        return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
    }
}

SYNAPTIK_EXPORT int32_t synaptik_metal_buffer_download(
        void *buffer,
        uint64_t buffer_offset,
        void *destination,
        uint64_t byte_count) {
    if (buffer == NULL) {
        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    }
    @try {
        SynaptikMetalBufferBox *box = (__bridge SynaptikMetalBufferBox *)buffer;
        int32_t validation =
                synaptik_metal_validate_copy(box, buffer_offset, destination, byte_count);
        if (validation != SYNAPTIK_METAL_STATUS_OK || byte_count == 0U) {
            return validation;
        }
        void *contents = box.buffer.contents;
        if (contents == NULL) {
            return SYNAPTIK_METAL_STATUS_COPY_FAILED;
        }
        memcpy(destination, (uint8_t *)contents + (size_t)buffer_offset, (size_t)byte_count);
        return SYNAPTIK_METAL_STATUS_OK;
    } @catch (__unused NSException *exception) {
        return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
    }
}
