#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

struct alignas(8) BinaryMeta {
    uint64_t dims[16]{};
    uint64_t leftStrides[16]{};
    uint64_t rightStrides[16]{};
    uint64_t elementCount{};
    uint64_t gridWidth{};
    uint64_t gridHeight{};
    uint32_t rank{};
    uint32_t reserved{};
};

struct alignas(8) PointMeta {
    uint64_t elementCount{};
    uint64_t gridWidth{};
    uint64_t gridHeight{};
    uint32_t scalar{};
    uint32_t reserved{};
};

struct alignas(8) ClampMeta {
    uint64_t elementCount{};
    uint64_t gridWidth{};
    uint64_t gridHeight{};
    uint32_t lower{};
    uint32_t upper{};
};

struct alignas(8) ReductionMeta {
    uint64_t dims[16]{};
    uint64_t strides[16]{};
    uint64_t outputCount{};
    uint64_t reducedCount{};
    uint64_t gridWidth{};
    uint64_t gridHeight{};
    uint32_t rank{};
    uint32_t axesMask{};
};

struct alignas(8) ScanMeta {
    uint64_t elementCount{};
    uint64_t axisLength{};
    uint64_t inner{};
    uint64_t lineCount{};
    uint64_t gridWidth{};
    uint64_t gridHeight{};
    uint32_t exclusive{};
    uint32_t reverse{};
};

static_assert(sizeof(BinaryMeta) == 416);
static_assert(sizeof(PointMeta) == 32);
static_assert(sizeof(ClampMeta) == 32);
static_assert(sizeof(ReductionMeta) == 296);
static_assert(sizeof(ScanMeta) == 56);

struct InputRecord {
    std::string name;
    id<MTLBuffer> buffer;
    std::vector<uint8_t> original;
};

struct OutputRecord {
    std::string role;
    std::string name;
    std::string shape;
    id<MTLBuffer> buffer;
    size_t count;
    size_t offset;
    bool boolean;
};

static id<MTLBuffer> makeBuffer(id<MTLDevice> device, const void *bytes, size_t length,
                                NSMutableArray<id<MTLBuffer>> *keepers) {
    id<MTLBuffer> buffer = [device newBufferWithBytes:bytes
                                               length:length
                                              options:MTLResourceStorageModeShared];
    if (buffer == nil) throw std::runtime_error("buffer allocation failed");
    [keepers addObject:buffer];
    return buffer;
}

static id<MTLBuffer> makeInput(id<MTLDevice> device, const std::string &name,
                               const std::vector<uint32_t> &words,
                               NSMutableArray<id<MTLBuffer>> *keepers,
                               std::vector<InputRecord> &inputs) {
    std::vector<uint8_t> bytes(words.size() * sizeof(uint32_t));
    std::memcpy(bytes.data(), words.data(), bytes.size());
    id<MTLBuffer> buffer = makeBuffer(device, bytes.data(), bytes.size(), keepers);
    inputs.push_back(InputRecord{name, buffer, bytes});
    return buffer;
}

static OutputRecord makeOutput32(id<MTLDevice> device, const std::string &role,
                                 const std::string &name, const std::string &shape, size_t count,
                                 NSMutableArray<id<MTLBuffer>> *keepers,
                                 std::vector<OutputRecord> &outputs) {
    std::vector<uint32_t> words(count + 2, 0xa5a5a5a5u);
    words.front() = 0x13579bdfu;
    words.back() = 0x2468ace0u;
    id<MTLBuffer> buffer = makeBuffer(device, words.data(), words.size() * sizeof(uint32_t), keepers);
    OutputRecord record{role, name, shape, buffer, count, sizeof(uint32_t), false};
    outputs.push_back(record);
    return record;
}

static OutputRecord makeOutput8(id<MTLDevice> device, const std::string &name,
                                const std::string &shape, size_t count,
                                NSMutableArray<id<MTLBuffer>> *keepers,
                                std::vector<OutputRecord> &outputs) {
    constexpr size_t offset = 16;
    std::vector<uint8_t> bytes(offset + count + 1, 0xccu);
    bytes[offset - 1] = 0x5au;
    bytes[offset + count] = 0xa5u;
    id<MTLBuffer> buffer = makeBuffer(device, bytes.data(), bytes.size(), keepers);
    OutputRecord record{"OUT8", name, shape, buffer, count, offset, true};
    outputs.push_back(record);
    return record;
}

static id<MTLComputePipelineState> pipeline(NSDictionary<NSString *, id<MTLComputePipelineState>> *pipes,
                                             const std::string &name) {
    NSString *key = [NSString stringWithUTF8String:name.c_str()];
    id<MTLComputePipelineState> value = pipes[key];
    if (value == nil) throw std::runtime_error("missing pipeline " + name);
    return value;
}

static MTLSize threadsFor(id<MTLComputePipelineState> state) {
    NSUInteger width = std::min<NSUInteger>(state.maxTotalThreadsPerThreadgroup, 128);
    return MTLSizeMake(std::max<NSUInteger>(width, 1), 1, 1);
}

static MTLSize gridFor(uint64_t count) {
    if (count == 0) throw std::runtime_error("zero-sized dispatch");
    uint64_t width = std::min<uint64_t>(count, UINT32_MAX);
    uint64_t remaining = count / width + (count % width != 0);
    uint64_t height = std::min<uint64_t>(remaining, UINT32_MAX);
    uint64_t depth = remaining / height + (remaining % height != 0);
    if (depth > UINT32_MAX) throw std::runtime_error("dispatch exceeds uint3 domain");
    return MTLSizeMake(static_cast<NSUInteger>(width), static_cast<NSUInteger>(height),
                       static_cast<NSUInteger>(depth));
}

static void setGrid(uint64_t count, uint64_t &width, uint64_t &height) {
    MTLSize grid = gridFor(count);
    width = grid.width;
    height = grid.height;
}

static uint64_t checkedMultiply(uint64_t left, uint64_t right) {
    if (left != 0 && right > UINT64_MAX / left)
        throw std::runtime_error("metadata product overflow");
    return left * right;
}

static BinaryMeta binaryMeta(const std::vector<uint64_t> &dims,
                             const std::vector<uint64_t> &leftStrides,
                             const std::vector<uint64_t> &rightStrides) {
    if (dims.empty() || dims.size() > 16 || leftStrides.size() != dims.size()
            || rightStrides.size() != dims.size()) throw std::runtime_error("invalid binary metadata");
    BinaryMeta meta{};
    meta.rank = static_cast<uint32_t>(dims.size());
    meta.elementCount = 1;
    for (size_t i = 0; i < dims.size(); ++i) {
        if (dims[i] == 0) throw std::runtime_error("zero binary dimension");
        meta.dims[i] = dims[i];
        meta.leftStrides[i] = leftStrides[i];
        meta.rightStrides[i] = rightStrides[i];
        meta.elementCount = checkedMultiply(meta.elementCount, dims[i]);
    }
    setGrid(meta.elementCount, meta.gridWidth, meta.gridHeight);
    return meta;
}

static ReductionMeta reductionMeta(const std::vector<uint64_t> &dims, uint32_t axesMask) {
    if (dims.empty() || dims.size() > 16 || (axesMask >> dims.size()) != 0u)
        throw std::runtime_error("invalid reduction metadata");
    ReductionMeta meta{};
    meta.rank = static_cast<uint32_t>(dims.size());
    meta.axesMask = axesMask;
    uint64_t stride = 1;
    meta.outputCount = 1;
    meta.reducedCount = 1;
    for (size_t reverse = 0; reverse < dims.size(); ++reverse) {
        size_t axis = dims.size() - 1 - reverse;
        if (dims[axis] == 0) throw std::runtime_error("zero reduction dimension");
        meta.dims[axis] = dims[axis];
        meta.strides[axis] = stride;
        stride = checkedMultiply(stride, dims[axis]);
        if ((axesMask & (1u << axis)) != 0u)
            meta.reducedCount = checkedMultiply(meta.reducedCount, dims[axis]);
        else
            meta.outputCount = checkedMultiply(meta.outputCount, dims[axis]);
    }
    setGrid(meta.outputCount, meta.gridWidth, meta.gridHeight);
    return meta;
}

static ScanMeta scanMeta(const std::vector<uint64_t> &dims, uint32_t axis,
                         bool exclusive, bool reverse) {
    if (dims.empty() || dims.size() > 16 || axis >= dims.size())
        throw std::runtime_error("invalid scan metadata");
    ScanMeta meta{};
    meta.elementCount = 1;
    for (uint64_t dim : dims) {
        if (dim == 0) throw std::runtime_error("zero scan dimension");
        meta.elementCount = checkedMultiply(meta.elementCount, dim);
    }
    meta.axisLength = dims[axis];
    meta.inner = 1;
    for (size_t i = axis + 1; i < dims.size(); ++i)
        meta.inner = checkedMultiply(meta.inner, dims[i]);
    meta.lineCount = meta.elementCount / meta.axisLength;
    setGrid(meta.lineCount, meta.gridWidth, meta.gridHeight);
    meta.exclusive = exclusive ? 1u : 0u;
    meta.reverse = reverse ? 1u : 0u;
    return meta;
}

static void encodeBinary(id<MTLComputeCommandEncoder> encoder,
                         NSDictionary<NSString *, id<MTLComputePipelineState>> *pipes,
                         const std::string &function, id<MTLBuffer> left, NSUInteger leftOffset,
                         id<MTLBuffer> right, NSUInteger rightOffset,
                         const OutputRecord &output, const BinaryMeta &meta) {
    id<MTLComputePipelineState> state = pipeline(pipes, function);
    [encoder setComputePipelineState:state];
    [encoder setBuffer:left offset:leftOffset atIndex:0];
    [encoder setBuffer:right offset:rightOffset atIndex:1];
    [encoder setBuffer:output.buffer offset:output.offset atIndex:2];
    [encoder setBytes:&meta length:sizeof(meta) atIndex:3];
    [encoder dispatchThreads:gridFor(meta.elementCount) threadsPerThreadgroup:threadsFor(state)];
}

static void encodeScalar(id<MTLComputeCommandEncoder> encoder,
                         NSDictionary<NSString *, id<MTLComputePipelineState>> *pipes,
                         const std::string &function, id<MTLBuffer> input, NSUInteger inputOffset,
                         const OutputRecord &output, uint32_t scalar, uint64_t count) {
    id<MTLComputePipelineState> state = pipeline(pipes, function);
    MTLSize grid = gridFor(count);
    PointMeta meta{count, grid.width, grid.height, scalar, 0};
    [encoder setComputePipelineState:state];
    [encoder setBuffer:input offset:inputOffset atIndex:0];
    [encoder setBuffer:output.buffer offset:output.offset atIndex:1];
    [encoder setBytes:&meta length:sizeof(meta) atIndex:2];
    [encoder dispatchThreads:grid threadsPerThreadgroup:threadsFor(state)];
}

static void encodeClamp(id<MTLComputeCommandEncoder> encoder,
                        NSDictionary<NSString *, id<MTLComputePipelineState>> *pipes,
                        id<MTLBuffer> input, const OutputRecord &output,
                        uint32_t lower, uint32_t upper, uint64_t count) {
    id<MTLComputePipelineState> state = pipeline(pipes, "clamp_fused");
    MTLSize grid = gridFor(count);
    ClampMeta meta{count, grid.width, grid.height, lower, upper};
    [encoder setComputePipelineState:state];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:output.buffer offset:output.offset atIndex:1];
    [encoder setBytes:&meta length:sizeof(meta) atIndex:2];
    [encoder dispatchThreads:grid threadsPerThreadgroup:threadsFor(state)];
}

static void encodeReduction(id<MTLComputeCommandEncoder> encoder,
                            NSDictionary<NSString *, id<MTLComputePipelineState>> *pipes,
                            const std::string &function, id<MTLBuffer> input,
                            const OutputRecord &output, const ReductionMeta &meta) {
    id<MTLComputePipelineState> state = pipeline(pipes, function);
    [encoder setComputePipelineState:state];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:output.buffer offset:output.offset atIndex:1];
    [encoder setBytes:&meta length:sizeof(meta) atIndex:2];
    [encoder dispatchThreads:gridFor(meta.outputCount) threadsPerThreadgroup:threadsFor(state)];
}

static void encodeScan(id<MTLComputeCommandEncoder> encoder,
                       NSDictionary<NSString *, id<MTLComputePipelineState>> *pipes,
                       const std::string &function, id<MTLBuffer> input,
                       const OutputRecord &output, const ScanMeta &meta) {
    id<MTLComputePipelineState> state = pipeline(pipes, function);
    [encoder setComputePipelineState:state];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:output.buffer offset:output.offset atIndex:1];
    [encoder setBytes:&meta length:sizeof(meta) atIndex:2];
    [encoder dispatchThreads:gridFor(meta.lineCount) threadsPerThreadgroup:threadsFor(state)];
}

static std::string hex32(uint32_t value) {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(8) << value;
    return out.str();
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        try {
            if (argc != 2) throw std::runtime_error("usage: oracle <metal-source>");
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (device == nil) throw std::runtime_error("no default Metal device");
            NSError *error = nil;
            NSString *source = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]
                                                           encoding:NSUTF8StringEncoding
                                                              error:&error];
            if (source == nil) throw std::runtime_error([[error description] UTF8String]);
            MTLCompileOptions *options = [MTLCompileOptions new];
            options.mathMode = MTLMathModeSafe;
            id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
            if (library == nil) throw std::runtime_error([[error description] UTF8String]);

            NSArray<NSString *> *functionNames = @[
                @"cmp_gt", @"cmp_ge", @"cmp_lt", @"cmp_le", @"cmp_eq", @"cmp_ne",
                @"tensor_min", @"tensor_max", @"scalar_min", @"scalar_max", @"clamp_fused",
                @"reduction_min", @"reduction_max", @"scan_sum", @"scan_prod"
            ];
            NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pipes = [NSMutableDictionary dictionary];
            for (NSString *name in functionNames) {
                id<MTLFunction> function = [library newFunctionWithName:name];
                if (function == nil) throw std::runtime_error("missing compiled Metal function");
                id<MTLComputePipelineState> state = [device newComputePipelineStateWithFunction:function error:&error];
                if (state == nil) throw std::runtime_error([[error description] UTF8String]);
                pipes[name] = state;
            }

            id<MTLCommandQueue> queue = [device newCommandQueue];
            if (queue == nil) throw std::runtime_error("command queue creation failed");
            id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            if (commandBuffer == nil || encoder == nil) throw std::runtime_error("command creation failed");

            NSMutableArray<id<MTLBuffer>> *keepers = [NSMutableArray array];
            std::vector<InputRecord> inputs;
            std::vector<OutputRecord> outputs;

            const std::vector<uint32_t> pairLeft = {
                0x00000000u,0x80000000u,0x00000001u,0x00000000u,0x80000001u,0x80000000u,
                0x007fffffu,0x00800000u,0x807fffffu,0x80800000u,0x3f800000u,0x3f800000u,
                0xbf800000u,0x7f7fffffu,0xff7fffffu,0x7f800000u,0x7fc12345u,0x3f800000u,
                0x7fa12345u,0xbf800000u,0xffc54321u,0x00000000u,0x7fc12345u,0x40000000u};
            const std::vector<uint32_t> pairRight = {
                0x80000000u,0x00000000u,0x00000000u,0x00000001u,0x80000000u,0x80000001u,
                0x00800000u,0x007fffffu,0x80800000u,0x807fffffu,0x3f800000u,0x3f800001u,
                0xbf7fffffu,0x7f800000u,0xff800000u,0xff800000u,0x3f800000u,0x7fc12345u,
                0xbf800000u,0x7fa12345u,0x00000000u,0xffc54321u,0xffc54321u,0xc0000000u};
            id<MTLBuffer> pairLeftBuffer = makeInput(device, "pair_left", pairLeft, keepers, inputs);
            id<MTLBuffer> pairRightBuffer = makeInput(device, "pair_right", pairRight, keepers, inputs);
            BinaryMeta pairMeta = binaryMeta({24}, {1}, {1});

            const std::vector<uint32_t> broadcastLeft = {
                0x3f800000u,0x40000000u,0x40400000u,0xbf800000u,0xc0000000u,0xc0400000u};
            const std::vector<uint32_t> broadcastRight = {
                0x00000000u,0x3f800000u,0x40800000u,0x40000000u,0x40000000u,0x40000000u,
                0xc0000000u,0xbf800000u,0x00000000u,0x41200000u,0xc1200000u,0x3f000000u};
            id<MTLBuffer> broadcastLeftBuffer = makeInput(device, "broadcast_left", broadcastLeft, keepers, inputs);
            id<MTLBuffer> broadcastRightBuffer = makeInput(device, "broadcast_right", broadcastRight, keepers, inputs);
            BinaryMeta broadcastMeta = binaryMeta({2,4,3}, {3,0,1}, {0,3,1});

            const std::array<std::string,6> cmpFunctions = {"cmp_gt","cmp_ge","cmp_lt","cmp_le","cmp_eq","cmp_ne"};
            for (const std::string &function : cmpFunctions) {
                OutputRecord pairs = makeOutput8(device, function + "_pairs", "[24]", 24, keepers, outputs);
                encodeBinary(encoder, pipes, function, pairLeftBuffer, 0, pairRightBuffer, 0, pairs, pairMeta);
                OutputRecord broad = makeOutput8(device, function + "_broadcast", "[2,4,3]", 24, keepers, outputs);
                encodeBinary(encoder, pipes, function, broadcastLeftBuffer, 0, broadcastRightBuffer, 0, broad, broadcastMeta);
            }
            for (const std::string &function : {std::string("tensor_min"), std::string("tensor_max")}) {
                OutputRecord pairs = makeOutput32(device, "OUT32", function + "_pairs", "[24]", 24, keepers, outputs);
                encodeBinary(encoder, pipes, function, pairLeftBuffer, 0, pairRightBuffer, 0, pairs, pairMeta);
                OutputRecord broad = makeOutput32(device, "OUT32", function + "_broadcast", "[2,4,3]", 24, keepers, outputs);
                encodeBinary(encoder, pipes, function, broadcastLeftBuffer, 0, broadcastRightBuffer, 0, broad, broadcastMeta);
            }

            std::vector<uint32_t> uniqueWords;
            std::set<uint32_t> seen;
            for (size_t i = 0; i < pairLeft.size(); ++i) {
                if (seen.insert(pairLeft[i]).second) uniqueWords.push_back(pairLeft[i]);
                if (seen.insert(pairRight[i]).second) uniqueWords.push_back(pairRight[i]);
            }
            id<MTLBuffer> uniqueBuffer = makeInput(device, "pointwise_unique", uniqueWords, keepers, inputs);
            const std::vector<uint32_t> scalars = {
                0x00000000u,0x80000000u,0x00000001u,0x80000001u,
                0x3f800000u,0xbf800000u,0x7f800000u,0x7fc12345u};
            for (uint32_t scalar : scalars) {
                for (const std::string &function : {std::string("scalar_min"), std::string("scalar_max")}) {
                    std::string name = function + "_s" + hex32(scalar);
                    OutputRecord output = makeOutput32(device, "OUT32", name, "[unique]", uniqueWords.size(), keepers, outputs);
                    encodeScalar(encoder, pipes, function, uniqueBuffer, 0, output, scalar, uniqueWords.size());
                }
            }

            const std::vector<std::pair<uint32_t,uint32_t>> clampBounds = {
                {0xbf800000u,0x3f800000u},{0x80000000u,0x00000000u},
                {0x00000000u,0x80000000u},{0x00000001u,0x3f800000u},
                {0xbf800000u,0x80000001u},{0x7fc12345u,0x3f800000u},
                {0xbf800000u,0x7fc12345u},{0x7f800000u,0x7f800000u}};
            for (auto [lower, upper] : clampBounds) {
                std::string suffix = "_lo" + hex32(lower) + "_hi" + hex32(upper);
                OutputRecord fused = makeOutput32(device, "OUT32", "clamp_fused" + suffix, "[unique]", uniqueWords.size(), keepers, outputs);
                encodeClamp(encoder, pipes, uniqueBuffer, fused, lower, upper, uniqueWords.size());
                OutputRecord middle = makeOutput32(device, "MID32", "clamp_composed_middle" + suffix, "[unique]", uniqueWords.size(), keepers, outputs);
                encodeScalar(encoder, pipes, "scalar_max", uniqueBuffer, 0, middle, lower, uniqueWords.size());
                OutputRecord composed = makeOutput32(device, "OUT32", "clamp_composed" + suffix, "[unique]", uniqueWords.size(), keepers, outputs);
                encodeScalar(encoder, pipes, "scalar_min", middle.buffer, middle.offset, composed, upper, uniqueWords.size());
            }

            const std::vector<uint32_t> aggregateWords = {
                0x00000000u,0x80000000u,0x00000001u,0x80000001u,0x00800000u,0x80800000u,0x3f800000u,0xbf800000u,
                0x80000000u,0x00000000u,0x007fffffu,0x807fffffu,0x7f7fffffu,0xff7fffffu,0x7f800000u,0xff800000u,
                0x7fc12345u,0x3f800000u,0xbf800000u,0x7fa12345u,0x00000000u,0x80000000u,0x40000000u,0xc0000000u,
                0xbf800000u,0x3f800000u,0x80800000u,0x00800000u,0x80000001u,0x00000001u,0x80000000u,0x00000000u,
                0xff800000u,0x7f800000u,0xff7fffffu,0x7f7fffffu,0x807fffffu,0x007fffffu,0x00000000u,0x80000000u,
                0xc0000000u,0x40000000u,0x80000000u,0x00000000u,0xffc54321u,0xbf800000u,0x3f800000u,0xffa54321u};
            id<MTLBuffer> aggregateBuffer = makeInput(device, "aggregate_scan", aggregateWords, keepers, inputs);
            struct ReductionForm { std::string name; uint32_t mask; std::string shape; };
            const std::vector<ReductionForm> forms = {
                {"full", 0x7u, "[]"}, {"axis1_drop", 0x2u, "[2,8]"},
                {"axis1_keep", 0x2u, "[2,1,8]"}, {"axes20_drop", 0x5u, "[3]"},
                {"axes20_keep", 0x5u, "[1,3,1]"}, {"identity", 0x0u, "[2,3,8]"}};
            for (const std::string &function : {std::string("reduction_min"), std::string("reduction_max")}) {
                for (const ReductionForm &form : forms) {
                    ReductionMeta meta = reductionMeta({2,3,8}, form.mask);
                    OutputRecord output = makeOutput32(device, "OUT32", function + "_base_" + form.name,
                                                       form.shape, meta.outputCount, keepers, outputs);
                    encodeReduction(encoder, pipes, function, aggregateBuffer, output, meta);
                }
            }

            const std::vector<std::pair<std::string,std::vector<uint32_t>>> reductionEdges = {
                {"zero_pos_neg", {0x00000000u,0x80000000u}},
                {"zero_neg_pos", {0x80000000u,0x00000000u}},
                {"single_pos_zero", {0x00000000u}}, {"single_neg_zero", {0x80000000u}},
                {"single_pos_sub", {0x00000001u}}, {"single_neg_sub", {0x80000001u}},
                {"single_pos_qnan", {0x7fc12345u}}, {"single_pos_snan", {0x7fa12345u}},
                {"single_neg_qnan", {0xffc54321u}}, {"single_neg_snan", {0xffa54321u}}};
            for (const auto &edge : reductionEdges) {
                id<MTLBuffer> input = makeInput(device, "reduction_" + edge.first, edge.second, keepers, inputs);
                ReductionMeta meta = reductionMeta({static_cast<uint64_t>(edge.second.size())}, 0x1u);
                for (const std::string &function : {std::string("reduction_min"), std::string("reduction_max")}) {
                    OutputRecord output = makeOutput32(device, "OUT32", function + "_" + edge.first,
                                                       "[]", 1, keepers, outputs);
                    encodeReduction(encoder, pipes, function, input, output, meta);
                }
            }

            struct ScanForm { std::string name; uint32_t axis; bool exclusive; bool reverse; };
            const std::vector<ScanForm> scanForms = {
                {"axis2_inc_fwd",2,false,false},{"axis2_exc_fwd",2,true,false},
                {"axis2_inc_rev",2,false,true},{"axis2_exc_rev",2,true,true},
                {"axis1_inc_fwd",1,false,false}};
            for (const std::string &function : {std::string("scan_sum"), std::string("scan_prod")}) {
                for (const ScanForm &form : scanForms) {
                    ScanMeta meta = scanMeta({2,3,8}, form.axis, form.exclusive, form.reverse);
                    OutputRecord output = makeOutput32(device, "OUT32", function + "_" + form.name,
                                                       "[2,3,8]", aggregateWords.size(), keepers, outputs);
                    encodeScan(encoder, pipes, function, aggregateBuffer, output, meta);
                }
            }

            [encoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            if (commandBuffer.status != MTLCommandBufferStatusCompleted)
                throw std::runtime_error([[commandBuffer.error description] UTF8String]);

            bool inputsPreserved = true;
            for (const InputRecord &input : inputs) {
                if (std::memcmp(input.original.data(), input.buffer.contents, input.original.size()) != 0)
                    inputsPreserved = false;
            }

            std::cout << "TASK=0052\n";
            std::cout << "DEVICE=" << [[device name] UTF8String] << "\n";
            std::cout << "DEVICE_CONTEXTS=1\n";
            std::cout << "COMMAND_BUFFERS=1\n";
            std::cout << "FAST_MATH=0\n";
            std::cout << "COMPILED_PIPELINES=" << functionNames.count << "\n";
            std::cout << "INPUT_COUNT=" << inputs.size() << "\n";
            std::cout << "OUTPUT_COUNT=" << outputs.size() << "\n";
            std::cout << "INPUTS_PRESERVED=" << (inputsPreserved ? "PASS" : "FAIL") << "\n";
            for (const OutputRecord &output : outputs) {
                bool canaryPass;
                std::cout << output.role << " name=" << output.name << " shape=" << output.shape
                          << " count=" << output.count << " values=";
                if (output.boolean) {
                    const uint8_t *bytes = static_cast<const uint8_t *>(output.buffer.contents);
                    canaryPass = bytes[output.offset - 1] == 0x5au
                              && bytes[output.offset + output.count] == 0xa5u;
                    for (size_t i = 0; i < output.count; ++i) {
                        if (i != 0) std::cout << ',';
                        std::cout << std::hex << std::setfill('0') << std::setw(2)
                                  << unsigned(bytes[output.offset + i]);
                    }
                } else {
                    const uint32_t *words = static_cast<const uint32_t *>(output.buffer.contents);
                    size_t base = output.offset / sizeof(uint32_t);
                    canaryPass = words[base - 1] == 0x13579bdfu
                              && words[base + output.count] == 0x2468ace0u;
                    for (size_t i = 0; i < output.count; ++i) {
                        if (i != 0) std::cout << ',';
                        std::cout << std::hex << std::setfill('0') << std::setw(8) << words[base + i];
                    }
                }
                std::cout << std::dec << " canary=" << (canaryPass ? "PASS" : "FAIL") << "\n";
            }
            std::cout << "ORACLE_PROCESS=PASS\n";
            return inputsPreserved ? 0 : 3;
        } catch (const std::exception &failure) {
            std::cerr << "ORACLE_PROCESS=FAIL reason=" << failure.what() << "\n";
            return 2;
        }
    }
}
