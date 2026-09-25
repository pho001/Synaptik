#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
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

static constexpr uint64_t POINT_COUNT = 1'048'576;
static constexpr uint64_t REDUCTION_OUTPUT_COUNT = 128;
static constexpr uint64_t SCAN_LINE_COUNT = 1024;
static constexpr uint64_t BATCH_FLOOR_NS = 25'000'000;
static constexpr uint64_t BATCH_LIMIT = 1'000'000;

static uint64_t checkedMultiply(uint64_t left, uint64_t right) {
    if (left != 0 && right > UINT64_MAX / left) throw std::runtime_error("metadata overflow");
    return left * right;
}

static MTLSize gridFor(uint64_t count) {
    if (count == 0) throw std::runtime_error("zero dispatch");
    uint64_t width = std::min<uint64_t>(count, UINT32_MAX);
    uint64_t remaining = count / width + (count % width != 0);
    uint64_t height = std::min<uint64_t>(remaining, UINT32_MAX);
    uint64_t depth = remaining / height + (remaining % height != 0);
    if (depth > UINT32_MAX) throw std::runtime_error("dispatch exceeds uint3");
    return MTLSizeMake(static_cast<NSUInteger>(width), static_cast<NSUInteger>(height),
                       static_cast<NSUInteger>(depth));
}

static void setGrid(uint64_t count, uint64_t &width, uint64_t &height) {
    MTLSize grid = gridFor(count);
    width = grid.width;
    height = grid.height;
}

static BinaryMeta binaryMeta(uint64_t count) {
    BinaryMeta meta{};
    meta.dims[0] = count;
    meta.leftStrides[0] = 1;
    meta.rightStrides[0] = 1;
    meta.elementCount = count;
    meta.rank = 1;
    setGrid(count, meta.gridWidth, meta.gridHeight);
    return meta;
}

static ReductionMeta reductionMeta() {
    const std::array<uint64_t, 3> dims = {64, 128, 128};
    ReductionMeta meta{};
    meta.rank = 3;
    meta.axesMask = 0x5u;
    uint64_t stride = 1;
    meta.outputCount = 1;
    meta.reducedCount = 1;
    for (size_t reverse = 0; reverse < dims.size(); ++reverse) {
        size_t axis = dims.size() - 1 - reverse;
        meta.dims[axis] = dims[axis];
        meta.strides[axis] = stride;
        stride = checkedMultiply(stride, dims[axis]);
        if ((meta.axesMask & (1u << axis)) != 0u)
            meta.reducedCount = checkedMultiply(meta.reducedCount, dims[axis]);
        else
            meta.outputCount = checkedMultiply(meta.outputCount, dims[axis]);
    }
    setGrid(meta.outputCount, meta.gridWidth, meta.gridHeight);
    return meta;
}

static ScanMeta scanMeta() {
    ScanMeta meta{};
    meta.elementCount = POINT_COUNT;
    meta.axisLength = 1024;
    meta.inner = 1;
    meta.lineCount = SCAN_LINE_COUNT;
    meta.exclusive = 1;
    meta.reverse = 1;
    setGrid(meta.lineCount, meta.gridWidth, meta.gridHeight);
    return meta;
}

static id<MTLBuffer> bufferWithBytes(id<MTLDevice> device, const void *bytes, size_t length) {
    id<MTLBuffer> result = [device newBufferWithBytes:bytes
                                               length:length
                                              options:MTLResourceStorageModeShared];
    if (result == nil) throw std::runtime_error("buffer allocation failed");
    return result;
}

static id<MTLBuffer> emptyBuffer(id<MTLDevice> device, size_t length) {
    id<MTLBuffer> result = [device newBufferWithLength:length options:MTLResourceStorageModeShared];
    if (result == nil) throw std::runtime_error("buffer allocation failed");
    std::memset(result.contents, 0xa5, length);
    return result;
}

template <typename T>
static id<MTLBuffer> metadataBuffer(id<MTLDevice> device, const T &value) {
    return bufferWithBytes(device, &value, sizeof(value));
}

static bool nanBits(uint32_t value) {
    return (value & 0x7f800000u) == 0x7f800000u && (value & 0x007fffffu) != 0u;
}

static bool zeroBits(uint32_t value) {
    return (value & 0x7fffffffu) == 0u;
}

static uint32_t orderedKey(uint32_t value) {
    return (value & 0x80000000u) != 0u ? ~value : (value ^ 0x80000000u);
}

static bool exactEqual(uint32_t left, uint32_t right) {
    if (nanBits(left) || nanBits(right)) return false;
    if (zeroBits(left) && zeroBits(right)) return true;
    return left == right;
}

static bool exactLess(uint32_t left, uint32_t right) {
    if (nanBits(left) || nanBits(right) || exactEqual(left, right)) return false;
    return orderedKey(left) < orderedKey(right);
}

static bool exactCompare(uint32_t left, uint32_t right, uint32_t operation) {
    switch (operation) {
        case 0: return exactLess(right, left);
        case 1: return exactLess(right, left) || exactEqual(left, right);
        case 2: return exactLess(left, right);
        case 3: return exactLess(left, right) || exactEqual(left, right);
        case 4: return exactEqual(left, right);
        default: return !exactEqual(left, right);
    }
}

static uint32_t exactExtreme(uint32_t left, uint32_t right, bool maximum) {
    if (nanBits(left)) return left;
    if (nanBits(right)) return right;
    if (zeroBits(left) && zeroBits(right)) {
        return maximum ? ((left & right) & 0x80000000u) : ((left | right) & 0x80000000u);
    }
    if (exactEqual(left, right)) return left;
    bool leftLess = exactLess(left, right);
    return maximum ? (leftLess ? right : left) : (leftLess ? left : right);
}

static uint32_t floatBits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float bitsFloat(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint32_t exactScanSite(uint32_t left, uint32_t right, bool product) {
    volatile float l = bitsFloat(left);
    volatile float r = bitsFloat(right);
    volatile float result = product ? l * r : l + r;
    return floatBits(result);
}

static uint64_t reductionOffset(uint64_t outputLinear, uint64_t reducedLinear,
                                const ReductionMeta &meta) {
    uint64_t outputRemaining = outputLinear;
    uint64_t reducedRemaining = reducedLinear;
    uint64_t offset = 0;
    for (int axis = static_cast<int>(meta.rank) - 1; axis >= 0; --axis) {
        uint64_t coordinate;
        if ((meta.axesMask & (1u << static_cast<uint32_t>(axis))) != 0u) {
            coordinate = reducedRemaining % meta.dims[axis];
            reducedRemaining /= meta.dims[axis];
        } else {
            coordinate = outputRemaining % meta.dims[axis];
            outputRemaining /= meta.dims[axis];
        }
        offset += coordinate * meta.strides[axis];
    }
    return offset;
}

enum class RouteKind { BINARY, SCALAR, CLAMP_FUSED, CLAMP_COMPOSED, REDUCTION, SCAN };

struct Candidate {
    std::string name;
    std::string operation;
    RouteKind kind;
    id<MTLComputePipelineState> firstPipeline;
    id<MTLComputePipelineState> secondPipeline;
    id<MTLBuffer> firstInput;
    id<MTLBuffer> secondInput;
    id<MTLBuffer> output;
    id<MTLBuffer> intermediate;
    id<MTLBuffer> metadata;
    uint64_t gridCount;
    uint64_t outputCount;
    uint32_t operationCode;
    uint32_t scalar;
    uint32_t lower;
    uint32_t upper;
    uint32_t dispatches;
    uint64_t temporaryBytes;
    std::vector<double> retainedNs;
};

static MTLSize threadsFor(id<MTLComputePipelineState> state) {
    NSUInteger width = std::min<NSUInteger>(state.maxTotalThreadsPerThreadgroup, 128);
    return MTLSizeMake(std::max<NSUInteger>(width, 1), 1, 1);
}

static void encodeCandidate(id<MTLComputeCommandEncoder> encoder, const Candidate &candidate) {
    [encoder setComputePipelineState:candidate.firstPipeline];
    [encoder setBuffer:candidate.firstInput offset:0 atIndex:0];
    switch (candidate.kind) {
        case RouteKind::BINARY:
            [encoder setBuffer:candidate.secondInput offset:0 atIndex:1];
            [encoder setBuffer:candidate.output offset:0 atIndex:2];
            [encoder setBuffer:candidate.metadata offset:0 atIndex:3];
            break;
        case RouteKind::SCALAR:
        case RouteKind::CLAMP_FUSED:
        case RouteKind::REDUCTION:
        case RouteKind::SCAN:
            [encoder setBuffer:candidate.output offset:0 atIndex:1];
            [encoder setBuffer:candidate.metadata offset:0 atIndex:2];
            break;
        case RouteKind::CLAMP_COMPOSED:
            [encoder setBuffer:candidate.intermediate offset:0 atIndex:1];
            [encoder setBuffer:candidate.metadata offset:0 atIndex:2];
            break;
    }
    [encoder dispatchThreads:gridFor(candidate.gridCount)
       threadsPerThreadgroup:threadsFor(candidate.firstPipeline)];
    if (candidate.kind == RouteKind::CLAMP_COMPOSED) {
        [encoder setComputePipelineState:candidate.secondPipeline];
        [encoder setBuffer:candidate.intermediate offset:0 atIndex:0];
        [encoder setBuffer:candidate.output offset:0 atIndex:1];
        [encoder setBuffer:candidate.secondInput offset:0 atIndex:2];
        [encoder dispatchThreads:gridFor(candidate.gridCount)
           threadsPerThreadgroup:threadsFor(candidate.secondPipeline)];
    }
}

static void executeOnce(id<MTLCommandQueue> queue, const Candidate &candidate) {
    @autoreleasepool {
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (command == nil || encoder == nil) throw std::runtime_error("command creation failed");
        encodeCandidate(encoder, candidate);
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            std::string message = command.error == nil ? "unknown Metal execution failure"
                : [[command.error description] UTF8String];
            throw std::runtime_error(message);
        }
    }
}

static void requireWords(const Candidate &candidate, const std::vector<uint32_t> &expected) {
    const uint32_t *actual = static_cast<const uint32_t *>(candidate.output.contents);
    for (size_t i = 0; i < expected.size(); ++i) {
        if (actual[i] != expected[i]) {
            throw std::runtime_error(candidate.name + " validation mismatch at " + std::to_string(i));
        }
    }
}

static void validateCandidate(const Candidate &candidate,
                              const std::vector<uint32_t> &left,
                              const std::vector<uint32_t> &right,
                              const std::vector<uint32_t> &scanInput,
                              const ReductionMeta &reduction,
                              const ScanMeta &scan) {
    if (candidate.kind == RouteKind::BINARY && candidate.operation.rfind("CMP_", 0) == 0) {
        const uint8_t *actual = static_cast<const uint8_t *>(candidate.output.contents);
        for (size_t i = 0; i < left.size(); ++i) {
            uint8_t expected = exactCompare(left[i], right[i], candidate.operationCode) ? 1 : 0;
            if (actual[i] != expected) {
                throw std::runtime_error(candidate.name + " validation mismatch at " + std::to_string(i));
            }
        }
        return;
    }
    if (candidate.kind == RouteKind::BINARY) {
        bool maximum = candidate.operation == "TENSOR_MAX";
        std::vector<uint32_t> expected(left.size());
        for (size_t i = 0; i < left.size(); ++i) expected[i] = exactExtreme(left[i], right[i], maximum);
        requireWords(candidate, expected);
        return;
    }
    if (candidate.kind == RouteKind::SCALAR) {
        bool maximum = candidate.operation == "SCALAR_MAX";
        std::vector<uint32_t> expected(left.size());
        for (size_t i = 0; i < left.size(); ++i) expected[i] = exactExtreme(left[i], candidate.scalar, maximum);
        requireWords(candidate, expected);
        return;
    }
    if (candidate.kind == RouteKind::CLAMP_FUSED || candidate.kind == RouteKind::CLAMP_COMPOSED) {
        std::vector<uint32_t> expected(left.size());
        for (size_t i = 0; i < left.size(); ++i) {
            expected[i] = exactExtreme(exactExtreme(left[i], candidate.lower, true), candidate.upper, false);
        }
        requireWords(candidate, expected);
        return;
    }
    if (candidate.kind == RouteKind::REDUCTION) {
        bool maximum = candidate.operation == "REDUCTION_MAX";
        std::vector<uint32_t> expected(reduction.outputCount);
        for (uint64_t output = 0; output < reduction.outputCount; ++output) {
            uint32_t accumulator = left[reductionOffset(output, 0, reduction)];
            for (uint64_t contributor = 1; contributor < reduction.reducedCount; ++contributor) {
                accumulator = exactExtreme(accumulator,
                    left[reductionOffset(output, contributor, reduction)], maximum);
            }
            expected[output] = accumulator;
        }
        requireWords(candidate, expected);
        return;
    }
    bool product = candidate.operation == "CUM_PROD";
    std::vector<uint32_t> expected(scan.elementCount);
    for (uint64_t line = 0; line < scan.lineCount; ++line) {
        uint64_t outer = line / scan.inner;
        uint64_t innerIndex = line % scan.inner;
        uint64_t base = outer * scan.axisLength * scan.inner + innerIndex;
        uint32_t accumulator = product ? 0x3f800000u : 0x00000000u;
        bool hasContributor = false;
        for (uint64_t step = 0; step < scan.axisLength; ++step) {
            uint64_t logical = scan.axisLength - 1 - step;
            uint64_t offset = base + logical * scan.inner;
            expected[offset] = hasContributor ? accumulator : (product ? 0x3f800000u : 0x00000000u);
            uint32_t value = scanInput[offset];
            if (!hasContributor) {
                accumulator = value;
                hasContributor = true;
            } else {
                accumulator = exactScanSite(accumulator, value, product);
            }
        }
    }
    requireWords(candidate, expected);
}

struct BatchResult {
    uint64_t durationNs;
    uint64_t iterations;
    double normalizedNs;
};

static BatchResult runBatch(id<MTLCommandQueue> queue, const Candidate &candidate) {
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
    uint64_t iterations = 0;
    uint64_t duration = 0;
    do {
        executeOnce(queue, candidate);
        ++iterations;
        duration = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
    } while (duration < BATCH_FLOOR_NS && iterations < BATCH_LIMIT);
    if (duration < BATCH_FLOOR_NS) throw std::runtime_error("batch ceiling reached before floor");
    return BatchResult{duration, iterations, static_cast<double>(duration) / iterations};
}

static double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return (samples[3] + samples[4]) / 2.0;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        try {
            if (argc != 2) throw std::runtime_error("usage: gate3 <retained-oracle.metal>");
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (device == nil) throw std::runtime_error("no default Metal device");
            std::string deviceName = [[device name] UTF8String];
            if (deviceName.find("M3 Max") == std::string::npos) {
                throw std::runtime_error("Gate 3 requires Apple M3 Max, got " + deviceName);
            }

            NSError *error = nil;
            NSString *source = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]
                                                           encoding:NSUTF8StringEncoding error:&error];
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
            NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pipelines =
                [NSMutableDictionary dictionary];
            for (NSString *name in functionNames) {
                id<MTLFunction> function = [library newFunctionWithName:name];
                if (function == nil) throw std::runtime_error("missing function");
                id<MTLComputePipelineState> state =
                    [device newComputePipelineStateWithFunction:function error:&error];
                if (state == nil) throw std::runtime_error([[error description] UTF8String]);
                pipelines[name] = state;
            }
            id<MTLCommandQueue> queue = [device newCommandQueue];
            if (queue == nil) throw std::runtime_error("queue creation failed");

            const std::array<uint32_t, 24> frozenLeft = {
                0x00000000u,0x80000000u,0x00000001u,0x00000000u,0x80000001u,0x80000000u,
                0x007fffffu,0x00800000u,0x807fffffu,0x80800000u,0x3f800000u,0x3f800000u,
                0xbf800000u,0x7f7fffffu,0xff7fffffu,0x7f800000u,0x7fc12345u,0x3f800000u,
                0x7fa12345u,0xbf800000u,0xffc54321u,0x00000000u,0x7fc12345u,0x40000000u};
            const std::array<uint32_t, 24> frozenRight = {
                0x80000000u,0x00000000u,0x00000000u,0x00000001u,0x80000000u,0x80000001u,
                0x00800000u,0x007fffffu,0x80800000u,0x807fffffu,0x3f800000u,0x3f800001u,
                0xbf7fffffu,0x7f800000u,0xff800000u,0xff800000u,0x3f800000u,0x7fc12345u,
                0xbf800000u,0x7fa12345u,0x00000000u,0xffc54321u,0xffc54321u,0xc0000000u};
            const std::array<uint32_t, 8> ordinary = {
                0x00000000u, 0x80000000u, 0x3f800000u, 0xbf800000u,
                0x40000000u, 0xc0000000u, 0x3f000000u, 0xbf000000u};

            std::vector<uint32_t> left(POINT_COUNT);
            std::vector<uint32_t> right(POINT_COUNT);
            for (uint64_t i = 0; i < POINT_COUNT; ++i) {
                left[i] = ordinary[i % ordinary.size()];
                right[i] = ordinary[(i * 5 + 3) % ordinary.size()];
            }
            std::copy(frozenLeft.begin(), frozenLeft.end(), left.begin());
            std::copy(frozenRight.begin(), frozenRight.end(), right.begin());
            std::vector<uint32_t> scanInput(POINT_COUNT);
            for (uint64_t i = 0; i < POINT_COUNT; ++i) scanInput[i] = ordinary[i % 6];

            id<MTLBuffer> leftBuffer = bufferWithBytes(device, left.data(), left.size() * 4);
            id<MTLBuffer> rightBuffer = bufferWithBytes(device, right.data(), right.size() * 4);
            id<MTLBuffer> scanBuffer = bufferWithBytes(device, scanInput.data(), scanInput.size() * 4);
            BinaryMeta binary = binaryMeta(POINT_COUNT);
            ReductionMeta reduction = reductionMeta();
            ScanMeta scan = scanMeta();
            id<MTLBuffer> binaryMetadata = metadataBuffer(device, binary);
            id<MTLBuffer> reductionMetadata = metadataBuffer(device, reduction);
            id<MTLBuffer> scanMetadata = metadataBuffer(device, scan);

            auto pipe = [&](NSString *name) -> id<MTLComputePipelineState> {
                id<MTLComputePipelineState> result = pipelines[name];
                if (result == nil) throw std::runtime_error("missing pipeline");
                return result;
            };
            std::vector<Candidate> candidates;
            const std::array<const char *, 6> cmpNames = {
                "custom_cmp_gt", "custom_cmp_ge", "custom_cmp_lt",
                "custom_cmp_le", "custom_cmp_eq", "custom_cmp_ne"};
            const std::array<NSString *, 6> cmpFunctions = {
                @"cmp_gt", @"cmp_ge", @"cmp_lt", @"cmp_le", @"cmp_eq", @"cmp_ne"};
            const std::array<const char *, 6> cmpOperations = {
                "CMP_GT", "CMP_GE", "CMP_LT", "CMP_LE", "CMP_EQ", "CMP_NE"};
            for (uint32_t i = 0; i < 6; ++i) {
                candidates.push_back(Candidate{cmpNames[i], cmpOperations[i], RouteKind::BINARY,
                    pipe(cmpFunctions[i]), nil, leftBuffer, rightBuffer,
                    emptyBuffer(device, POINT_COUNT), nil, binaryMetadata, POINT_COUNT, POINT_COUNT,
                    i, 0, 0, 0, 1, 0, {}});
            }
            for (bool maximum : {false, true}) {
                NSString *function = maximum ? @"tensor_max" : @"tensor_min";
                candidates.push_back(Candidate{maximum ? "custom_tensor_max" : "custom_tensor_min",
                    maximum ? "TENSOR_MAX" : "TENSOR_MIN", RouteKind::BINARY, pipe(function), nil,
                    leftBuffer, rightBuffer, emptyBuffer(device, POINT_COUNT * 4), nil, binaryMetadata,
                    POINT_COUNT, POINT_COUNT, 0, 0, 0, 0, 1, 0, {}});
            }
            const uint32_t scalar = 0x3f800000u;
            for (bool maximum : {false, true}) {
                NSString *function = maximum ? @"scalar_max" : @"scalar_min";
                PointMeta point{POINT_COUNT, binary.gridWidth, binary.gridHeight, scalar, 0};
                id<MTLBuffer> pointMetadata = metadataBuffer(device, point);
                candidates.push_back(Candidate{maximum ? "custom_scalar_max" : "custom_scalar_min",
                    maximum ? "SCALAR_MAX" : "SCALAR_MIN", RouteKind::SCALAR, pipe(function), nil,
                    leftBuffer, nil, emptyBuffer(device, POINT_COUNT * 4), nil, pointMetadata,
                    POINT_COUNT, POINT_COUNT, 0, scalar, 0, 0, 1, 0, {}});
            }
            const uint32_t lower = 0xbf800000u;
            const uint32_t upper = 0x3f800000u;
            ClampMeta clamp{POINT_COUNT, binary.gridWidth, binary.gridHeight, lower, upper};
            id<MTLBuffer> clampMetadata = metadataBuffer(device, clamp);
            candidates.push_back(Candidate{"custom_clamp_fused", "CLAMP", RouteKind::CLAMP_FUSED,
                pipe(@"clamp_fused"), nil, leftBuffer, nil, emptyBuffer(device, POINT_COUNT * 4), nil,
                clampMetadata, POINT_COUNT, POINT_COUNT, 0, 0, lower, upper, 1, 0, {}});
            PointMeta lowerPoint{POINT_COUNT, binary.gridWidth, binary.gridHeight, lower, 0};
            PointMeta upperPoint{POINT_COUNT, binary.gridWidth, binary.gridHeight, upper, 0};
            id<MTLBuffer> lowerMetadata = metadataBuffer(device, lowerPoint);
            id<MTLBuffer> upperMetadata = metadataBuffer(device, upperPoint);
            candidates.push_back(Candidate{"composed_custom_clamp", "CLAMP", RouteKind::CLAMP_COMPOSED,
                pipe(@"scalar_max"), pipe(@"scalar_min"), leftBuffer, upperMetadata,
                emptyBuffer(device, POINT_COUNT * 4), emptyBuffer(device, POINT_COUNT * 4),
                lowerMetadata, POINT_COUNT, POINT_COUNT, 0, 0, lower, upper, 2, POINT_COUNT * 4, {}});
            for (bool maximum : {false, true}) {
                NSString *function = maximum ? @"reduction_max" : @"reduction_min";
                candidates.push_back(Candidate{maximum ? "custom_reduction_max" : "custom_reduction_min",
                    maximum ? "REDUCTION_MAX" : "REDUCTION_MIN", RouteKind::REDUCTION, pipe(function), nil,
                    leftBuffer, nil, emptyBuffer(device, REDUCTION_OUTPUT_COUNT * 4), nil,
                    reductionMetadata, REDUCTION_OUTPUT_COUNT, REDUCTION_OUTPUT_COUNT,
                    0, 0, 0, 0, 1, 0, {}});
            }
            for (bool product : {false, true}) {
                NSString *function = product ? @"scan_prod" : @"scan_sum";
                candidates.push_back(Candidate{product ? "custom_scan_prod" : "custom_scan_sum",
                    product ? "CUM_PROD" : "CUM_SUM", RouteKind::SCAN, pipe(function), nil,
                    scanBuffer, nil, emptyBuffer(device, POINT_COUNT * 4), nil, scanMetadata,
                    SCAN_LINE_COUNT, POINT_COUNT, 0, 0, 0, 0, 1, 0, {}});
            }
            if (candidates.size() != 16) throw std::runtime_error("candidate inventory mismatch");

            for (Candidate &candidate : candidates) {
                executeOnce(queue, candidate);
                validateCandidate(candidate, left, right, scanInput, reduction, scan);
            }
            if (std::memcmp(leftBuffer.contents, left.data(), left.size() * 4) != 0
                    || std::memcmp(rightBuffer.contents, right.data(), right.size() * 4) != 0
                    || std::memcmp(scanBuffer.contents, scanInput.data(), scanInput.size() * 4) != 0) {
                throw std::runtime_error("input preservation failure");
            }

            for (int round = 0; round < 4; ++round) {
                for (size_t ordinal = 0; ordinal < candidates.size(); ++ordinal) {
                    size_t index = round % 2 == 0 ? ordinal : candidates.size() - 1 - ordinal;
                    (void)runBatch(queue, candidates[index]);
                }
            }

            struct RawSample { size_t index; int round; size_t order; BatchResult batch; };
            std::vector<RawSample> rawSamples;
            for (int round = 0; round < 8; ++round) {
                for (size_t ordinal = 0; ordinal < candidates.size(); ++ordinal) {
                    size_t index = round % 2 == 0 ? ordinal : candidates.size() - 1 - ordinal;
                    BatchResult batch = runBatch(queue, candidates[index]);
                    candidates[index].retainedNs.push_back(batch.normalizedNs);
                    rawSamples.push_back(RawSample{index, round, ordinal, batch});
                }
            }

            std::cout << "TASK=0052_GATE3\n";
            std::cout << "PROTOCOL=custom-only-v1\n";
            std::cout << "DEVICE=" << deviceName << "\n";
            std::cout << "CANDIDATE_COUNT=" << candidates.size() << "\n";
            std::cout << "WARMUP_ROUNDS=4\n";
            std::cout << "RETAINED_ROUNDS=8\n";
            std::cout << "BATCH_FLOOR_NS=" << BATCH_FLOOR_NS << "\n";
            std::cout << "BATCH_LIMIT=" << BATCH_LIMIT << "\n";
            std::cout << "FAST_MATH=0\n";
            std::cout << "VALIDATION=PASS\n";
            std::cout << "INPUTS_PRESERVED=PASS\n";
            std::cout << std::fixed << std::setprecision(3);
            for (const RawSample &sample : rawSamples) {
                const Candidate &candidate = candidates[sample.index];
                std::cout << "SAMPLE candidate=" << candidate.name
                          << " round=" << sample.round
                          << " order=" << sample.order
                          << " duration_ns=" << sample.batch.durationNs
                          << " iterations=" << sample.batch.iterations
                          << " normalized_ns=" << sample.batch.normalizedNs << "\n";
            }
            for (const Candidate &candidate : candidates) {
                std::cout << "CANDIDATE=" << candidate.name
                          << " operation=" << candidate.operation
                          << " median_ns=" << median(candidate.retainedNs)
                          << " dispatches=" << candidate.dispatches
                          << " route_temp_bytes=" << candidate.temporaryBytes
                          << " validation=PASS source_owned=PASS\n";
            }
            const Candidate &fused = candidates[10];
            const Candidate &composed = candidates[11];
            double fusedMedian = median(fused.retainedNs);
            double composedMedian = median(composed.retainedNs);
            const Candidate *clampWinner;
            if (fusedMedian != composedMedian) clampWinner = fusedMedian < composedMedian ? &fused : &composed;
            else if (fused.dispatches != composed.dispatches)
                clampWinner = fused.dispatches < composed.dispatches ? &fused : &composed;
            else if (fused.temporaryBytes != composed.temporaryBytes)
                clampWinner = fused.temporaryBytes < composed.temporaryBytes ? &fused : &composed;
            else
                clampWinner = fused.name < composed.name ? &fused : &composed;
            std::cout << "CLAMP_WINNER=" << clampWinner->name << "\n";
            std::cout << "GATE3_VERDICT=PASS\n";
            return 0;
        } catch (const std::exception &failure) {
            std::cerr << "ERROR=" << failure.what() << "\n";
            return 1;
        }
    }
}
