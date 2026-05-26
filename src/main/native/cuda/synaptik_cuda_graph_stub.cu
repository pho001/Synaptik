#include <cuda_runtime_api.h>

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct SynaptikCudaContext {
    int device;
};

struct SynaptikCudaExecutable {
    struct Node {
        int type;
        int input0Kind;
        int input0Index;
        int input1Kind;
        int input1Index;
        int input0Rank;
        int input0Dim0;
        int input0Dim1;
        int input0Dim2;
        int input0Dim3;
        int input0ElementCount;
        int input1Rank;
        int input1Dim0;
        int input1Dim1;
        int input1Dim2;
        int input1Dim3;
        int input1ElementCount;
        int outputRank;
        int outputDim0;
        int outputDim1;
        int outputDim2;
        int outputDim3;
        int elementCount;
        int reductionAxis;
        bool reductionKeepDims;
        float scalarValue;
    };
    std::vector<Node> nodes;
    std::vector<int> outputNodeIndices;
};

struct SynaptikCudaBuffer {
    void* data;
    int byteLength;
};

static std::string g_unavailable_reason = "CUDA runtime has not been probed.";

const char* stable_reason(const std::string& reason) {
    g_unavailable_reason = reason;
    return g_unavailable_reason.c_str();
}

bool cuda_runtime_available() {
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) {
        stable_reason(std::string("CUDA runtime probe failed: ") + cudaGetErrorString(status));
        return false;
    }
    if (count <= 0) {
        stable_reason("CUDA device count is zero.");
        return false;
    }
    stable_reason("");
    return true;
}

int element_count(int rank, int dim0, int dim1, int dim2, int dim3) {
    int count = dim0;
    if (rank >= 2) {
        count *= dim1;
    }
    if (rank >= 3) {
        count *= dim2;
    }
    if (rank >= 4) {
        count *= dim3;
    }
    return count;
}

__global__ void relu_kernel(const float* input, float* output, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        float value = input[idx];
        output[idx] = value > 0.0f ? value : 0.0f;
    }
}

__global__ void add_kernel(const float* left, const float* right, float* output, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        output[idx] = left[idx] + right[idx];
    }
}

__device__ int flat_index4(const int* coords, int rank, int dim1, int dim2, int dim3);

__device__ int broadcast_index4(
        int outputIndex,
        int outputRank,
        int outputDim0,
        int outputDim1,
        int outputDim2,
        int outputDim3,
        int inputRank,
        int inputDim0,
        int inputDim1,
        int inputDim2,
        int inputDim3
) {
    int outputDims[4] = {outputDim0, outputDim1, outputDim2, outputDim3};
    int inputDims[4] = {inputDim0, inputDim1, inputDim2, inputDim3};
    int outputCoords[4] = {0, 0, 0, 0};
    int remaining = outputIndex;
    for (int dim = outputRank - 1; dim >= 0; dim--) {
        outputCoords[dim] = remaining % outputDims[dim];
        remaining /= outputDims[dim];
    }
    int inputCoords[4] = {0, 0, 0, 0};
    int rankOffset = outputRank - inputRank;
    for (int dim = 0; dim < inputRank; dim++) {
        int outputDim = dim + rankOffset;
        inputCoords[dim] = inputDims[dim] == 1 ? 0 : outputCoords[outputDim];
    }
    return flat_index4(inputCoords, inputRank, inputDim1, inputDim2, inputDim3);
}

__global__ void binary_broadcast_kernel(
        const float* left,
        const float* right,
        float* output,
        int count,
        int opType,
        int outputRank,
        int outputDim0,
        int outputDim1,
        int outputDim2,
        int outputDim3,
        int leftRank,
        int leftDim0,
        int leftDim1,
        int leftDim2,
        int leftDim3,
        int rightRank,
        int rightDim0,
        int rightDim1,
        int rightDim2,
        int rightDim3
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    int leftIndex = broadcast_index4(
            idx,
            outputRank,
            outputDim0,
            outputDim1,
            outputDim2,
            outputDim3,
            leftRank,
            leftDim0,
            leftDim1,
            leftDim2,
            leftDim3
    );
    int rightIndex = broadcast_index4(
            idx,
            outputRank,
            outputDim0,
            outputDim1,
            outputDim2,
            outputDim3,
            rightRank,
            rightDim0,
            rightDim1,
            rightDim2,
            rightDim3
    );
    float a = left[leftIndex];
    float b = right[rightIndex];
    if (opType == 3) {
        output[idx] = a + b;
    } else if (opType == 4) {
        output[idx] = a - b;
    } else if (opType == 5) {
        output[idx] = a * b;
    } else {
        output[idx] = a / b;
    }
}

__global__ void unary_f32_kernel(const float* input, float* output, int count, int opType, float scalarValue) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    float value = input[idx];
    if (opType == 7) {
        output[idx] = value > 0.0f ? value : 0.0f;
    } else if (opType == 14) {
        output[idx] = sqrtf(value);
    } else if (opType == 15) {
        output[idx] = 1.0f / value;
    } else if (opType == 40) {
        output[idx] = value + scalarValue;
    } else {
        output[idx] = value;
    }
}

__device__ int flat_index4(const int* coords, int rank, int dim1, int dim2, int dim3) {
    int index = coords[0];
    if (rank >= 2) {
        index = index * dim1 + coords[1];
    }
    if (rank >= 3) {
        index = index * dim2 + coords[2];
    }
    if (rank >= 4) {
        index = index * dim3 + coords[3];
    }
    return index;
}

__global__ void reduction_kernel(
        const float* input,
        float* output,
        int outputCount,
        int opType,
        int axis,
        int inputRank,
        int inputDim0,
        int inputDim1,
        int inputDim2,
        int inputDim3,
        int outputRank,
        int outputDim0,
        int outputDim1,
        int outputDim2,
        int outputDim3,
        bool keepDims
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outputCount) {
        return;
    }
    int inputDims[4] = {inputDim0, inputDim1, inputDim2, inputDim3};
    int outputDims[4] = {outputDim0, outputDim1, outputDim2, outputDim3};
    int outputCoords[4] = {0, 0, 0, 0};
    int remaining = idx;
    for (int dim = outputRank - 1; dim >= 0; dim--) {
        outputCoords[dim] = remaining % outputDims[dim];
        remaining /= outputDims[dim];
    }
    int reduceCount = inputDims[axis];
    float acc = 0.0f;
    if (opType == 38) {
        acc = CUDART_INF_F;
    } else if (opType == 39) {
        acc = -CUDART_INF_F;
    }
    for (int r = 0; r < reduceCount; r++) {
        int inputCoords[4] = {0, 0, 0, 0};
        int outDim = 0;
        for (int dim = 0; dim < inputRank; dim++) {
            if (dim == axis) {
                inputCoords[dim] = r;
            } else if (keepDims) {
                inputCoords[dim] = outputCoords[dim];
            } else {
                inputCoords[dim] = outputCoords[outDim++];
            }
        }
        int inputIndex = flat_index4(inputCoords, inputRank, inputDim1, inputDim2, inputDim3);
        float value = input[inputIndex];
        if (opType == 38) {
            acc = fminf(acc, value);
        } else if (opType == 39) {
            acc = fmaxf(acc, value);
        } else {
            acc += value;
        }
    }
    output[idx] = opType == 37 ? acc / static_cast<float>(reduceCount) : acc;
}

__device__ int positive_mod(int value, int divisor) {
    int remainder = value % divisor;
    return remainder < 0 ? remainder + divisor : remainder;
}

__global__ void unfold_axis_f32_kernel(
        const float* input,
        float* output,
        int outputCount,
        int axis,
        int size,
        int step,
        int inputRank,
        int inputDim0,
        int inputDim1,
        int inputDim2,
        int inputDim3,
        int outputRank,
        int outputDim0,
        int outputDim1,
        int outputDim2,
        int outputDim3
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outputCount) {
        return;
    }
    int inputDims[4] = {inputDim0, inputDim1, inputDim2, inputDim3};
    int outputDims[4] = {outputDim0, outputDim1, outputDim2, outputDim3};
    int outputCoords[4] = {0, 0, 0, 0};
    int remaining = idx;
    for (int dim = outputRank - 1; dim >= 0; dim--) {
        outputCoords[dim] = remaining % outputDims[dim];
        remaining /= outputDims[dim];
    }
    int windowOffset = outputCoords[inputRank];
    if (windowOffset >= size) {
        output[idx] = 0.0f;
        return;
    }
    int inputCoords[4] = {0, 0, 0, 0};
    for (int dim = 0; dim < inputRank; dim++) {
        inputCoords[dim] = dim == axis ? outputCoords[dim] * step + windowOffset : outputCoords[dim];
        if (inputCoords[dim] < 0 || inputCoords[dim] >= inputDims[dim]) {
            output[idx] = 0.0f;
            return;
        }
    }
    int inputIndex = flat_index4(inputCoords, inputRank, inputDim1, inputDim2, inputDim3);
    output[idx] = input[inputIndex];
}

__global__ void unfold2d_f32_kernel(
        const float* input,
        float* output,
        int outputCount,
        int kernelH,
        int kernelW,
        int strideH,
        int strideW,
        int padH,
        int padW,
        int batch,
        int channels,
        int height,
        int width,
        int outH,
        int outW
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outputCount) {
        return;
    }
    int windowCount = outH * outW;
    int kernelArea = kernelH * kernelW;
    int l = idx % windowCount;
    int columnChannel = (idx / windowCount) % (channels * kernelArea);
    int n = idx / (windowCount * channels * kernelArea);
    if (n >= batch) {
        return;
    }
    int c = columnChannel / kernelArea;
    int k = columnChannel % kernelArea;
    int kh = k / kernelW;
    int kw = k % kernelW;
    int oy = l / outW;
    int ox = l % outW;
    int iy = oy * strideH - padH + kh;
    int ix = ox * strideW - padW + kw;
    if (iy < 0 || iy >= height || ix < 0 || ix >= width) {
        output[idx] = 0.0f;
        return;
    }
    int inputIndex = ((n * channels + c) * height + iy) * width + ix;
    output[idx] = input[inputIndex];
}

__global__ void fold2d_f32_kernel(
        const float* input,
        float* output,
        int outputCount,
        int kernelH,
        int kernelW,
        int strideH,
        int strideW,
        int padH,
        int padW,
        int batch,
        int channels,
        int height,
        int width,
        int outH,
        int outW
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outputCount) {
        return;
    }
    int x = idx % width;
    int y = (idx / width) % height;
    int c = (idx / (width * height)) % channels;
    int n = idx / (width * height * channels);
    if (n >= batch) {
        return;
    }
    int windowCount = outH * outW;
    int kernelArea = kernelH * kernelW;
    float acc = 0.0f;
    for (int kh = 0; kh < kernelH; kh++) {
        int yNumerator = y + padH - kh;
        if (yNumerator < 0 || yNumerator % strideH != 0) {
            continue;
        }
        int oy = yNumerator / strideH;
        if (oy < 0 || oy >= outH) {
            continue;
        }
        for (int kw = 0; kw < kernelW; kw++) {
            int xNumerator = x + padW - kw;
            if (xNumerator < 0 || xNumerator % strideW != 0) {
                continue;
            }
            int ox = xNumerator / strideW;
            if (ox < 0 || ox >= outW) {
                continue;
            }
            int k = kh * kernelW + kw;
            int columnChannel = c * kernelArea + k;
            int inputIndex = (n * (channels * kernelArea) + columnChannel) * windowCount + oy * outW + ox;
            acc += input[inputIndex];
        }
    }
    output[idx] = acc;
}

bool validate_buffer(SynaptikCudaBuffer* buffer, int byteLength) {
    return buffer != nullptr && buffer->data != nullptr && buffer->byteLength >= byteLength;
}

int decode_reduction_axis(const float* scalarValues, int index) {
    if (scalarValues == nullptr) {
        return 0;
    }
    int encoded = 0;
    std::memcpy(&encoded, &scalarValues[index], sizeof(float));
    int axis = encoded & 0xFFFF;
    if ((axis & 0x8000) != 0) {
        axis |= 0xFFFF0000;
    }
    return axis;
}

bool decode_reduction_keep_dims(const float* scalarValues, int index) {
    if (scalarValues == nullptr) {
        return false;
    }
    int encoded = 0;
    std::memcpy(&encoded, &scalarValues[index], sizeof(float));
    return (encoded & (1 << 16)) != 0;
}

void set_node_shape(SynaptikCudaExecutable::Node& node, int rank, int dim0, int dim1, int dim2, int dim3) {
    node.outputRank = rank;
    node.outputDim0 = dim0;
    node.outputDim1 = dim1;
    node.outputDim2 = dim2;
    node.outputDim3 = dim3;
    node.elementCount = element_count(rank, dim0, dim1, dim2, dim3);
}

void set_input0_shape_from_external(
        SynaptikCudaExecutable::Node& node,
        int externalIndex,
        const int* externalInputRanks,
        const int* externalInputDim0,
        const int* externalInputDim1,
        const int* externalInputDim2,
        const int* externalInputDim3
) {
    node.input0Rank = externalInputRanks == nullptr ? 1 : externalInputRanks[externalIndex];
    node.input0Dim0 = externalInputDim0 == nullptr ? 1 : externalInputDim0[externalIndex];
    node.input0Dim1 = externalInputDim1 == nullptr ? 1 : externalInputDim1[externalIndex];
    node.input0Dim2 = externalInputDim2 == nullptr ? 1 : externalInputDim2[externalIndex];
    node.input0Dim3 = externalInputDim3 == nullptr ? 1 : externalInputDim3[externalIndex];
    node.input0ElementCount = element_count(node.input0Rank, node.input0Dim0, node.input0Dim1, node.input0Dim2, node.input0Dim3);
}

void set_input0_shape_from_node(SynaptikCudaExecutable::Node& node, const SynaptikCudaExecutable::Node& source) {
    node.input0Rank = source.outputRank;
    node.input0Dim0 = source.outputDim0;
    node.input0Dim1 = source.outputDim1;
    node.input0Dim2 = source.outputDim2;
    node.input0Dim3 = source.outputDim3;
    node.input0ElementCount = source.elementCount;
}

void set_input1_shape_from_external(
        SynaptikCudaExecutable::Node& node,
        int externalIndex,
        const int* externalInputRanks,
        const int* externalInputDim0,
        const int* externalInputDim1,
        const int* externalInputDim2,
        const int* externalInputDim3
) {
    node.input1Rank = externalInputRanks == nullptr ? 1 : externalInputRanks[externalIndex];
    node.input1Dim0 = externalInputDim0 == nullptr ? 1 : externalInputDim0[externalIndex];
    node.input1Dim1 = externalInputDim1 == nullptr ? 1 : externalInputDim1[externalIndex];
    node.input1Dim2 = externalInputDim2 == nullptr ? 1 : externalInputDim2[externalIndex];
    node.input1Dim3 = externalInputDim3 == nullptr ? 1 : externalInputDim3[externalIndex];
    node.input1ElementCount = element_count(node.input1Rank, node.input1Dim0, node.input1Dim1, node.input1Dim2, node.input1Dim3);
}

void set_input1_shape_from_node(SynaptikCudaExecutable::Node& node, const SynaptikCudaExecutable::Node& source) {
    node.input1Rank = source.outputRank;
    node.input1Dim0 = source.outputDim0;
    node.input1Dim1 = source.outputDim1;
    node.input1Dim2 = source.outputDim2;
    node.input1Dim3 = source.outputDim3;
    node.input1ElementCount = source.elementCount;
}

bool suffix_broadcast_supported(
        int inputRank,
        int inputDim0,
        int inputDim1,
        int inputDim2,
        int inputDim3,
        int outputRank,
        int outputDim0,
        int outputDim1,
        int outputDim2,
        int outputDim3
) {
    if (inputRank < 1 || inputRank > 4 || outputRank < 1 || outputRank > 4 || inputRank > outputRank) {
        return false;
    }
    int inputDims[4] = {inputDim0, inputDim1, inputDim2, inputDim3};
    int outputDims[4] = {outputDim0, outputDim1, outputDim2, outputDim3};
    int rankOffset = outputRank - inputRank;
    for (int dim = 0; dim < inputRank; dim++) {
        int inputDim = inputDims[dim];
        int outputDim = outputDims[dim + rankOffset];
        if (inputDim != 1 && inputDim != outputDim) {
            return false;
        }
    }
    return true;
}

bool is_binary_op(int opType) {
    return opType == 3 || opType == 4 || opType == 5 || opType == 6;
}

bool is_unary_op(int opType) {
    return opType == 7 || opType == 14 || opType == 15 || opType == 40;
}

int decode_int_scalar(float scalarValue) {
    int encoded = 0;
    std::memcpy(&encoded, &scalarValue, sizeof(float));
    return encoded;
}

void decode_unfold_axis_mode(float scalarValue, int& axis, int& size, int& step) {
    int encoded = decode_int_scalar(scalarValue);
    axis = encoded & 0xF;
    size = (encoded >> 4) & 0xFFF;
    step = (encoded >> 16) & 0xFFF;
}

void decode_window2d_mode(float scalarValue, int& kernelH, int& kernelW, int& strideH, int& strideW, int& padH, int& padW) {
    int encoded = decode_int_scalar(scalarValue);
    kernelH = encoded & 0xF;
    kernelW = (encoded >> 4) & 0xF;
    strideH = (encoded >> 8) & 0xF;
    strideW = (encoded >> 12) & 0xF;
    padH = (encoded >> 16) & 0xF;
    padW = (encoded >> 20) & 0xF;
}

} // namespace

extern "C" int synaptik_cuda_graph_available(void) {
    return cuda_runtime_available() ? 1 : 0;
}

extern "C" const char* synaptik_cuda_graph_unavailable_reason(void) {
    if (g_unavailable_reason.empty()) {
        return "";
    }
    return g_unavailable_reason.c_str();
}

extern "C" int synaptik_cuda_graph_layout_abi_version(void) {
    return 2;
}

extern "C" int synaptik_cuda_graph_validate_layout_abi_v2(
        int binding_count,
        const int* ranks,
        const int* dtypes,
        const long long* storage_offsets,
        const long long* logical_element_counts,
        const long long* logical_byte_lengths,
        const long long* physical_byte_spans,
        const int* access_modes,
        const int* layout_classes,
        const void* const* native_handles,
        const int* shape_offsets,
        const long long* shape_values,
        const int* stride_offsets,
        const long long* stride_values) {
    if (binding_count < 0) {
        return 1;
    }
    if (binding_count == 0) {
        return 0;
    }
    if (ranks == nullptr || dtypes == nullptr || storage_offsets == nullptr
            || logical_element_counts == nullptr || logical_byte_lengths == nullptr
            || physical_byte_spans == nullptr || access_modes == nullptr || layout_classes == nullptr
            || native_handles == nullptr || shape_offsets == nullptr || shape_values == nullptr
            || stride_offsets == nullptr || stride_values == nullptr) {
        return 1;
    }
    for (int i = 0; i < binding_count; i++) {
        if (ranks[i] <= 0 || physical_byte_spans[i] < 0 || native_handles[i] == nullptr) {
            return 1;
        }
    }
    return 0;
}

extern "C" void* synaptik_cuda_graph_create_context(void) {
    if (!cuda_runtime_available()) {
        return nullptr;
    }
    auto* context = new SynaptikCudaContext();
    context->device = 0;
    cudaError_t status = cudaSetDevice(context->device);
    if (status != cudaSuccess) {
        stable_reason(std::string("CUDA set device failed: ") + cudaGetErrorString(status));
        delete context;
        return nullptr;
    }
    return context;
}

extern "C" void* synaptik_cuda_graph_compile_partition_f32(
        void* context,
        int externalInputCount,
        const int* externalInputRanks,
        const int* externalInputDTypes,
        const int* externalInputDim0,
        const int* externalInputDim1,
        const int* externalInputDim2,
        const int* externalInputDim3,
        int nodeCount,
        const int* nodeTypes,
        const int* input0Kinds,
        const int* input0Indices,
        const int* input1Kinds,
        const int* input1Indices,
        const int* input2Kinds,
        const int* input2Indices,
        const int* input3Kinds,
        const int* input3Indices,
        const int* input4Kinds,
        const int* input4Indices,
        const float* scalarValues,
        const int* outputRanks,
        const int* outputDim0,
        const int* outputDim1,
        const int* outputDim2,
        const int* outputDim3,
        int outputCount,
        const int* outputNodeIndices
) {
    (void) externalInputCount;
    (void) externalInputRanks;
    (void) externalInputDTypes;
    (void) externalInputDim0;
    (void) externalInputDim1;
    (void) externalInputDim2;
    (void) externalInputDim3;
    (void) nodeTypes;
    (void) input0Kinds;
    (void) input0Indices;
    (void) input1Kinds;
    (void) input1Indices;
    (void) input2Kinds;
    (void) input2Indices;
    (void) input3Kinds;
    (void) input3Indices;
    (void) input4Kinds;
    (void) input4Indices;
    if (context == nullptr) {
        stable_reason("CUDA compile requested without a context.");
        return nullptr;
    }
    if (nodeCount <= 0 || outputCount <= 0) {
        stable_reason("CUDA compile requires at least one node and one output.");
        return nullptr;
    }
    auto* executable = new SynaptikCudaExecutable();
    executable->nodes.reserve(nodeCount);
    for (int i = 0; i < nodeCount; i++) {
        SynaptikCudaExecutable::Node node{};
        node.type = nodeTypes == nullptr ? 0 : nodeTypes[i];
        node.input0Kind = input0Kinds == nullptr ? 0 : input0Kinds[i];
        node.input0Index = input0Indices == nullptr ? -1 : input0Indices[i];
        node.input1Kind = input1Kinds == nullptr ? 0 : input1Kinds[i];
        node.input1Index = input1Indices == nullptr ? -1 : input1Indices[i];
        set_node_shape(
                node,
                outputRanks == nullptr ? 1 : outputRanks[i],
                outputDim0 == nullptr ? 1 : outputDim0[i],
                outputDim1 == nullptr ? 1 : outputDim1[i],
                outputDim2 == nullptr ? 1 : outputDim2[i],
                outputDim3 == nullptr ? 1 : outputDim3[i]
        );
        node.input0Rank = node.outputRank;
        node.input0Dim0 = node.outputDim0;
        node.input0Dim1 = node.outputDim1;
        node.input0Dim2 = node.outputDim2;
        node.input0Dim3 = node.outputDim3;
        node.input0ElementCount = node.elementCount;
        node.input1Rank = node.outputRank;
        node.input1Dim0 = node.outputDim0;
        node.input1Dim1 = node.outputDim1;
        node.input1Dim2 = node.outputDim2;
        node.input1Dim3 = node.outputDim3;
        node.input1ElementCount = node.elementCount;
        if (node.input0Kind == 1 && node.input0Index >= 0 && node.input0Index < externalInputCount) {
            set_input0_shape_from_external(
                    node,
                    node.input0Index,
                    externalInputRanks,
                    externalInputDim0,
                    externalInputDim1,
                    externalInputDim2,
                    externalInputDim3
            );
        } else if (node.input0Kind == 2 && node.input0Index >= 0 && node.input0Index < static_cast<int>(executable->nodes.size())) {
            set_input0_shape_from_node(node, executable->nodes[node.input0Index]);
        }
        if (node.input1Kind == 1 && node.input1Index >= 0 && node.input1Index < externalInputCount) {
            set_input1_shape_from_external(
                    node,
                    node.input1Index,
                    externalInputRanks,
                    externalInputDim0,
                    externalInputDim1,
                    externalInputDim2,
                    externalInputDim3
            );
        } else if (node.input1Kind == 2 && node.input1Index >= 0 && node.input1Index < static_cast<int>(executable->nodes.size())) {
            set_input1_shape_from_node(node, executable->nodes[node.input1Index]);
        }
        node.scalarValue = scalarValues == nullptr ? 0.0f : scalarValues[i];
        node.reductionAxis = decode_reduction_axis(scalarValues, i);
        node.reductionKeepDims = decode_reduction_keep_dims(scalarValues, i);
        if (node.reductionAxis < 0) {
            node.reductionAxis += node.input0Rank;
        }
        if (node.reductionAxis < 0 || node.reductionAxis >= node.input0Rank) {
            node.reductionAxis = 0;
        }
        executable->nodes.push_back(node);
    }
    executable->outputNodeIndices.reserve(outputCount);
    for (int i = 0; i < outputCount; i++) {
        executable->outputNodeIndices.push_back(outputNodeIndices == nullptr ? i : outputNodeIndices[i]);
    }
    return executable;
}

extern "C" int synaptik_cuda_graph_execute_partition_f32(
        void* context,
        void* executable,
        const float** externalInputs,
        int externalInputCount,
        float** outputs,
        int outputCount
) {
    (void) externalInputs;
    (void) externalInputCount;
    (void) outputs;
    (void) outputCount;
    if (context == nullptr || executable == nullptr) {
        stable_reason("CUDA execute requested without context or executable.");
        return 2;
    }
    stable_reason("CUDA graph execution is not implemented by the Phase 6 probe shim.");
    return 1;
}

extern "C" void* synaptik_cuda_graph_create_buffer(
        void* context,
        const void* initialData,
        int byteLength
) {
    if (context == nullptr) {
        stable_reason("CUDA create_buffer requested without a context.");
        return nullptr;
    }
    if (byteLength <= 0) {
        stable_reason("CUDA create_buffer requires positive byte length.");
        return nullptr;
    }
    auto* buffer = new SynaptikCudaBuffer();
    buffer->data = nullptr;
    buffer->byteLength = byteLength;
    cudaError_t status = cudaMalloc(&buffer->data, static_cast<size_t>(byteLength));
    if (status != cudaSuccess) {
        stable_reason(std::string("CUDA buffer allocation failed: ") + cudaGetErrorString(status));
        delete buffer;
        return nullptr;
    }
    if (initialData != nullptr) {
        status = cudaMemcpy(buffer->data, initialData, static_cast<size_t>(byteLength), cudaMemcpyHostToDevice);
        if (status != cudaSuccess) {
            stable_reason(std::string("CUDA buffer upload failed: ") + cudaGetErrorString(status));
            cudaFree(buffer->data);
            delete buffer;
            return nullptr;
        }
    }
    return buffer;
}

extern "C" int synaptik_cuda_graph_read_buffer(
        void* context,
        void* buffer,
        void* destination,
        int byteLength
) {
    if (context == nullptr || buffer == nullptr || destination == nullptr) {
        stable_reason("CUDA read_buffer requested without context, buffer, or destination.");
        return 2;
    }
    auto* cudaBuffer = static_cast<SynaptikCudaBuffer*>(buffer);
    if (!validate_buffer(cudaBuffer, byteLength)) {
        stable_reason("CUDA read_buffer requested with invalid or undersized buffer.");
        return 3;
    }
    cudaError_t status = cudaMemcpy(destination, cudaBuffer->data, static_cast<size_t>(byteLength), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        stable_reason(std::string("CUDA read_buffer failed: ") + cudaGetErrorString(status));
        return 4;
    }
    return 0;
}

extern "C" void synaptik_cuda_graph_destroy_buffer(void* buffer) {
    if (buffer == nullptr) {
        return;
    }
    auto* cudaBuffer = static_cast<SynaptikCudaBuffer*>(buffer);
    if (cudaBuffer->data != nullptr) {
        cudaFree(cudaBuffer->data);
    }
    delete cudaBuffer;
}

__global__ void layout_contiguous_f32_kernel(
        const float* source,
        float* destination,
        int64_t logicalElementCount,
        int rank,
        const int64_t* shape,
        const int64_t* strides,
        int64_t storageOffset
) {
    int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= logicalElementCount) {
        return;
    }
    int64_t sourceIndex = storageOffset;
    int64_t remaining = linear;
    for (int dim = rank - 1; dim >= 0; dim--) {
        int64_t coordinate = remaining % shape[dim];
        remaining /= shape[dim];
        sourceIndex += coordinate * strides[dim];
    }
    destination[linear] = source[sourceIndex];
}

extern "C" int synaptik_cuda_graph_layout_contiguous_f32_buffer(
        void* context,
        void* sourceBuffer,
        void* destinationBuffer,
        int rank,
        int64_t* shape,
        int64_t* strides,
        int64_t storageOffset,
        int64_t logicalElementCount,
        int64_t sourcePhysicalByteSpan,
        int64_t destinationByteLength
) {
    if (context == nullptr || sourceBuffer == nullptr || destinationBuffer == nullptr) {
        stable_reason("CUDA layout materialization requires context and buffers.");
        return 1;
    }
    if (rank <= 0 || rank > 8 || shape == nullptr || strides == nullptr) {
        stable_reason("CUDA layout materialization received invalid rank metadata.");
        return 2;
    }
    if (storageOffset < 0 || logicalElementCount < 0 || sourcePhysicalByteSpan < 0 || destinationByteLength < 0) {
        stable_reason("CUDA layout materialization received negative layout metadata.");
        return 3;
    }
    if (destinationByteLength < logicalElementCount * static_cast<int64_t>(sizeof(float))) {
        stable_reason("CUDA layout materialization destination buffer is too small.");
        return 4;
    }
    auto* source = static_cast<SynaptikCudaBuffer*>(sourceBuffer);
    auto* destination = static_cast<SynaptikCudaBuffer*>(destinationBuffer);
    if (!validate_buffer(source, static_cast<int>(sourcePhysicalByteSpan))
            || !validate_buffer(destination, static_cast<int>(destinationByteLength))) {
        stable_reason("CUDA layout materialization received invalid buffers.");
        return 5;
    }
    for (int i = 0; i < rank; i++) {
        if (shape[i] <= 0 || strides[i] < 0) {
            stable_reason("CUDA layout materialization received unsupported shape/stride metadata.");
            return 6;
        }
    }
    int64_t metadataBytes = static_cast<int64_t>(rank) * static_cast<int64_t>(sizeof(int64_t));
    int64_t* deviceShape = nullptr;
    int64_t* deviceStrides = nullptr;
    cudaError_t status = cudaMalloc(&deviceShape, static_cast<size_t>(metadataBytes));
    if (status != cudaSuccess) {
        stable_reason(std::string("CUDA layout materialization shape allocation failed: ") + cudaGetErrorString(status));
        return 7;
    }
    status = cudaMalloc(&deviceStrides, static_cast<size_t>(metadataBytes));
    if (status != cudaSuccess) {
        cudaFree(deviceShape);
        stable_reason(std::string("CUDA layout materialization stride allocation failed: ") + cudaGetErrorString(status));
        return 8;
    }
    cudaMemcpy(deviceShape, shape, static_cast<size_t>(metadataBytes), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceStrides, strides, static_cast<size_t>(metadataBytes), cudaMemcpyHostToDevice);
    int threads = 256;
    int blocks = static_cast<int>((logicalElementCount + threads - 1) / threads);
    layout_contiguous_f32_kernel<<<blocks, threads>>>(
            static_cast<const float*>(source->data),
            static_cast<float*>(destination->data),
            logicalElementCount,
            rank,
            deviceShape,
            deviceStrides,
            storageOffset
    );
    status = cudaGetLastError();
    cudaFree(deviceShape);
    cudaFree(deviceStrides);
    if (status != cudaSuccess) {
        stable_reason(std::string("CUDA layout materialization kernel launch failed: ") + cudaGetErrorString(status));
        return 9;
    }
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        stable_reason(std::string("CUDA layout materialization kernel execution failed: ") + cudaGetErrorString(status));
        return 10;
    }
    return 0;
}

extern "C" int synaptik_cuda_graph_execute_partition_f32_buffers(
        void* context,
        void* executable,
        void** inputBuffers,
        int inputCount,
        void** outputBuffers,
        int outputCount
) {
    if (context == nullptr || executable == nullptr) {
        stable_reason("CUDA buffer execute requested without context or executable.");
        return 2;
    }
    if (outputBuffers == nullptr || outputCount <= 0) {
        stable_reason("CUDA buffer execute requires at least one output buffer.");
        return 3;
    }
    auto* cudaExecutable = static_cast<SynaptikCudaExecutable*>(executable);
    std::vector<SynaptikCudaBuffer*> nodeOutputs(cudaExecutable->nodes.size(), nullptr);
    std::vector<SynaptikCudaBuffer*> temporaryOutputs;
    auto cleanupTemporaryOutputs = [&temporaryOutputs]() {
        for (auto* buffer : temporaryOutputs) {
            if (buffer != nullptr) {
                if (buffer->data != nullptr) {
                    cudaFree(buffer->data);
                }
                delete buffer;
            }
        }
        temporaryOutputs.clear();
    };
    if (outputCount != static_cast<int>(cudaExecutable->outputNodeIndices.size())) {
        stable_reason("CUDA buffer execute output count does not match executable outputs.");
        return 4;
    }
    for (int i = 0; i < static_cast<int>(cudaExecutable->nodes.size()); i++) {
        const auto& node = cudaExecutable->nodes[i];
        int outputSlot = -1;
        for (int j = 0; j < outputCount; j++) {
            if (cudaExecutable->outputNodeIndices[j] == i) {
                outputSlot = j;
                break;
            }
        }
        SynaptikCudaBuffer* output = nullptr;
        if (outputSlot >= 0) {
            output = static_cast<SynaptikCudaBuffer*>(outputBuffers[outputSlot]);
            if (!validate_buffer(output, node.elementCount * static_cast<int>(sizeof(float)))) {
                cleanupTemporaryOutputs();
                stable_reason("CUDA buffer execute received invalid output buffer.");
                return 5;
            }
        } else {
            output = new SynaptikCudaBuffer();
            output->byteLength = node.elementCount * static_cast<int>(sizeof(float));
            output->data = nullptr;
            cudaError_t allocStatus = cudaMalloc(&output->data, static_cast<size_t>(output->byteLength));
            if (allocStatus != cudaSuccess) {
                delete output;
                cleanupTemporaryOutputs();
                stable_reason(std::string("CUDA buffer execute intermediate allocation failed: ") + cudaGetErrorString(allocStatus));
                return 5;
            }
            temporaryOutputs.push_back(output);
        }
        const float* input0 = nullptr;
        const float* input1 = nullptr;
        if (node.input0Kind == 1) {
            if (inputBuffers == nullptr || node.input0Index < 0 || node.input0Index >= inputCount) {
                cleanupTemporaryOutputs();
                stable_reason("CUDA buffer execute input0 external index is invalid.");
                return 6;
            }
            auto* input = static_cast<SynaptikCudaBuffer*>(inputBuffers[node.input0Index]);
            int requiredBytes = node.input0ElementCount * static_cast<int>(sizeof(float));
            if (!validate_buffer(input, requiredBytes)) {
                cleanupTemporaryOutputs();
                stable_reason("CUDA buffer execute received invalid input0 buffer.");
                return 7;
            }
            input0 = static_cast<const float*>(input->data);
        } else if (node.input0Kind == 2) {
            if (node.input0Index < 0 || node.input0Index >= static_cast<int>(nodeOutputs.size()) || nodeOutputs[node.input0Index] == nullptr) {
                cleanupTemporaryOutputs();
                stable_reason("CUDA buffer execute input0 node index is invalid.");
                return 8;
            }
            input0 = static_cast<const float*>(nodeOutputs[node.input0Index]->data);
        }
        if (node.input1Kind == 1) {
            if (inputBuffers == nullptr || node.input1Index < 0 || node.input1Index >= inputCount) {
                cleanupTemporaryOutputs();
                stable_reason("CUDA buffer execute input1 external index is invalid.");
                return 9;
            }
            auto* input = static_cast<SynaptikCudaBuffer*>(inputBuffers[node.input1Index]);
            if (!validate_buffer(input, node.input1ElementCount * static_cast<int>(sizeof(float)))) {
                cleanupTemporaryOutputs();
                stable_reason("CUDA buffer execute received invalid input1 buffer.");
                return 10;
            }
            input1 = static_cast<const float*>(input->data);
        } else if (node.input1Kind == 2) {
            if (node.input1Index < 0 || node.input1Index >= static_cast<int>(nodeOutputs.size()) || nodeOutputs[node.input1Index] == nullptr) {
                cleanupTemporaryOutputs();
                stable_reason("CUDA buffer execute input1 node index is invalid.");
                return 11;
            }
            input1 = static_cast<const float*>(nodeOutputs[node.input1Index]->data);
        }
        int threads = 256;
        int blocks = (node.elementCount + threads - 1) / threads;
        if (is_unary_op(node.type) && input0 != nullptr) {
            unary_f32_kernel<<<blocks, threads>>>(
                    input0,
                    static_cast<float*>(output->data),
                    node.elementCount,
                    node.type,
                    node.scalarValue
            );
        } else if (is_binary_op(node.type) && input0 != nullptr && input1 != nullptr) {
            if (!suffix_broadcast_supported(
                        node.input0Rank,
                        node.input0Dim0,
                        node.input0Dim1,
                        node.input0Dim2,
                        node.input0Dim3,
                        node.outputRank,
                        node.outputDim0,
                        node.outputDim1,
                        node.outputDim2,
                        node.outputDim3)
                    || !suffix_broadcast_supported(
                        node.input1Rank,
                        node.input1Dim0,
                        node.input1Dim1,
                        node.input1Dim2,
                        node.input1Dim3,
                        node.outputRank,
                        node.outputDim0,
                        node.outputDim1,
                        node.outputDim2,
                        node.outputDim3)) {
                cleanupTemporaryOutputs();
                stable_reason("CUDA buffer execute unsupported broadcast for binary primitive.");
                return 12;
            }
            binary_broadcast_kernel<<<blocks, threads>>>(
                    input0,
                    input1,
                    static_cast<float*>(output->data),
                    node.elementCount,
                    node.type,
                    node.outputRank,
                    node.outputDim0,
                    node.outputDim1,
                    node.outputDim2,
                    node.outputDim3,
                    node.input0Rank,
                    node.input0Dim0,
                    node.input0Dim1,
                    node.input0Dim2,
                    node.input0Dim3,
                    node.input1Rank,
                    node.input1Dim0,
                    node.input1Dim1,
                    node.input1Dim2,
                    node.input1Dim3
            );
        } else if (node.type >= 36 && node.type <= 39 && input0 != nullptr) {
            reduction_kernel<<<blocks, threads>>>(
                    input0,
                    static_cast<float*>(output->data),
                    node.elementCount,
                    node.type,
                    node.reductionAxis,
                    node.input0Rank,
                    node.input0Dim0,
                    node.input0Dim1,
                    node.input0Dim2,
                    node.input0Dim3,
                    node.outputRank,
                    node.outputDim0,
                    node.outputDim1,
                    node.outputDim2,
                    node.outputDim3,
                    node.reductionKeepDims
            );
        } else if (node.type == 90 && input0 != nullptr) {
            int axis = 0;
            int size = 0;
            int step = 0;
            decode_unfold_axis_mode(node.scalarValue, axis, size, step);
            unfold_axis_f32_kernel<<<blocks, threads>>>(
                    input0,
                    static_cast<float*>(output->data),
                    node.elementCount,
                    axis,
                    size,
                    step,
                    node.input0Rank,
                    node.input0Dim0,
                    node.input0Dim1,
                    node.input0Dim2,
                    node.input0Dim3,
                    node.outputRank,
                    node.outputDim0,
                    node.outputDim1,
                    node.outputDim2,
                    node.outputDim3
            );
        } else if ((node.type == 91 || node.type == 92) && input0 != nullptr) {
            int kernelH = 0;
            int kernelW = 0;
            int strideH = 0;
            int strideW = 0;
            int padH = 0;
            int padW = 0;
            decode_window2d_mode(node.scalarValue, kernelH, kernelW, strideH, strideW, padH, padW);
            int batch = node.type == 91 ? node.input0Dim0 : node.outputDim0;
            int channels = node.type == 91 ? node.input0Dim1 : node.outputDim1;
            int height = node.type == 91 ? node.input0Dim2 : node.outputDim2;
            int width = node.type == 91 ? node.input0Dim3 : node.outputDim3;
            int outH = (height + 2 * padH - kernelH) / strideH + 1;
            int outW = (width + 2 * padW - kernelW) / strideW + 1;
            if (kernelH <= 0 || kernelW <= 0 || strideH <= 0 || strideW <= 0 || outH <= 0 || outW <= 0) {
                cleanupTemporaryOutputs();
                stable_reason("CUDA buffer execute received invalid window layout metadata.");
                return 12;
            }
            if (node.type == 91) {
                unfold2d_f32_kernel<<<blocks, threads>>>(
                        input0,
                        static_cast<float*>(output->data),
                        node.elementCount,
                        kernelH,
                        kernelW,
                        strideH,
                        strideW,
                        padH,
                        padW,
                        batch,
                        channels,
                        height,
                        width,
                        outH,
                        outW
                );
            } else {
                fold2d_f32_kernel<<<blocks, threads>>>(
                        input0,
                        static_cast<float*>(output->data),
                        node.elementCount,
                        kernelH,
                        kernelW,
                        strideH,
                        strideW,
                        padH,
                        padW,
                        batch,
                        channels,
                        height,
                        width,
                        outH,
                        outW
                );
            }
        } else {
            cleanupTemporaryOutputs();
            stable_reason("CUDA buffer execute supports unary, broadcast binary, ADD_SCALAR, dense FLOAT32 reductions, and scoped window layout primitives.");
            return 12;
        }
        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) {
            cleanupTemporaryOutputs();
            stable_reason(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(status));
            return 13;
        }
        status = cudaDeviceSynchronize();
        if (status != cudaSuccess) {
            cleanupTemporaryOutputs();
            stable_reason(std::string("CUDA kernel execution failed: ") + cudaGetErrorString(status));
            return 14;
        }
        nodeOutputs[i] = output;
    }
    cleanupTemporaryOutputs();
    return 0;
}

extern "C" void synaptik_cuda_graph_destroy_context(void* context) {
    delete static_cast<SynaptikCudaContext*>(context);
}

extern "C" void synaptik_cuda_graph_destroy_executable(void* executable) {
    delete static_cast<SynaptikCudaExecutable*>(executable);
}
