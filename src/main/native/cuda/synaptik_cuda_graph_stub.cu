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
        int elementCount;
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

bool validate_buffer(SynaptikCudaBuffer* buffer, int byteLength) {
    return buffer != nullptr && buffer->data != nullptr && buffer->byteLength >= byteLength;
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
    (void) scalarValues;
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
        node.elementCount = element_count(
                outputRanks == nullptr ? 1 : outputRanks[i],
                outputDim0 == nullptr ? 1 : outputDim0[i],
                outputDim1 == nullptr ? 1 : outputDim1[i],
                outputDim2 == nullptr ? 1 : outputDim2[i],
                outputDim3 == nullptr ? 1 : outputDim3[i]
        );
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
    for (int i = 0; i < static_cast<int>(cudaExecutable->nodes.size()); i++) {
        const auto& node = cudaExecutable->nodes[i];
        if (i >= outputCount) {
            stable_reason("CUDA buffer execute currently expects one output buffer per node.");
            return 4;
        }
        auto* output = static_cast<SynaptikCudaBuffer*>(outputBuffers[i]);
        if (!validate_buffer(output, node.elementCount * static_cast<int>(sizeof(float)))) {
            stable_reason("CUDA buffer execute received invalid output buffer.");
            return 5;
        }
        const float* input0 = nullptr;
        const float* input1 = nullptr;
        if (node.input0Kind == 1) {
            if (inputBuffers == nullptr || node.input0Index < 0 || node.input0Index >= inputCount) {
                stable_reason("CUDA buffer execute input0 external index is invalid.");
                return 6;
            }
            auto* input = static_cast<SynaptikCudaBuffer*>(inputBuffers[node.input0Index]);
            if (!validate_buffer(input, node.elementCount * static_cast<int>(sizeof(float)))) {
                stable_reason("CUDA buffer execute received invalid input0 buffer.");
                return 7;
            }
            input0 = static_cast<const float*>(input->data);
        } else if (node.input0Kind == 2) {
            if (node.input0Index < 0 || node.input0Index >= static_cast<int>(nodeOutputs.size()) || nodeOutputs[node.input0Index] == nullptr) {
                stable_reason("CUDA buffer execute input0 node index is invalid.");
                return 8;
            }
            input0 = static_cast<const float*>(nodeOutputs[node.input0Index]->data);
        }
        if (node.input1Kind == 1) {
            if (inputBuffers == nullptr || node.input1Index < 0 || node.input1Index >= inputCount) {
                stable_reason("CUDA buffer execute input1 external index is invalid.");
                return 9;
            }
            auto* input = static_cast<SynaptikCudaBuffer*>(inputBuffers[node.input1Index]);
            if (!validate_buffer(input, node.elementCount * static_cast<int>(sizeof(float)))) {
                stable_reason("CUDA buffer execute received invalid input1 buffer.");
                return 10;
            }
            input1 = static_cast<const float*>(input->data);
        } else if (node.input1Kind == 2) {
            if (node.input1Index < 0 || node.input1Index >= static_cast<int>(nodeOutputs.size()) || nodeOutputs[node.input1Index] == nullptr) {
                stable_reason("CUDA buffer execute input1 node index is invalid.");
                return 11;
            }
            input1 = static_cast<const float*>(nodeOutputs[node.input1Index]->data);
        }
        int threads = 256;
        int blocks = (node.elementCount + threads - 1) / threads;
        if (node.type == 7 && input0 != nullptr) {
            relu_kernel<<<blocks, threads>>>(input0, static_cast<float*>(output->data), node.elementCount);
        } else if (node.type == 3 && input0 != nullptr && input1 != nullptr) {
            add_kernel<<<blocks, threads>>>(input0, input1, static_cast<float*>(output->data), node.elementCount);
        } else {
            stable_reason("CUDA buffer execute supports only RELU and ADD in Phase 7.");
            return 12;
        }
        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) {
            stable_reason(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(status));
            return 13;
        }
        status = cudaDeviceSynchronize();
        if (status != cudaSuccess) {
            stable_reason(std::string("CUDA kernel execution failed: ") + cudaGetErrorString(status));
            return 14;
        }
        nodeOutputs[i] = output;
    }
    return 0;
}

extern "C" void synaptik_cuda_graph_destroy_context(void* context) {
    delete static_cast<SynaptikCudaContext*>(context);
}

extern "C" void synaptik_cuda_graph_destroy_executable(void* executable) {
    delete static_cast<SynaptikCudaExecutable*>(executable);
}
