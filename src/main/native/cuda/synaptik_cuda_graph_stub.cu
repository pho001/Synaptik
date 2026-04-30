#include <cuda_runtime_api.h>

#include <cstdlib>
#include <cstring>
#include <string>

namespace {

struct SynaptikCudaContext {
    int device;
};

struct SynaptikCudaExecutable {
    int nodeCount;
    int outputCount;
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
    (void) outputRanks;
    (void) outputDim0;
    (void) outputDim1;
    (void) outputDim2;
    (void) outputDim3;
    (void) outputNodeIndices;
    if (context == nullptr) {
        stable_reason("CUDA compile requested without a context.");
        return nullptr;
    }
    if (nodeCount <= 0 || outputCount <= 0) {
        stable_reason("CUDA compile requires at least one node and one output.");
        return nullptr;
    }
    auto* executable = new SynaptikCudaExecutable();
    executable->nodeCount = nodeCount;
    executable->outputCount = outputCount;
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

extern "C" void synaptik_cuda_graph_destroy_context(void* context) {
    delete static_cast<SynaptikCudaContext*>(context);
}

extern "C" void synaptik_cuda_graph_destroy_executable(void* executable) {
    delete static_cast<SynaptikCudaExecutable*>(executable);
}
