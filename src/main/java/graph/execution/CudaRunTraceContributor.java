package graph.execution;

import backend.cuda.exec.PreparedCudaExecutable;

import java.util.LinkedHashMap;

final class CudaRunTraceContributor implements BackendRunTraceContributor {
    @Override
    public void contribute(BackendRunTraceContext context, LinkedHashMap<String, Object> attrs) {
        if (!(context.metadata().acceleratorExecutable() instanceof PreparedCudaExecutable cuda)) {
            return;
        }
        var cudaStats = cuda.lastExecutionStats();
        attrs.put("cudaBridgeAvailable", cuda.bridge().isAvailable());
        attrs.put("cudaBridgeContextAvailable", cuda.bridgeContext().available());
        attrs.put("cudaBridgeExecutableAvailable", cuda.bridgeExecutable().available());
        attrs.put("cudaSupportsBufferBindings", cuda.bridge().supportsBufferBindings());
        attrs.put("cudaUsedCpuFallback", cudaStats.usedCpuFallback());
        attrs.put("cudaFallbackReason", cudaStats.fallbackReason());
        attrs.put("cudaExecutionPath", cudaStats.executionPath().name());
        attrs.put("cudaExternalInputCount", cudaStats.externalInputCount());
        attrs.put("cudaOutputCount", cudaStats.outputCount());
        attrs.put("cudaInputBytes", cudaStats.inputBytes());
        attrs.put("cudaOutputBytes", cudaStats.outputBytes());
        attrs.put("cudaJavaToNativeCopyNs", cudaStats.javaToNativeCopyNs());
        attrs.put("cudaNativeExecuteNs", cudaStats.nativeExecuteNs());
        attrs.put("cudaNativeDeviceCopyNs", cudaStats.nativeDeviceCopyNs());
        attrs.put("cudaNativeToJavaCopyNs", cudaStats.nativeToJavaCopyNs());
        attrs.put("cudaBridgeTotalNs", cudaStats.totalNs());
        attrs.put("acceleratorInputBytes", cudaStats.inputBytes());
        attrs.put("acceleratorOutputBytes", cudaStats.outputBytes());
        attrs.put("acceleratorJavaToNativeCopyNs", cudaStats.javaToNativeCopyNs());
        attrs.put("acceleratorNativeToJavaCopyNs", cudaStats.nativeToJavaCopyNs());
        attrs.put("acceleratorNativeDeviceCopyNs", cudaStats.nativeDeviceCopyNs());
    }
}
