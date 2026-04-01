package graph.execution;

import backend.ComputeBackend;
import backend.kernels.cpu.CpuKernel;
import backend.kernels.cpu.CpuNodeExecutionPlan;
import backend.kernels.cpu.fused.CompiledFusedKernel;

import java.util.Objects;

public record CompiledNodeExecutionMetadata(
        ComputeBackend backend,
        CpuKernel cpuKernel,
        CpuNodeExecutionPlan cpuPlan,
        CompiledFusedKernel fusedKernel
) {
    public CompiledNodeExecutionMetadata {
        Objects.requireNonNull(backend, "backend cannot be null");
    }
}
