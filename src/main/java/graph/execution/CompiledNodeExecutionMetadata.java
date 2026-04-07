package graph.execution;

import backend.ComputeBackend;
import backend.kernels.cpu.CpuKernel;
import backend.kernels.cpu.CpuNodeExecutionPlan;
import backend.kernels.cpu.CpuNodeWorkspace;
import graph.fused.PreparedFusedExecutable;

import java.util.Objects;

public record CompiledNodeExecutionMetadata(
        ComputeBackend backend,
        CpuKernel cpuKernel,
        CpuNodeExecutionPlan cpuPlan,
        PreparedFusedExecutable fusedExecutable,
        CpuNodeWorkspace cpuWorkspace
) {
    public CompiledNodeExecutionMetadata {
        Objects.requireNonNull(backend, "backend cannot be null");
    }
}
