package graph.execution;

import backend.ComputeBackend;
import backend.accelerator.exec.PartitionExecutionRole;
import backend.accelerator.exec.PreparedAcceleratorExecutable;
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
        CpuNodeWorkspace cpuWorkspace,
        PreparedAcceleratorExecutable acceleratorExecutable,
        PartitionExecutionRole partitionRole
) {
    public CompiledNodeExecutionMetadata {
        Objects.requireNonNull(backend, "backend cannot be null");
        partitionRole = partitionRole == null ? PartitionExecutionRole.NONE : partitionRole;
    }
}
