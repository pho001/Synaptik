package graph.execution;

import backend.ComputeBackend;
import backend.accelerator.exec.PartitionExecutionRole;
import backend.accelerator.exec.PreparedAcceleratorExecutable;
import backend.cpu.kernels.CpuKernel;
import backend.cpu.kernels.CpuNodeExecutionPlan;
import backend.cpu.kernels.CpuNodeWorkspace;
import backend.cpu.fused.exec.PreparedFusedExecutable;
import operations.Operation;

import java.util.List;
import java.util.Objects;

public record CompiledNodeExecutionMetadata(
        ComputeBackend backend,
        CpuKernel cpuKernel,
        CpuNodeExecutionPlan cpuPlan,
        PreparedFusedExecutable fusedExecutable,
        CpuNodeWorkspace cpuWorkspace,
        PreparedAcceleratorExecutable acceleratorExecutable,
        Operation executionOperation,
        List<Integer> executionInputNodeIds,
        PartitionExecutionRole partitionRole
) {
    public CompiledNodeExecutionMetadata {
        Objects.requireNonNull(backend, "backend cannot be null");
        executionInputNodeIds = List.copyOf(executionInputNodeIds == null ? List.of() : executionInputNodeIds);
        partitionRole = partitionRole == null ? PartitionExecutionRole.NONE : partitionRole;
    }

    public CompiledNodeExecutionMetadata(
            ComputeBackend backend,
            CpuKernel cpuKernel,
            CpuNodeExecutionPlan cpuPlan,
            PreparedFusedExecutable fusedExecutable,
            CpuNodeWorkspace cpuWorkspace,
            PreparedAcceleratorExecutable acceleratorExecutable,
            PartitionExecutionRole partitionRole
    ) {
        this(backend, cpuKernel, cpuPlan, fusedExecutable, cpuWorkspace, acceleratorExecutable, null, List.of(), partitionRole);
    }
}
