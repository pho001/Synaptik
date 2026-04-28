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

/**
 * Prepare-time execution metadata for a compiled node.
 *
 * @param backend backend selected for execution
 * @param cpuKernel CPU kernel selected for the node, when applicable
 * @param cpuPlan CPU execution plan, when applicable
 * @param fusedExecutable prepared fused executable, when applicable
 * @param cpuWorkspace CPU workspace template, when applicable
 * @param acceleratorExecutable prepared accelerator executable, when applicable
 * @param executionOperation operation to execute instead of the compiled semantic operation, when present
 * @param executionInputNodeIds node ids to use as execution inputs
 * @param partitionRole role of this node in partitioned execution
 */
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

    /**
     * Creates metadata for a node whose execution operation and input ids match compile-time defaults.
     *
     * @param backend backend selected for execution
     * @param cpuKernel CPU kernel selected for the node, when applicable
     * @param cpuPlan CPU execution plan, when applicable
     * @param fusedExecutable prepared fused executable, when applicable
     * @param cpuWorkspace CPU workspace template, when applicable
     * @param acceleratorExecutable prepared accelerator executable, when applicable
     * @param partitionRole role of this node in partitioned execution
     */
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
