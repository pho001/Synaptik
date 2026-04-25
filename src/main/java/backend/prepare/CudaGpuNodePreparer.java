package backend.prepare;

import backend.ComputeBackend;
import backend.accelerator.exec.PartitionExecutionRole;
import graph.CompiledNode;
import graph.execution.CompiledNodeExecutionMetadata;

final class CudaGpuNodePreparer {
    private final CpuNodePreparer cpuPreparer;

    CudaGpuNodePreparer(CpuNodePreparer cpuPreparer) {
        this.cpuPreparer = cpuPreparer;
    }

    CompiledNodeExecutionMetadata prepare(CompiledNode node, BackendPrepareContext context) {
        PartitionExecutionRole role = context.partitionRoleFor(node.id());
        if (role == PartitionExecutionRole.INTERIOR) {
            return new CompiledNodeExecutionMetadata(ComputeBackend.GPU_CUDA, null, null, null, null, null, role);
        }
        return cpuPreparer.prepareAsCpu(node, context);
    }
}
