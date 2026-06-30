package backend.cpu;

import backend.cpu.fused.plan.FusedVectorFallbackReason;
import backend.cpu.fused.exec.PreparedFusedExecutable;
import backend.cpu.kernels.CpuKernel;
import backend.cpu.plan.CpuNodeExecutionPlan;
import backend.cpu.execution.CpuNodeWorkspace;
import backend.cpu.plan.CpuPreparedInput;
import backend.runtime.ExecutionContext;
import graph.model.CompiledNode;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.plan.PreparedExecutionArtifact;
import graph.execution.plan.PreparedRuntimeStateAllocator;
import trace.backend.StepTraceContribution;
import tensor.Tensor;

public record CpuFusedExecutionArtifact(
        CpuKernel cpuKernel,
        CpuNodeExecutionPlan cpuPlan,
        PreparedFusedExecutable fusedExecutable,
        CpuNodeWorkspace cpuWorkspace,
        FusedVectorFallbackReason vectorFallbackReason
) implements PreparedExecutionArtifact {
    public CpuFusedExecutionArtifact {
        vectorFallbackReason = vectorFallbackReason == null ? FusedVectorFallbackReason.NONE : vectorFallbackReason;
    }

    @Override
    public void allocateRuntimeState(int nodeId, PreparedRuntimeStateAllocator allocator) {
        if (allocator == null) {
            return;
        }
        if (cpuWorkspace != null) {
            allocator.putWorkspace(nodeId, allocator.forkWorkspace(cpuWorkspace, cpuWorkspace::fork));
        }
        if (cpuPlan == null || cpuPlan.layoutPlan().preparedInputs().isEmpty()) {
            return;
        }
        for (CpuPreparedInput preparedInput : cpuPlan.layoutPlan().preparedInputs()) {
            allocator.putPreparedInputTensor(
                    nodeId,
                    preparedInput.inputIndex(),
                    runtimePreparedInput(preparedInput.runtimeTensor())
            );
        }
    }

    @Override
    public StepTraceContribution traceContribution(
            CompiledNode node,
            CompiledNodeExecutionMetadata metadata,
            ExecutionContext context
    ) {
        return CpuStepTraceContributor.contribute(node, metadata, context);
    }

    private static Tensor runtimePreparedInput(Tensor template) {
        Tensor runtimePrepared = new Tensor(
                template.getShapeUnsafe().clone(),
                template.getStridesUnsafe().clone(),
                template.getStorageOffsetUnsafe(),
                null,
                null,
                template.getLabel(),
                template.getDataType()
        );
        runtimePrepared.setRequiresGrad(template.getRequiresGrad());
        return runtimePrepared;
    }
}
