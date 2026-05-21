package backend.cpu;

import backend.cpu.region.PreparedCpuRegionExecutable;
import backend.runtime.ExecutionContext;
import graph.CompiledNode;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.plan.PreparedExecutionArtifact;
import graph.execution.plan.PreparedRuntimeStateAllocator;
import graph.execution.trace.StepTraceContribution;

public record CpuRegionExecutionArtifact(
        PreparedCpuRegionExecutable executable
) implements PreparedExecutionArtifact {
    @Override
    public void allocateRuntimeState(int nodeId, PreparedRuntimeStateAllocator allocator) {
        if (executable == null) {
            return;
        }
        for (var step : executable.nativeSteps()) {
            if (step != null && step.metadata().artifact() != null) {
                step.metadata().artifact().allocateRuntimeState(step.compiledNode().id(), allocator);
            }
        }
        for (var step : executable.fallbackSteps()) {
            if (step != null && step.metadata().artifact() != null) {
                step.metadata().artifact().allocateRuntimeState(step.compiledNode().id(), allocator);
            }
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
}
