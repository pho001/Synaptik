package graph.execution.trace.contrib;

import backend.runtime.ExecutionContext;
import graph.CompiledNode;
import graph.execution.PreparedNodeExecution;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.trace.MatMulTraceMetadata;

record BackendRunTraceContext(
        CompiledNode node,
        PreparedNodeExecution step,
        ExecutionContext executionContext,
        MatMulTraceMetadata matMul
) {
    CompiledNodeExecutionMetadata metadata() {
        return step.metadata();
    }
}
