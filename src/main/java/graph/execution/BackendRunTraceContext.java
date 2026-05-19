package graph.execution;

import backend.runtime.ExecutionContext;
import graph.CompiledNode;
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
