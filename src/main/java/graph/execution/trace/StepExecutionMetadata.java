package graph.execution.trace;

import java.util.Map;

public record StepExecutionMetadata(
        String kind,
        Map<String, Object> attributes,
        LayoutTraceMetadata layout,
        DispatchTraceMetadata dispatch,
        ReductionTraceMetadata reduction,
        MatMulTraceMetadata matMul,
        FusedTraceMetadata fused
) {
    public StepExecutionMetadata {
        kind = (kind == null || kind.isBlank()) ? "generic" : kind;
        attributes = attributes == null ? Map.of() : Map.copyOf(attributes);
    }

    public static StepExecutionMetadata none() {
        return new StepExecutionMetadata("none", Map.of(), null, null, null, null, null);
    }
}
