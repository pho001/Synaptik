package graph.execution.trace;

import java.util.Map;

/**
 * Structured metadata attached to an execution step.
 *
 * <p>Only sections relevant to a step are populated. Generic attributes carry stable scalar fields while specialized
 * records describe compute precision, layout, dispatch, reduction, matmul, convolution, or fusion decisions.
 *
 * @param kind coarse metadata kind
 * @param attributes generic metadata attributes
 * @param compute compute precision metadata
 * @param layout layout metadata
 * @param dispatch dispatch scheduler metadata
 * @param reduction reduction metadata
 * @param matMul matrix multiplication metadata
 * @param conv convolution metadata
 * @param fused fused-kernel metadata
 */
public record StepExecutionMetadata(
        String kind,
        Map<String, Object> attributes,
        ComputeTraceMetadata compute,
        LayoutTraceMetadata layout,
        DispatchTraceMetadata dispatch,
        ReductionTraceMetadata reduction,
        MatMulTraceMetadata matMul,
        ConvTraceMetadata conv,
        FusedTraceMetadata fused
) {
    public StepExecutionMetadata {
        kind = (kind == null || kind.isBlank()) ? "generic" : kind;
        attributes = attributes == null ? Map.of() : Map.copyOf(attributes);
    }

    /**
     * Returns an empty metadata value.
     *
     * @return empty metadata
     */
    public static StepExecutionMetadata none() {
        return new StepExecutionMetadata("none", Map.of(), null, null, null, null, null, null, null);
    }
}
