package graph.execution.trace;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Backend-owned trace metadata contribution for one prepared execution step.
 */
public record StepTraceContribution(
        String kernel,
        Map<String, Object> attributes,
        ComputeTraceMetadata compute,
        LayoutTraceMetadata layout,
        DispatchTraceMetadata dispatch,
        ReductionTraceMetadata reduction,
        MatMulTraceMetadata matMul,
        ConvTraceMetadata conv,
        FusedTraceMetadata fused
) {
    public StepTraceContribution {
        kernel = kernel == null ? "" : kernel;
        attributes = attributes == null
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(attributes));
    }

    public static StepTraceContribution empty() {
        return new StepTraceContribution("", Map.of(), null, null, null, null, null, null, null);
    }
}
