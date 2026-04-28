package graph.execution.trace;

import tensor.DataType;

import java.util.List;

/**
 * Trace entry for a single prepared execution step.
 *
 * @param index step index within the run
 * @param label compiled node label
 * @param opType operation type name
 * @param shape output shape
 * @param dataType output dtype
 * @param backend backend selected for the step
 * @param kernel kernel implementation name, when available
 * @param durationNs step duration in nanoseconds
 * @param metadata structured step metadata
 */
public record ExecutionStepTrace(
        int index,
        String label,
        String opType,
        List<Integer> shape,
        DataType dataType,
        String backend,
        String kernel,
        long durationNs,
        StepExecutionMetadata metadata
) {
    public ExecutionStepTrace {
        label = label == null ? "" : label;
        opType = opType == null ? "UNKNOWN" : opType;
        shape = shape == null ? List.of() : List.copyOf(shape);
        backend = backend == null ? "" : backend;
        kernel = kernel == null ? "" : kernel;
        metadata = metadata == null ? StepExecutionMetadata.none() : metadata;
    }
}
