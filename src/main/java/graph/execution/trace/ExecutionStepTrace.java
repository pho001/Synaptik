package graph.execution.trace;

import tensor.DataType;

import java.util.List;

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
