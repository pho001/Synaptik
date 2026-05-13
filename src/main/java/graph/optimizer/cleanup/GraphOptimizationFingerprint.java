package graph.optimizer.cleanup;

import operations.Operation;
import tensor.DataType;
import tensor.Tensor;

import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.List;

/**
 * Stable structural fingerprint for cleanup fixpoint convergence checks.
 */
public record GraphOptimizationFingerprint(String value) {
    public static GraphOptimizationFingerprint capture(List<Tensor> graph, Tensor forwardOutput) {
        IdentityHashMap<Tensor, Integer> ids = new IdentityHashMap<>();
        for (int i = 0; i < graph.size(); i++) {
            ids.put(graph.get(i), i);
        }
        StringBuilder out = new StringBuilder(graph.size() * 96);
        out.append("forward=").append(ids.getOrDefault(forwardOutput, -1)).append(';');
        for (int i = 0; i < graph.size(); i++) {
            Tensor tensor = graph.get(i);
            Operation operation = tensor.getOperation();
            out.append(i)
                    .append(':')
                    .append(operation == null ? "LEAF" : operation.opType().name())
                    .append(':')
                    .append(operation == null ? "" : operation.getExpression())
                    .append(':')
                    .append(tensor.getDataType())
                    .append(':')
                    .append(Arrays.toString(tensor.getShapeUnsafe()))
                    .append(':')
                    .append(Arrays.toString(tensor.getStridesUnsafe()))
                    .append(':')
                    .append(tensor.getStorageOffsetUnsafe())
                    .append(':')
                    .append(tensor.getRequiresGrad())
                    .append(':')
                    .append(tensor.isBackward());
            List<Tensor> inputs = tensor.getPrevTensors();
            if (inputs != null && !inputs.isEmpty()) {
                out.append(":in=");
                for (Tensor input : inputs) {
                    out.append(ids.getOrDefault(input, -1)).append(',');
                }
            }
            if (operation == null && !tensor.getRequiresGrad()) {
                out.append(":const=").append(constantHash(tensor));
            }
            out.append('|');
        }
        return new GraphOptimizationFingerprint(out.toString());
    }

    private static int constantHash(Tensor tensor) {
        if (tensor.getFlatDataSize() > 32) {
            return tensor.getFlatDataSize();
        }
        DataType dataType = tensor.getDataType();
        return switch (dataType) {
            case BOOL -> Arrays.hashCode(tensor.toBooleanArrayCopy());
            case INT32 -> Arrays.hashCode(tensor.getInt32Data());
            case INT64 -> Arrays.hashCode(tensor.getInt64Data());
            case FLOAT32, FLOAT64, BFLOAT16 -> Arrays.hashCode(tensor.toDoubleArrayCopy());
        };
    }
}
