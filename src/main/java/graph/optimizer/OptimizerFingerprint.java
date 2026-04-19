package graph.optimizer;

import tensor.Tensor;

import java.util.List;
import java.util.Objects;

final class OptimizerFingerprint {
    private OptimizerFingerprint() {
    }

    static String of(List<Tensor> graph) {
        Objects.requireNonNull(graph, "graph cannot be null");
        if (graph.isEmpty()) {
            return "graph:empty";
        }
        java.util.IdentityHashMap<Tensor, Integer> ids = new java.util.IdentityHashMap<>();
        for (int i = 0; i < graph.size(); i++) {
            ids.put(graph.get(i), i);
        }
        StringBuilder out = new StringBuilder(graph.size() * 32);
        out.append("nodes=").append(graph.size());
        for (Tensor tensor : graph) {
            out.append('|');
            out.append(':').append(tensor.isBackward());
            out.append(':').append(tensor.getDataType());
            if (tensor.getOperation() == null) {
                out.append(":LEAF");
            } else {
                out.append(':').append(tensor.getOperation().getClass().getName());
                out.append(':').append(tensor.getOperation().opType());
                out.append(':').append(tensor.getOperation().getExpression());
            }
            out.append(':').append(java.util.Arrays.toString(tensor.getShapeUnsafe()));
            out.append(':').append(java.util.Arrays.toString(tensor.getStridesUnsafe()));
            out.append(':').append(tensor.getStorageOffsetUnsafe());
            out.append(':').append(tensor.getRequiresGrad());
            out.append(":grad=");
            Tensor gradient = tensor.getGradient();
            if (gradient == null) {
                out.append("null");
            } else {
                Integer gradientId = ids.get(gradient);
                out.append(gradientId == null ? "external" : gradientId);
                if (gradient.getOperation() == null && gradient.getFlatDataSize() == 1) {
                    out.append(':').append(Double.doubleToLongBits(gradient.scalarAsDouble()));
                }
            }
            if (tensor.getOperation() == null && tensor.getFlatDataSize() == 1) {
                out.append(":scalar=").append(Double.doubleToLongBits(tensor.scalarAsDouble()));
            }
            List<Tensor> prev = tensor.getPrevTensors();
            out.append(":inputs=");
            if (prev != null) {
                for (Tensor input : prev) {
                    Integer id = ids.get(input);
                    out.append(id == null ? ("x" + System.identityHashCode(input)) : id).append(',');
                }
            }
        }
        return out.toString();
    }
}
