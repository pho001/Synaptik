package graph.execution.state;

import tensor.DataType;
import tensor.Tensor;

import java.util.IdentityHashMap;
import java.util.Map;

/**
 * Run-scoped runtime tensor identity and node-id lookup.
 */
final class RuntimeTensorStore {
    private final Map<Integer, Tensor> runtimeTensorByNodeId;
    private final Map<Tensor, Integer> runtimeNodeIdByTensor;

    RuntimeTensorStore(
            Map<Integer, Tensor> runtimeTensorByNodeId,
            Map<Tensor, Integer> runtimeNodeIdByTensor
    ) {
        this.runtimeTensorByNodeId = Map.copyOf(runtimeTensorByNodeId);
        this.runtimeNodeIdByTensor = new IdentityHashMap<>(runtimeNodeIdByTensor);
    }

    Tensor runtimeTensorForNodeId(int nodeId) {
        Tensor tensor = runtimeTensorByNodeId.get(nodeId);
        if (tensor == null) {
            throw new IllegalStateException("Missing runtime tensor for nodeId=" + nodeId);
        }
        return tensor;
    }

    Integer nodeIdForRuntimeTensor(Tensor tensor) {
        return tensor == null ? null : runtimeNodeIdByTensor.get(tensor);
    }

    long logicalByteLength(int nodeId) {
        Tensor tensor = runtimeTensorForNodeId(nodeId);
        return (long) tensor.getFlatDataSize() * elementByteSize(tensor.getDataType());
    }

    private static int elementByteSize(DataType dataType) {
        if (dataType == null) {
            return 0;
        }
        return switch (dataType) {
            case FLOAT64 -> Double.BYTES;
            case FLOAT32 -> Float.BYTES;
            case BFLOAT16 -> Short.BYTES;
            case BOOL -> Byte.BYTES;
            case INT32 -> Integer.BYTES;
            case INT64 -> Long.BYTES;
        };
    }
}
