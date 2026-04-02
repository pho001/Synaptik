package graph.optimizer.memory;

import tensor.DataType;
import tensor.Tensor;

public record ReusableInterval(
        Tensor owner,
        int birthIndex,
        int lastReadIndex,
        int size,
        DataType dataType,
        MemoryRole role
) {
}
