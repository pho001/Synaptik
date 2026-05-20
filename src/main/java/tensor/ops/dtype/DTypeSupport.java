package tensor.ops.dtype;

import tensor.DataType;
import tensor.Tensor;

final class DTypeSupport {
    private DTypeSupport() {
    }

    static boolean isFloating(DataType type) {
        return type == DataType.FLOAT64 || type == DataType.FLOAT32 || type == DataType.BFLOAT16;
    }
}
