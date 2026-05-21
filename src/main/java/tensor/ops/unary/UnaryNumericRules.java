package tensor.ops.unary;

import tensor.DataType;
import tensor.Tensor;

final class UnaryNumericRules {
    private UnaryNumericRules() {
    }

    static void requireNumeric(Tensor input, String opName) {
        if (input == null) {
            throw new IllegalArgumentException(opName + " input cannot be null");
        }
        if (input.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException(opName + " requires numeric input.");
        }
    }
}
