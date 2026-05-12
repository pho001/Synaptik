package operations.dtype;

import operations.Operation;
import tensor.DataType;

import java.util.Objects;

/**
 * Converts tensor values to a target dtype as an explicit graph operation.
 */
public final class cast implements Operation {
    private final DataType targetType;

    public cast(DataType targetType) {
        this.targetType = Objects.requireNonNull(targetType, "targetType cannot be null");
    }

    public DataType getTargetType() {
        return targetType;
    }

    @Override
    public OpType opType() {
        return OpType.CAST;
    }

    @Override
    public String getExpression() {
        return "cast(" + targetType + ")";
    }
}
