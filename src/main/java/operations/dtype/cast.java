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
    public OpArityClass arityClass() {
        return OpArityClass.SPECIAL;
    }

    @Override
    public boolean isFusable() {
        return false;
    }

    @Override
    public OpSemanticFamily semanticFamily() {
        return OpSemanticFamily.SPECIAL;
    }

    @Override
    public OpComputationalCost computationalCost() {
        return OpComputationalCost.CHEAP;
    }

    @Override
    public OpResultKind resultKind() {
        return OpResultKind.NUMERIC;
    }

    @Override
    public String getExpression() {
        return "cast(" + targetType + ")";
    }
}
