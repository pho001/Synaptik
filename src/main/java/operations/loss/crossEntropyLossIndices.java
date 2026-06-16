package operations.loss;
import operations.Operation;

import tensor.loss.LossReduction;

import java.util.Objects;

/**
 * Computes cross-entropy loss for integer class-index targets.
 *
 * <p>The descriptor records the class-score dimension, reduction policy, and an
 * optional ignore index. For example, sequence losses can ignore padded target
 * positions while averaging only valid entries.</p>
 */
public final class crossEntropyLossIndices implements Operation {
    private final int classDimension;
    private final LossReduction reduction;
    private final Integer ignoreIndex;

    /**
     * Creates an index-target cross-entropy descriptor.
     *
     * @param classDimension dimension containing class scores
     * @param reduction reduction applied to per-target losses; defaults to
     *        {@link LossReduction#MEAN} when {@code null}
     * @param ignoreIndex optional target value excluded from the loss, or
     *        {@code null} when every target is valid
     */
    public crossEntropyLossIndices(int classDimension, LossReduction reduction, Integer ignoreIndex) {
        this.classDimension = classDimension;
        this.reduction = Objects.requireNonNullElse(reduction, LossReduction.MEAN);
        this.ignoreIndex = ignoreIndex;
    }

    @Override
    public OpType opType() {
        return OpType.CROSS_ENTROPY_LOSS_INDICES;
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
        return OpComputationalCost.EXPENSIVE;
    }

    @Override
    public OpControlTrait controlTrait() {
        return OpControlTrait.NONE;
    }

    @Override
    public OpResultKind resultKind() {
        return OpResultKind.NUMERIC;
    }

    @Override
    public String getExpression() {
        return "crossEntropyLossFromIndices";
    }

    /**
     * Returns the class-score dimension.
     *
     * @return class dimension supplied by the tensor front end
     */
    public int getClassDimension() {
        return classDimension;
    }

    /**
     * Returns the configured loss reduction.
     *
     * @return non-null reduction mode
     */
    public LossReduction getReduction() {
        return reduction;
    }

    /**
     * Returns the ignored target index, if configured.
     *
     * @return ignored index, or {@code null} when no target is ignored
     */
    public Integer getIgnoreIndex() {
        return ignoreIndex;
    }

    /**
     * Indicates whether an ignore index is configured.
     *
     * @return {@code true} when {@link #getIgnoreIndex()} is non-null
     */
    public boolean hasIgnoreIndex() {
        return ignoreIndex != null;
    }
}
