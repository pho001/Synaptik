package operations;

import tensor.loss.LossReduction;

import java.util.Objects;

public final class crossEntropyLossIndices implements Operation {
    private final int classDimension;
    private final LossReduction reduction;
    private final Integer ignoreIndex;

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
    public String getExpression() {
        return "crossEntropyLossFromIndices";
    }

    public int getClassDimension() {
        return classDimension;
    }

    public LossReduction getReduction() {
        return reduction;
    }

    public Integer getIgnoreIndex() {
        return ignoreIndex;
    }

    public boolean hasIgnoreIndex() {
        return ignoreIndex != null;
    }
}
