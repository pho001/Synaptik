package backend.cpu.fused.plan;

import backend.cpu.fused.codegen.FusedExpressionPlan;
import backend.cpu.fused.optimize.FusedDispatchFamily;
import operations.Operation;

public final class FusedOperation implements Operation {
    private final String expression;
    private final int precisionMode;
    private final boolean lowCostHint;
    private final FusedDispatchFamily dispatchFamily;
    private final String schedulerSignature;
    private final int dispatchScale;
    private final FusedExpressionPlan plan;

    public FusedOperation(
            String expression,
            int precisionMode,
            boolean lowCostHint,
            FusedDispatchFamily dispatchFamily,
            String schedulerSignature,
            int dispatchScale,
            FusedExpressionPlan plan
    ) {
        if (expression == null || expression.isBlank()) {
            throw new IllegalArgumentException("expression cannot be blank");
        }
        if (schedulerSignature == null || schedulerSignature.isBlank()) {
            throw new IllegalArgumentException("schedulerSignature cannot be blank");
        }
        if (dispatchFamily == null) {
            throw new IllegalArgumentException("dispatchFamily cannot be null");
        }
        if (plan == null) {
            throw new IllegalArgumentException("plan cannot be null");
        }
        if (dispatchScale < 1) {
            throw new IllegalArgumentException("dispatchScale must be >= 1");
        }

        this.expression = expression;
        this.precisionMode = precisionMode;
        this.lowCostHint = lowCostHint;
        this.dispatchFamily = dispatchFamily;
        this.schedulerSignature = schedulerSignature;
        this.dispatchScale = dispatchScale;
        this.plan = plan;
    }

    @Override
    public OpType opType() {
        return OpType.FUSED;
    }

    @Override
    public String getExpression() {
        return expression;
    }

    @Override
    public boolean isCheap() {
        return lowCostHint && dispatchScale == 1;
    }

    public int getPrecisionMode() {
        return precisionMode;
    }

    public boolean isLowCostHint() {
        return lowCostHint;
    }

    public FusedDispatchFamily getDispatchFamily() {
        return dispatchFamily;
    }

    public String getSchedulerSignature() {
        return schedulerSignature;
    }

    public int getDispatchScale() {
        return dispatchScale;
    }

    public FusedExpressionPlan getPlan() {
        return plan;
    }
}
