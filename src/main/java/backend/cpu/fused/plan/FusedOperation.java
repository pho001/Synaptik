package backend.cpu.fused.plan;

import backend.cpu.fused.ir.FusedExpressionPlan;
import backend.cpu.fused.numeric.FusedComputeKind;
import backend.cpu.fused.numeric.FusedNumericContract;
import backend.cpu.fused.numeric.FusedValueLane;
import backend.cpu.fused.runtime.FusedDTypeOps;
import operations.Operation;

/**
 * Operation descriptor for a lowered CPU fused expression.
 */
public final class FusedOperation implements Operation {
    private final String expression;
    private final FusedNumericContract numericContract;
    private final boolean lowCostHint;
    private final FusedDispatchFamily dispatchFamily;
    private final String schedulerSignature;
    private final int dispatchScale;
    private final FusedExpressionPlan plan;

    /**
     * Creates a fused operation descriptor.
     */
    public FusedOperation(
            String expression,
            FusedNumericContract numericContract,
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
        if (numericContract == null) {
            throw new IllegalArgumentException("numericContract cannot be null");
        }
        this.numericContract = numericContract;
        this.lowCostHint = lowCostHint;
        this.dispatchFamily = dispatchFamily;
        this.schedulerSignature = schedulerSignature;
        this.dispatchScale = dispatchScale;
        this.plan = plan;
    }

    /**
     * Returns {@link OpType#FUSED}.
     */
    @Override
    public OpType opType() {
        return OpType.FUSED;
    }

    /**
     * Returns a compact expression label for diagnostics.
     */
    @Override
    public String getExpression() {
        return expression;
    }

    /**
     * Returns whether scheduler policy may treat this fused operation as cheap.
     */
    @Override
    public boolean isCheap() {
        return lowCostHint && dispatchScale == 1;
    }

    /**
     * Returns the numeric precision mode used by generated fused code.
     */
    public int getPrecisionMode() {
        return numericContract.computeKind() == FusedComputeKind.F64
                ? FusedDTypeOps.MODE_F64
                : numericContract.outputValueLane() == FusedValueLane.BF16
                        ? FusedDTypeOps.MODE_BF16
                        : FusedDTypeOps.MODE_F32;
    }

    /**
     * Returns the explicit numeric storage/compute contract for fused execution.
     */
    public FusedNumericContract getNumericContract() {
        return numericContract;
    }

    /**
     * Returns the cost-model cheap hint before dispatch-scale adjustment.
     */
    public boolean isLowCostHint() {
        return lowCostHint;
    }

    /**
     * Returns the dispatch family selected by the fused cost model.
     */
    public FusedDispatchFamily getDispatchFamily() {
        return dispatchFamily;
    }

    /**
     * Returns the cache and scheduler signature for this fused expression.
     */
    public String getSchedulerSignature() {
        return schedulerSignature;
    }

    /**
     * Returns the dispatch-scale multiplier chosen from plan complexity.
     */
    public int getDispatchScale() {
        return dispatchScale;
    }

    /**
     * Returns the lowered expression plan consumed by code generation.
     */
    public FusedExpressionPlan getPlan() {
        return plan;
    }
}
