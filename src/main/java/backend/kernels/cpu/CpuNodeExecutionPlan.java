package backend.kernels.cpu;

import backend.CpuLayoutPlan;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;
import java.util.Objects;

public record CpuNodeExecutionPlan(
        CpuLayoutPlan layoutPlan,
        CpuComputeMode computeMode,
        boolean publishFloatContinuation,
        ResolvedDispatchHints dispatchHints,
        ResolvedReductionHints reductionHints,
        ResolvedMatMulHints matMulHints
) {
    public CpuNodeExecutionPlan {
        Objects.requireNonNull(layoutPlan, "layoutPlan cannot be null");
        computeMode = computeMode == null ? CpuComputeMode.F64 : computeMode;
    }

    public List<Tensor> apply(List<Tensor> originalInputs) {
        return layoutPlan.apply(originalInputs);
    }

    public boolean stridedPath() {
        return layoutPlan.stridedPath();
    }

    public DataType targetType() {
        return layoutPlan.targetType();
    }

    public ResolvedBroadcastPlan broadcastPlan() {
        return layoutPlan.broadcastPlan();
    }

    public ResolvedWhereBroadcastPlan whereBroadcastPlan() {
        return layoutPlan.whereBroadcastPlan();
    }
}
