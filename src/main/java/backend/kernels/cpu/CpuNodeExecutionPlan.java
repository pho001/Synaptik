package backend.kernels.cpu;

import backend.CpuLayoutPlan;
import backend.runtime.ExecutionContext;
import backend.kernels.cpu.elementwise.plan.ResolvedDispatchHints;
import backend.kernels.cpu.layout.plan.ResolvedBroadcastPlan;
import backend.kernels.cpu.layout.plan.ResolvedWhereBroadcastPlan;
import backend.kernels.cpu.linalg.attention.plan.ResolvedScaledDotProductAttentionPlan;
import backend.kernels.cpu.linalg.matmul.exec.PreparedMatMulExecutable;
import backend.kernels.cpu.linalg.matmul.plan.ResolvedMatMulHints;
import backend.kernels.cpu.nn.conv2d.plan.ResolvedConv2dHints;
import backend.kernels.cpu.reduction.plan.ResolvedReductionHints;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;
import java.util.Objects;

public record CpuNodeExecutionPlan(
        CpuLayoutPlan layoutPlan,
        ResolvedCpuComputeContract computeContract,
        boolean publishFloatContinuation,
        int plannedWorkers,
        int contiguousMaterializeThreshold,
        ResolvedDispatchHints dispatchHints,
        ResolvedReductionHints reductionHints,
        ResolvedMatMulHints matMulHints,
        PreparedMatMulExecutable matMulExecutable,
        ResolvedConv2dHints conv2dHints,
        ResolvedScaledDotProductAttentionPlan attentionPlan
) {
    public CpuNodeExecutionPlan {
        Objects.requireNonNull(layoutPlan, "layoutPlan cannot be null");
        plannedWorkers = Math.max(1, plannedWorkers);
        contiguousMaterializeThreshold = Math.max(0, contiguousMaterializeThreshold);
        computeContract = computeContract == null
                ? new ResolvedCpuComputeContract(layoutPlan.targetType(), CpuComputeDType.F64, CpuExecutionBackend.CPU_GENERIC, CpuAccumulateDType.NONE)
                : computeContract;
    }

    public List<Tensor> apply(int nodeId, List<Tensor> originalInputs, ExecutionContext executionContext) {
        return layoutPlan.apply(nodeId, originalInputs, executionContext);
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
