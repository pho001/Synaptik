package backend.cpu.kernels;

import backend.cpu.plan.CpuLayoutPlan;
import backend.runtime.ExecutionContext;
import backend.cpu.kernels.elementwise.plan.ResolvedDispatchHints;
import backend.cpu.kernels.layout.plan.ResolvedBroadcastPlan;
import backend.cpu.kernels.layout.plan.ResolvedWhereBroadcastPlan;
import backend.cpu.kernels.linalg.attention.plan.ResolvedScaledDotProductAttentionPlan;
import backend.cpu.kernels.linalg.matmul.exec.PreparedMatMulExecutable;
import backend.cpu.kernels.linalg.matmul.plan.ResolvedMatMulHints;
import backend.cpu.kernels.nn.conv2d.plan.ResolvedConv2dHints;
import backend.cpu.kernels.reduction.plan.ResolvedReductionHints;
import backend.cpu.nativecpu.PreparedNativeCpuPlan;
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
        ResolvedScaledDotProductAttentionPlan attentionPlan,
        PreparedNativeCpuPlan nativeCpuPlan
) {
    public CpuNodeExecutionPlan {
        Objects.requireNonNull(layoutPlan, "layoutPlan cannot be null");
        plannedWorkers = Math.max(1, plannedWorkers);
        contiguousMaterializeThreshold = Math.max(0, contiguousMaterializeThreshold);
        computeContract = computeContract == null
                ? new ResolvedCpuComputeContract(layoutPlan.targetType(), CpuComputeDType.F64, CpuExecutionBackend.CPU_GENERIC, CpuAccumulateDType.NONE)
                : computeContract;
    }

    public CpuNodeExecutionPlan(
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
        this(
                layoutPlan,
                computeContract,
                publishFloatContinuation,
                plannedWorkers,
                contiguousMaterializeThreshold,
                dispatchHints,
                reductionHints,
                matMulHints,
                matMulExecutable,
                conv2dHints,
                attentionPlan,
                null
        );
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

    public CpuNodeExecutionPlan withNativeCpuPlan(PreparedNativeCpuPlan plan) {
        return new CpuNodeExecutionPlan(
                layoutPlan,
                computeContract,
                publishFloatContinuation,
                plannedWorkers,
                contiguousMaterializeThreshold,
                dispatchHints,
                reductionHints,
                matMulHints,
                matMulExecutable,
                conv2dHints,
                attentionPlan,
                plan
        );
    }
}
