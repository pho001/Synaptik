package backend.kernels.cpu.plan;

import backend.CpuLayoutPlan;
import backend.kernels.cpu.CpuNodeExecutionPlan;
import backend.kernels.cpu.elementwise.plan.ResolvedDispatchHints;
import backend.kernels.cpu.elementwise.strided.StridedLayoutDecision;
import backend.kernels.cpu.layout.plan.ResolvedBroadcastPlan;
import backend.kernels.cpu.layout.plan.ResolvedWhereBroadcastPlan;
import backend.kernels.cpu.elementwise.strided.StridedPathEligibility;
import backend.kernels.cpu.layout.BroadcastPlanResolver;
import backend.kernels.cpu.layout.PreparedInputPlanner;
import backend.kernels.cpu.layout.PreparedInputsResult;
import config.runtime.BlasConfig;
import config.runtime.Conv2dConfig;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;
import java.util.Objects;

public final class CpuPlanAssembler {
    private CpuPlanAssembler() {
    }

    public static CpuNodeExecutionPlan buildExecutionPlan(
            Operation op,
            List<Tensor> inputs,
            Tensor node,
            CpuExecutionPlanner planner,
            BlasConfig blasConfig,
            Conv2dConfig conv2dConfig,
            boolean publishFloatContinuation,
            ResolvedDispatchHints dispatchHintsOverride
    ) {
        Objects.requireNonNull(node, "node cannot be null");
        Objects.requireNonNull(planner, "planner cannot be null");
        Objects.requireNonNull(blasConfig, "blasConfig cannot be null");
        Objects.requireNonNull(conv2dConfig, "conv2dConfig cannot be null");

        List<Tensor> safeInputs = inputs == null ? List.of() : List.copyOf(inputs);
        PreparedTypeContract typeContract = CpuTypeContractResolver.resolve(op, node, safeInputs);
        DataType targetType = typeContract.outputType();

        StridedLayoutDecision layoutDecision = StridedPathEligibility.resolve(op, safeInputs, node, targetType, planner);
        PreparedInputsResult prepared = PreparedInputPlanner.plan(op, safeInputs, node, typeContract, planner, layoutDecision);

        ResolvedBroadcastPlan broadcastPlan = BroadcastPlanResolver.resolve(op, prepared.runtimeInputs(), node);
        ResolvedWhereBroadcastPlan whereBroadcastPlan = BroadcastPlanResolver.resolveWhere(op, prepared.runtimeInputs(), node);

        CpuLayoutPlan layoutPlan = new CpuLayoutPlan(
                layoutDecision,
                targetType,
                planner.contiguousMaterializeThreshold(),
                broadcastPlan,
                whereBroadcastPlan,
                prepared.preparedInputs(),
                prepared.runtimeInputs()
        );

        ResolvedCpuOperationPlans operationPlans = CpuOperationPlanResolver.resolve(
                op,
                prepared.runtimeInputs(),
                node,
                planner,
                blasConfig,
                conv2dConfig,
                dispatchHintsOverride
        );
        return new CpuNodeExecutionPlan(
                layoutPlan,
                operationPlans.computeContract(),
                publishFloatContinuation,
                planner.plannedWorkers(),
                planner.contiguousMaterializeThreshold(),
                operationPlans.dispatchHints(),
                operationPlans.reductionHints(),
                operationPlans.matMulHints(),
                operationPlans.conv2dHints(),
                operationPlans.attentionPlan()
        );
    }
}
