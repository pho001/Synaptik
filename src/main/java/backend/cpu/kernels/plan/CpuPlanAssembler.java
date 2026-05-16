package backend.cpu.kernels.plan;

import backend.cpu.plan.CpuLayoutPlan;
import backend.cpu.kernels.CpuNodeExecutionPlan;
import backend.cpu.kernels.elementwise.plan.ResolvedDispatchHints;
import backend.cpu.kernels.elementwise.strided.StridedLayoutDecision;
import backend.cpu.kernels.layout.plan.ResolvedBroadcastPlan;
import backend.cpu.kernels.layout.plan.ResolvedWhereBroadcastPlan;
import backend.cpu.kernels.elementwise.strided.StridedPathEligibility;
import backend.cpu.kernels.layout.BroadcastPlanResolver;
import backend.cpu.kernels.layout.PreparedInputPlanner;
import backend.cpu.kernels.layout.PreparedInputsResult;
import backend.cpu.kernels.linalg.matmul.exec.PreparedMatMulExecutable;
import backend.cpu.kernels.linalg.matmul.exec.PreparedMatMulExecutableFactory;
import config.runtime.BlasConfig;
import config.runtime.Conv2dConfig;
import config.runtime.CpuStorageProfile;
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
            CpuStorageProfile cpuStorageProfile,
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
                cpuStorageProfile,
                publishFloatContinuation,
                dispatchHintsOverride
        );
        PreparedMatMulExecutable matMulExecutable = PreparedMatMulExecutableFactory.create(
                op,
                node,
                operationPlans.matMulHints(),
                publishFloatContinuation
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
                matMulExecutable,
                operationPlans.conv2dHints(),
                operationPlans.attentionPlan()
        );
    }
}
