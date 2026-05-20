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
import backend.cpu.nativecpu.NativeCpuPlanResolver;
import backend.cpu.nativecpu.PreparedNativeCpuPlan;
import config.runtime.BlasConfig;
import config.runtime.Conv2dConfig;
import config.runtime.CpuStorageProfile;
import graph.compile.descriptor.CompiledTensorDescriptor;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import operations.Operation;

import java.util.List;
import java.util.Objects;

public final class CpuPlanAssembler {
    private CpuPlanAssembler() {
    }

    public static CpuNodeExecutionPlan buildExecutionPlan(
            Operation op,
            List<CompiledTensorDescriptor> inputDescriptors,
            CompiledTensorDescriptor nodeDescriptor,
            CompiledTensorDescriptorIndex descriptorIndex,
            CpuExecutionPlanner planner,
            BlasConfig blasConfig,
            Conv2dConfig conv2dConfig,
            CpuStorageProfile cpuStorageProfile,
            boolean publishFloatContinuation,
            ResolvedDispatchHints dispatchHintsOverride
    ) {
        Objects.requireNonNull(planner, "planner cannot be null");
        Objects.requireNonNull(blasConfig, "blasConfig cannot be null");
        Objects.requireNonNull(conv2dConfig, "conv2dConfig cannot be null");

        List<CompiledTensorDescriptor> safeInputDescriptors = inputDescriptors == null ? List.of() : List.copyOf(inputDescriptors);
        CompiledTensorDescriptor safeNodeDescriptor = Objects.requireNonNull(nodeDescriptor, "nodeDescriptor cannot be null");
        PreparedTypeContract typeContract = CpuTypeContractResolver.resolve(op, safeNodeDescriptor, safeInputDescriptors);
        tensor.DataType targetType = typeContract.outputType();

        StridedLayoutDecision layoutDecision = StridedPathEligibility.resolve(op, safeInputDescriptors, safeNodeDescriptor, targetType, planner);
        PreparedInputsResult prepared = PreparedInputPlanner.plan(op, safeInputDescriptors, safeNodeDescriptor, typeContract, planner, layoutDecision);

        ResolvedBroadcastPlan broadcastPlan = BroadcastPlanResolver.resolve(op, prepared.runtimeInputDescriptors(), safeNodeDescriptor);
        ResolvedWhereBroadcastPlan whereBroadcastPlan = BroadcastPlanResolver.resolveWhere(op, prepared.runtimeInputDescriptors(), safeNodeDescriptor);

        CpuLayoutPlan layoutPlan = new CpuLayoutPlan(
                layoutDecision,
                targetType,
                planner.contiguousMaterializeThreshold(),
                broadcastPlan,
                whereBroadcastPlan,
                prepared.preparedInputs()
        );

        ResolvedCpuOperationPlans operationPlans = CpuOperationPlanResolver.resolve(
                op,
                prepared.runtimeInputDescriptors(),
                safeInputDescriptors,
                safeNodeDescriptor,
                descriptorIndex,
                planner,
                blasConfig,
                conv2dConfig,
                cpuStorageProfile,
                publishFloatContinuation,
                dispatchHintsOverride
        );
        PreparedMatMulExecutable matMulExecutable = PreparedMatMulExecutableFactory.create(
                op,
                safeNodeDescriptor,
                operationPlans.matMulHints(),
                publishFloatContinuation
        );
        CpuNodeExecutionPlan basePlan = new CpuNodeExecutionPlan(
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
        PreparedNativeCpuPlan nativeCpuPlan = NativeCpuPlanResolver.resolve(
                op,
                prepared.runtimeInputDescriptors(),
                safeNodeDescriptor.dataType(),
                basePlan,
                matMulExecutable,
                cpuStorageProfile
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
                operationPlans.attentionPlan(),
                nativeCpuPlan
        );
    }
}
