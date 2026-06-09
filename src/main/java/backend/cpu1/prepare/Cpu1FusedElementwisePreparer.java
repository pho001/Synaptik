package backend.cpu1.prepare;

import backend.ComputeBackend;
import backend.cpu1.fused.ir.Cpu1FusedExpressionPlan;
import backend.cpu1.fused.ir.Cpu1FusedIrBuilder;
import backend.cpu1.kernels.Cpu1LayoutKind;
import backend.cpu1.kernels.fused.codegen.Cpu1FusedCodegenKernel;
import backend.cpu1.kernels.fused.codegen.Cpu1FusedCodegenKernelFactory;
import backend.cpu1.kernels.fused.codegen.Cpu1FusedCodegenLoopKind;
import backend.cpu1.kernels.fused.codegen.Cpu1FusedCodegenPlan;
import backend.cpu1.kernels.fused.codegen.Cpu1FusedCodegenRejectionReason;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.launch.Cpu1LaunchPolicy;
import backend.cpu1.launch.Cpu1ParallelLaunch;
import backend.cpu1.launch.Cpu1SingleThreadLaunch;
import backend.cpu1.prepare.dispatch.Cpu1DispatchPolicy;
import backend.cpu1.prepare.dispatch.Cpu1FusedDispatchDecision;
import backend.cpu1.storage.Cpu1StorageKind;
import backend.lowering.LoweredExecutionUnit;
import backend.prepare.BackendPrepareContext;
import config.runtime.CpuStorageProfile;
import config.runtime.RuntimeConfig;
import graph.CompiledNode;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.plan.InputResidencyRequirement;
import graph.execution.plan.OutputResidencyEffect;
import operations.Operation;
import tensor.DataType;

import java.util.ArrayList;
import java.util.List;

/**
 * Prepares a lowered cpu1 fused elementwise unit without retaining Operation objects in the result.
 */
public final class Cpu1FusedElementwisePreparer {
    private final RuntimeConfig runtimeConfig;
    private final Cpu1DispatchPolicy dispatchPolicy;

    public Cpu1FusedElementwisePreparer(RuntimeConfig runtimeConfig) {
        if (runtimeConfig == null) {
            throw new IllegalArgumentException("runtimeConfig cannot be null");
        }
        this.runtimeConfig = runtimeConfig;
        this.dispatchPolicy = new Cpu1DispatchPolicy();
    }

    public CompiledNodeExecutionMetadata prepare(
            CompiledNode outputNode,
            LoweredExecutionUnit loweredUnit,
            BackendPrepareContext context
    ) {
        Cpu1PreparedFusedElementwiseUnit preparedUnit = prepareUnit(outputNode, loweredUnit, context);
        return new CompiledNodeExecutionMetadata(
                ComputeBackend.CPU,
                null,
                preparedUnit.inputNodeIds(),
                new Cpu1PreparedArtifact(preparedUnit),
                InputResidencyRequirement.cpuReadableAll(),
                OutputResidencyEffect.cpuCurrentPreserveNative()
        );
    }

    public Cpu1PreparedFusedElementwiseUnit prepareUnit(
            CompiledNode outputNode,
            LoweredExecutionUnit loweredUnit,
            BackendPrepareContext context
    ) {
        if (outputNode == null) {
            throw new IllegalArgumentException("outputNode cannot be null");
        }
        if (loweredUnit == null) {
            throw new IllegalArgumentException("loweredUnit cannot be null");
        }
        if (context == null) {
            throw new IllegalArgumentException("context cannot be null");
        }
        validate(outputNode, loweredUnit, context);
        List<Operation> sourceOperations = sourceOperations(loweredUnit, context);
        Cpu1FusedExpressionPlan plan = Cpu1FusedIrBuilder.build(
                loweredUnit.orderedNodeIds(),
                context::compiledNode,
                context.descriptorIndex()
        );
        boolean useFastExp = runtimeConfig.approximation().useFastExp(context.supportsBackward());
        boolean useFastTanh = runtimeConfig.approximation().useFastTanh(context.supportsBackward());
        Cpu1PrepareConfig config = Cpu1PrepareConfig
                .automatic(runtimeConfig, Runtime.getRuntime().availableProcessors(), storageKindFromRuntime())
                .withApproximation(useFastExp, useFastTanh);
        DataType computeType = computeType(plan);
        Cpu1FusedDispatchDecision dispatchDecision = dispatchPolicy.decideFusedElementwise(
                plan,
                sourceOperations,
                computeType,
                outputNode.flatDataSize(),
                config
        );
        Cpu1LaunchConfig launchConfig = dispatchDecision.launchConfig();
        Cpu1LayoutKind layoutKind = layoutKind(plan, outputNode);
        Cpu1FusedCodegenLoopKind loopKind = Cpu1FusedCodegenLoopKind.select(plan, layoutKind, dispatchDecision);
        Cpu1FusedCodegenPlan codegenPlan = Cpu1FusedCodegenPlan.from(
                plan,
                computeType,
                layoutKind,
                dispatchDecision.storageKind(),
                loopKind,
                config
        );
        Cpu1FusedCodegenRejectionReason rejectionReason = codegenPlan.rejectionReason();
        if (rejectionReason != Cpu1FusedCodegenRejectionReason.NONE) {
            throw Cpu1FusedCodegenKernelFactory.rejection(codegenPlan, rejectionReason);
        }
        Cpu1FusedCodegenKernel generatedKernel = Cpu1FusedCodegenKernelFactory.prepareKernel(codegenPlan);
        return new Cpu1PreparedFusedElementwiseUnit(
                loweredUnit.unitId(),
                loweredUnit.orderedNodeIds(),
                plan.inputs().stream().map(input -> input.nodeId()).toList(),
                outputNode.id(),
                outputNode.dataType(),
                outputNode.flatDataSize(),
                outputNode.shape(),
                plan,
                layoutKind,
                dispatchDecision.storageKind(),
                launchPolicy(launchConfig),
                launchConfig,
                dispatchDecision,
                rejectionReason,
                generatedKernel,
                useFastExp,
                useFastTanh
        );
    }

    private static void validate(
            CompiledNode outputNode,
            LoweredExecutionUnit loweredUnit,
            BackendPrepareContext context
    ) {
        if (loweredUnit.orderedNodeIds().isEmpty()) {
            throw new IllegalStateException("cpu1 fused unit has no ordered nodes: " + loweredUnit.unitId());
        }
        if (loweredUnit.orderedNodeIds().getLast() != outputNode.id()) {
            throw new IllegalStateException("cpu1 fused output must be last ordered node. outputNodeId="
                    + outputNode.id() + ", orderedNodeIds=" + loweredUnit.orderedNodeIds());
        }
        for (int nodeId : loweredUnit.orderedNodeIds()) {
            CompiledNode node = context.compiledNode(nodeId);
            if (node == null) {
                throw new IllegalStateException("cpu1 fused unit references missing nodeId=" + nodeId);
            }
            Operation operation = node.operation();
            if (operation == null || !operation.isFusable()) {
                throw new UnsupportedOperationException("cpu1 fused unit contains non-fusable nodeId=" + nodeId);
            }
        }
    }

    private static List<Operation> sourceOperations(
            LoweredExecutionUnit loweredUnit,
            BackendPrepareContext context
    ) {
        List<Operation> operations = new ArrayList<>(loweredUnit.orderedNodeIds().size());
        for (int nodeId : loweredUnit.orderedNodeIds()) {
            CompiledNode node = context.compiledNode(nodeId);
            operations.add(node == null ? null : node.operation());
        }
        return List.copyOf(operations);
    }

    private static DataType computeType(Cpu1FusedExpressionPlan plan) {
        DataType result = null;
        for (var input : plan.inputs()) {
            result = promote(result, input.dataType());
        }
        for (var node : plan.nodes()) {
            result = promote(result, node.outputType());
        }
        return result == null || result == DataType.BOOL ? DataType.FLOAT32 : result;
    }

    private static DataType promote(DataType current, DataType next) {
        if (next == null || next == DataType.BOOL) {
            return current;
        }
        if (next == DataType.FLOAT64) {
            return DataType.FLOAT64;
        }
        if (next == DataType.FLOAT32) {
            return current == DataType.FLOAT64 ? current : DataType.FLOAT32;
        }
        if (next == DataType.BFLOAT16 && current == null) {
            return DataType.BFLOAT16;
        }
        return current;
    }

    private static Cpu1LayoutKind layoutKind(Cpu1FusedExpressionPlan plan, CompiledNode outputNode) {
        if (outputNode.contiguous() && plan.usesOnlyLinearInputs()) {
            return Cpu1LayoutKind.CONTIGUOUS;
        }
        return switch (outputNode.shape().length) {
            case 2 -> Cpu1LayoutKind.STRIDED_RANK2;
            case 3 -> Cpu1LayoutKind.STRIDED_RANK3;
            case 4 -> Cpu1LayoutKind.STRIDED_RANK4;
            default -> Cpu1LayoutKind.STRIDED_GENERIC;
        };
    }

    private static Cpu1LaunchPolicy launchPolicy(Cpu1LaunchConfig launchConfig) {
        if (launchConfig.workerCount() == 1) {
            return new Cpu1SingleThreadLaunch(launchConfig);
        }
        return new Cpu1ParallelLaunch(launchConfig);
    }

    private Cpu1StorageKind storageKindFromRuntime() {
        return runtimeConfig.cpuStorageProfile() == CpuStorageProfile.CPU_NATIVE
                ? Cpu1StorageKind.MEMORY_SEGMENT
                : Cpu1StorageKind.JAVA_ARRAY;
    }
}
