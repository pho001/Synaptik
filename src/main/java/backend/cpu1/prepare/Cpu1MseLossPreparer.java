package backend.cpu1.prepare;

import backend.ComputeBackend;
import backend.cpu1.exec.Cpu1WorkspaceSpec;
import backend.cpu1.kernels.loss.mse.Cpu1MseLossKernelId;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.launch.Cpu1RangeLauncher;
import backend.cpu1.storage.Cpu1StorageKind;
import backend.lowering.LoweredExecutionUnit;
import backend.lowering.region.CpuSpecializedPrimitivePayload;
import backend.lowering.region.RegionExecutionPlan;
import backend.prepare.BackendPrepareContext;
import config.backend.CpuKernelConfig;
import config.runtime.CpuStorageProfile;
import config.runtime.RuntimeConfig;
import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptor;
import graph.compile.planning.region.specialization.RegionSpecializationCandidate;
import graph.compile.planning.region.specialization.RegionSpecializationKind;
import graph.compile.planning.value.GraphValueRef;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.plan.InputResidencyRequirement;
import graph.execution.plan.OutputResidencyEffect;
import operations.Operation;
import tensor.DataType;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public final class Cpu1MseLossPreparer {
    private final RuntimeConfig runtimeConfig;

    public Cpu1MseLossPreparer(RuntimeConfig runtimeConfig) {
        this.runtimeConfig = Objects.requireNonNull(runtimeConfig, "runtimeConfig cannot be null");
    }

    public CompiledNodeExecutionMetadata prepare(
            CompiledNode outputNode,
            LoweredExecutionUnit loweredUnit,
            BackendPrepareContext context
    ) {
        Objects.requireNonNull(outputNode, "outputNode cannot be null");
        Objects.requireNonNull(loweredUnit, "loweredUnit cannot be null");
        Objects.requireNonNull(context, "context cannot be null");

        RegionExecutionPlan plan = loweredUnit.requireRegionPlan();
        RegionSpecializationCandidate candidate = requireMseCandidate(plan);
        if (candidate.outputValueRef().nodeId() != outputNode.id()) {
            throw new IllegalStateException("MSE specialization output node mismatch. candidate="
                    + candidate.outputValueRef().nodeId() + ", outputNode=" + outputNode.id());
        }
        validateStructure(candidate, context);

        List<Integer> inputNodeIds = candidate.inputValueRefs().stream()
                .map(GraphValueRef::nodeId)
                .toList();
        int predictionNodeId = inputNodeIds.get(0);
        int targetNodeId = inputNodeIds.get(1);
        CompiledTensorDescriptor prediction = context.descriptor(predictionNodeId);
        CompiledTensorDescriptor target = context.descriptor(targetNodeId);
        CompiledTensorDescriptor output = context.descriptor(outputNode.id());
        requireDenseScalarContract(outputNode, prediction, target, output);

        Operation.OpType reductionOpType = outputNode.operation().opType();
        int elementCount = Math.toIntExact(prediction.logicalElementCount());
        Cpu1StorageKind storageKind = storageKind(output.dataType(), runtimeConfig.cpuStorageProfile());
        Cpu1LaunchConfig launchConfig = launchConfig(elementCount, runtimeConfig);
        Cpu1PreparedMseLossUnit preparedUnit = new Cpu1PreparedMseLossUnit(
                outputNode.id(),
                predictionNodeId,
                targetNodeId,
                output.dataType(),
                storageKind,
                kernelId(output.dataType(), reductionOpType),
                reductionOpType,
                elementCount,
                candidate.orderedNodeIds(),
                launchConfig,
                workspaceSpec(storageKind, output.dataType(), launchConfig, elementCount)
        );
        return new CompiledNodeExecutionMetadata(
                ComputeBackend.CPU,
                null,
                inputNodeIds,
                new Cpu1PreparedArtifact(preparedUnit),
                inputResidencyRequirement(storageKind),
                OutputResidencyEffect.cpuCurrentPreserveNative()
        );
    }

    private static RegionSpecializationCandidate requireMseCandidate(RegionExecutionPlan plan) {
        if (!(plan.backendPayload() instanceof CpuSpecializedPrimitivePayload payload)) {
            throw new IllegalStateException("CPU specialized MSE prepare requires CpuSpecializedPrimitivePayload.");
        }
        RegionSpecializationCandidate candidate = payload.candidate();
        if (candidate.kind() != RegionSpecializationKind.MSE_LOSS) {
            throw new UnsupportedOperationException("cpu1 specialized preparer does not support " + candidate.kind());
        }
        if (candidate.inputValueRefs().size() != 2) {
            throw new UnsupportedOperationException("cpu1 MSE_LOSS expects exactly two external inputs, got "
                    + candidate.inputValueRefs().size());
        }
        return candidate;
    }

    private static void validateStructure(
            RegionSpecializationCandidate candidate,
            BackendPrepareContext context
    ) {
        if (candidate.orderedNodeIds().size() != 3) {
            throw new UnsupportedOperationException("cpu1 MSE_LOSS supports SUB->MUL->single scalar SUM/MEAN only, got nodes "
                    + candidate.orderedNodeIds());
        }
        CompiledNode diff = context.compiledNode(candidate.orderedNodeIds().get(0));
        CompiledNode square = context.compiledNode(candidate.orderedNodeIds().get(1));
        CompiledNode reduction = context.compiledNode(candidate.orderedNodeIds().get(2));
        if (opType(diff) != Operation.OpType.SUB) {
            throw new UnsupportedOperationException("cpu1 MSE_LOSS first node must be SUB.");
        }
        if (opType(square) != Operation.OpType.MUL
                || square.inputIds().size() != 2
                || square.inputIds().get(0) != diff.id()
                || square.inputIds().get(1) != diff.id()) {
            throw new UnsupportedOperationException("cpu1 MSE_LOSS second node must be MUL(diff,diff).");
        }
        Operation.OpType reductionOpType = opType(reduction);
        if (reductionOpType != Operation.OpType.SUM && reductionOpType != Operation.OpType.MEAN) {
            throw new UnsupportedOperationException("cpu1 MSE_LOSS reduction must be SUM or MEAN.");
        }
        List<Integer> expectedInputs = candidate.inputValueRefs().stream()
                .map(GraphValueRef::nodeId)
                .toList();
        if (!diff.inputIds().equals(expectedInputs)) {
            throw new UnsupportedOperationException("cpu1 MSE_LOSS SUB inputs do not match candidate inputs.");
        }
        if (reduction.inputIds().size() != 1 || reduction.inputIds().getFirst() != square.id()) {
            throw new UnsupportedOperationException("cpu1 MSE_LOSS reduction must consume the squared diff.");
        }
    }

    private static void requireDenseScalarContract(
            CompiledNode outputNode,
            CompiledTensorDescriptor prediction,
            CompiledTensorDescriptor target,
            CompiledTensorDescriptor output
    ) {
        if (prediction.dataType() != target.dataType() || prediction.dataType() != output.dataType()) {
            throw new UnsupportedOperationException("cpu1 MSE_LOSS requires matching pred/target/output dtype, got "
                    + prediction.dataType() + "/" + target.dataType() + "/" + output.dataType());
        }
        if (!isSupportedDataType(output.dataType())) {
            throw new UnsupportedOperationException("cpu1 MSE_LOSS does not support dtype " + output.dataType());
        }
        if (!Arrays.equals(prediction.shape(), target.shape())) {
            throw new UnsupportedOperationException("cpu1 MSE_LOSS requires equal pred/target shapes, got "
                    + Arrays.toString(prediction.shape()) + " and " + Arrays.toString(target.shape()));
        }
        if (!prediction.denseContiguousWithoutOffset() || !target.denseContiguousWithoutOffset()) {
            throw new UnsupportedOperationException("cpu1 MSE_LOSS requires dense contiguous pred/target inputs.");
        }
        if (output.logicalElementCount() != 1 || outputNode.flatDataSize() != 1) {
            throw new UnsupportedOperationException("cpu1 MSE_LOSS requires scalar output, got elementCount="
                    + output.logicalElementCount());
        }
        if (prediction.logicalElementCount() <= 0 || prediction.logicalElementCount() > Integer.MAX_VALUE) {
            throw new UnsupportedOperationException("cpu1 MSE_LOSS input element count is unsupported: "
                    + prediction.logicalElementCount());
        }
    }

    private static Cpu1MseLossKernelId kernelId(DataType dataType, Operation.OpType reductionOpType) {
        boolean mean = reductionOpType == Operation.OpType.MEAN;
        return switch (dataType) {
            case FLOAT32 -> mean
                    ? Cpu1MseLossKernelId.MSE_MEAN_F32_DENSE_SCALAR
                    : Cpu1MseLossKernelId.MSE_SUM_F32_DENSE_SCALAR;
            case FLOAT64 -> mean
                    ? Cpu1MseLossKernelId.MSE_MEAN_F64_DENSE_SCALAR
                    : Cpu1MseLossKernelId.MSE_SUM_F64_DENSE_SCALAR;
            case BFLOAT16 -> mean
                    ? Cpu1MseLossKernelId.MSE_MEAN_BF16_DENSE_SCALAR
                    : Cpu1MseLossKernelId.MSE_SUM_BF16_DENSE_SCALAR;
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException("cpu1 MSE_LOSS requires floating dtype.");
        };
    }

    private static boolean isSupportedDataType(DataType dataType) {
        return dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64 || dataType == DataType.BFLOAT16;
    }

    private static Cpu1StorageKind storageKind(DataType dataType, CpuStorageProfile storageProfile) {
        if (storageProfile != CpuStorageProfile.CPU_NATIVE) {
            return Cpu1StorageKind.JAVA_ARRAY;
        }
        if (dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64) {
            return Cpu1StorageKind.MEMORY_SEGMENT;
        }
        throw new UnsupportedOperationException("cpu1 MSE_LOSS MEMORY_SEGMENT supports FLOAT32/FLOAT64 only; "
                + "BFLOAT16 remains on the JAVA_ARRAY path.");
    }

    private static Cpu1LaunchConfig launchConfig(int elementCount, RuntimeConfig runtimeConfig) {
        CpuKernelConfig cpuKernelConfig = runtimeConfig.cpuKernelConfig();
        int maxWorkers = Math.max(1, Runtime.getRuntime().availableProcessors());
        if (maxWorkers == 1 || elementCount < cpuKernelConfig.reductionParallelMinSize()) {
            return Cpu1LaunchConfig.singleThread();
        }
        int plannedWorkers = Math.min(maxWorkers, elementCount);
        int chunkSize = computeChunkSize(
                elementCount,
                cpuKernelConfig.highCostTargetChunksPerWorker(),
                cpuKernelConfig.minReductionChunkSize(),
                plannedWorkers
        );
        return Cpu1LaunchConfig.parallel(plannedWorkers, chunkSize);
    }

    private static int computeChunkSize(
            int elementCount,
            int targetChunksPerWorker,
            int minChunkSize,
            int plannedWorkers
    ) {
        int workers = Math.max(1, plannedWorkers);
        int targets = Math.max(workers, workers * Math.max(1, targetChunksPerWorker));
        int candidate = (Math.max(1, elementCount) + targets - 1) / targets;
        return Math.max(Math.max(1, minChunkSize), candidate);
    }

    private static Cpu1WorkspaceSpec workspaceSpec(
            Cpu1StorageKind storageKind,
            DataType dataType,
            Cpu1LaunchConfig launchConfig,
            int elementCount
    ) {
        int partialSumSlots = launchConfig.workerCount() == 1
                ? 0
                : Cpu1RangeLauncher.slotCount(elementCount, launchConfig);
        DataType nativeOutputDataType = storageKind == Cpu1StorageKind.MEMORY_SEGMENT ? dataType : null;
        int nativeOutputElements = storageKind == Cpu1StorageKind.MEMORY_SEGMENT ? 1 : 0;
        if (partialSumSlots == 0 && nativeOutputDataType == null) {
            return Cpu1WorkspaceSpec.none();
        }
        return new Cpu1WorkspaceSpec(
                0,
                partialSumSlots,
                0,
                0L,
                false,
                nativeOutputDataType,
                nativeOutputElements
        );
    }

    private static InputResidencyRequirement inputResidencyRequirement(Cpu1StorageKind storageKind) {
        return storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                ? InputResidencyRequirement.none()
                : InputResidencyRequirement.cpuReadableAll();
    }

    private static Operation.OpType opType(CompiledNode node) {
        return node == null || node.operation() == null ? Operation.OpType.UNKNOWN : node.operation().opType();
    }
}
