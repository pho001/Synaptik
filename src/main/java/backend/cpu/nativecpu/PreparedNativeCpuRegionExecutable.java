package backend.cpu.nativecpu;

import tensor.TensorInternalAccess;

import backend.ComputeBackend;
import backend.cpu.CpuBackend;
import backend.cpu.kernels.CpuDTypeOps;
import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.elementwise.unary.support.CpuPowSupport;
import backend.cpu.kernels.elementwise.where.WhereElementwiseKernel;
import backend.cpu.nativecpu.layout.NativeCpuStorageFamily;
import backend.cpu.nativecpu.layout.NativeSegmentStridedKernels;
import backend.cpu.nativecpu.layout.NativeSegmentView;
import backend.cpu.nativecpu.layout.TensorPhysicalView;
import backend.cpu.region.PreparedCpuRegionExecutable;
import backend.lowering.region.RegionExecutionGroup;
import backend.lowering.region.RegionExecutionKind;
import backend.lowering.region.RegionExecutionPlan;
import backend.lowering.region.RegionStorageContract;
import backend.memory.CpuMaterializationReason;
import backend.runtime.ExecutionContext;
import config.runtime.CpuStorageProfile;
import config.runtime.NativeCpuFailurePolicy;
import graph.execution.CompiledNodeExecutionMetadata;
import graph.execution.PreparedNodeExecution;
import operations.Operation;
import operations.elementwise.unary.clampMax;
import operations.elementwise.unary.clampMin;
import operations.elementwise.unary.mulScalar;
import operations.elementwise.unary.pow;
import operations.linalg.linear;
import operations.reduction.mean;
import operations.reduction.reduceAll;
import operations.reduction.reduceAny;
import operations.reduction.reduceMax;
import operations.reduction.reduceMin;
import operations.reduction.sum;
import tensor.layout.BroadcastPlan;
import tensor.layout.BroadcastPlanner;
import tensor.DataType;
import tensor.storage.NativeBFloat16Storage;
import tensor.storage.NativeFloat32Storage;
import tensor.storage.NativeFloat64Storage;
import tensor.storage.NativeTensorStorage;
import tensor.Tensor;
import tensor.layout.WhereBroadcastPlan;
import tensor.layout.WhereBroadcastPlanner;
import utils.FastTranscendentals;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;
import java.util.stream.Collectors;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;

/**
 * Executes one provider-backed CPU native region from its anchor node.
 *
 * <p>The executable owns a private native subplan and a private CPU-array fallback subplan. The fallback
 * never appears as top-level prepared steps, so the graph still has one executable anchor for the region.</p>
 */
public final class PreparedNativeCpuRegionExecutable implements PreparedCpuRegionExecutable {
    private static final CpuBackend CPU_BACKEND = new CpuBackend();
    private static final WhereElementwiseKernel REGION_WHERE_KERNEL = new WhereElementwiseKernel() {
        @Override
        public double applyF64(byte condition, double ifTrue, double ifFalse) {
            return condition != 0 ? ifTrue : ifFalse;
        }

        @Override
        public float applyF32(byte condition, float ifTrue, float ifFalse) {
            return condition != 0 ? ifTrue : ifFalse;
        }

        @Override
        public float applyBF16(byte condition, float ifTrue, float ifFalse) {
            return condition != 0 ? ifTrue : ifFalse;
        }
    };

    private final RegionExecutionPlan regionExecutionPlan;
    private final List<PreparedNodeExecution> nativeSteps;
    private final List<PreparedNodeExecution> fallbackSteps;
    private final Map<Integer, PreparedNodeExecution> nativeStepsByNodeId;
    private String lastRoute = "NOT_EXECUTED";
    private String lastFallbackReason = "";
    private int lastRegionLocalKernelCount;
    private int lastRegionLocalViewCount;
    private int lastExecutedGroupCount;

    public PreparedNativeCpuRegionExecutable(
            RegionExecutionPlan regionExecutionPlan,
            List<PreparedNodeExecution> nativeSteps,
            List<PreparedNodeExecution> fallbackSteps
    ) {
        this.regionExecutionPlan = Objects.requireNonNull(regionExecutionPlan, "regionExecutionPlan cannot be null");
        this.nativeSteps = List.copyOf(nativeSteps == null ? List.of() : nativeSteps);
        this.fallbackSteps = List.copyOf(fallbackSteps == null ? List.of() : fallbackSteps);
        if (this.nativeSteps.isEmpty()) {
            throw new IllegalArgumentException("nativeSteps cannot be empty");
        }
        if (this.fallbackSteps.isEmpty()) {
            throw new IllegalArgumentException("fallbackSteps cannot be empty");
        }
        this.nativeStepsByNodeId = this.nativeSteps.stream()
                .collect(Collectors.toUnmodifiableMap(step -> step.compiledNode().id(), Function.identity()));
    }

    @Override
    public void execute(ExecutionContext context) {
        Objects.requireNonNull(context, "context cannot be null");
        lastRoute = "NATIVE";
        lastFallbackReason = "";
        lastRegionLocalKernelCount = 0;
        lastRegionLocalViewCount = 0;
        lastExecutedGroupCount = 0;
        try {
            if (regionExecutionPlan.executionGroups().isEmpty()) {
                for (PreparedNodeExecution step : nativeSteps) {
                    executeNativeStep(step, context);
                }
            } else {
                for (RegionExecutionGroup group : regionExecutionPlan.executionGroups()) {
                    executeGroup(group, context);
                    lastExecutedGroupCount++;
                }
            }
            verifyNativeBoundaryOutputs(context);
        } catch (NativeRegionFallbackSignal signal) {
            fallbackOrThrow(context, signal.getMessage(), null);
        } catch (RuntimeException ex) {
            fallbackOrThrow(context, "native-cpu-region-runtime-failure:" + safeMessage(ex), ex);
        }
    }

    private void executeGroup(RegionExecutionGroup group, ExecutionContext context) {
        RegionExecutionKind kind = group.executionKind();
        for (int nodeId : group.orderedNodeIds()) {
            PreparedNodeExecution step = nativeStepsByNodeId.get(nodeId);
            if (step == null) {
                throw new NativeRegionFallbackSignal("native-cpu-region-missing-step:node-" + nodeId);
            }
            switch (kind) {
                case PROVIDER_CALL -> executeProviderStep(step, context);
                case DIRECT_KERNEL, FUSED_KERNEL -> executeRegionLocalStep(step, context);
                case VIEW -> executeRegionLocalViewStep(step, context);
                default -> throw new NativeRegionFallbackSignal(
                        "native-cpu-region-unsupported-group-kind:" + kind.name().toLowerCase()
                );
            }
        }
    }

    private void executeProviderStep(PreparedNodeExecution step, ExecutionContext context) {
        if (tryExecuteRegionLocalKernel(step, context)) {
            lastRegionLocalKernelCount++;
            return;
        }
        executeNativeStep(step, context);
    }

    private void executeRegionLocalStep(PreparedNodeExecution step, ExecutionContext context) {
        if (!tryExecuteRegionLocalKernel(step, context)) {
            throw new NativeRegionFallbackSignal("native-cpu-region-local-kernel-missing:node-" + step.compiledNode().id());
        }
        lastRegionLocalKernelCount++;
    }

    private void executeRegionLocalViewStep(PreparedNodeExecution step, ExecutionContext context) {
        if (!tryExecuteRegionLocalView(step, context)) {
            throw new NativeRegionFallbackSignal("native-cpu-region-local-view-missing:node-" + step.compiledNode().id());
        }
        lastRegionLocalViewCount++;
    }

    private void executeNativeStep(PreparedNodeExecution step, ExecutionContext context) {
        if (tryExecuteRegionLocalKernel(step, context)) {
            lastRegionLocalKernelCount++;
            return;
        }
        CPU_BACKEND.execute(step.compiledNode(), step.metadata(), context);
        if (!expectsNativeOutput(step)) {
            return;
        }
        var residency = context.residencyForNodeId(step.compiledNode().id());
        if (residency == null || !residency.nativeCurrent()) {
            throw new NativeRegionFallbackSignal("native-cpu-region-step-fallback:node-" + step.compiledNode().id());
        }
    }

    private boolean tryExecuteRegionLocalKernel(PreparedNodeExecution step, ExecutionContext context) {
        Operation op = step.executionOperation();
        if (op == null) {
            return false;
        }
        if (op.opType() == Operation.OpType.CAST) {
            return tryExecuteCast(step, context, op);
        }
        if (op.opType() == Operation.OpType.CONTIGUOUS) {
            return tryExecuteContiguousCopy(step, context, op);
        }
        if (step.compiledNode().dataType() == DataType.BOOL) {
            if (isCompareOp(op.opType())) {
                return tryExecuteCompare(step, context, op);
            }
            return switch (op.opType()) {
                case LOGICAL_NOT -> tryExecuteSegmentUnary(step, context, op);
                case LOGICAL_AND, LOGICAL_OR -> tryExecuteSegmentBinary(step, context, op);
                case REDUCE_ALL, REDUCE_ANY -> tryExecuteSegmentReduction(step, context, op);
                default -> false;
            };
        }
        return switch (step.compiledNode().dataType()) {
            case FLOAT32 -> switch (op.opType()) {
                case LINEAR -> tryExecuteLinearF32(step, context, op);
                case RELU, NEG, LOG, EXP, FAST_EXP, SQRT, ABS, TANH, FAST_TANH, SIGMOID, INV, MUL_SCALAR,
                     CLAMP_MIN, CLAMP_MAX, FLOOR, CEIL, SIGN, POW ->
                        tryExecuteUnaryF32(step, context, op);
                case ADD, SUB, MUL, DIV, MIN, MAX, POW_TENSOR -> tryExecuteBroadcastBinaryF32(step, context, op);
                case SUM, MEAN, REDUCE_MIN, REDUCE_MAX -> tryExecuteReductionF32(step, context, op);
                case WHERE -> tryExecuteWhereF32(step, context, op);
                default -> false;
            };
            case FLOAT64 -> switch (op.opType()) {
                case LINEAR -> tryExecuteLinearF64(step, context, op);
                case NEG, MUL_SCALAR, RELU, LOG, EXP, FAST_EXP, SQRT, ABS, TANH, FAST_TANH, SIGMOID, INV,
                     CLAMP_MIN, CLAMP_MAX, FLOOR, CEIL, SIGN, POW ->
                        tryExecuteUnaryF64(step, context, op);
                case ADD, SUB, MUL, DIV, MIN, MAX, POW_TENSOR -> tryExecuteBroadcastBinaryF64(step, context, op);
                case SUM, MEAN, REDUCE_MIN, REDUCE_MAX -> tryExecuteReductionF64(step, context, op);
                case WHERE -> tryExecuteWhereF64(step, context, op);
                default -> false;
            };
            case BFLOAT16 -> switch (op.opType()) {
                case NEG, MUL_SCALAR, RELU, ABS, CLAMP_MIN, CLAMP_MAX -> tryExecuteUnaryBF16(step, context, op);
                case ADD, SUB, MUL, DIV, MIN, MAX -> tryExecuteBroadcastBinaryBF16(step, context, op);
                case SUM, MEAN -> tryExecuteSegmentReduction(step, context, op);
                case WHERE -> tryExecuteWhereBF16(step, context, op);
                default -> false;
            };
            default -> false;
        };
    }

    private boolean tryExecuteUnaryF64(PreparedNodeExecution step, ExecutionContext context, Operation op) {
        if (tryExecuteSegmentUnary(step, context, op)) {
            return true;
        }
        List<Integer> inputIds = inputIds(step);
        if (inputIds.size() != 1) {
            return false;
        }
        Tensor outTensor = context.runtimeTensorForNodeId(step.compiledNode().id());
        NativeFloat64Storage input = requireF64Storage(context, inputIds.getFirst(), "native-cpu-region-" + opLabel(op));
        NativeFloat64Storage out = allocateF64(context, step, opLabel(op));
        int size = outTensor.getFlatDataSize();
        for (int i = 0; i < size; i++) {
            out.setFloat64At(i, applyUnaryF64(op, input.getFloat64At(i), context));
        }
        context.attachNativeStorage(
                step.compiledNode().id(),
                out,
                "native CPU region local " + opLabel(op).toUpperCase() + " wrote FLOAT64 output"
        );
        return true;
    }

    private boolean tryExecuteUnaryBF16(PreparedNodeExecution step, ExecutionContext context, Operation op) {
        if (tryExecuteSegmentUnary(step, context, op)) {
            return true;
        }
        List<Integer> inputIds = inputIds(step);
        if (inputIds.size() != 1) {
            return false;
        }
        Tensor outTensor = context.runtimeTensorForNodeId(step.compiledNode().id());
        NativeBFloat16Storage input = requireBF16Storage(context, inputIds.getFirst(), "native-cpu-region-" + opLabel(op));
        NativeBFloat16Storage out = allocateBF16(context, step, opLabel(op));
        int size = outTensor.getFlatDataSize();
        for (int i = 0; i < size; i++) {
            float value = CpuDTypeOps.fromBFloat16Bits(input.getBFloat16BitsAt(i));
            out.setBFloat16BitsAt(i, CpuDTypeOps.toBFloat16Bits(applyUnaryBF16(op, value)));
        }
        context.attachNativeStorage(
                step.compiledNode().id(),
                out,
                "native CPU region local " + opLabel(op).toUpperCase() + " wrote BFLOAT16 output"
        );
        return true;
    }

    private boolean tryExecuteRegionLocalView(PreparedNodeExecution step, ExecutionContext context) {
        Operation op = step.executionOperation();
        if (op == null || !supportsRegionLocalViewOp(op.opType())) {
            return false;
        }
        if (!supportsRegionLocalViewDType(step.compiledNode().dataType())) {
            throw new NativeRegionFallbackSignal("native-cpu-region-view-dtype-unsupported:"
                    + step.compiledNode().dataType().name().toLowerCase());
        }
        List<Integer> inputIds = inputIds(step);
        if (inputIds.size() != 1) {
            throw new NativeRegionFallbackSignal("native-cpu-region-view-input-count:" + opLabel(op));
        }
        int sourceNodeId = inputIds.getFirst();
        Tensor input = context.runtimeTensorForNodeId(sourceNodeId);
        Tensor out = context.runtimeTensorForNodeId(step.compiledNode().id());
        if (input.getDataType() != out.getDataType()) {
            throw new NativeRegionFallbackSignal("native-cpu-region-view-dtype-mismatch:" + opLabel(op));
        }
        try {
            context.aliasNativeStorage(
                    step.compiledNode().id(),
                    sourceNodeId,
                    "native CPU region local " + opLabel(op).toUpperCase() + " view aliases node-" + sourceNodeId
            );
            return true;
        } catch (RuntimeException ex) {
            throw new NativeRegionFallbackSignal("native-cpu-region-view-alias-failed:"
                    + opLabel(op) + ":" + safeMessage(ex));
        }
    }

    private boolean tryExecuteCast(PreparedNodeExecution step, ExecutionContext context, Operation op) {
        List<Integer> inputIds = inputIds(step);
        if (inputIds.size() != 1) {
            return false;
        }
        Tensor inputTensor = context.runtimeTensorForNodeId(inputIds.getFirst());
        Tensor outTensor = context.runtimeTensorForNodeId(step.compiledNode().id());
        if (inputTensor.getFlatDataSize() != outTensor.getFlatDataSize()
                || inputTensor.hasStorageOffset()
                || !isNativeRegionCast(inputTensor.getDataType(), outTensor.getDataType())) {
            return false;
        }
        NativeTensorStorage input = context.requireNativeReadable(inputIds.getFirst(), CpuMaterializationReason.CPU_CONSUMER);
        NativeTensorStorage out = context.allocateNativeStorage(
                outTensor.getDataType(),
                outTensor.getFlatDataSize(),
                "node-" + step.compiledNode().id() + ":" + outTensor.getLabel() + ":native-region-cast"
        );
        if (inputTensor.getDataType() == DataType.FLOAT32 && outTensor.getDataType() == DataType.BFLOAT16) {
            NativeBFloat16Kernels.fromFloat32(
                    requireStorage(input, NativeFloat32Storage.class, opLabel(op)),
                    requireStorage(out, NativeBFloat16Storage.class, opLabel(op)),
                    outTensor.getFlatDataSize()
            );
        } else if (inputTensor.getDataType() == DataType.BFLOAT16 && outTensor.getDataType() == DataType.FLOAT32) {
            NativeBFloat16Kernels.toFloat32(
                    requireStorage(input, NativeBFloat16Storage.class, opLabel(op)),
                    requireStorage(out, NativeFloat32Storage.class, opLabel(op)),
                    outTensor.getFlatDataSize()
            );
        } else {
            return false;
        }
        context.attachNativeStorage(
                step.compiledNode().id(),
                out,
                "native CPU region local CAST wrote " + outTensor.getDataType() + " output"
        );
        return true;
    }

    private boolean tryExecuteContiguousCopy(PreparedNodeExecution step, ExecutionContext context, Operation op) {
        List<Integer> inputIds = inputIds(step);
        if (inputIds.size() != 1) {
            return false;
        }
        Tensor inputTensor = context.runtimeTensorForNodeId(inputIds.getFirst());
        Tensor outTensor = context.runtimeTensorForNodeId(step.compiledNode().id());
        if (inputTensor.getDataType() != outTensor.getDataType()
                || inputTensor.getFlatDataSize() != outTensor.getFlatDataSize()
                || !supportsNativeContiguousCopyDType(outTensor.getDataType())) {
            return false;
        }
        NativeTensorStorage input = context.requireNativeReadable(inputIds.getFirst(), CpuMaterializationReason.CPU_CONSUMER);
        NativeTensorStorage out = context.allocateNativeStorage(
                outTensor.getDataType(),
                outTensor.getFlatDataSize(),
                "node-" + step.compiledNode().id() + ":" + outTensor.getLabel() + ":native-region-contiguous"
        );
        copyNativeStorage(input, out, outTensor.getFlatDataSize(), outTensor.getDataType(), opLabel(op));
        context.attachNativeStorage(
                step.compiledNode().id(),
                out,
                "native CPU region local CONTIGUOUS copied " + outTensor.getDataType() + " output"
        );
        return true;
    }

    private boolean tryExecuteLinearF32(
            PreparedNodeExecution step,
            ExecutionContext context,
            Operation op
    ) {
        if (!(op instanceof linear linearOp) || step.metadata().cpuPlan() == null
                || step.metadata().cpuPlan().matMulExecutable() == null
                || !step.metadata().cpuPlan().matMulExecutable().acceptsNativeInputs()) {
            return false;
        }
        List<Integer> inputIds = inputIds(step);
        if (inputIds.size() < 2 || (linearOp.hasBias() && inputIds.size() < 3)) {
            return false;
        }
        Tensor input = context.runtimeTensorForNodeId(inputIds.get(0));
        Tensor weight = context.runtimeTensorForNodeId(inputIds.get(1));
        Tensor out = context.runtimeTensorForNodeId(step.compiledNode().id());
        CpuKernelContext kernelContext = new CpuKernelContext(
                step.compiledNode().id(),
                inputIds,
                step.metadata().cpuPlan(),
                context,
                step.metadata(),
                List.of(),
                op
        );
        step.metadata().cpuPlan().matMulExecutable().execute(input, weight, out, kernelContext);
        NativeFloat32Storage outStorage = requireF32Storage(
                context,
                step.compiledNode().id(),
                "native-cpu-region-linear"
        );
        if (linearOp.hasBias()) {
            Tensor biasTensor = context.runtimeTensorForNodeId(inputIds.get(2));
            int lastDim = out.getShapeUnsafe().length == 0
                    ? out.getFlatDataSize()
                    : out.getShapeUnsafe()[out.getShapeUnsafe().length - 1];
            if (biasTensor.getFlatDataSize() != lastDim) {
                throw new NativeRegionFallbackSignal("native-cpu-region-linear-bias-shape");
            }
            NativeFloat32Storage bias = requireF32Storage(context, inputIds.get(2), "native-cpu-region-linear-bias");
            addBiasInPlace(outStorage, bias, out.getFlatDataSize(), lastDim);
            outStorage.markModified();
            context.attachNativeStorage(
                    step.compiledNode().id(),
                    outStorage,
                    "native CPU region local LINEAR bias wrote FLOAT32 output"
            );
        }
        return true;
    }

    private boolean tryExecuteLinearF64(
            PreparedNodeExecution step,
            ExecutionContext context,
            Operation op
    ) {
        if (!(op instanceof linear linearOp) || step.metadata().cpuPlan() == null
                || step.metadata().cpuPlan().matMulExecutable() == null
                || !step.metadata().cpuPlan().matMulExecutable().acceptsNativeInputs()) {
            return false;
        }
        List<Integer> inputIds = inputIds(step);
        if (inputIds.size() < 2 || (linearOp.hasBias() && inputIds.size() < 3)) {
            return false;
        }
        Tensor input = context.runtimeTensorForNodeId(inputIds.get(0));
        Tensor weight = context.runtimeTensorForNodeId(inputIds.get(1));
        Tensor out = context.runtimeTensorForNodeId(step.compiledNode().id());
        CpuKernelContext kernelContext = new CpuKernelContext(
                step.compiledNode().id(),
                inputIds,
                step.metadata().cpuPlan(),
                context,
                step.metadata(),
                List.of(),
                op
        );
        step.metadata().cpuPlan().matMulExecutable().execute(input, weight, out, kernelContext);
        NativeFloat64Storage outStorage = requireF64Storage(
                context,
                step.compiledNode().id(),
                "native-cpu-region-linear"
        );
        if (linearOp.hasBias()) {
            Tensor biasTensor = context.runtimeTensorForNodeId(inputIds.get(2));
            int lastDim = out.getShapeUnsafe().length == 0
                    ? out.getFlatDataSize()
                    : out.getShapeUnsafe()[out.getShapeUnsafe().length - 1];
            if (biasTensor.getFlatDataSize() != lastDim) {
                throw new NativeRegionFallbackSignal("native-cpu-region-linear-bias-shape");
            }
            NativeFloat64Storage bias = requireF64Storage(context, inputIds.get(2), "native-cpu-region-linear-bias");
            addBiasInPlace(outStorage, bias, out.getFlatDataSize(), lastDim);
            outStorage.markModified();
            context.attachNativeStorage(
                    step.compiledNode().id(),
                    outStorage,
                    "native CPU region local LINEAR bias wrote FLOAT64 output"
            );
        }
        return true;
    }

    private boolean tryExecuteUnaryF32(PreparedNodeExecution step, ExecutionContext context, Operation op) {
        if (tryExecuteSegmentUnary(step, context, op)) {
            return true;
        }
        List<Integer> inputIds = inputIds(step);
        if (inputIds.size() != 1) {
            return false;
        }
        Tensor outTensor = context.runtimeTensorForNodeId(step.compiledNode().id());
        NativeFloat32Storage input = requireF32Storage(context, inputIds.getFirst(), "native-cpu-region-" + opLabel(op));
        NativeFloat32Storage out = allocateF32(context, step, opLabel(op));
        int size = outTensor.getFlatDataSize();
        for (int i = 0; i < size; i++) {
            long offset = (long) i * Float.BYTES;
            float value = input.segment().get(JAVA_FLOAT, offset);
            out.segment().set(JAVA_FLOAT, offset, applyUnaryF32(op, value, context));
        }
        out.markModified();
        context.attachNativeStorage(
                step.compiledNode().id(),
                out,
                "native CPU region local " + opLabel(op).toUpperCase() + " wrote FLOAT32 output"
        );
        return true;
    }

    private boolean tryExecuteWhereF32(PreparedNodeExecution step, ExecutionContext context, Operation op) {
        if (tryExecuteSegmentWhere(step, context, op)) {
            return true;
        }
        List<Integer> inputIds = inputIds(step);
        if (inputIds.size() != 3) {
            return false;
        }
        int conditionNodeId = inputIds.get(0);
        int trueNodeId = inputIds.get(1);
        int falseNodeId = inputIds.get(2);
        Tensor conditionTensor = context.runtimeTensorForNodeId(conditionNodeId);
        Tensor trueTensor = context.runtimeTensorForNodeId(trueNodeId);
        Tensor falseTensor = context.runtimeTensorForNodeId(falseNodeId);
        Tensor outTensor = context.runtimeTensorForNodeId(step.compiledNode().id());
        int outSize = outTensor.getFlatDataSize();
        if (conditionTensor.getDataType() != DataType.BOOL
                || trueTensor.getDataType() != DataType.FLOAT32
                || falseTensor.getDataType() != DataType.FLOAT32) {
            return false;
        }
        if (conditionTensor.hasStorageOffset()
                || trueTensor.hasStorageOffset()
                || falseTensor.hasStorageOffset()) {
            return false;
        }
        WhereBroadcastPlan plan;
        try {
            plan = WhereBroadcastPlanner.plan(conditionTensor, trueTensor, falseTensor);
        } catch (IllegalArgumentException ignored) {
            return false;
        }
        int[] outShape = plan.outShape();
        int[] outStrides = plan.outStrides();
        int[] condEffStrides = plan.condEffStrides();
        int[] trueEffStrides = plan.trueEffStrides();
        int[] falseEffStrides = plan.falseEffStrides();
        if (flatSize(outShape) != outSize || !Arrays.equals(outShape, outTensor.getShapeUnsafe())) {
            return false;
        }
        context.requireCpuReadable(conditionNodeId, CpuMaterializationReason.CPU_CONSUMER);
        byte[] condition = TensorInternalAccess.boolData(conditionTensor);
        NativeFloat32Storage ifTrue = requireF32Storage(context, trueNodeId, "native-cpu-region-where");
        NativeFloat32Storage ifFalse = requireF32Storage(context, falseNodeId, "native-cpu-region-where");
        NativeFloat32Storage out = allocateF32(context, step, opLabel(op));
        for (int i = 0; i < outSize; i++) {
            long offset = (long) i * Float.BYTES;
            int conditionIndex = broadcastedFlatIndex(i, outShape, outStrides, condEffStrides);
            int trueIndex = broadcastedFlatIndex(i, outShape, outStrides, trueEffStrides);
            int falseIndex = broadcastedFlatIndex(i, outShape, outStrides, falseEffStrides);
            out.segment().set(
                    JAVA_FLOAT,
                    offset,
                    condition[conditionIndex] != 0
                            ? ifTrue.getFloat32At(trueIndex)
                            : ifFalse.getFloat32At(falseIndex)
            );
        }
        out.markModified();
        context.attachNativeStorage(
                step.compiledNode().id(),
                out,
                "native CPU region local WHERE wrote FLOAT32 output"
        );
        return true;
    }

    private boolean tryExecuteWhereF64(PreparedNodeExecution step, ExecutionContext context, Operation op) {
        if (tryExecuteSegmentWhere(step, context, op)) {
            return true;
        }
        List<Integer> inputIds = inputIds(step);
        if (inputIds.size() != 3) {
            return false;
        }
        int conditionNodeId = inputIds.get(0);
        int trueNodeId = inputIds.get(1);
        int falseNodeId = inputIds.get(2);
        Tensor conditionTensor = context.runtimeTensorForNodeId(conditionNodeId);
        Tensor trueTensor = context.runtimeTensorForNodeId(trueNodeId);
        Tensor falseTensor = context.runtimeTensorForNodeId(falseNodeId);
        Tensor outTensor = context.runtimeTensorForNodeId(step.compiledNode().id());
        int outSize = outTensor.getFlatDataSize();
        if (conditionTensor.getDataType() != DataType.BOOL
                || trueTensor.getDataType() != DataType.FLOAT64
                || falseTensor.getDataType() != DataType.FLOAT64) {
            return false;
        }
        if (conditionTensor.hasStorageOffset()
                || trueTensor.hasStorageOffset()
                || falseTensor.hasStorageOffset()) {
            return false;
        }
        WhereBroadcastPlan plan;
        try {
            plan = WhereBroadcastPlanner.plan(conditionTensor, trueTensor, falseTensor);
        } catch (IllegalArgumentException ignored) {
            return false;
        }
        int[] outShape = plan.outShape();
        int[] outStrides = plan.outStrides();
        int[] condEffStrides = plan.condEffStrides();
        int[] trueEffStrides = plan.trueEffStrides();
        int[] falseEffStrides = plan.falseEffStrides();
        if (flatSize(outShape) != outSize || !Arrays.equals(outShape, outTensor.getShapeUnsafe())) {
            return false;
        }
        context.requireCpuReadable(conditionNodeId, CpuMaterializationReason.CPU_CONSUMER);
        byte[] condition = TensorInternalAccess.boolData(conditionTensor);
        NativeFloat64Storage ifTrue = requireF64Storage(context, trueNodeId, "native-cpu-region-where");
        NativeFloat64Storage ifFalse = requireF64Storage(context, falseNodeId, "native-cpu-region-where");
        NativeFloat64Storage out = allocateF64(context, step, opLabel(op));
        for (int i = 0; i < outSize; i++) {
            int conditionIndex = broadcastedFlatIndex(i, outShape, outStrides, condEffStrides);
            int trueIndex = broadcastedFlatIndex(i, outShape, outStrides, trueEffStrides);
            int falseIndex = broadcastedFlatIndex(i, outShape, outStrides, falseEffStrides);
            out.setFloat64At(
                    i,
                    condition[conditionIndex] != 0
                            ? ifTrue.getFloat64At(trueIndex)
                            : ifFalse.getFloat64At(falseIndex)
            );
        }
        context.attachNativeStorage(
                step.compiledNode().id(),
                out,
                "native CPU region local WHERE wrote FLOAT64 output"
        );
        return true;
    }

    private boolean tryExecuteWhereBF16(PreparedNodeExecution step, ExecutionContext context, Operation op) {
        return tryExecuteSegmentWhere(step, context, op);
    }

    private boolean tryExecuteCompare(PreparedNodeExecution step, ExecutionContext context, Operation op) {
        List<Integer> inputIds = inputIds(step);
        if (inputIds.size() != 2) {
            return false;
        }
        if (tryExecuteSegmentCompare(step, context, op)) {
            return true;
        }
        Tensor outTensor = context.runtimeTensorForNodeId(step.compiledNode().id());
        Tensor leftTensor = context.runtimeTensorForNodeId(inputIds.get(0));
        Tensor rightTensor = context.runtimeTensorForNodeId(inputIds.get(1));
        int outSize = outTensor.getFlatDataSize();
        if (outTensor.getDataType() != DataType.BOOL
                || leftTensor.getDataType() != rightTensor.getDataType()
                || (leftTensor.getDataType() != DataType.FLOAT32
                && leftTensor.getDataType() != DataType.FLOAT64
                && leftTensor.getDataType() != DataType.BFLOAT16)
                || leftTensor.hasStorageOffset()
                || rightTensor.hasStorageOffset()) {
            return false;
        }
        BroadcastPlan plan;
        try {
            plan = BroadcastPlanner.plan(leftTensor, rightTensor);
        } catch (IllegalArgumentException ignored) {
            return false;
        }
        int[] outShape = plan.outShape();
        int[] outStrides = plan.outStrides();
        int[] leftEffStrides = plan.aEffStrides();
        int[] rightEffStrides = plan.bEffStrides();
        if (plan.flatSize() != outSize || !Arrays.equals(outShape, outTensor.getShapeUnsafe())) {
            return false;
        }
        byte[] out = TensorInternalAccess.boolData(outTensor);
        if (leftTensor.getDataType() == DataType.FLOAT32) {
            NativeFloat32Storage left = requireF32Storage(context, inputIds.get(0), "native-cpu-region-" + opLabel(op));
            NativeFloat32Storage right = requireF32Storage(context, inputIds.get(1), "native-cpu-region-" + opLabel(op));
            for (int i = 0; i < outSize; i++) {
                int leftIndex = broadcastedFlatIndex(i, outShape, outStrides, leftEffStrides);
                int rightIndex = broadcastedFlatIndex(i, outShape, outStrides, rightEffStrides);
                out[i] = applyCompareF32(op.opType(), left.getFloat32At(leftIndex), right.getFloat32At(rightIndex));
            }
        } else if (leftTensor.getDataType() == DataType.FLOAT64) {
            NativeFloat64Storage left = requireF64Storage(context, inputIds.get(0), "native-cpu-region-" + opLabel(op));
            NativeFloat64Storage right = requireF64Storage(context, inputIds.get(1), "native-cpu-region-" + opLabel(op));
            for (int i = 0; i < outSize; i++) {
                int leftIndex = broadcastedFlatIndex(i, outShape, outStrides, leftEffStrides);
                int rightIndex = broadcastedFlatIndex(i, outShape, outStrides, rightEffStrides);
                out[i] = applyCompareF64(op.opType(), left.getFloat64At(leftIndex), right.getFloat64At(rightIndex));
            }
        } else {
            NativeBFloat16Storage left = requireBF16Storage(context, inputIds.get(0), "native-cpu-region-" + opLabel(op));
            NativeBFloat16Storage right = requireBF16Storage(context, inputIds.get(1), "native-cpu-region-" + opLabel(op));
            for (int i = 0; i < outSize; i++) {
                int leftIndex = broadcastedFlatIndex(i, outShape, outStrides, leftEffStrides);
                int rightIndex = broadcastedFlatIndex(i, outShape, outStrides, rightEffStrides);
                out[i] = applyCompareF32(
                        op.opType(),
                        CpuDTypeOps.fromBFloat16Bits(left.getBFloat16BitsAt(leftIndex)),
                        CpuDTypeOps.fromBFloat16Bits(right.getBFloat16BitsAt(rightIndex))
                );
            }
        }
        TensorInternalAccess.markStorageModified(outTensor);
        context.markCpuCurrent(
                step.compiledNode().id(),
                "native CPU region local " + opLabel(op).toUpperCase() + " wrote BOOL CPU array output"
        );
        return true;
    }

    private boolean tryExecuteReductionF32(PreparedNodeExecution step, ExecutionContext context, Operation op) {
        if (tryExecuteSegmentReduction(step, context, op)) {
            return true;
        }
        List<Integer> inputIds = inputIds(step);
        if (inputIds.size() != 1) {
            return false;
        }
        int dimension = reductionDimension(op);
        if (dimension < -1) {
            return false;
        }
        int inputNodeId = inputIds.getFirst();
        Tensor inputTensor = context.runtimeTensorForNodeId(inputNodeId);
        Tensor outTensor = context.runtimeTensorForNodeId(step.compiledNode().id());
        int[] shape = inputTensor.getShapeUnsafe();
        if (shape == null || shape.length == 0 || dimension >= shape.length) {
            return false;
        }
        if (inputTensor.hasStorageOffset() || !inputTensor.isContiguous() || inputTensor.getFlatDataSize() <= 0) {
            return false;
        }
        int expectedOutputSize = expectedReductionOutputSize(shape, dimension);
        if (expectedOutputSize != outTensor.getFlatDataSize()) {
            return false;
        }
        NativeFloat32Storage input = requireF32Storage(context, inputNodeId, "native-cpu-region-" + opLabel(op));
        NativeFloat32Storage out = allocateF32(context, step, opLabel(op));
        if (dimension == -1) {
            out.setFloat32At(0, reduceAllF32(op.opType(), input, inputTensor.getFlatDataSize()));
        } else {
            reduceAxisF32(op.opType(), input, out, shape, dimension);
        }
        out.markModified();
        context.attachNativeStorage(
                step.compiledNode().id(),
                out,
                "native CPU region local " + opLabel(op).toUpperCase() + " wrote FLOAT32 output"
        );
        return true;
    }

    private boolean tryExecuteReductionF64(PreparedNodeExecution step, ExecutionContext context, Operation op) {
        if (tryExecuteSegmentReduction(step, context, op)) {
            return true;
        }
        List<Integer> inputIds = inputIds(step);
        if (inputIds.size() != 1) {
            return false;
        }
        int dimension = reductionDimension(op);
        if (dimension < -1) {
            return false;
        }
        int inputNodeId = inputIds.getFirst();
        Tensor inputTensor = context.runtimeTensorForNodeId(inputNodeId);
        Tensor outTensor = context.runtimeTensorForNodeId(step.compiledNode().id());
        int[] shape = inputTensor.getShapeUnsafe();
        if (shape == null || shape.length == 0 || dimension >= shape.length) {
            return false;
        }
        if (inputTensor.hasStorageOffset() || !inputTensor.isContiguous() || inputTensor.getFlatDataSize() <= 0) {
            return false;
        }
        int expectedOutputSize = expectedReductionOutputSize(shape, dimension);
        if (expectedOutputSize != outTensor.getFlatDataSize()) {
            return false;
        }
        NativeFloat64Storage input = requireF64Storage(context, inputNodeId, "native-cpu-region-" + opLabel(op));
        NativeFloat64Storage out = allocateF64(context, step, opLabel(op));
        if (dimension == -1) {
            out.setFloat64At(0, reduceAllF64(op.opType(), input, inputTensor.getFlatDataSize()));
        } else {
            reduceAxisF64(op.opType(), input, out, shape, dimension);
        }
        context.attachNativeStorage(
                step.compiledNode().id(),
                out,
                "native CPU region local " + opLabel(op).toUpperCase() + " wrote FLOAT64 output"
        );
        return true;
    }

    private boolean tryExecuteBroadcastBinaryF32(PreparedNodeExecution step, ExecutionContext context, Operation op) {
        if (tryExecuteSegmentBinary(step, context, op)) {
            return true;
        }
        List<Integer> inputIds = inputIds(step);
        if (inputIds.size() != 2) {
            return false;
        }
        Tensor outTensor = context.runtimeTensorForNodeId(step.compiledNode().id());
        Tensor leftTensor = context.runtimeTensorForNodeId(inputIds.get(0));
        Tensor rightTensor = context.runtimeTensorForNodeId(inputIds.get(1));
        int outSize = outTensor.getFlatDataSize();
        if (leftTensor.hasStorageOffset() || rightTensor.hasStorageOffset()) {
            return false;
        }
        BroadcastPlan plan;
        try {
            plan = BroadcastPlanner.plan(leftTensor, rightTensor);
        } catch (IllegalArgumentException ignored) {
            return false;
        }
        int[] outShape = plan.outShape();
        int[] outStrides = plan.outStrides();
        int[] leftEffStrides = plan.aEffStrides();
        int[] rightEffStrides = plan.bEffStrides();
        if (plan.flatSize() != outSize || !Arrays.equals(outShape, outTensor.getShapeUnsafe())) {
            return false;
        }
        NativeFloat32Storage left = requireF32Storage(context, inputIds.get(0), "native-cpu-region-" + opLabel(op));
        NativeFloat32Storage right = requireF32Storage(context, inputIds.get(1), "native-cpu-region-" + opLabel(op));
        NativeFloat32Storage out = allocateF32(context, step, opLabel(op));
        for (int i = 0; i < outSize; i++) {
            long outOffset = (long) i * Float.BYTES;
            int leftIndex = broadcastedFlatIndex(i, outShape, outStrides, leftEffStrides);
            int rightIndex = broadcastedFlatIndex(i, outShape, outStrides, rightEffStrides);
            out.segment().set(
                    JAVA_FLOAT,
                    outOffset,
                    applyDenseBinaryF32(
                            op.opType(),
                            left.getFloat32At(leftIndex),
                            right.getFloat32At(rightIndex)
                    )
            );
        }
        out.markModified();
        context.attachNativeStorage(
                step.compiledNode().id(),
                out,
                "native CPU region local " + opLabel(op).toUpperCase() + " wrote FLOAT32 output"
        );
        return true;
    }

    private boolean tryExecuteBroadcastBinaryF64(PreparedNodeExecution step, ExecutionContext context, Operation op) {
        if (tryExecuteSegmentBinary(step, context, op)) {
            return true;
        }
        List<Integer> inputIds = inputIds(step);
        if (inputIds.size() != 2) {
            return false;
        }
        Tensor outTensor = context.runtimeTensorForNodeId(step.compiledNode().id());
        Tensor leftTensor = context.runtimeTensorForNodeId(inputIds.get(0));
        Tensor rightTensor = context.runtimeTensorForNodeId(inputIds.get(1));
        int outSize = outTensor.getFlatDataSize();
        if (leftTensor.hasStorageOffset() || rightTensor.hasStorageOffset()) {
            return false;
        }
        BroadcastPlan plan;
        try {
            plan = BroadcastPlanner.plan(leftTensor, rightTensor);
        } catch (IllegalArgumentException ignored) {
            return false;
        }
        int[] outShape = plan.outShape();
        int[] outStrides = plan.outStrides();
        int[] leftEffStrides = plan.aEffStrides();
        int[] rightEffStrides = plan.bEffStrides();
        if (plan.flatSize() != outSize || !Arrays.equals(outShape, outTensor.getShapeUnsafe())) {
            return false;
        }
        NativeFloat64Storage left = requireF64Storage(context, inputIds.get(0), "native-cpu-region-" + opLabel(op));
        NativeFloat64Storage right = requireF64Storage(context, inputIds.get(1), "native-cpu-region-" + opLabel(op));
        NativeFloat64Storage out = allocateF64(context, step, opLabel(op));
        for (int i = 0; i < outSize; i++) {
            int leftIndex = broadcastedFlatIndex(i, outShape, outStrides, leftEffStrides);
            int rightIndex = broadcastedFlatIndex(i, outShape, outStrides, rightEffStrides);
            out.setFloat64At(i, applyDenseBinaryF64(
                    op.opType(),
                    left.getFloat64At(leftIndex),
                    right.getFloat64At(rightIndex)
            ));
        }
        context.attachNativeStorage(
                step.compiledNode().id(),
                out,
                "native CPU region local " + opLabel(op).toUpperCase() + " wrote FLOAT64 output"
        );
        return true;
    }

    private boolean tryExecuteBroadcastBinaryBF16(PreparedNodeExecution step, ExecutionContext context, Operation op) {
        if (tryExecuteSegmentBinary(step, context, op)) {
            return true;
        }
        List<Integer> inputIds = inputIds(step);
        if (inputIds.size() != 2) {
            return false;
        }
        Tensor outTensor = context.runtimeTensorForNodeId(step.compiledNode().id());
        Tensor leftTensor = context.runtimeTensorForNodeId(inputIds.get(0));
        Tensor rightTensor = context.runtimeTensorForNodeId(inputIds.get(1));
        int outSize = outTensor.getFlatDataSize();
        if (leftTensor.hasStorageOffset() || rightTensor.hasStorageOffset()) {
            return false;
        }
        BroadcastPlan plan;
        try {
            plan = BroadcastPlanner.plan(leftTensor, rightTensor);
        } catch (IllegalArgumentException ignored) {
            return false;
        }
        int[] outShape = plan.outShape();
        int[] outStrides = plan.outStrides();
        int[] leftEffStrides = plan.aEffStrides();
        int[] rightEffStrides = plan.bEffStrides();
        if (plan.flatSize() != outSize || !Arrays.equals(outShape, outTensor.getShapeUnsafe())) {
            return false;
        }
        NativeBFloat16Storage left = requireBF16Storage(context, inputIds.get(0), "native-cpu-region-" + opLabel(op));
        NativeBFloat16Storage right = requireBF16Storage(context, inputIds.get(1), "native-cpu-region-" + opLabel(op));
        NativeBFloat16Storage out = allocateBF16(context, step, opLabel(op));
        for (int i = 0; i < outSize; i++) {
            int leftIndex = broadcastedFlatIndex(i, outShape, outStrides, leftEffStrides);
            int rightIndex = broadcastedFlatIndex(i, outShape, outStrides, rightEffStrides);
            float leftValue = CpuDTypeOps.fromBFloat16Bits(left.getBFloat16BitsAt(leftIndex));
            float rightValue = CpuDTypeOps.fromBFloat16Bits(right.getBFloat16BitsAt(rightIndex));
            out.setBFloat16BitsAt(i, CpuDTypeOps.toBFloat16Bits(applyDenseBinaryBF16(
                    op.opType(),
                    leftValue,
                    rightValue
            )));
        }
        context.attachNativeStorage(
                step.compiledNode().id(),
                out,
                "native CPU region local " + opLabel(op).toUpperCase() + " wrote BFLOAT16 output"
        );
        return true;
    }

    private boolean tryExecuteSegmentUnary(PreparedNodeExecution step, ExecutionContext context, Operation op) {
        List<Integer> inputIds = inputIds(step);
        DataType dataType = step.compiledNode().dataType();
        if (op == null || inputIds.size() != 1 || !NativeSegmentStridedKernels.supportsUnary(op, dataType)) {
            return false;
        }
        Tensor inputTensor = context.runtimeTensorForNodeId(inputIds.getFirst());
        Tensor outTensor = context.runtimeTensorForNodeId(step.compiledNode().id());
        if (!sameShape(inputTensor, outTensor)) {
            return false;
        }
        try {
            NativeSegmentView inputView = nativeSegmentView(context, inputIds.getFirst());
            NativeSegmentOutput output = allocateSegmentOutput(context, step, opLabel(op));
            NativeSegmentStridedKernels.runUnary(
                    op,
                    inputView,
                    output.view(),
                    context.useFastExpApprox(),
                    context.useFastTanhApprox()
            );
            context.attachNativeStorage(
                    step.compiledNode().id(),
                    output.storage(),
                    "native CPU region segment " + opLabel(op).toUpperCase() + " wrote " + dataType + " output"
            );
            return true;
        } catch (RuntimeException ex) {
            throw new NativeRegionFallbackSignal("native-cpu-region-segment-unary-failed:"
                    + opLabel(op) + ":" + safeMessage(ex));
        }
    }

    private boolean tryExecuteSegmentBinary(PreparedNodeExecution step, ExecutionContext context, Operation op) {
        List<Integer> inputIds = inputIds(step);
        DataType dataType = step.compiledNode().dataType();
        if (op == null || inputIds.size() != 2 || !NativeSegmentStridedKernels.supportsBinary(op, dataType)) {
            return false;
        }
        Tensor leftTensor = context.runtimeTensorForNodeId(inputIds.get(0));
        Tensor rightTensor = context.runtimeTensorForNodeId(inputIds.get(1));
        Tensor outTensor = context.runtimeTensorForNodeId(step.compiledNode().id());
        BroadcastPlan plan;
        try {
            plan = BroadcastPlanner.plan(leftTensor, rightTensor);
        } catch (IllegalArgumentException ignored) {
            return false;
        }
        int[] outShape = plan.outShape();
        if (plan.flatSize() != outTensor.getFlatDataSize()
                || !Arrays.equals(outShape, outTensor.getShapeUnsafe())) {
            return false;
        }
        try {
            NativeSegmentView leftView = nativeSegmentView(context, inputIds.get(0), outShape, plan.aEffStrides());
            NativeSegmentView rightView = nativeSegmentView(context, inputIds.get(1), outShape, plan.bEffStrides());
            NativeSegmentOutput output = allocateSegmentOutput(context, step, opLabel(op));
            NativeSegmentStridedKernels.runBinary(op, leftView, rightView, output.view());
            context.attachNativeStorage(
                    step.compiledNode().id(),
                    output.storage(),
                    "native CPU region segment " + opLabel(op).toUpperCase() + " wrote " + dataType + " output"
            );
            return true;
        } catch (RuntimeException ex) {
            throw new NativeRegionFallbackSignal("native-cpu-region-segment-binary-failed:"
                    + opLabel(op) + ":" + safeMessage(ex));
        }
    }

    private boolean tryExecuteSegmentWhere(PreparedNodeExecution step, ExecutionContext context, Operation op) {
        List<Integer> inputIds = inputIds(step);
        DataType dataType = step.compiledNode().dataType();
        if (op == null || op.opType() != Operation.OpType.WHERE || inputIds.size() != 3
                || (dataType != DataType.FLOAT32 && dataType != DataType.FLOAT64 && dataType != DataType.BFLOAT16)) {
            return false;
        }
        int conditionNodeId = inputIds.get(0);
        int trueNodeId = inputIds.get(1);
        int falseNodeId = inputIds.get(2);
        Tensor conditionTensor = context.runtimeTensorForNodeId(conditionNodeId);
        Tensor trueTensor = context.runtimeTensorForNodeId(trueNodeId);
        Tensor falseTensor = context.runtimeTensorForNodeId(falseNodeId);
        Tensor outTensor = context.runtimeTensorForNodeId(step.compiledNode().id());
        if (conditionTensor.getDataType() != DataType.BOOL
                || trueTensor.getDataType() != dataType
                || falseTensor.getDataType() != dataType) {
            return false;
        }
        WhereBroadcastPlan plan;
        try {
            plan = WhereBroadcastPlanner.plan(conditionTensor, trueTensor, falseTensor);
        } catch (IllegalArgumentException ignored) {
            return false;
        }
        int[] outShape = plan.outShape();
        if (flatSize(outShape) != outTensor.getFlatDataSize()
                || !Arrays.equals(outShape, outTensor.getShapeUnsafe())) {
            return false;
        }
        try {
            NativeSegmentView trueView = nativeSegmentView(context, trueNodeId, outShape, plan.trueEffStrides());
            NativeSegmentView falseView = nativeSegmentView(context, falseNodeId, outShape, plan.falseEffStrides());
            NativeSegmentOutput output = allocateSegmentOutput(context, step, opLabel(op));
            NativeSegmentView nativeConditionView = nativeSegmentViewIfCurrent(
                    context,
                    conditionNodeId,
                    outShape,
                    plan.condEffStrides()
            );
            if (nativeConditionView != null) {
                NativeSegmentStridedKernels.runWhere(
                        REGION_WHERE_KERNEL,
                        nativeConditionView,
                        trueView,
                        falseView,
                        output.view()
                );
            } else {
                context.requireCpuReadable(conditionNodeId, CpuMaterializationReason.CPU_CONSUMER);
                byte[] condition = TensorInternalAccess.boolData(conditionTensor);
                TensorPhysicalView conditionView = cpuArrayPhysicalView(
                        conditionNodeId,
                        conditionTensor,
                        outShape,
                        plan.condEffStrides()
                );
                NativeSegmentStridedKernels.runWhere(
                        REGION_WHERE_KERNEL,
                        condition,
                        conditionView,
                        trueView,
                        falseView,
                        output.view()
                );
            }
            context.attachNativeStorage(
                    step.compiledNode().id(),
                    output.storage(),
                    "native CPU region segment WHERE wrote " + dataType + " output"
            );
            return true;
        } catch (RuntimeException ex) {
            throw new NativeRegionFallbackSignal("native-cpu-region-segment-where-failed:"
                    + safeMessage(ex));
        }
    }

    private boolean tryExecuteSegmentCompare(PreparedNodeExecution step, ExecutionContext context, Operation op) {
        List<Integer> inputIds = inputIds(step);
        if (op == null || inputIds.size() != 2) {
            return false;
        }
        Tensor outTensor = context.runtimeTensorForNodeId(step.compiledNode().id());
        Tensor leftTensor = context.runtimeTensorForNodeId(inputIds.get(0));
        Tensor rightTensor = context.runtimeTensorForNodeId(inputIds.get(1));
        DataType dataType = leftTensor.getDataType();
        if (outTensor.getDataType() != DataType.BOOL
                || dataType != rightTensor.getDataType()
                || !NativeSegmentStridedKernels.supportsCompare(op, dataType)) {
            return false;
        }
        BroadcastPlan plan;
        try {
            plan = BroadcastPlanner.plan(leftTensor, rightTensor);
        } catch (IllegalArgumentException ignored) {
            return false;
        }
        int[] outShape = plan.outShape();
        if (plan.flatSize() != outTensor.getFlatDataSize()
                || !Arrays.equals(outShape, outTensor.getShapeUnsafe())) {
            return false;
        }
        try {
            NativeSegmentView leftView = nativeSegmentView(context, inputIds.get(0), outShape, plan.aEffStrides());
            NativeSegmentView rightView = nativeSegmentView(context, inputIds.get(1), outShape, plan.bEffStrides());
            if (hasRegionLocalConsumer(step.compiledNode().id())) {
                NativeSegmentOutput output = allocateSegmentOutput(context, step, opLabel(op));
                NativeSegmentStridedKernels.runCompare(op, leftView, rightView, output.view());
                context.attachNativeStorage(
                        step.compiledNode().id(),
                        output.storage(),
                        "native CPU region segment " + opLabel(op).toUpperCase() + " wrote BOOL_MASK_NATIVE output"
                );
                return true;
            }
            TensorPhysicalView outputView = cpuArrayPhysicalView(
                    step.compiledNode().id(),
                    outTensor,
                    outShape,
                    outTensor.getStridesUnsafe()
            );
            byte[] output = TensorInternalAccess.boolData(outTensor);
            NativeSegmentStridedKernels.runCompare(op, leftView, rightView, output, outputView);
            TensorInternalAccess.markStorageModified(outTensor);
            context.markCpuCurrent(
                    step.compiledNode().id(),
                    "native CPU region segment " + opLabel(op).toUpperCase() + " wrote BOOL CPU array output"
            );
            return true;
        } catch (RuntimeException ex) {
            throw new NativeRegionFallbackSignal("native-cpu-region-segment-compare-failed:"
                    + opLabel(op) + ":" + safeMessage(ex));
        }
    }

    private boolean tryExecuteSegmentReduction(PreparedNodeExecution step, ExecutionContext context, Operation op) {
        if (op == null) {
            return false;
        }
        List<Integer> inputIds = inputIds(step);
        int dimension = reductionDimension(op);
        DataType dataType = step.compiledNode().dataType();
        if (inputIds.size() != 1
                || dimension < -1
                || !NativeSegmentStridedKernels.supportsReduction(op.opType(), dataType)) {
            return false;
        }
        try {
            NativeSegmentView inputView = nativeSegmentView(context, inputIds.getFirst());
            NativeSegmentOutput output = allocateSegmentOutput(context, step, opLabel(op));
            NativeSegmentStridedKernels.runReduction(op.opType(), inputView, output.view(), dimension);
            context.attachNativeStorage(
                    step.compiledNode().id(),
                    output.storage(),
                    "native CPU region segment " + opLabel(op).toUpperCase() + " wrote " + dataType + " output"
            );
            return true;
        } catch (RuntimeException ex) {
            throw new NativeRegionFallbackSignal("native-cpu-region-segment-reduction-failed:"
                    + opLabel(op) + ":" + safeMessage(ex));
        }
    }

    private NativeSegmentView nativeSegmentView(ExecutionContext context, int nodeId) {
        Tensor tensor = context.runtimeTensorForNodeId(nodeId);
        NativeTensorStorage storage = context.requireNativeReadable(nodeId, CpuMaterializationReason.CPU_CONSUMER);
        return NativeSegmentView.from(physicalView(nodeId, tensor), storage);
    }

    private NativeSegmentView nativeSegmentView(
            ExecutionContext context,
            int nodeId,
            int[] shape,
            int[] effectiveStrides
    ) {
        Tensor tensor = context.runtimeTensorForNodeId(nodeId);
        NativeTensorStorage storage = context.requireNativeReadable(nodeId, CpuMaterializationReason.CPU_CONSUMER);
        return NativeSegmentView.from(physicalView(nodeId, tensor, shape, effectiveStrides), storage);
    }

    private NativeSegmentView nativeSegmentViewIfCurrent(
            ExecutionContext context,
            int nodeId,
            int[] shape,
            int[] effectiveStrides
    ) {
        var residency = context.residencyForNodeId(nodeId);
        if (residency == null || !residency.nativeCurrent()) {
            return null;
        }
        NativeTensorStorage storage = context.nativeStorageForNodeId(nodeId);
        if (storage == null) {
            return null;
        }
        Tensor tensor = context.runtimeTensorForNodeId(nodeId);
        return NativeSegmentView.from(physicalView(nodeId, tensor, shape, effectiveStrides), storage);
    }

    private NativeSegmentOutput allocateSegmentOutput(
            ExecutionContext context,
            PreparedNodeExecution step,
            String label
    ) {
        Tensor tensor = context.runtimeTensorForNodeId(step.compiledNode().id());
        NativeTensorStorage storage = context.allocateNativeStorage(
                tensor.getDataType(),
                tensor.getFlatDataSize(),
                "node-" + step.compiledNode().id() + ":" + tensor.getLabel() + ":native-region-segment-" + label
        );
        return new NativeSegmentOutput(NativeSegmentView.from(physicalView(step.compiledNode().id(), tensor), storage), storage);
    }

    private static TensorPhysicalView physicalView(int nodeId, Tensor tensor) {
        return physicalView(nodeId, tensor, tensor.getShapeUnsafe(), tensor.getStridesUnsafe());
    }

    private static TensorPhysicalView physicalView(int nodeId, Tensor tensor, int[] shape, int[] strides) {
        return TensorPhysicalView.of(
                nodeId,
                tensor.getDataType(),
                shape,
                strides,
                tensor.getStorageOffsetUnsafe(),
                NativeCpuStorageFamily.CPU_NATIVE
        );
    }

    private static TensorPhysicalView cpuArrayPhysicalView(int nodeId, Tensor tensor, int[] shape, int[] strides) {
        return TensorPhysicalView.of(
                nodeId,
                tensor.getDataType(),
                shape,
                strides,
                tensor.getStorageOffsetUnsafe(),
                NativeCpuStorageFamily.CPU_ARRAY
        );
    }

    private static boolean sameShape(Tensor left, Tensor right) {
        return left != null && right != null && Arrays.equals(left.getShapeUnsafe(), right.getShapeUnsafe());
    }

    private record NativeSegmentOutput(NativeSegmentView view, NativeTensorStorage storage) {
    }

    private boolean hasRegionLocalConsumer(int nodeId) {
        return regionExecutionPlan.nodePlans().stream()
                .anyMatch(plan -> plan.nodeId() != nodeId && plan.inputNodeIds().contains(nodeId));
    }

    private static void addBiasInPlace(
            NativeFloat32Storage out,
            NativeFloat32Storage bias,
            int outSize,
            int lastDim
    ) {
        for (int i = 0; i < outSize; i++) {
            long outOffset = (long) i * Float.BYTES;
            long biasOffset = (long) (i % lastDim) * Float.BYTES;
            out.segment().set(
                    JAVA_FLOAT,
                    outOffset,
                    out.segment().get(JAVA_FLOAT, outOffset) + bias.segment().get(JAVA_FLOAT, biasOffset)
            );
        }
    }

    private static void addBiasInPlace(
            NativeFloat64Storage out,
            NativeFloat64Storage bias,
            int outSize,
            int lastDim
    ) {
        for (int i = 0; i < outSize; i++) {
            out.setFloat64At(i, out.getFloat64At(i) + bias.getFloat64At(i % lastDim));
        }
    }

    private static float applyUnaryF32(Operation op, float value, ExecutionContext context) {
        return switch (op.opType()) {
            case MUL_SCALAR -> value * ((mulScalar) op).getScalarF32();
            case NEG -> -value;
            case RELU -> Math.max(0.0f, value);
            case CLAMP_MIN -> Math.max(value, ((clampMin) op).getMinValueF32());
            case CLAMP_MAX -> Math.min(value, ((clampMax) op).getMaxValueF32());
            case LOG -> (float) Math.log(value);
            case EXP -> context.useFastExpApprox() ? FastTranscendentals.fastExpF32(value) : (float) Math.exp(value);
            case FAST_EXP -> FastTranscendentals.fastExpF32(value);
            case SQRT -> (float) Math.sqrt(value);
            case ABS -> Math.abs(value);
            case FLOOR -> (float) Math.floor(value);
            case CEIL -> (float) Math.ceil(value);
            case SIGN -> Math.signum(value);
            case POW -> CpuPowSupport.applyF32(value, ((pow) op).getExponentF32());
            case TANH -> context.useFastTanhApprox() ? FastTranscendentals.fastTanhF32(value) : (float) Math.tanh(value);
            case FAST_TANH -> FastTranscendentals.fastTanhF32(value);
            case SIGMOID -> 1.0f / (1.0f + (float) Math.exp(-value));
            case INV -> 1.0f / value;
            default -> throw new NativeRegionFallbackSignal("native-cpu-region-unary-unsupported:" + opLabel(op));
        };
    }

    private static double applyUnaryF64(Operation op, double value, ExecutionContext context) {
        return switch (op.opType()) {
            case MUL_SCALAR -> value * ((mulScalar) op).getScalar();
            case NEG -> -value;
            case RELU -> Math.max(0.0d, value);
            case CLAMP_MIN -> Math.max(value, ((clampMin) op).getMinValue());
            case CLAMP_MAX -> Math.min(value, ((clampMax) op).getMaxValue());
            case LOG -> Math.log(value);
            case EXP -> context.useFastExpApprox() ? FastTranscendentals.fastExpF64(value) : Math.exp(value);
            case FAST_EXP -> FastTranscendentals.fastExpF64(value);
            case SQRT -> Math.sqrt(value);
            case ABS -> Math.abs(value);
            case FLOOR -> Math.floor(value);
            case CEIL -> Math.ceil(value);
            case SIGN -> Math.signum(value);
            case POW -> CpuPowSupport.applyF64(value, ((pow) op).getExponent());
            case TANH -> context.useFastTanhApprox() ? FastTranscendentals.fastTanhF64(value) : Math.tanh(value);
            case FAST_TANH -> FastTranscendentals.fastTanhF64(value);
            case SIGMOID -> 1.0d / (1.0d + Math.exp(-value));
            case INV -> 1.0d / value;
            default -> throw new NativeRegionFallbackSignal("native-cpu-region-unary-unsupported:" + opLabel(op));
        };
    }

    private static float applyUnaryBF16(Operation op, float value) {
        return switch (op.opType()) {
            case MUL_SCALAR -> value * ((mulScalar) op).getScalarF32();
            case NEG -> -value;
            case RELU -> Math.max(0.0f, value);
            case CLAMP_MIN -> Math.max(value, ((clampMin) op).getMinValueF32());
            case CLAMP_MAX -> Math.min(value, ((clampMax) op).getMaxValueF32());
            case ABS -> Math.abs(value);
            default -> throw new NativeRegionFallbackSignal("native-cpu-region-unary-unsupported:" + opLabel(op));
        };
    }

    private static float applyDenseBinaryF32(Operation.OpType opType, float left, float right) {
        return switch (opType) {
            case ADD -> left + right;
            case SUB -> left - right;
            case MUL -> left * right;
            case DIV -> left / right;
            case MIN -> Math.min(left, right);
            case MAX -> Math.max(left, right);
            case POW_TENSOR -> CpuPowSupport.applyF32(left, right);
            default -> throw new NativeRegionFallbackSignal("native-cpu-region-binary-unsupported:"
                    + opType.name().toLowerCase());
        };
    }

    private static double applyDenseBinaryF64(Operation.OpType opType, double left, double right) {
        return switch (opType) {
            case ADD -> left + right;
            case SUB -> left - right;
            case MUL -> left * right;
            case DIV -> left / right;
            case MIN -> Math.min(left, right);
            case MAX -> Math.max(left, right);
            case POW_TENSOR -> CpuPowSupport.applyF64(left, right);
            default -> throw new NativeRegionFallbackSignal("native-cpu-region-binary-unsupported:"
                    + opType.name().toLowerCase());
        };
    }

    private static float applyDenseBinaryBF16(Operation.OpType opType, float left, float right) {
        return switch (opType) {
            case ADD -> left + right;
            case SUB -> left - right;
            case MUL -> left * right;
            case DIV -> left / right;
            case MIN -> Math.min(left, right);
            case MAX -> Math.max(left, right);
            default -> throw new NativeRegionFallbackSignal("native-cpu-region-binary-unsupported:"
                    + opType.name().toLowerCase());
        };
    }

    private static byte applyCompareF32(Operation.OpType opType, float left, float right) {
        return applyCompare(opType, left, right) ? (byte) 1 : (byte) 0;
    }

    private static byte applyCompareF64(Operation.OpType opType, double left, double right) {
        return applyCompare(opType, left, right) ? (byte) 1 : (byte) 0;
    }

    private static boolean applyCompare(Operation.OpType opType, double left, double right) {
        return switch (opType) {
            case GT -> left > right;
            case GE -> left >= right;
            case LT -> left < right;
            case LE -> left <= right;
            case EQ -> left == right;
            case NE -> left != right;
            default -> throw new NativeRegionFallbackSignal("native-cpu-region-compare-unsupported:"
                    + opType.name().toLowerCase());
        };
    }

    private static void copyNativeStorage(
            NativeTensorStorage input,
            NativeTensorStorage out,
            int size,
            DataType dataType,
            String label
    ) {
        switch (dataType) {
            case FLOAT32 -> {
                NativeFloat32Storage f32Input = requireStorage(input, NativeFloat32Storage.class, label);
                NativeFloat32Storage f32Out = requireStorage(out, NativeFloat32Storage.class, label);
                for (int i = 0; i < size; i++) {
                    f32Out.setFloat32At(i, f32Input.getFloat32At(i));
                }
            }
            case FLOAT64 -> {
                NativeFloat64Storage f64Input = requireStorage(input, NativeFloat64Storage.class, label);
                NativeFloat64Storage f64Out = requireStorage(out, NativeFloat64Storage.class, label);
                for (int i = 0; i < size; i++) {
                    f64Out.setFloat64At(i, f64Input.getFloat64At(i));
                }
            }
            case BFLOAT16 -> {
                NativeBFloat16Storage bf16Input = requireStorage(input, NativeBFloat16Storage.class, label);
                NativeBFloat16Storage bf16Out = requireStorage(out, NativeBFloat16Storage.class, label);
                for (int i = 0; i < size; i++) {
                    bf16Out.setBFloat16BitsAt(i, bf16Input.getBFloat16BitsAt(i));
                }
            }
            default -> throw new NativeRegionFallbackSignal("native-cpu-region-contiguous-dtype-unsupported:"
                    + dataType.name().toLowerCase());
        }
    }

    private static <T extends NativeTensorStorage> T requireStorage(
            NativeTensorStorage storage,
            Class<T> type,
            String label
    ) {
        if (type.isInstance(storage)) {
            return type.cast(storage);
        }
        throw new NativeRegionFallbackSignal("native-cpu-region-" + label + "-storage-type-mismatch");
    }

    private static int broadcastedFlatIndex(
            int outIndex,
            int[] outShape,
            int[] outStrides,
            int[] inputEffStrides
    ) {
        int rem = outIndex;
        int inputIndex = 0;
        for (int dim = 0; dim < outShape.length; dim++) {
            int coord = rem / outStrides[dim];
            rem %= outStrides[dim];
            inputIndex += coord * inputEffStrides[dim];
        }
        return inputIndex;
    }

    private static int flatSize(int[] shape) {
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        return size;
    }

    private static float reduceAllF32(Operation.OpType opType, NativeFloat32Storage input, int size) {
        double sum = 0.0d;
        for (int i = 0; i < size; i++) {
            sum += input.getFloat32At(i);
        }
        if (opType == Operation.OpType.MEAN) {
            sum /= size;
        }
        return (float) sum;
    }

    private static double reduceAllF64(Operation.OpType opType, NativeFloat64Storage input, int size) {
        double sum = 0.0d;
        for (int i = 0; i < size; i++) {
            sum += input.getFloat64At(i);
        }
        if (opType == Operation.OpType.MEAN) {
            sum /= size;
        }
        return sum;
    }

    private static void reduceAxisF32(
            Operation.OpType opType,
            NativeFloat32Storage input,
            NativeFloat32Storage out,
            int[] shape,
            int dimension
    ) {
        int reducedSize = shape[dimension];
        int axisStride = denseStride(shape, dimension);
        int outSize = expectedReductionOutputSize(shape, dimension);
        int[] outDenseStrides = denseStridesExcludingDim(shape, dimension);
        for (int outIndex = 0; outIndex < outSize; outIndex++) {
            int inputBase = inputBaseOffset(outIndex, shape, outDenseStrides, dimension);
            double sum = 0.0d;
            for (int k = 0; k < reducedSize; k++) {
                sum += input.getFloat32At(inputBase + k * axisStride);
            }
            if (opType == Operation.OpType.MEAN) {
                sum /= reducedSize;
            }
            out.setFloat32At(outIndex, (float) sum);
        }
    }

    private static void reduceAxisF64(
            Operation.OpType opType,
            NativeFloat64Storage input,
            NativeFloat64Storage out,
            int[] shape,
            int dimension
    ) {
        int reducedSize = shape[dimension];
        int axisStride = denseStride(shape, dimension);
        int outSize = expectedReductionOutputSize(shape, dimension);
        int[] outDenseStrides = denseStridesExcludingDim(shape, dimension);
        for (int outIndex = 0; outIndex < outSize; outIndex++) {
            int inputBase = inputBaseOffset(outIndex, shape, outDenseStrides, dimension);
            double sum = 0.0d;
            for (int k = 0; k < reducedSize; k++) {
                sum += input.getFloat64At(inputBase + k * axisStride);
            }
            if (opType == Operation.OpType.MEAN) {
                sum /= reducedSize;
            }
            out.setFloat64At(outIndex, sum);
        }
    }

    private static int inputBaseOffset(int outIndex, int[] shape, int[] outDenseStrides, int dimension) {
        int rem = outIndex;
        int base = 0;
        int outAxis = 0;
        for (int dim = 0; dim < shape.length; dim++) {
            if (dim == dimension) {
                continue;
            }
            int coord = rem / outDenseStrides[outAxis];
            rem %= outDenseStrides[outAxis];
            base += coord * denseStride(shape, dim);
            outAxis++;
        }
        return base;
    }

    private static int[] denseStridesExcludingDim(int[] shape, int dimension) {
        int[] strides = new int[Math.max(0, shape.length - 1)];
        int stride = 1;
        for (int dim = shape.length - 1; dim >= 0; dim--) {
            if (dim == dimension) {
                continue;
            }
            strides[dim < dimension ? dim : dim - 1] = stride;
            stride *= shape[dim];
        }
        return strides;
    }

    private static int denseStride(int[] shape, int dimension) {
        int stride = 1;
        for (int dim = dimension + 1; dim < shape.length; dim++) {
            stride *= shape[dim];
        }
        return stride;
    }

    private static int expectedReductionOutputSize(int[] shape, int dimension) {
        if (dimension == -1) {
            return 1;
        }
        int size = 1;
        for (int dim = 0; dim < shape.length; dim++) {
            if (dim != dimension) {
                size *= shape[dim];
            }
        }
        return size;
    }

    private static int reductionDimension(Operation op) {
        if (op instanceof sum reduction) {
            return reduction.getDimension();
        }
        if (op instanceof mean reduction) {
            return reduction.getDimension();
        }
        if (op instanceof reduceAll reduction) {
            return reduction.getDimension();
        }
        if (op instanceof reduceAny reduction) {
            return reduction.getDimension();
        }
        if (op instanceof reduceMin reduction) {
            return reduction.getDimension();
        }
        if (op instanceof reduceMax reduction) {
            return reduction.getDimension();
        }
        return Integer.MIN_VALUE;
    }

    private static NativeFloat32Storage requireF32Storage(ExecutionContext context, int nodeId, String op) {
        NativeTensorStorage storage = context.requireNativeReadable(nodeId, CpuMaterializationReason.CPU_CONSUMER);
        if (storage instanceof NativeFloat32Storage f32) {
            return f32;
        }
        throw new NativeRegionFallbackSignal(op + "-requires-float32-native-storage");
    }

    private static NativeFloat64Storage requireF64Storage(ExecutionContext context, int nodeId, String op) {
        NativeTensorStorage storage = context.requireNativeReadable(nodeId, CpuMaterializationReason.CPU_CONSUMER);
        if (storage instanceof NativeFloat64Storage f64) {
            return f64;
        }
        throw new NativeRegionFallbackSignal(op + "-requires-float64-native-storage");
    }

    private static NativeBFloat16Storage requireBF16Storage(ExecutionContext context, int nodeId, String op) {
        NativeTensorStorage storage = context.requireNativeReadable(nodeId, CpuMaterializationReason.CPU_CONSUMER);
        if (storage instanceof NativeBFloat16Storage bf16) {
            return bf16;
        }
        throw new NativeRegionFallbackSignal(op + "-requires-bfloat16-native-storage");
    }

    private static NativeFloat32Storage allocateF32(
            ExecutionContext context,
            PreparedNodeExecution step,
            String label
    ) {
        Tensor outTensor = context.runtimeTensorForNodeId(step.compiledNode().id());
        return (NativeFloat32Storage) context.allocateNativeStorage(
                DataType.FLOAT32,
                outTensor.getFlatDataSize(),
                "node-" + step.compiledNode().id() + ":" + outTensor.getLabel() + ":native-region-f32-" + label
        );
    }

    private static NativeFloat64Storage allocateF64(
            ExecutionContext context,
            PreparedNodeExecution step,
            String label
    ) {
        Tensor outTensor = context.runtimeTensorForNodeId(step.compiledNode().id());
        return (NativeFloat64Storage) context.allocateNativeStorage(
                DataType.FLOAT64,
                outTensor.getFlatDataSize(),
                "node-" + step.compiledNode().id() + ":" + outTensor.getLabel() + ":native-region-f64-" + label
        );
    }

    private static NativeBFloat16Storage allocateBF16(
            ExecutionContext context,
            PreparedNodeExecution step,
            String label
    ) {
        Tensor outTensor = context.runtimeTensorForNodeId(step.compiledNode().id());
        return (NativeBFloat16Storage) context.allocateNativeStorage(
                DataType.BFLOAT16,
                outTensor.getFlatDataSize(),
                "node-" + step.compiledNode().id() + ":" + outTensor.getLabel() + ":native-region-bf16-" + label
        );
    }

    private boolean expectsNativeOutput(PreparedNodeExecution step) {
        if (step == null || step.metadata().backend() != ComputeBackend.CPU || step.metadata().cpuPlan() == null) {
            return false;
        }
        PreparedNativeCpuPlan plan = step.metadata().cpuPlan().nativeCpuPlan();
        return plan != null && plan.allowsNativeInputs();
    }

    private void fallbackOrThrow(ExecutionContext context, String reason, RuntimeException cause) {
        String safeReason = reason == null || reason.isBlank() ? "native-cpu-region-fallback" : reason;
        lastFallbackReason = safeReason;
        if (requiresNative(context)) {
            IllegalStateException failure = new IllegalStateException(
                    "Native CPU region execution required but region fell back: "
                            + regionExecutionPlan.regionId() + " reason=" + safeReason
            );
            if (cause != null) {
                failure.initCause(cause);
            }
            throw failure;
        }
        lastRoute = "FALLBACK_TO_ARRAY";
        ExecutionContext fallbackContext = fallbackContext(context);
        for (PreparedNodeExecution step : fallbackSteps) {
            requireCpuReadableInputs(step, fallbackContext);
            CPU_BACKEND.execute(step.compiledNode(), step.metadata(), fallbackContext);
            fallbackContext.markCpuCurrent(step.compiledNode().id(), "native CPU region fallback wrote CPU array");
        }
        verifyCpuBoundaryOutputs(fallbackContext);
    }

    private static ExecutionContext fallbackContext(ExecutionContext context) {
        if (context.runtimeConfig() == null) {
            return context;
        }
        return context.withRuntimeConfig(context.runtimeConfig().withCpuStorageProfile(CpuStorageProfile.CPU_ARRAY));
    }

    private void verifyNativeBoundaryOutputs(ExecutionContext context) {
        List<Integer> boundaryOutputNodeIds = boundaryOutputNodeIds();
        for (int nodeId : boundaryOutputNodeIds) {
            if (!boundaryRequiresNativeStorage(nodeId)) {
                context.requireCpuReadable(nodeId, CpuMaterializationReason.CPU_CONSUMER);
                var cpuResidency = context.residencyForNodeId(nodeId);
                if (cpuResidency == null || !cpuResidency.cpuCurrent()) {
                    throw new NativeRegionFallbackSignal("native-cpu-region-boundary-not-cpu-array:node-" + nodeId);
                }
                continue;
            }
            var residency = context.residencyForNodeId(nodeId);
            if (residency == null || !residency.nativeCurrent()) {
                throw new NativeRegionFallbackSignal("native-cpu-region-boundary-not-native:node-" + nodeId);
            }
        }
    }

    private void verifyCpuBoundaryOutputs(ExecutionContext context) {
        List<Integer> boundaryOutputNodeIds = boundaryOutputNodeIds();
        for (int nodeId : boundaryOutputNodeIds) {
            context.requireCpuReadable(nodeId, CpuMaterializationReason.CPU_CONSUMER);
            var residency = context.residencyForNodeId(nodeId);
            if (residency == null || !residency.cpuCurrent()) {
                throw new IllegalStateException("Native CPU region fallback did not publish CPU boundary output: node-"
                        + nodeId);
            }
        }
    }

    private List<Integer> boundaryOutputNodeIds() {
        return regionExecutionPlan.boundaryOutputNodeIds().isEmpty()
                ? List.of(regionExecutionPlan.anchorNodeId())
                : regionExecutionPlan.boundaryOutputNodeIds();
    }

    private boolean boundaryRequiresNativeStorage(int nodeId) {
        return regionExecutionPlan.nodePlans().stream()
                .filter(plan -> plan.nodeId() == nodeId)
                .findFirst()
                .map(plan -> plan.storageContract() != RegionStorageContract.CPU_ARRAY
                        && plan.storageContract() != RegionStorageContract.MIXED_BOUNDARY)
                .orElse(true);
    }

    private static void requireCpuReadableInputs(PreparedNodeExecution step, ExecutionContext context) {
        for (int inputId : inputIds(step)) {
            Tensor ignored = context.runtimeTensorForNodeId(inputId);
            if (ignored != null) {
                context.requireCpuReadable(inputId, CpuMaterializationReason.CPU_CONSUMER);
            }
        }
    }

    private static List<Integer> inputIds(PreparedNodeExecution step) {
        return step.metadata().executionInputNodeIds().isEmpty()
                ? step.compiledNode().inputIds()
                : step.metadata().executionInputNodeIds();
    }

    private static boolean requiresNative(ExecutionContext context) {
        return context.runtimeConfig() != null
                && context.runtimeConfig().nativeCpuFailurePolicy() == NativeCpuFailurePolicy.REQUIRE_NATIVE;
    }

    private static boolean supportsRegionLocalViewDType(DataType dataType) {
        return dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64 || dataType == DataType.BFLOAT16;
    }

    private static boolean supportsNativeContiguousCopyDType(DataType dataType) {
        return dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64 || dataType == DataType.BFLOAT16;
    }

    private static boolean isNativeRegionCast(DataType input, DataType output) {
        return input == DataType.FLOAT32 && output == DataType.BFLOAT16
                || input == DataType.BFLOAT16 && output == DataType.FLOAT32;
    }

    private static boolean supportsRegionLocalViewOp(Operation.OpType opType) {
        return opType == Operation.OpType.NOOP
                || opType == Operation.OpType.RESHAPE
                || opType == Operation.OpType.PERMUTE
                || opType == Operation.OpType.EXPAND
                || opType == Operation.OpType.SELECT
                || opType == Operation.OpType.SLICE
                || opType == Operation.OpType.EXPAND_DIMS
                || opType == Operation.OpType.SQUEEZE;
    }

    private static boolean isCompareOp(Operation.OpType opType) {
        return opType == Operation.OpType.GT
                || opType == Operation.OpType.GE
                || opType == Operation.OpType.LT
                || opType == Operation.OpType.LE
                || opType == Operation.OpType.EQ
                || opType == Operation.OpType.NE;
    }

    private static String opLabel(Operation op) {
        return op == null ? "unknown" : op.opType().name().toLowerCase();
    }

    @Override
    public RegionExecutionPlan regionExecutionPlan() {
        return regionExecutionPlan;
    }

    @Override
    public List<PreparedNodeExecution> nativeSteps() {
        return nativeSteps;
    }

    @Override
    public List<PreparedNodeExecution> fallbackSteps() {
        return fallbackSteps;
    }

    @Override
    public String lastRoute() {
        return lastRoute;
    }

    @Override
    public String lastFallbackReason() {
        return lastFallbackReason;
    }

    @Override
    public int lastRegionLocalKernelCount() {
        return lastRegionLocalKernelCount;
    }

    @Override
    public int lastRegionLocalViewCount() {
        return lastRegionLocalViewCount;
    }

    public int lastExecutedGroupCount() {
        return lastExecutedGroupCount;
    }

    private static String safeMessage(RuntimeException ex) {
        String message = ex == null ? "" : ex.getMessage();
        return message == null || message.isBlank() ? ex.getClass().getSimpleName() : message;
    }

    private static final class NativeRegionFallbackSignal extends RuntimeException {
        private NativeRegionFallbackSignal(String message) {
            super(message);
        }
    }
}
