package backend.cpu.nativecpu;

import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.CpuNodeExecutionPlan;
import backend.cpu.kernels.CpuDTypeOps;
import backend.memory.CpuMaterializationReason;
import config.runtime.CpuStorageProfile;
import config.runtime.NativeCpuFailurePolicy;
import config.runtime.RuntimeConfig;
import operations.Operation;
import operations.reduction.mean;
import operations.reduction.reduceMax;
import operations.reduction.reduceMin;
import operations.reduction.sum;
import tensor.DataType;
import tensor.NativeBFloat16Storage;
import tensor.NativeFloat32Storage;
import tensor.NativeFloat64Storage;
import tensor.NativeTensorStorage;
import tensor.Tensor;

/**
 * Native CPU reduction slice for dense F32/F64 reductions.
 */
public final class NativeCpuReductionExecutor {
    private NativeCpuReductionExecutor() {
    }

    public static boolean acceptsNativeInputs(Operation op, DataType dataType, CpuNodeExecutionPlan plan, RuntimeConfig runtimeConfig) {
        if (!nativeRequested(runtimeConfig) || op == null || !supportsNativeReductionDType(dataType) || plan == null || plan.stridedPath()) {
            return false;
        }
        Operation.OpType opType = op.opType();
        return isSupportedReduction(dataType, opType) && reductionDimension(op) >= -1;
    }

    public static boolean tryRunSumLike(
            Operation.OpType opType,
            Tensor input,
            Tensor node,
            int dimension,
            CpuKernelContext context
    ) {
        if (!nativeRequested(context)) {
            return false;
        }
        return tryRunReduction(opType, input, node, dimension, context);
    }

    public static boolean tryRunMinMax(
            Operation.OpType opType,
            Tensor input,
            Tensor node,
            int dimension,
            CpuKernelContext context
    ) {
        if (!nativeRequested(context)) {
            return false;
        }
        return tryRunReduction(opType, input, node, dimension, context);
    }

    private static boolean tryRunReduction(
            Operation.OpType opType,
            Tensor input,
            Tensor node,
            int dimension,
            CpuKernelContext context
    ) {
        NativeCpuKernelFact fact = NativeCpuKernelFacts.factFor(opTypeOrUnknown(opType), node.getDataType());
        if (!isSupportedReduction(node.getDataType(), opType)) {
            return fallback(context, fact, opType, "native-kernel-unsupported:" + opLabel(opType));
        }
        if (!supportsNativeReductionDType(node.getDataType()) || input.getDataType() != node.getDataType()) {
            return fallback(context, fact, opType, "native-storage-dtype-unsupported:" + node.getDataType().name().toLowerCase());
        }
        if (context.nodePlan().stridedPath()) {
            return fallback(context, fact, opType, "native-kernel-ineligible:" + opLabel(opType) + "-strided");
        }
        if (!input.isContiguous() || input.hasStorageOffset()) {
            return fallback(context, fact, opType, "native-kernel-ineligible:" + opLabel(opType) + "-strided");
        }
        int[] shape = input.getShapeUnsafe();
        if (shape == null || shape.length == 0 || dimension < -1 || dimension >= shape.length) {
            return fallback(context, fact, opType, "native-kernel-ineligible:" + opLabel(opType) + "-axis");
        }
        if (input.getFlatDataSize() <= 0 || expectedOutputSize(shape, dimension) != node.getFlatDataSize()) {
            return fallback(context, fact, opType, "native-kernel-ineligible:" + opLabel(opType) + "-shape");
        }
        try {
            NativeTensorStorage out;
            if (node.getDataType() == DataType.FLOAT64) {
                NativeFloat64Storage in = requireF64NativeInput(context, opLabel(opType).toUpperCase());
                NativeFloat64Storage f64Out = allocateF64(node, context, opLabel(opType));
                if (dimension == -1) {
                    f64Out.setFloat64At(0, reduceAllF64(opType, in, input.getFlatDataSize()));
                } else {
                    reduceAxisF64(opType, in, f64Out, shape, dimension);
                }
                out = f64Out;
            } else if (node.getDataType() == DataType.BFLOAT16) {
                NativeBFloat16Storage in = requireBF16NativeInput(context, opLabel(opType).toUpperCase());
                NativeBFloat16Storage bf16Out = allocateBF16(node, context, opLabel(opType));
                if (dimension == -1) {
                    bf16Out.setBFloat16BitsAt(0, CpuDTypeOps.toBFloat16Bits(reduceAllBF16(opType, in, input.getFlatDataSize())));
                } else {
                    reduceAxisBF16(opType, in, bf16Out, shape, dimension);
                }
                out = bf16Out;
            } else {
                NativeFloat32Storage in = requireF32NativeInput(context, opLabel(opType).toUpperCase());
                NativeFloat32Storage f32Out = allocateF32(node, context, opLabel(opType));
                if (dimension == -1) {
                    f32Out.setFloat32At(0, reduceAllF32(opType, in, input.getFlatDataSize()));
                } else {
                    reduceAxisF32(opType, in, f32Out, shape, dimension);
                }
                out = f32Out;
            }
            context.executionContext().attachNativeStorage(context.nodeId(), out, "native CPU " + opLabel(opType).toUpperCase() + " wrote " + node.getDataType() + " output");
            publishTrace(context, fact, "CPU_NATIVE", "");
            return true;
        } catch (Throwable t) {
            return fallback(context, fact, opType, "native-kernel-failed:" + opLabel(opType) + ":" + safeMessage(t));
        }
    }

    private static float reduceAllF32(Operation.OpType opType, NativeFloat32Storage input, int size) {
        if (opType == Operation.OpType.REDUCE_MIN || opType == Operation.OpType.REDUCE_MAX) {
            float best = input.getFloat32At(0);
            for (int i = 1; i < size; i++) {
                float value = input.getFloat32At(i);
                best = opType == Operation.OpType.REDUCE_MAX ? Math.max(best, value) : Math.min(best, value);
            }
            return best;
        }
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
        if (opType == Operation.OpType.REDUCE_MIN || opType == Operation.OpType.REDUCE_MAX) {
            double best = input.getFloat64At(0);
            for (int i = 1; i < size; i++) {
                double value = input.getFloat64At(i);
                best = opType == Operation.OpType.REDUCE_MAX ? Math.max(best, value) : Math.min(best, value);
            }
            return best;
        }
        double sum = 0.0d;
        for (int i = 0; i < size; i++) {
            sum += input.getFloat64At(i);
        }
        if (opType == Operation.OpType.MEAN) {
            sum /= size;
        }
        return sum;
    }

    private static float reduceAllBF16(Operation.OpType opType, NativeBFloat16Storage input, int size) {
        double sum = 0.0d;
        for (int i = 0; i < size; i++) {
            sum += CpuDTypeOps.fromBFloat16Bits(input.getBFloat16BitsAt(i));
        }
        if (opType == Operation.OpType.MEAN) {
            sum /= size;
        }
        return (float) sum;
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
        int outSize = expectedOutputSize(shape, dimension);
        int[] outDenseStrides = denseStridesExcludingDim(shape, dimension);
        for (int outIndex = 0; outIndex < outSize; outIndex++) {
            int inputBase = inputBaseOffset(outIndex, shape, outDenseStrides, dimension);
            out.setFloat32At(outIndex, reduceAxisF32Value(opType, input, inputBase, axisStride, reducedSize));
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
        int outSize = expectedOutputSize(shape, dimension);
        int[] outDenseStrides = denseStridesExcludingDim(shape, dimension);
        for (int outIndex = 0; outIndex < outSize; outIndex++) {
            int inputBase = inputBaseOffset(outIndex, shape, outDenseStrides, dimension);
            out.setFloat64At(outIndex, reduceAxisF64Value(opType, input, inputBase, axisStride, reducedSize));
        }
    }

    private static void reduceAxisBF16(
            Operation.OpType opType,
            NativeBFloat16Storage input,
            NativeBFloat16Storage out,
            int[] shape,
            int dimension
    ) {
        int reducedSize = shape[dimension];
        int axisStride = denseStride(shape, dimension);
        int outSize = expectedOutputSize(shape, dimension);
        int[] outDenseStrides = denseStridesExcludingDim(shape, dimension);
        for (int outIndex = 0; outIndex < outSize; outIndex++) {
            int inputBase = inputBaseOffset(outIndex, shape, outDenseStrides, dimension);
            out.setBFloat16BitsAt(
                    outIndex,
                    CpuDTypeOps.toBFloat16Bits(reduceAxisBF16Value(opType, input, inputBase, axisStride, reducedSize))
            );
        }
    }

    private static float reduceAxisF32Value(
            Operation.OpType opType,
            NativeFloat32Storage input,
            int inputBase,
            int axisStride,
            int reducedSize
    ) {
        if (opType == Operation.OpType.REDUCE_MIN || opType == Operation.OpType.REDUCE_MAX) {
            float best = input.getFloat32At(inputBase);
            for (int k = 1; k < reducedSize; k++) {
                float value = input.getFloat32At(inputBase + k * axisStride);
                best = opType == Operation.OpType.REDUCE_MAX ? Math.max(best, value) : Math.min(best, value);
            }
            return best;
        }
        double sum = 0.0d;
        for (int k = 0; k < reducedSize; k++) {
            sum += input.getFloat32At(inputBase + k * axisStride);
        }
        if (opType == Operation.OpType.MEAN) {
            sum /= reducedSize;
        }
        return (float) sum;
    }

    private static double reduceAxisF64Value(
            Operation.OpType opType,
            NativeFloat64Storage input,
            int inputBase,
            int axisStride,
            int reducedSize
    ) {
        if (opType == Operation.OpType.REDUCE_MIN || opType == Operation.OpType.REDUCE_MAX) {
            double best = input.getFloat64At(inputBase);
            for (int k = 1; k < reducedSize; k++) {
                double value = input.getFloat64At(inputBase + k * axisStride);
                best = opType == Operation.OpType.REDUCE_MAX ? Math.max(best, value) : Math.min(best, value);
            }
            return best;
        }
        double sum = 0.0d;
        for (int k = 0; k < reducedSize; k++) {
            sum += input.getFloat64At(inputBase + k * axisStride);
        }
        if (opType == Operation.OpType.MEAN) {
            sum /= reducedSize;
        }
        return sum;
    }

    private static float reduceAxisBF16Value(
            Operation.OpType opType,
            NativeBFloat16Storage input,
            int inputBase,
            int axisStride,
            int reducedSize
    ) {
        double sum = 0.0d;
        for (int k = 0; k < reducedSize; k++) {
            sum += CpuDTypeOps.fromBFloat16Bits(input.getBFloat16BitsAt(inputBase + k * axisStride));
        }
        if (opType == Operation.OpType.MEAN) {
            sum /= reducedSize;
        }
        return (float) sum;
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

    private static int expectedOutputSize(int[] shape, int dimension) {
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

    private static boolean fallback(CpuKernelContext context, NativeCpuKernelFact fact, Operation.OpType opType, String reason) {
        handleRequireNative(context, opLabel(opType) + " reduction", reason);
        requireCpuReadableInputs(context);
        publishTrace(context, fact, "CPU_ARRAY", reason);
        return false;
    }

    private static void handleRequireNative(CpuKernelContext context, String family, String reason) {
        if (context.executionContext().runtimeConfig().nativeCpuFailurePolicy() == NativeCpuFailurePolicy.REQUIRE_NATIVE) {
            throw new IllegalStateException("Native CPU execution required but " + family + " fell back to Java: " + reason);
        }
    }

    private static void requireCpuReadableInputs(CpuKernelContext context) {
        for (int inputNodeId : context.inputNodeIds()) {
            context.executionContext().requireCpuReadable(inputNodeId, CpuMaterializationReason.CPU_CONSUMER);
        }
    }

    private static NativeFloat32Storage requireF32NativeInput(CpuKernelContext context, String op) {
        int inputNodeId = context.inputNodeIds().getFirst();
        NativeTensorStorage storage = context.executionContext().requireNativeReadable(inputNodeId, CpuMaterializationReason.CPU_CONSUMER);
        if (storage instanceof NativeFloat32Storage f32) {
            return f32;
        }
        throw new IllegalStateException("native " + op + " requires FLOAT32 native input storage");
    }

    private static NativeFloat64Storage requireF64NativeInput(CpuKernelContext context, String op) {
        int inputNodeId = context.inputNodeIds().getFirst();
        NativeTensorStorage storage = context.executionContext().requireNativeReadable(inputNodeId, CpuMaterializationReason.CPU_CONSUMER);
        if (storage instanceof NativeFloat64Storage f64) {
            return f64;
        }
        throw new IllegalStateException("native " + op + " requires FLOAT64 native input storage");
    }

    private static NativeBFloat16Storage requireBF16NativeInput(CpuKernelContext context, String op) {
        int inputNodeId = context.inputNodeIds().getFirst();
        NativeTensorStorage storage = context.executionContext().requireNativeReadable(inputNodeId, CpuMaterializationReason.CPU_CONSUMER);
        if (storage instanceof NativeBFloat16Storage bf16) {
            return bf16;
        }
        throw new IllegalStateException("native " + op + " requires BFLOAT16 native input storage");
    }

    private static NativeFloat32Storage allocateF32(Tensor node, CpuKernelContext context, String label) {
        return (NativeFloat32Storage) context.executionContext().allocateNativeStorage(
                DataType.FLOAT32,
                node.getFlatDataSize(),
                "node-" + context.nodeId() + ":" + node.getLabel() + ":native-f32-" + label
        );
    }

    private static NativeFloat64Storage allocateF64(Tensor node, CpuKernelContext context, String label) {
        return (NativeFloat64Storage) context.executionContext().allocateNativeStorage(
                DataType.FLOAT64,
                node.getFlatDataSize(),
                "node-" + context.nodeId() + ":" + node.getLabel() + ":native-f64-" + label
        );
    }

    private static NativeBFloat16Storage allocateBF16(Tensor node, CpuKernelContext context, String label) {
        return (NativeBFloat16Storage) context.executionContext().allocateNativeStorage(
                DataType.BFLOAT16,
                node.getFlatDataSize(),
                "node-" + context.nodeId() + ":" + node.getLabel() + ":native-bf16-" + label
        );
    }

    private static void publishTrace(CpuKernelContext context, NativeCpuKernelFact fact, String actualCpuStorage, String fallbackReason) {
        var runtime = context.executionContext().runtimeConfig();
        Tensor runtimeTensor = context.executionContext().runtimeTensorForNodeId(context.nodeId());
        boolean bf16Promoted = runtimeTensor.getDataType() == DataType.BFLOAT16
                && "CPU_NATIVE".equals(actualCpuStorage)
                && (fallbackReason == null || fallbackReason.isBlank());
        context.putRuntimeState(
                runtimeTensor,
                new NativeCpuTraceState(
                        runtime.cpuStorageProfile().name(),
                        runtime.nativeCpuFailurePolicy().name(),
                        "CPU_NATIVE",
                        actualCpuStorage,
                        fact.status().name(),
                        fact.family().name(),
                        fallbackReason,
                        bf16Promoted ? "BF16" : "",
                        bf16Promoted ? "F32_PROMOTED" : ""
                )
        );
    }

    private static boolean nativeRequested(CpuKernelContext context) {
        return NativeCpuRuntimePolicy.nativeRequested(context);
    }

    private static boolean nativeRequested(RuntimeConfig runtimeConfig) {
        return runtimeConfig != null && runtimeConfig.cpuStorageProfile() == CpuStorageProfile.CPU_NATIVE;
    }

    private static boolean supportsNativeReductionDType(DataType dataType) {
        return dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64 || dataType == DataType.BFLOAT16;
    }

    private static boolean isSupportedReduction(DataType dataType, Operation.OpType opType) {
        if (dataType == DataType.BFLOAT16) {
            return opType == Operation.OpType.SUM || opType == Operation.OpType.MEAN;
        }
        return (dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64)
                && (opType == Operation.OpType.SUM
                || opType == Operation.OpType.MEAN
                || opType == Operation.OpType.REDUCE_MIN
                || opType == Operation.OpType.REDUCE_MAX);
    }

    private static int reductionDimension(Operation op) {
        if (op instanceof sum reduction) {
            return reduction.getDimension();
        }
        if (op instanceof mean reduction) {
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

    private static Operation.OpType opTypeOrUnknown(Operation.OpType opType) {
        return opType == null ? Operation.OpType.UNKNOWN : opType;
    }

    private static String opLabel(Operation.OpType opType) {
        return opTypeOrUnknown(opType).name().toLowerCase();
    }

    private static String safeMessage(Throwable t) {
        String message = t.getMessage();
        return message == null || message.isBlank() ? t.getClass().getSimpleName() : message;
    }
}
