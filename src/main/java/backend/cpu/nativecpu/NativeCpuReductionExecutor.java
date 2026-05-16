package backend.cpu.nativecpu;

import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.CpuNodeExecutionPlan;
import backend.memory.CpuMaterializationReason;
import config.runtime.CpuStorageProfile;
import config.runtime.NativeCpuFailurePolicy;
import config.runtime.RuntimeConfig;
import operations.Operation;
import operations.reduction.mean;
import operations.reduction.sum;
import tensor.DataType;
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
        return (opType == Operation.OpType.SUM || opType == Operation.OpType.MEAN) && reductionDimension(op) >= -1;
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
        NativeCpuKernelFact fact = NativeCpuKernelFacts.factFor(opTypeOrUnknown(opType), node.getDataType());
        if (opType != Operation.OpType.SUM && opType != Operation.OpType.MEAN) {
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
        int outSize = expectedOutputSize(shape, dimension);
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
        int outSize = expectedOutputSize(shape, dimension);
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

    private static void publishTrace(CpuKernelContext context, NativeCpuKernelFact fact, String actualCpuStorage, String fallbackReason) {
        var runtime = context.executionContext().runtimeConfig();
        context.putRuntimeState(
                context.executionContext().runtimeTensorForNodeId(context.nodeId()),
                new NativeCpuTraceState(
                        runtime.cpuStorageProfile().name(),
                        runtime.nativeCpuFailurePolicy().name(),
                        "CPU_NATIVE",
                        actualCpuStorage,
                        fact.status().name(),
                        fact.family().name(),
                        fallbackReason
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
        return dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64;
    }

    private static int reductionDimension(Operation op) {
        if (op instanceof sum reduction) {
            return reduction.getDimension();
        }
        if (op instanceof mean reduction) {
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
