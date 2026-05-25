package backend.cpu.kernels.reduction;

import backend.cpu.kernels.CpuDTypeOps;
import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.CpuNativeTraceSupport;
import backend.cpu.nativecpu.layout.NativeCpuStorageFamily;
import backend.cpu.nativecpu.layout.NativeSegmentStridedKernels;
import backend.cpu.nativecpu.layout.NativeSegmentView;
import backend.cpu.nativecpu.layout.TensorPhysicalView;
import backend.memory.CpuMaterializationReason;
import config.runtime.CpuStorageProfile;
import config.runtime.NativeCpuFailurePolicy;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;
import tensor.storage.NativeTensorStorage;

final class ReductionStorageLoops {
    private ReductionStorageLoops() {
    }

    static boolean tryRunSumLike(
            Operation.OpType opType,
            Tensor input,
            Tensor node,
            int dimension,
            CpuKernelContext context
    ) {
        if (!nativeRequested(context)) {
            return false;
        }
        return tryRun(opType, input, node, dimension, context);
    }

    static boolean tryRunMinMax(
            Operation.OpType opType,
            Tensor input,
            Tensor node,
            int dimension,
            CpuKernelContext context
    ) {
        if (!nativeRequested(context)) {
            return false;
        }
        return tryRun(opType, input, node, dimension, context);
    }

    static boolean tryRunBool(
            Operation.OpType opType,
            Tensor input,
            Tensor node,
            int dimension,
            CpuKernelContext context
    ) {
        if (!nativeRequested(context)) {
            return false;
        }
        return tryRun(opType, input, node, dimension, context);
    }

    private static boolean tryRun(
            Operation.OpType opType,
            Tensor input,
            Tensor node,
            int dimension,
            CpuKernelContext context
    ) {
        Operation.OpType safeOpType = opTypeOrUnknown(opType);
        String reason = nativeIneligibleReason(safeOpType, input, node, dimension, context);
        if (!reason.isBlank()) {
            return fallback(context, safeOpType, reason);
        }
        try {
            NativeTensorStorage inputStorage = requireNativeInput(context, input.getDataType(), opLabel(safeOpType).toUpperCase());
            NativeTensorStorage outputStorage = allocateNativeOutput(node, context, opLabel(safeOpType));
            NativeSegmentStridedKernels.runReduction(
                    safeOpType,
                    nativeView(context.inputNodeIds().getFirst(), input, inputStorage),
                    nativeView(context.nodeId(), node, outputStorage),
                    dimension
            );
            context.executionContext().attachNativeStorage(
                    context.nodeId(),
                    outputStorage,
                    "reduction storage loop " + opLabel(safeOpType).toUpperCase() + " wrote " + node.getDataType() + " native output"
            );
            CpuNativeTraceSupport.publishSegmentScalar(context, CpuNativeTraceSupport.CPU_NATIVE, "");
            return true;
        } catch (Throwable t) {
            return fallback(
                    context,
                    safeOpType,
                    "native-kernel-failed:" + opLabel(safeOpType) + ":" + safeMessage(t)
            );
        }
    }

    private static String nativeIneligibleReason(
            Operation.OpType opType,
            Tensor input,
            Tensor node,
            int dimension,
            CpuKernelContext context
    ) {
        String label = opLabel(opType);
        if (context == null || context.nodePlan() == null) {
            return "native-kernel-ineligible:" + label + "-plan";
        }
        if (input == null || node == null || context.inputNodeIds().size() != 1) {
            return "native-kernel-ineligible:" + label + "-input-count";
        }
        if (!NativeSegmentStridedKernels.supportsReduction(opType, node.getDataType())) {
            return unsupportedReductionReason(opType, node.getDataType());
        }
        if (input.getDataType() != node.getDataType()) {
            return "native-storage-dtype-unsupported:" + node.getDataType().name().toLowerCase();
        }
        if (context.nodePlan().stridedPath() || !denseTensor(input) || !denseTensor(node)) {
            return "native-kernel-ineligible:" + label + "-strided";
        }
        int[] shape = input.getShapeUnsafe();
        if (shape == null || shape.length == 0 || dimension < -1 || dimension >= shape.length) {
            return "native-kernel-ineligible:" + label + "-axis";
        }
        if (input.getFlatDataSize() <= 0 || expectedOutputSize(shape, dimension) != node.getFlatDataSize()) {
            return "native-kernel-ineligible:" + label + "-shape";
        }
        return "";
    }

    private static String unsupportedReductionReason(Operation.OpType opType, DataType dataType) {
        if (dataType == DataType.BFLOAT16 && (opType == Operation.OpType.REDUCE_MIN || opType == Operation.OpType.REDUCE_MAX)) {
            return "native-bf16-reduce-minmax-output-policy-unsupported";
        }
        if (!supportsNativeReductionDType(dataType)) {
            return "native-storage-dtype-unsupported:" + dataType.name().toLowerCase();
        }
        return "native-kernel-unsupported:" + opLabel(opType);
    }

    private static boolean fallback(CpuKernelContext context, Operation.OpType opType, String reason) {
        requireFallbackAllowed(context, opLabel(opType) + " reduction", reason);
        requireCpuReadableInputs(context);
        CpuNativeTraceSupport.publishSegmentScalar(context, CpuNativeTraceSupport.CPU_ARRAY, reason);
        return false;
    }

    private static NativeTensorStorage requireNativeInput(CpuKernelContext context, DataType dtype, String op) {
        NativeTensorStorage storage = context.executionContext().requireNativeReadable(
                context.inputNodeIds().getFirst(),
                CpuMaterializationReason.CPU_CONSUMER
        );
        if (storage.getType() != dtype) {
            throw new IllegalStateException("native " + op + " requires " + dtype + " native input storage");
        }
        return storage;
    }

    private static NativeTensorStorage allocateNativeOutput(Tensor node, CpuKernelContext context, String label) {
        return context.executionContext().allocateNativeStorage(
                node.getDataType(),
                node.getFlatDataSize(),
                "node-" + context.nodeId() + ":" + node.getLabel() + ":native-" + label
        );
    }

    private static NativeSegmentView nativeView(int nodeId, Tensor tensor, NativeTensorStorage storage) {
        return NativeSegmentView.from(
                TensorPhysicalView.of(
                        nodeId,
                        tensor.getDataType(),
                        tensor.getShapeUnsafe(),
                        tensor.getStridesUnsafe(),
                        tensor.getStorageOffsetUnsafe(),
                        NativeCpuStorageFamily.CPU_NATIVE
                ),
                storage
        );
    }

    private static boolean denseTensor(Tensor tensor) {
        return tensor != null && tensor.isContiguous() && !tensor.hasStorageOffset();
    }

    private static int expectedOutputSize(int[] shape, int dimension) {
        if (dimension == -1) {
            return 1;
        }
        int size = 1;
        for (int dim = 0; dim < shape.length; dim++) {
            if (dim != dimension) {
                size = Math.multiplyExact(size, shape[dim]);
            }
        }
        return size;
    }

    private static void requireFallbackAllowed(CpuKernelContext context, String family, String reason) {
        if (context.executionContext().runtimeConfig().nativeCpuFailurePolicy() == NativeCpuFailurePolicy.REQUIRE_NATIVE) {
            throw new IllegalStateException("Native CPU execution required but " + family + " fell back to Java: " + reason);
        }
    }

    private static void requireCpuReadableInputs(CpuKernelContext context) {
        for (int inputNodeId : context.inputNodeIds()) {
            context.executionContext().requireCpuReadable(inputNodeId, CpuMaterializationReason.CPU_CONSUMER);
        }
    }

    private static boolean nativeRequested(CpuKernelContext context) {
        return context != null
                && context.executionContext().runtimeConfig() != null
                && context.executionContext().runtimeConfig().cpuStorageProfile() == CpuStorageProfile.CPU_NATIVE;
    }

    private static boolean supportsNativeReductionDType(DataType dataType) {
        return dataType == DataType.FLOAT32
                || dataType == DataType.FLOAT64
                || dataType == DataType.BFLOAT16
                || dataType == DataType.BOOL;
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
