package backend.cpu.kernels.elementwise.compare;

import backend.cpu.kernels.CpuDTypeOps;
import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.CpuNativeTraceSupport;
import backend.cpu.kernels.elementwise.ElementwiseLoops;
import backend.memory.CpuMaterializationReason;
import config.runtime.CpuStorageProfile;
import config.runtime.NativeCpuFailurePolicy;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.storage.NativeTensorStorage;

import java.lang.foreign.MemorySegment;
import java.util.List;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

final class CompareStorageLoops {
    private CompareStorageLoops() {
    }

    static void execute(CompareElementwiseKernel kernel, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!nativeRequested(context)) {
            ElementwiseLoops.runCompare(kernel, inputs.get(0), inputs.get(1), node, context);
            return;
        }
        Operation op = context.executionOperation();
        Operation.OpType opType = opType(op);
        String reason = nativeIneligibleReason(opType, inputs, node, context);
        if (!reason.isBlank()) {
            fallbackToArray(kernel, inputs, node, context, reason);
            return;
        }
        try {
            Tensor left = inputs.get(0);
            Tensor right = inputs.get(1);
            NativeTensorStorage leftStorage = requireNativeInput(context, 0, left.getDataType(), opLabel(opType).toUpperCase());
            NativeTensorStorage rightStorage = requireNativeInput(context, 1, left.getDataType(), opLabel(opType).toUpperCase());
            byte[] out = TensorInternalAccess.boolData(node);
            runDense(kernel, left.getDataType(), leftStorage.segment(), rightStorage.segment(), out, node.getFlatDataSize());
            TensorInternalAccess.markStorageModified(node);
            CpuNativeTraceSupport.publishSegmentScalar(context, CpuNativeTraceSupport.CPU_ARRAY, "");
        } catch (Throwable t) {
            fallbackToArray(kernel, inputs, node, context,
                    "native-kernel-failed:" + opLabel(opType) + ":" + safeMessage(t));
        }
    }

    private static void runDense(
            CompareElementwiseKernel kernel,
            DataType dtype,
            MemorySegment left,
            MemorySegment right,
            byte[] output,
            int size
    ) {
        switch (dtype) {
            case FLOAT32 -> runDenseF32(kernel, left, right, output, size);
            case FLOAT64 -> runDenseF64(kernel, left, right, output, size);
            case BFLOAT16 -> runDenseBF16(kernel, left, right, output, size);
            case BOOL, INT32, INT64 -> throw new UnsupportedOperationException("Compare storage loop does not support dtype: " + dtype);
        }
    }

    private static void runDenseF32(
            CompareElementwiseKernel kernel,
            MemorySegment left,
            MemorySegment right,
            byte[] output,
            int size
    ) {
        for (int i = 0; i < size; i++) {
            long offset = (long) i * Float.BYTES;
            output[i] = kernel.testF32(left.get(JAVA_FLOAT, offset), right.get(JAVA_FLOAT, offset))
                    ? (byte) 1
                    : (byte) 0;
        }
    }

    private static void runDenseF64(
            CompareElementwiseKernel kernel,
            MemorySegment left,
            MemorySegment right,
            byte[] output,
            int size
    ) {
        for (int i = 0; i < size; i++) {
            long offset = (long) i * Double.BYTES;
            output[i] = kernel.testF64(left.get(JAVA_DOUBLE, offset), right.get(JAVA_DOUBLE, offset))
                    ? (byte) 1
                    : (byte) 0;
        }
    }

    private static void runDenseBF16(
            CompareElementwiseKernel kernel,
            MemorySegment left,
            MemorySegment right,
            byte[] output,
            int size
    ) {
        for (int i = 0; i < size; i++) {
            long offset = (long) i * Short.BYTES;
            float leftValue = CpuDTypeOps.fromBFloat16Bits(left.get(JAVA_SHORT, offset));
            float rightValue = CpuDTypeOps.fromBFloat16Bits(right.get(JAVA_SHORT, offset));
            output[i] = kernel.testBF16(leftValue, rightValue) ? (byte) 1 : (byte) 0;
        }
    }

    private static void fallbackToArray(
            CompareElementwiseKernel kernel,
            List<Tensor> inputs,
            Tensor node,
            CpuKernelContext context,
            String reason
    ) {
        requireFallbackAllowed(context, "compare elementwise", reason);
        requireCpuReadableInputs(context);
        CpuNativeTraceSupport.publishSegmentScalar(context, CpuNativeTraceSupport.CPU_ARRAY, reason);
        ElementwiseLoops.runCompare(kernel, inputs.get(0), inputs.get(1), node, context);
    }

    private static String nativeIneligibleReason(
            Operation.OpType opType,
            List<Tensor> inputs,
            Tensor node,
            CpuKernelContext context
    ) {
        String label = opLabel(opType);
        if (context == null || context.nodePlan() == null) {
            return "native-kernel-ineligible:" + label + "-plan";
        }
        if (!isCompareOp(opType)) {
            return "native-kernel-unsupported:compare";
        }
        if (node == null || node.getDataType() != DataType.BOOL) {
            return "native-kernel-ineligible:" + label + "-output-dtype";
        }
        if (inputs == null || inputs.size() != 2 || context.inputNodeIds().size() != 2) {
            return "native-kernel-ineligible:" + label + "-input-count";
        }
        Tensor left = inputs.get(0);
        Tensor right = inputs.get(1);
        if (!supportsInputDType(left.getDataType()) || left.getDataType() != right.getDataType()) {
            return "native-storage-dtype-unsupported:" + left.getDataType().name().toLowerCase();
        }
        if (context.broadcastPlan() != null && !context.broadcastPlan().isNoBroadcast()) {
            return "native-kernel-ineligible:" + label + "-broadcast";
        }
        if (context.nodePlan().stridedPath()
                || !denseTensor(left)
                || !denseTensor(right)
                || !denseTensor(node)) {
            return "native-kernel-ineligible:" + label + "-strided";
        }
        if (left.getFlatDataSize() != right.getFlatDataSize() || left.getFlatDataSize() != node.getFlatDataSize()) {
            return "native-kernel-ineligible:" + label + "-shape";
        }
        return "";
    }

    private static NativeTensorStorage requireNativeInput(CpuKernelContext context, int inputIndex, DataType dtype, String op) {
        NativeTensorStorage storage = context.executionContext().requireNativeReadable(
                context.inputNodeIds().get(inputIndex),
                CpuMaterializationReason.CPU_CONSUMER
        );
        if (storage.getType() != dtype) {
            throw new IllegalStateException("native " + op + " compare requires " + dtype + " native input storage");
        }
        return storage;
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

    private static boolean denseTensor(Tensor tensor) {
        return tensor != null && tensor.isContiguous() && !tensor.hasStorageOffset();
    }

    private static boolean supportsInputDType(DataType dataType) {
        return dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64 || dataType == DataType.BFLOAT16;
    }

    private static boolean isCompareOp(Operation.OpType opType) {
        return opType == Operation.OpType.GT
                || opType == Operation.OpType.GE
                || opType == Operation.OpType.LT
                || opType == Operation.OpType.LE
                || opType == Operation.OpType.EQ
                || opType == Operation.OpType.NE;
    }

    private static Operation.OpType opType(Operation op) {
        return op == null ? Operation.OpType.UNKNOWN : op.opType();
    }

    private static String opLabel(Operation.OpType opType) {
        return opType == null ? Operation.OpType.UNKNOWN.name().toLowerCase() : opType.name().toLowerCase();
    }

    private static String safeMessage(Throwable t) {
        String message = t.getMessage();
        return message == null || message.isBlank() ? t.getClass().getSimpleName() : message;
    }
}
