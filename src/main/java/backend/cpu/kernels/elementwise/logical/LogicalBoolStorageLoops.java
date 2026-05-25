package backend.cpu.kernels.elementwise.logical;

import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.CpuNativeTraceSupport;
import backend.cpu.kernels.elementwise.ElementwiseLoops;
import backend.memory.CpuMaterializationReason;
import config.runtime.CpuStorageProfile;
import config.runtime.NativeCpuFailurePolicy;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;
import tensor.storage.NativeTensorStorage;

import java.lang.foreign.MemorySegment;
import java.util.List;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;

final class LogicalBoolStorageLoops {
    private LogicalBoolStorageLoops() {
    }

    static void executeBinary(
            LogicalBinaryElementwiseKernel kernel,
            List<Tensor> inputs,
            Tensor node,
            CpuKernelContext context
    ) {
        if (!nativeRequested(context)) {
            ElementwiseLoops.runLogicalBinary(kernel, inputs.get(0), inputs.get(1), node, context);
            return;
        }
        Operation op = context.executionOperation();
        Operation.OpType opType = opType(op);
        String reason = nativeBinaryIneligibleReason(opType, inputs, node, context);
        if (!reason.isBlank()) {
            fallbackBinary(kernel, inputs, node, context, reason);
            return;
        }
        try {
            NativeTensorStorage leftStorage = requireNativeInput(context, 0, opLabel(opType).toUpperCase());
            NativeTensorStorage rightStorage = requireNativeInput(context, 1, opLabel(opType).toUpperCase());
            NativeTensorStorage outputStorage = allocateNativeOutput(node, context, opLabel(opType));
            runBinaryDense(kernel, leftStorage.segment(), rightStorage.segment(), outputStorage.segment(), node.getFlatDataSize());
            outputStorage.markModified();
            context.executionContext().attachNativeStorage(
                    context.nodeId(),
                    outputStorage,
                    "logical storage loop " + opLabel(opType).toUpperCase() + " wrote BOOL native output"
            );
            CpuNativeTraceSupport.publishSegmentScalar(context, CpuNativeTraceSupport.CPU_NATIVE, "");
        } catch (Throwable t) {
            fallbackBinary(kernel, inputs, node, context,
                    "native-kernel-failed:" + opLabel(opType) + ":" + safeMessage(t));
        }
    }

    static void executeUnary(
            LogicalUnaryElementwiseKernel kernel,
            List<Tensor> inputs,
            Tensor node,
            CpuKernelContext context
    ) {
        if (!nativeRequested(context)) {
            ElementwiseLoops.runLogicalUnary(kernel, inputs.get(0), node, context);
            return;
        }
        Operation op = context.executionOperation();
        Operation.OpType opType = opType(op);
        String reason = nativeUnaryIneligibleReason(opType, inputs, node, context);
        if (!reason.isBlank()) {
            fallbackUnary(kernel, inputs, node, context, reason);
            return;
        }
        try {
            NativeTensorStorage inputStorage = requireNativeInput(context, 0, "LOGICAL_NOT");
            NativeTensorStorage outputStorage = allocateNativeOutput(node, context, "logical_not");
            runUnaryDense(kernel, inputStorage.segment(), outputStorage.segment(), node.getFlatDataSize());
            outputStorage.markModified();
            context.executionContext().attachNativeStorage(
                    context.nodeId(),
                    outputStorage,
                    "logical storage loop LOGICAL_NOT wrote BOOL native output"
            );
            CpuNativeTraceSupport.publishSegmentScalar(context, CpuNativeTraceSupport.CPU_NATIVE, "");
        } catch (Throwable t) {
            fallbackUnary(kernel, inputs, node, context,
                    "native-kernel-failed:logical_not:" + safeMessage(t));
        }
    }

    private static void runBinaryDense(
            LogicalBinaryElementwiseKernel kernel,
            MemorySegment left,
            MemorySegment right,
            MemorySegment output,
            int size
    ) {
        for (int i = 0; i < size; i++) {
            long offset = i;
            output.set(JAVA_BYTE, offset, kernel.apply(left.get(JAVA_BYTE, offset), right.get(JAVA_BYTE, offset)));
        }
    }

    private static void runUnaryDense(
            LogicalUnaryElementwiseKernel kernel,
            MemorySegment input,
            MemorySegment output,
            int size
    ) {
        for (int i = 0; i < size; i++) {
            output.set(JAVA_BYTE, i, kernel.apply(input.get(JAVA_BYTE, i)));
        }
    }

    private static void fallbackBinary(
            LogicalBinaryElementwiseKernel kernel,
            List<Tensor> inputs,
            Tensor node,
            CpuKernelContext context,
            String reason
    ) {
        requireFallbackAllowed(context, "BOOL logical binary", reason);
        requireCpuReadableInputs(context);
        CpuNativeTraceSupport.publishSegmentScalar(context, CpuNativeTraceSupport.CPU_ARRAY, reason);
        ElementwiseLoops.runLogicalBinary(kernel, inputs.get(0), inputs.get(1), node, context);
    }

    private static void fallbackUnary(
            LogicalUnaryElementwiseKernel kernel,
            List<Tensor> inputs,
            Tensor node,
            CpuKernelContext context,
            String reason
    ) {
        requireFallbackAllowed(context, "BOOL logical unary", reason);
        requireCpuReadableInputs(context);
        CpuNativeTraceSupport.publishSegmentScalar(context, CpuNativeTraceSupport.CPU_ARRAY, reason);
        ElementwiseLoops.runLogicalUnary(kernel, inputs.get(0), node, context);
    }

    private static String nativeBinaryIneligibleReason(
            Operation.OpType opType,
            List<Tensor> inputs,
            Tensor node,
            CpuKernelContext context
    ) {
        String label = opLabel(opType);
        if (context == null || context.nodePlan() == null) {
            return "native-kernel-ineligible:" + label + "-plan";
        }
        if (!isLogicalBinary(opType)) {
            return "native-kernel-unsupported:" + label;
        }
        if (inputs == null || inputs.size() != 2 || context.inputNodeIds().size() != 2) {
            return "native-kernel-ineligible:" + label + "-input-count";
        }
        if (inputs.get(0).getDataType() != DataType.BOOL
                || inputs.get(1).getDataType() != DataType.BOOL
                || node == null
                || node.getDataType() != DataType.BOOL) {
            return "native-storage-dtype-unsupported:bool";
        }
        if (context.broadcastPlan() != null && !context.broadcastPlan().isNoBroadcast()) {
            return "native-kernel-ineligible:" + label + "-broadcast";
        }
        if (context.nodePlan().stridedPath()
                || !denseTensor(inputs.get(0))
                || !denseTensor(inputs.get(1))
                || !denseTensor(node)) {
            return "native-kernel-ineligible:" + label + "-strided";
        }
        if (inputs.get(0).getFlatDataSize() != node.getFlatDataSize()
                || inputs.get(1).getFlatDataSize() != node.getFlatDataSize()) {
            return "native-kernel-ineligible:" + label + "-shape";
        }
        return "";
    }

    private static String nativeUnaryIneligibleReason(
            Operation.OpType opType,
            List<Tensor> inputs,
            Tensor node,
            CpuKernelContext context
    ) {
        if (context == null || context.nodePlan() == null) {
            return "native-kernel-ineligible:logical_not-plan";
        }
        if (opType != Operation.OpType.LOGICAL_NOT) {
            return "native-kernel-unsupported:" + opLabel(opType);
        }
        if (inputs == null || inputs.size() != 1 || context.inputNodeIds().size() != 1) {
            return "native-kernel-ineligible:logical_not-input-count";
        }
        if (inputs.getFirst().getDataType() != DataType.BOOL || node == null || node.getDataType() != DataType.BOOL) {
            return "native-storage-dtype-unsupported:bool";
        }
        if (context.nodePlan().stridedPath() || !denseTensor(inputs.getFirst()) || !denseTensor(node)) {
            return "native-kernel-ineligible:logical_not-strided";
        }
        if (inputs.getFirst().getFlatDataSize() != node.getFlatDataSize()) {
            return "native-kernel-ineligible:logical_not-shape";
        }
        return "";
    }

    private static NativeTensorStorage requireNativeInput(CpuKernelContext context, int inputIndex, String op) {
        NativeTensorStorage storage = context.executionContext().requireNativeReadable(
                context.inputNodeIds().get(inputIndex),
                CpuMaterializationReason.CPU_CONSUMER
        );
        if (storage.getType() != DataType.BOOL) {
            throw new IllegalStateException("native " + op + " requires BOOL native input storage");
        }
        return storage;
    }

    private static NativeTensorStorage allocateNativeOutput(Tensor node, CpuKernelContext context, String label) {
        return context.executionContext().allocateNativeStorage(
                DataType.BOOL,
                node.getFlatDataSize(),
                "node-" + context.nodeId() + ":" + node.getLabel() + ":native-bool-" + label
        );
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

    private static boolean isLogicalBinary(Operation.OpType opType) {
        return opType == Operation.OpType.LOGICAL_AND || opType == Operation.OpType.LOGICAL_OR;
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
