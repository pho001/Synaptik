package backend.cpu.kernels.layout;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.nativecpu.CpuNativeTraceSupport;
import backend.memory.CpuMaterializationReason;
import config.runtime.CpuStorageProfile;
import config.runtime.NativeCpuFailurePolicy;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

final class CpuLayoutNativeViewSupport {
    private CpuLayoutNativeViewSupport() {
    }

    static boolean tryRunNativeView(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!nativeRequested(context)) {
            return false;
        }
        Operation.OpType opType = opType(op);
        if (!supportsViewOp(opType) || !supportsViewDType(node.getDataType())) {
            return false;
        }
        if (inputs == null || inputs.size() != 1 || context.inputNodeIds().size() != 1) {
            return fallbackViewOnly(context, "view-only layout",
                    "native-kernel-ineligible:" + opLabel(opType) + "-input-count");
        }
        Tensor input = inputs.getFirst();
        if (input.getDataType() != node.getDataType()) {
            return fallbackViewOnly(context, "view-only layout",
                    "native-kernel-ineligible:" + opLabel(opType) + "-dtype");
        }
        if (opType == Operation.OpType.RESHAPE && !input.isContiguous()) {
            return fallbackViewOnly(context, "view-only layout", "native-kernel-ineligible:reshape-strided");
        }
        try {
            int sourceNodeId = context.inputNodeIds().getFirst();
            context.executionContext().aliasNativeStorage(
                    context.nodeId(),
                    sourceNodeId,
                    "layout storage loop " + opLabel(opType).toUpperCase() + " view aliases node-" + sourceNodeId
            );
            CpuNativeTraceSupport.publishViewOnly(context, CpuNativeTraceSupport.CPU_NATIVE, "");
            return true;
        } catch (Throwable t) {
            return fallbackViewOnly(context, "view-only layout",
                    "native-kernel-failed:" + opLabel(opType) + ":" + safeMessage(t));
        }
    }

    private static boolean fallbackViewOnly(CpuKernelContext context, String family, String reason) {
        requireFallbackAllowed(context, family, reason);
        requireCpuReadableInputs(context);
        CpuNativeTraceSupport.publishViewOnly(context, CpuNativeTraceSupport.CPU_ARRAY, reason);
        return false;
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

    private static boolean supportsViewDType(DataType dataType) {
        return dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64 || dataType == DataType.BFLOAT16;
    }

    private static boolean supportsViewOp(Operation.OpType opType) {
        return opType == Operation.OpType.NOOP
                || opType == Operation.OpType.RESHAPE
                || opType == Operation.OpType.PERMUTE
                || opType == Operation.OpType.EXPAND
                || opType == Operation.OpType.SELECT
                || opType == Operation.OpType.SLICE
                || opType == Operation.OpType.EXPAND_DIMS
                || opType == Operation.OpType.SQUEEZE;
    }

    private static boolean nativeRequested(CpuKernelContext context) {
        return context != null
                && context.executionContext().runtimeConfig() != null
                && context.executionContext().runtimeConfig().cpuStorageProfile() == CpuStorageProfile.CPU_NATIVE;
    }

    private static Operation.OpType opType(Operation op) {
        return op == null ? Operation.OpType.UNKNOWN : op.opType();
    }

    private static String opLabel(Operation.OpType opType) {
        return opType == null ? "unknown" : opType.name().toLowerCase();
    }

    private static String safeMessage(Throwable t) {
        String message = t.getMessage();
        return message == null || message.isBlank() ? t.getClass().getSimpleName() : message;
    }
}
