package backend.cpu.nativecpu;

import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.CpuNodeExecutionPlan;
import backend.memory.CpuMaterializationReason;
import config.runtime.CpuStorageProfile;
import config.runtime.NativeCpuFailurePolicy;
import config.runtime.RuntimeConfig;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

/**
 * Native CPU view-only slice for shape-only aliases over dense native storage.
 */
public final class NativeCpuViewExecutor {
    private NativeCpuViewExecutor() {
    }

    public static boolean acceptsNativeInputs(Operation op, DataType dataType, CpuNodeExecutionPlan plan, RuntimeConfig runtimeConfig) {
        return nativeRequested(runtimeConfig)
                && op != null
                && supportsViewOp(op.opType())
                && supportsViewDType(dataType)
                && plan != null;
    }

    public static boolean tryRunView(List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!nativeRequested(context)) {
            return false;
        }
        Operation op = context.executionOperation();
        if (op == null || !supportsViewOp(op.opType())) {
            return false;
        }
        NativeCpuKernelFact fact = NativeCpuKernelFacts.factFor(op.opType(), node.getDataType());
        if (!supportsViewDType(node.getDataType())) {
            return false;
        }
        if (inputs == null || inputs.size() != 1 || context.inputNodeIds().size() != 1) {
            return fallback(context, fact, op, "native-kernel-ineligible:" + opLabel(op) + "-input-count");
        }
        Tensor input = inputs.getFirst();
        if (input.getDataType() != node.getDataType()) {
            return fallback(context, fact, op, "native-kernel-ineligible:" + opLabel(op) + "-dtype");
        }
        if (input.getFlatDataSize() != node.getFlatDataSize()) {
            return fallback(context, fact, op, "native-kernel-ineligible:" + opLabel(op) + "-shape");
        }
        try {
            int sourceNodeId = context.inputNodeIds().getFirst();
            context.executionContext().aliasNativeStorage(
                    context.nodeId(),
                    sourceNodeId,
                    "native CPU " + opLabel(op).toUpperCase() + " view aliases node-" + sourceNodeId
            );
            publishTrace(context, fact, "CPU_NATIVE", "");
            return true;
        } catch (Throwable t) {
            return fallback(context, fact, op, "native-kernel-failed:" + opLabel(op) + ":" + safeMessage(t));
        }
    }

    private static boolean fallback(CpuKernelContext context, NativeCpuKernelFact fact, Operation op, String reason) {
        handleRequireNative(context, "view-only layout", reason);
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

    private static boolean supportsViewDType(DataType dataType) {
        return dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64 || dataType == DataType.BFLOAT16;
    }

    private static boolean supportsViewOp(Operation.OpType opType) {
        return opType == Operation.OpType.NOOP
                || opType == Operation.OpType.RESHAPE
                || opType == Operation.OpType.EXPAND_DIMS
                || opType == Operation.OpType.SQUEEZE;
    }

    private static boolean nativeRequested(CpuKernelContext context) {
        return NativeCpuRuntimePolicy.nativeRequested(context);
    }

    private static boolean nativeRequested(RuntimeConfig runtimeConfig) {
        return runtimeConfig != null && runtimeConfig.cpuStorageProfile() == CpuStorageProfile.CPU_NATIVE;
    }

    private static String opLabel(Operation op) {
        return op == null ? "unknown" : op.opType().name().toLowerCase();
    }

    private static String safeMessage(Throwable t) {
        String message = t.getMessage();
        return message == null || message.isBlank() ? t.getClass().getSimpleName() : message;
    }
}
