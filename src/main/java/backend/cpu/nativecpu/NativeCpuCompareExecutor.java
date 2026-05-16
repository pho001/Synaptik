package backend.cpu.nativecpu;

import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.CpuNodeExecutionPlan;
import backend.cpu.kernels.elementwise.compare.CompareElementwiseKernel;
import backend.cpu.kernels.layout.plan.ResolvedBroadcastPlan;
import backend.memory.CpuMaterializationReason;
import config.runtime.CpuStorageProfile;
import config.runtime.NativeCpuFailurePolicy;
import config.runtime.RuntimeConfig;
import operations.Operation;
import tensor.DataType;
import tensor.NativeFloat32Storage;
import tensor.NativeFloat64Storage;
import tensor.NativeTensorStorage;
import tensor.Tensor;

import java.util.List;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;

/**
 * Native CPU compare slice for dense contiguous F32/F64 inputs with BOOL array output.
 */
public final class NativeCpuCompareExecutor {
    private NativeCpuCompareExecutor() {
    }

    public static boolean acceptsNativeInputs(Operation op, DataType dataType, CpuNodeExecutionPlan plan, RuntimeConfig runtimeConfig) {
        if (!nativeRequested(runtimeConfig) || op == null || dataType != DataType.BOOL || plan == null || plan.stridedPath()) {
            return false;
        }
        if (!isCompareOp(op.opType())) {
            return false;
        }
        ResolvedBroadcastPlan broadcastPlan = plan.broadcastPlan();
        return broadcastPlan == null || broadcastPlan.isNoBroadcast();
    }

    public static boolean tryRunCompare(CompareElementwiseKernel kernel, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!nativeRequested(context)) {
            return false;
        }
        Operation op = context.executionOperation();
        NativeCpuKernelFact fact = NativeCpuKernelFacts.factFor(opType(op), node.getDataType());
        if (op == null || !isCompareOp(op.opType())) {
            return fallback(context, fact, "native-kernel-unsupported:compare");
        }
        if (node.getDataType() != DataType.BOOL) {
            return fallback(context, fact, "native-kernel-ineligible:" + opLabel(op) + "-output-dtype");
        }
        if (inputs == null || inputs.size() != 2) {
            return fallback(context, fact, "native-kernel-ineligible:" + opLabel(op) + "-input-count");
        }
        Tensor left = inputs.get(0);
        Tensor right = inputs.get(1);
        if (!supportsInputDType(left.getDataType()) || left.getDataType() != right.getDataType()) {
            return fallback(context, fact, "native-storage-dtype-unsupported:" + left.getDataType().name().toLowerCase());
        }
        ResolvedBroadcastPlan broadcastPlan = context.broadcastPlan();
        if (broadcastPlan != null && !broadcastPlan.isNoBroadcast()) {
            return fallback(context, fact, "native-kernel-ineligible:" + opLabel(op) + "-broadcast");
        }
        if (context.nodePlan().stridedPath() || !left.isContiguous() || !right.isContiguous()
                || left.hasStorageOffset() || right.hasStorageOffset()) {
            return fallback(context, fact, "native-kernel-ineligible:" + opLabel(op) + "-strided");
        }
        if (left.getFlatDataSize() != right.getFlatDataSize() || left.getFlatDataSize() != node.getFlatDataSize()) {
            return fallback(context, fact, "native-kernel-ineligible:" + opLabel(op) + "-shape");
        }
        try {
            byte[] out = node.getBoolData();
            if (left.getDataType() == DataType.FLOAT64) {
                runCompareF64(kernel, requireF64NativeInput(context, 0, opLabel(op).toUpperCase()),
                        requireF64NativeInput(context, 1, opLabel(op).toUpperCase()), out, node.getFlatDataSize());
            } else {
                runCompareF32(kernel, requireF32NativeInput(context, 0, opLabel(op).toUpperCase()),
                        requireF32NativeInput(context, 1, opLabel(op).toUpperCase()), out, node.getFlatDataSize());
            }
            node.markStorageModified();
            publishTrace(context, fact, "CPU_ARRAY", "");
            return true;
        } catch (Throwable t) {
            return fallback(context, fact, "native-kernel-failed:" + opLabel(op) + ":" + safeMessage(t));
        }
    }

    private static void runCompareF32(
            CompareElementwiseKernel kernel,
            NativeFloat32Storage left,
            NativeFloat32Storage right,
            byte[] out,
            int size
    ) {
        for (int i = 0; i < size; i++) {
            long offset = (long) i * Float.BYTES;
            out[i] = kernel.testF32(left.segment().get(JAVA_FLOAT, offset), right.segment().get(JAVA_FLOAT, offset))
                    ? (byte) 1
                    : (byte) 0;
        }
    }

    private static void runCompareF64(
            CompareElementwiseKernel kernel,
            NativeFloat64Storage left,
            NativeFloat64Storage right,
            byte[] out,
            int size
    ) {
        for (int i = 0; i < size; i++) {
            long offset = (long) i * Double.BYTES;
            out[i] = kernel.testF64(left.segment().get(JAVA_DOUBLE, offset), right.segment().get(JAVA_DOUBLE, offset))
                    ? (byte) 1
                    : (byte) 0;
        }
    }

    private static boolean fallback(CpuKernelContext context, NativeCpuKernelFact fact, String reason) {
        handleRequireNative(context, "compare elementwise", reason);
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

    private static NativeFloat32Storage requireF32NativeInput(CpuKernelContext context, int inputIndex, String op) {
        int inputNodeId = context.inputNodeIds().get(inputIndex);
        NativeTensorStorage storage = context.executionContext().requireNativeReadable(inputNodeId, CpuMaterializationReason.CPU_CONSUMER);
        if (storage instanceof NativeFloat32Storage f32) {
            return f32;
        }
        throw new IllegalStateException("native " + op + " compare requires FLOAT32 native input storage");
    }

    private static NativeFloat64Storage requireF64NativeInput(CpuKernelContext context, int inputIndex, String op) {
        int inputNodeId = context.inputNodeIds().get(inputIndex);
        NativeTensorStorage storage = context.executionContext().requireNativeReadable(inputNodeId, CpuMaterializationReason.CPU_CONSUMER);
        if (storage instanceof NativeFloat64Storage f64) {
            return f64;
        }
        throw new IllegalStateException("native " + op + " compare requires FLOAT64 native input storage");
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

    private static boolean supportsInputDType(DataType dataType) {
        return dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64;
    }

    private static boolean isCompareOp(Operation.OpType opType) {
        return opType == Operation.OpType.GT
                || opType == Operation.OpType.GE
                || opType == Operation.OpType.LT
                || opType == Operation.OpType.LE
                || opType == Operation.OpType.EQ
                || opType == Operation.OpType.NE;
    }

    private static boolean nativeRequested(CpuKernelContext context) {
        return context != null && nativeRequested(context.executionContext().runtimeConfig());
    }

    private static boolean nativeRequested(RuntimeConfig runtimeConfig) {
        return runtimeConfig != null && runtimeConfig.cpuStorageProfile() == CpuStorageProfile.CPU_NATIVE;
    }

    private static Operation.OpType opType(Operation op) {
        return op == null ? Operation.OpType.UNKNOWN : op.opType();
    }

    private static String opLabel(Operation op) {
        return opType(op).name().toLowerCase();
    }

    private static String safeMessage(Throwable t) {
        String message = t.getMessage();
        return message == null || message.isBlank() ? t.getClass().getSimpleName() : message;
    }
}
