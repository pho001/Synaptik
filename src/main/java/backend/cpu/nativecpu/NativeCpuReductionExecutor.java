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
import tensor.NativeTensorStorage;
import tensor.Tensor;

/**
 * Native CPU reduction slice for dense all-axis F32 reductions.
 */
public final class NativeCpuReductionExecutor {
    private NativeCpuReductionExecutor() {
    }

    public static boolean acceptsNativeInputs(Operation op, DataType dataType, CpuNodeExecutionPlan plan, RuntimeConfig runtimeConfig) {
        if (!nativeRequested(runtimeConfig) || op == null || dataType != DataType.FLOAT32 || plan == null || plan.stridedPath()) {
            return false;
        }
        Operation.OpType opType = op.opType();
        return (opType == Operation.OpType.SUM || opType == Operation.OpType.MEAN) && reductionDimension(op) == -1;
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
        if (node.getDataType() != DataType.FLOAT32 || input.getDataType() != DataType.FLOAT32) {
            return fallback(context, fact, opType, "native-storage-dtype-unsupported:" + node.getDataType().name().toLowerCase());
        }
        if (context.nodePlan().stridedPath()) {
            return fallback(context, fact, opType, "native-kernel-ineligible:" + opLabel(opType) + "-strided");
        }
        if (dimension != -1) {
            return fallback(context, fact, opType, "native-kernel-ineligible:" + opLabel(opType) + "-axis");
        }
        if (node.getFlatDataSize() != 1 || input.getFlatDataSize() <= 0) {
            return fallback(context, fact, opType, "native-kernel-ineligible:" + opLabel(opType) + "-shape");
        }
        try {
            NativeFloat32Storage in = requireF32NativeInput(context, opLabel(opType).toUpperCase());
            NativeFloat32Storage out = allocateF32(node, context, opLabel(opType));
            float value = reduceAllF32(opType, in, input.getFlatDataSize());
            out.setFloat32At(0, value);
            context.executionContext().attachNativeStorage(context.nodeId(), out, "native CPU " + opLabel(opType).toUpperCase() + " wrote FLOAT32 output");
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

    private static NativeFloat32Storage allocateF32(Tensor node, CpuKernelContext context, String label) {
        return (NativeFloat32Storage) new NativeCpuStorageFactory().allocate(
                DataType.FLOAT32,
                node.getFlatDataSize(),
                "node-" + context.nodeId() + ":" + node.getLabel() + ":native-f32-" + label
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
        return context != null && nativeRequested(context.executionContext().runtimeConfig());
    }

    private static boolean nativeRequested(RuntimeConfig runtimeConfig) {
        return runtimeConfig != null && runtimeConfig.cpuStorageProfile() == CpuStorageProfile.CPU_NATIVE;
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
