package backend.cpu.nativecpu;

import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.CpuNodeExecutionPlan;
import backend.memory.CpuMaterializationReason;
import config.runtime.CpuStorageProfile;
import config.runtime.NativeCpuFailurePolicy;
import config.runtime.RuntimeConfig;
import operations.Operation;
import tensor.DataType;
import tensor.storage.NativeBFloat16Storage;
import tensor.storage.NativeFloat32Storage;
import tensor.storage.NativeFloat64Storage;
import tensor.storage.NativeTensorStorage;
import tensor.Tensor;

import java.lang.foreign.MemorySegment;
import java.util.List;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;

/**
 * Native CPU dense copy slice for CONTIGUOUS materialization.
 */
public final class NativeCpuContiguousExecutor {
    private NativeCpuContiguousExecutor() {
    }

    public static boolean acceptsNativeInputs(Operation op, DataType dataType, CpuNodeExecutionPlan plan, RuntimeConfig runtimeConfig) {
        return nativeRequested(runtimeConfig)
                && op != null
                && op.opType() == Operation.OpType.CONTIGUOUS
                && supportsNativeCopyDType(dataType)
                && plan != null;
    }

    public static boolean tryRunContiguous(List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!nativeRequested(context)) {
            return false;
        }
        Operation op = context.executionOperation();
        NativeCpuKernelFact fact = NativeCpuKernelFacts.factFor(opType(op), node.getDataType());
        if (op == null || op.opType() != Operation.OpType.CONTIGUOUS) {
            return fallback(context, fact, "native-kernel-unsupported:contiguous");
        }
        if (!supportsNativeCopyDType(node.getDataType())) {
            return fallback(context, fact, "native-storage-dtype-unsupported:" + node.getDataType().name().toLowerCase());
        }
        if (inputs == null || inputs.size() != 1 || context.inputNodeIds().size() != 1) {
            return fallback(context, fact, "native-kernel-ineligible:contiguous-input-count");
        }
        Tensor input = inputs.getFirst();
        if (input.getDataType() != node.getDataType()) {
            return fallback(context, fact, "native-kernel-ineligible:contiguous-dtype");
        }
        if (input.getFlatDataSize() != node.getFlatDataSize()) {
            return fallback(context, fact, "native-kernel-ineligible:contiguous-shape");
        }
        if (!input.isContiguous() || input.hasStorageOffset()) {
            return fallback(context, fact, "native-kernel-ineligible:contiguous-strided");
        }
        try {
            NativeTensorStorage in = context.executionContext()
                    .requireNativeReadable(context.inputNodeIds().getFirst(), CpuMaterializationReason.CPU_CONSUMER);
            NativeTensorStorage out = allocateNativeOutput(node, context);
            copy(in, out, node.getFlatDataSize());
            context.executionContext().attachNativeStorage(context.nodeId(), out, "native CPU CONTIGUOUS wrote " + node.getDataType() + " output");
            publishTrace(context, fact, "CPU_NATIVE", "");
            return true;
        } catch (Throwable t) {
            return fallback(context, fact, "native-kernel-failed:contiguous:" + safeMessage(t));
        }
    }

    private static void copy(NativeTensorStorage input, NativeTensorStorage output, int size) {
        if (input instanceof NativeFloat32Storage && output instanceof NativeFloat32Storage) {
            MemorySegment.copy(input.segment(), JAVA_FLOAT, 0L, output.segment(), JAVA_FLOAT, 0L, size);
            output.markModified();
            return;
        }
        if (input instanceof NativeFloat64Storage && output instanceof NativeFloat64Storage) {
            MemorySegment.copy(input.segment(), JAVA_DOUBLE, 0L, output.segment(), JAVA_DOUBLE, 0L, size);
            output.markModified();
            return;
        }
        if (input instanceof NativeBFloat16Storage bf16Input && output instanceof NativeBFloat16Storage bf16Output) {
            NativeBFloat16Kernels.copy(bf16Input, bf16Output, size);
            return;
        }
        throw new IllegalStateException("native CONTIGUOUS storage dtype mismatch. input="
                + input.getType() + ", output=" + output.getType());
    }

    private static boolean fallback(CpuKernelContext context, NativeCpuKernelFact fact, String reason) {
        handleRequireNative(context, "contiguous", reason);
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

    private static NativeTensorStorage allocateNativeOutput(Tensor node, CpuKernelContext context) {
        return context.executionContext().allocateNativeStorage(
                node.getDataType(),
                node.getFlatDataSize(),
                "node-" + context.nodeId() + ":" + node.getLabel() + ":native-contiguous"
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

    private static boolean supportsNativeCopyDType(DataType dataType) {
        return dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64 || dataType == DataType.BFLOAT16;
    }

    private static boolean nativeRequested(CpuKernelContext context) {
        return NativeCpuRuntimePolicy.nativeRequested(context);
    }

    private static boolean nativeRequested(RuntimeConfig runtimeConfig) {
        return runtimeConfig != null && runtimeConfig.cpuStorageProfile() == CpuStorageProfile.CPU_NATIVE;
    }

    private static Operation.OpType opType(Operation op) {
        return op == null ? Operation.OpType.UNKNOWN : op.opType();
    }

    private static String safeMessage(Throwable t) {
        String message = t.getMessage();
        return message == null || message.isBlank() ? t.getClass().getSimpleName() : message;
    }
}
