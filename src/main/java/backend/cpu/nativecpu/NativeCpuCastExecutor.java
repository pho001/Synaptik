package backend.cpu.nativecpu;

import backend.cpu.kernels.CpuDTypeOps;
import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.CpuNodeExecutionPlan;
import backend.memory.CpuMaterializationReason;
import config.runtime.CpuStorageProfile;
import config.runtime.NativeCpuFailurePolicy;
import config.runtime.RuntimeConfig;
import operations.Operation;
import operations.dtype.cast;
import tensor.DataType;
import tensor.NativeBFloat16Storage;
import tensor.NativeFloat32Storage;
import tensor.NativeTensorStorage;
import tensor.Tensor;

import java.util.List;

/**
 * Native CPU cast slice for BF16/F32 conversion.
 */
public final class NativeCpuCastExecutor {
    private NativeCpuCastExecutor() {
    }

    public static boolean acceptsNativeInputs(Operation op, DataType dataType, CpuNodeExecutionPlan plan, RuntimeConfig runtimeConfig) {
        if (!nativeRequested(runtimeConfig) || op == null || op.opType() != Operation.OpType.CAST
                || plan == null || plan.stridedPath()) {
            return false;
        }
        if (op instanceof cast castOp && castOp.getTargetType() != dataType) {
            return false;
        }
        return dataType == DataType.FLOAT32 || dataType == DataType.BFLOAT16;
    }

    public static boolean tryRunCast(List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!nativeRequested(context)) {
            return false;
        }
        Operation op = context.executionOperation();
        NativeCpuKernelFact fact = NativeCpuKernelFacts.factFor(opType(op), node.getDataType());
        if (op == null || op.opType() != Operation.OpType.CAST) {
            return fallback(context, fact, "native-kernel-unsupported:cast");
        }
        if (inputs == null || inputs.size() != 1) {
            return fallback(context, fact, "native-kernel-ineligible:cast-input-count");
        }
        Tensor input = inputs.getFirst();
        if (!supportedCast(input.getDataType(), node.getDataType())) {
            return fallback(context, fact, "native-kernel-ineligible:cast-dtype");
        }
        if (context.nodePlan().stridedPath() || !input.isContiguous() || input.hasStorageOffset()) {
            return fallback(context, fact, "native-kernel-ineligible:cast-strided");
        }
        if (input.getFlatDataSize() != node.getFlatDataSize()) {
            return fallback(context, fact, "native-kernel-ineligible:cast-shape");
        }
        try {
            if (input.getDataType() == DataType.FLOAT32) {
                NativeFloat32Storage in = requireF32NativeInput(context);
                NativeTensorStorage out = allocateNativeOutput(node, context);
                runF32ToBF16(in, (NativeBFloat16Storage) out, node.getFlatDataSize());
                context.executionContext().attachNativeStorage(context.nodeId(), out, "native CPU CAST wrote " + node.getDataType() + " output");
            } else {
                NativeBFloat16Storage in = requireBF16NativeInput(context);
                NativeTensorStorage out = allocateNativeOutput(node, context);
                runBF16ToF32(in, (NativeFloat32Storage) out, node.getFlatDataSize());
                context.executionContext().attachNativeStorage(context.nodeId(), out, "native CPU CAST wrote " + node.getDataType() + " output");
            }
            publishTrace(context, fact, "CPU_NATIVE", "");
            return true;
        } catch (Throwable t) {
            return fallback(context, fact, "native-kernel-failed:cast:" + safeMessage(t));
        }
    }

    private static boolean supportedCast(DataType input, DataType output) {
        return input == DataType.FLOAT32 && output == DataType.BFLOAT16
                || input == DataType.BFLOAT16 && output == DataType.FLOAT32;
    }

    private static void runF32ToBF16(NativeFloat32Storage input, NativeBFloat16Storage out, int size) {
        for (int i = 0; i < size; i++) {
            out.setBFloat16BitsAt(i, CpuDTypeOps.toBFloat16Bits(input.getFloat32At(i)));
        }
    }

    private static void runBF16ToF32(NativeBFloat16Storage input, NativeFloat32Storage out, int size) {
        for (int i = 0; i < size; i++) {
            out.setFloat32At(i, CpuDTypeOps.fromBFloat16Bits(input.getBFloat16BitsAt(i)));
        }
    }

    private static boolean fallback(CpuKernelContext context, NativeCpuKernelFact fact, String reason) {
        handleRequireNative(context, "cast", reason);
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

    private static NativeFloat32Storage requireF32NativeInput(CpuKernelContext context) {
        NativeTensorStorage storage = context.executionContext()
                .requireNativeReadable(context.inputNodeIds().getFirst(), CpuMaterializationReason.CPU_CONSUMER);
        if (storage instanceof NativeFloat32Storage f32) {
            return f32;
        }
        throw new IllegalStateException("native CAST requires FLOAT32 native input storage");
    }

    private static NativeBFloat16Storage requireBF16NativeInput(CpuKernelContext context) {
        NativeTensorStorage storage = context.executionContext()
                .requireNativeReadable(context.inputNodeIds().getFirst(), CpuMaterializationReason.CPU_CONSUMER);
        if (storage instanceof NativeBFloat16Storage bf16) {
            return bf16;
        }
        throw new IllegalStateException("native CAST requires BFLOAT16 native input storage");
    }

    private static NativeTensorStorage allocateNativeOutput(Tensor node, CpuKernelContext context) {
        return new NativeCpuStorageFactory().allocate(
                node.getDataType(),
                node.getFlatDataSize(),
                "node-" + context.nodeId() + ":" + node.getLabel() + ":native-cast"
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

    private static Operation.OpType opType(Operation op) {
        return op == null ? Operation.OpType.UNKNOWN : op.opType();
    }

    private static String safeMessage(Throwable t) {
        String message = t.getMessage();
        return message == null || message.isBlank() ? t.getClass().getSimpleName() : message;
    }
}
