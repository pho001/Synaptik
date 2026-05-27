package backend.cpu.kernels.layout;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import backend.cpu.nativecpu.CpuNativeTraceSupport;
import backend.memory.CpuMaterializationReason;
import config.runtime.CpuStorageProfile;
import config.runtime.NativeCpuFailurePolicy;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.dtype.TensorDTypeOps;
import tensor.storage.NativeTensorStorage;

import java.lang.foreign.MemorySegment;
import java.util.List;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

public final class CpuCastKernel implements CpuStorageAwareKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        cast(call.operation(), call.inputTensors(), call.outputTensor(), call.context());
        return CpuKernelResult.completed();
    }

    private static void cast(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException("cast expects exactly one input.");
        }
        if (tryRunNativeCast(op, inputs, node, context)) {
            return;
        }
        Tensor input = inputs.getFirst();
        int size = node.getFlatDataSize();
        if (input.getFlatDataSize() != size) {
            throw new IllegalArgumentException("cast requires input and output to have the same flat size.");
        }
        switch (node.getDataType()) {
            case FLOAT64 -> {
                double[] out = TensorInternalAccess.float64Data(node);
                for (int i = 0; i < size; i++) {
                    out[i] = input.getByFlatIndex(i);
                }
            }
            case FLOAT32 -> {
                float[] out = TensorInternalAccess.float32Data(node);
                for (int i = 0; i < size; i++) {
                    out[i] = (float) input.getByFlatIndex(i);
                }
            }
            case BFLOAT16 -> {
                short[] out = TensorInternalAccess.bfloat16Data(node);
                for (int i = 0; i < size; i++) {
                    out[i] = TensorDTypeOps.toBFloat16Bits((float) input.getByFlatIndex(i));
                }
            }
            case INT32 -> {
                int[] out = TensorInternalAccess.int32Data(node);
                for (int i = 0; i < size; i++) {
                    out[i] = (int) input.getByFlatIndex(i);
                }
            }
            case INT64 -> {
                long[] out = TensorInternalAccess.int64Data(node);
                for (int i = 0; i < size; i++) {
                    out[i] = (long) input.getByFlatIndex(i);
                }
            }
            case BOOL -> {
                byte[] out = TensorInternalAccess.boolData(node);
                for (int i = 0; i < size; i++) {
                    out[i] = input.getByFlatIndex(i) == 0.0d ? (byte) 0 : (byte) 1;
                }
            }
        }
        TensorInternalAccess.markStorageModified(node);
    }

    private static boolean tryRunNativeCast(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!nativeRequested(context)) {
            return false;
        }
        Operation.OpType opType = opType(op);
        String reason = nativeCastIneligibleReason(opType, inputs, node, context);
        if (!reason.isBlank()) {
            return fallbackSegmentScalar(context, "cast", reason);
        }
        try {
            Tensor input = inputs.getFirst();
            NativeTensorStorage inputStorage = requireNativeInput(context, 0, input.getDataType(), "CAST");
            NativeTensorStorage outputStorage = allocateNativeOutput(node, context, "native-cast");
            castDense(inputStorage, outputStorage, node.getFlatDataSize());
            context.executionContext().attachNativeStorage(
                    context.nodeId(),
                    outputStorage,
                    "layout storage loop CAST wrote " + node.getDataType() + " native output"
            );
            CpuNativeTraceSupport.publishSegmentScalar(context, CpuNativeTraceSupport.CPU_NATIVE, "");
            return true;
        } catch (Throwable t) {
            return fallbackSegmentScalar(context, "cast", "native-kernel-failed:cast:" + safeMessage(t));
        }
    }

    private static String nativeCastIneligibleReason(
            Operation.OpType opType,
            List<Tensor> inputs,
            Tensor node,
            CpuKernelContext context
    ) {
        if (context == null || context.nodePlan() == null) {
            return "native-kernel-ineligible:cast-plan";
        }
        if (opType != Operation.OpType.CAST) {
            return "native-kernel-unsupported:cast";
        }
        if (inputs == null || inputs.size() != 1 || context.inputNodeIds().size() != 1) {
            return "native-kernel-ineligible:cast-input-count";
        }
        Tensor input = inputs.getFirst();
        if (!supportedCast(input.getDataType(), node.getDataType())) {
            return "native-kernel-ineligible:cast-dtype";
        }
        if (context.nodePlan().stridedPath() || !denseTensor(input) || !denseTensor(node)) {
            return "native-kernel-ineligible:cast-strided";
        }
        if (input.getFlatDataSize() != node.getFlatDataSize()) {
            return "native-kernel-ineligible:cast-shape";
        }
        return "";
    }

    private static void castDense(NativeTensorStorage input, NativeTensorStorage output, int size) {
        if (input.getType() == DataType.FLOAT32 && output.getType() == DataType.BFLOAT16) {
            MemorySegment in = input.segment();
            MemorySegment out = output.segment();
            for (int i = 0; i < size; i++) {
                out.set(JAVA_SHORT, (long) i * Short.BYTES,
                        TensorDTypeOps.toBFloat16Bits(in.get(JAVA_FLOAT, (long) i * Float.BYTES)));
            }
            output.markModified();
            return;
        }
        if (input.getType() == DataType.BFLOAT16 && output.getType() == DataType.FLOAT32) {
            MemorySegment in = input.segment();
            MemorySegment out = output.segment();
            for (int i = 0; i < size; i++) {
                out.set(JAVA_FLOAT, (long) i * Float.BYTES,
                        TensorDTypeOps.fromBFloat16Bits(in.get(JAVA_SHORT, (long) i * Short.BYTES)));
            }
            output.markModified();
            return;
        }
        throw new IllegalStateException("native CAST storage dtype mismatch. input="
                + input.getType() + ", output=" + output.getType());
    }

    private static boolean fallbackSegmentScalar(CpuKernelContext context, String family, String reason) {
        requireFallbackAllowed(context, family, reason);
        requireCpuReadableInputs(context);
        CpuNativeTraceSupport.publishSegmentScalar(context, CpuNativeTraceSupport.CPU_ARRAY, reason);
        return false;
    }

    private static NativeTensorStorage requireNativeInput(CpuKernelContext context, int inputIndex, DataType dtype, String op) {
        NativeTensorStorage storage = context.executionContext().requireNativeReadable(
                context.inputNodeIds().get(inputIndex),
                CpuMaterializationReason.CPU_CONSUMER
        );
        if (storage.getType() != dtype) {
            throw new IllegalStateException("native " + op + " input dtype mismatch. expected="
                    + dtype + ", actual=" + storage.getType());
        }
        return storage;
    }

    private static NativeTensorStorage allocateNativeOutput(Tensor node, CpuKernelContext context, String label) {
        return context.executionContext().allocateNativeStorage(
                node.getDataType(),
                node.getFlatDataSize(),
                "node-" + context.nodeId() + ":" + node.getLabel() + ":" + label
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

    private static boolean supportedCast(DataType input, DataType output) {
        return input == DataType.FLOAT32 && output == DataType.BFLOAT16
                || input == DataType.BFLOAT16 && output == DataType.FLOAT32;
    }

    private static boolean denseTensor(Tensor tensor) {
        return tensor != null && tensor.isContiguous() && !tensor.hasStorageOffset();
    }

    private static boolean nativeRequested(CpuKernelContext context) {
        return context != null
                && context.executionContext().runtimeConfig() != null
                && context.executionContext().runtimeConfig().cpuStorageProfile() == CpuStorageProfile.CPU_NATIVE;
    }

    private static Operation.OpType opType(Operation op) {
        return op == null ? Operation.OpType.UNKNOWN : op.opType();
    }

    private static String safeMessage(Throwable t) {
        String message = t.getMessage();
        return message == null || message.isBlank() ? t.getClass().getSimpleName() : message;
    }
}
