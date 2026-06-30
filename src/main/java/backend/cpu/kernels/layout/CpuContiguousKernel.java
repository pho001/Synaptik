package backend.cpu.kernels.layout;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import backend.cpu.nativecpu.CpuNativeTraceSupport;
import runtime.contract.CpuMaterializationReason;
import config.runtime.CpuStorageProfile;
import config.runtime.NativeCpuFailurePolicy;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;
import tensor.layout.TensorRemap;
import tensor.storage.NativeTensorStorage;

import java.lang.foreign.MemorySegment;
import java.util.List;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

public class CpuContiguousKernel implements CpuStorageAwareKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        contiguous(call.operation(), call.inputTensors(), call.outputTensor(), call.context());
        return CpuKernelResult.completed();
    }

    private static void contiguous(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (tryRunNativeContiguous(op, inputs, node, context)) {
            return;
        }
        if (inputs == null || inputs.isEmpty()) {
            return;
        }
        TensorRemap.apply(inputs.getFirst(), node, context.contiguousMaterializeThreshold());
    }

    private static boolean tryRunNativeContiguous(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!nativeRequested(context)) {
            return false;
        }
        Operation.OpType opType = opType(op);
        String reason = nativeContiguousIneligibleReason(opType, inputs, node, context);
        if (!reason.isBlank()) {
            return fallbackNativeMicrokernel(context, "contiguous", reason);
        }
        try {
            Tensor input = inputs.getFirst();
            NativeTensorStorage inputStorage = requireNativeInput(context, 0, input.getDataType(), "CONTIGUOUS");
            NativeTensorStorage outputStorage = allocateNativeOutput(node, context, "native-contiguous");
            copyDense(inputStorage, outputStorage, node.getFlatDataSize());
            context.executionContext().attachNativeStorage(
                    context.nodeId(),
                    outputStorage,
                    "layout storage loop CONTIGUOUS wrote " + node.getDataType() + " native output"
            );
            CpuNativeTraceSupport.publishNativeMicrokernel(context, CpuNativeTraceSupport.CPU_NATIVE, "");
            return true;
        } catch (Throwable t) {
            return fallbackNativeMicrokernel(context, "contiguous",
                    "native-kernel-failed:contiguous:" + safeMessage(t));
        }
    }

    private static String nativeContiguousIneligibleReason(
            Operation.OpType opType,
            List<Tensor> inputs,
            Tensor node,
            CpuKernelContext context
    ) {
        if (context == null || context.nodePlan() == null) {
            return "native-kernel-ineligible:contiguous-plan";
        }
        if (opType != Operation.OpType.CONTIGUOUS) {
            return "native-kernel-unsupported:contiguous";
        }
        if (!supportsNativeCopyDType(node.getDataType())) {
            return "native-storage-dtype-unsupported:" + node.getDataType().name().toLowerCase();
        }
        if (inputs == null || inputs.size() != 1 || context.inputNodeIds().size() != 1) {
            return "native-kernel-ineligible:contiguous-input-count";
        }
        Tensor input = inputs.getFirst();
        if (input.getDataType() != node.getDataType()) {
            return "native-kernel-ineligible:contiguous-dtype";
        }
        if (input.getFlatDataSize() != node.getFlatDataSize()) {
            return "native-kernel-ineligible:contiguous-shape";
        }
        if (!denseTensor(input) || !denseTensor(node)) {
            return "native-kernel-ineligible:contiguous-strided";
        }
        return "";
    }

    private static void copyDense(NativeTensorStorage input, NativeTensorStorage output, int size) {
        if (input.getType() == DataType.FLOAT32 && output.getType() == DataType.FLOAT32) {
            MemorySegment.copy(input.segment(), JAVA_FLOAT, 0L, output.segment(), JAVA_FLOAT, 0L, size);
            output.markModified();
            return;
        }
        if (input.getType() == DataType.FLOAT64 && output.getType() == DataType.FLOAT64) {
            MemorySegment.copy(input.segment(), JAVA_DOUBLE, 0L, output.segment(), JAVA_DOUBLE, 0L, size);
            output.markModified();
            return;
        }
        if (input.getType() == DataType.BFLOAT16 && output.getType() == DataType.BFLOAT16) {
            MemorySegment.copy(input.segment(), JAVA_SHORT, 0L, output.segment(), JAVA_SHORT, 0L, size);
            output.markModified();
            return;
        }
        throw new IllegalStateException("native CONTIGUOUS storage dtype mismatch. input="
                + input.getType() + ", output=" + output.getType());
    }

    private static boolean fallbackNativeMicrokernel(CpuKernelContext context, String family, String reason) {
        requireFallbackAllowed(context, family, reason);
        requireCpuReadableInputs(context);
        CpuNativeTraceSupport.publishNativeMicrokernel(context, CpuNativeTraceSupport.CPU_ARRAY, reason);
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

    private static boolean supportsNativeCopyDType(DataType dataType) {
        return dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64 || dataType == DataType.BFLOAT16;
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
