package backend.cpu.kernels.elementwise;

import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.storage.CpuStorageView;
import backend.memory.CpuMaterializationReason;
import config.runtime.CpuStorageProfile;
import config.runtime.NativeCpuFailurePolicy;
import tensor.DataType;
import tensor.Tensor;
import tensor.storage.NativeTensorStorage;

public final class ElementwiseNativeSupport {
    private ElementwiseNativeSupport() {
    }

    public static boolean nativeRequested(CpuKernelContext context) {
        return context != null
                && context.executionContext().runtimeConfig() != null
                && context.executionContext().runtimeConfig().cpuStorageProfile() == CpuStorageProfile.CPU_NATIVE;
    }

    public static void requireFallbackAllowed(CpuKernelContext context, String family, String reason) {
        if (context.executionContext().runtimeConfig().nativeCpuFailurePolicy() == NativeCpuFailurePolicy.REQUIRE_NATIVE) {
            throw new IllegalStateException("Native CPU execution required but " + family + " fell back to Java: " + reason);
        }
    }

    public static void requireCpuReadableInputs(CpuKernelContext context) {
        for (int inputNodeId : context.inputNodeIds()) {
            context.executionContext().requireCpuReadable(inputNodeId, CpuMaterializationReason.CPU_CONSUMER);
        }
    }

    public static NativeTensorStorage requireNativeInput(CpuKernelContext context, int inputIndex, DataType dtype, String op) {
        NativeTensorStorage storage = context.executionContext().requireNativeReadable(
                context.inputNodeIds().get(inputIndex),
                CpuMaterializationReason.CPU_CONSUMER
        );
        if (storage.getType() != dtype) {
            throw new IllegalStateException(op + " native input dtype mismatch. expected=" + dtype + ", actual=" + storage.getType());
        }
        return storage;
    }

    public static NativeTensorStorage allocateNativeOutput(Tensor node, CpuKernelContext context, String label) {
        return context.executionContext().allocateNativeStorage(
                node.getDataType(),
                node.getFlatDataSize(),
                "node-" + context.nodeId() + ":" + node.getLabel() + ":" + label
        );
    }

    public static CpuStorageView segmentView(Tensor tensor, NativeTensorStorage storage) {
        return CpuStorageView.segment(
                tensor.getDataType(),
                storage.segment(),
                tensor.getShapeUnsafe(),
                tensor.getStridesUnsafe(),
                tensor.getStorageOffsetUnsafe(),
                tensor.getFlatDataSize()
        );
    }

    public static boolean isDenseView(Tensor tensor) {
        return tensor.isContiguous() && tensor.getStorageOffsetUnsafe() == 0;
    }

    public static boolean isDenseView(CpuStorageView view) {
        if (view.storageOffset() != 0) {
            return false;
        }
        int[] shape = view.shape();
        int[] strides = view.strides();
        int expected = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            if (strides[i] != expected) {
                return false;
            }
            expected = Math.multiplyExact(expected, shape[i]);
        }
        return true;
    }

    public static String safeMessage(Throwable t) {
        String message = t.getMessage();
        return message == null || message.isBlank() ? t.getClass().getSimpleName() : message;
    }
}
