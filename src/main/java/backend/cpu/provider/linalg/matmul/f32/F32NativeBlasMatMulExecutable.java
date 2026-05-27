package backend.cpu.provider.linalg.matmul.f32;

import tensor.TensorInternalAccess;

import backend.blas.OpenBlasRuntime;
import backend.blas.OpenBlasSegmentGemm;
import backend.cpu.execution.CpuKernelContext;
import backend.cpu.kernels.linalg.matmul.f32.F32MatMulJavaBackend;
import backend.cpu.provider.linalg.matmul.PreparedMatMulExecutable;
import backend.cpu.plan.linalg.matmul.MatMulExecutionRoute;
import backend.cpu.plan.linalg.matmul.ResolvedMatMulHints;
import backend.memory.CpuMaterializationReason;
import config.runtime.NativeCpuFailurePolicy;
import tensor.storage.NativeFloat32Storage;
import tensor.storage.NativeTensorStorage;
import tensor.Tensor;

import java.util.Arrays;

public final class F32NativeBlasMatMulExecutable implements PreparedMatMulExecutable {
    private final ResolvedMatMulHints hints;
    private MatMulExecutionRoute lastRoute = MatMulExecutionRoute.OPENBLAS_NATIVE_SEGMENT;
    private long lastCopyInBytes;
    private long lastCopyOutBytes;
    private String lastFallbackReason = "";

    public F32NativeBlasMatMulExecutable(ResolvedMatMulHints hints) {
        this.hints = hints;
    }

    @Override
    public boolean acceptsNativeInputs() {
        return true;
    }

    @Override
    public void execute(Tensor a, Tensor b, Tensor node, CpuKernelContext context) {
        resetTrace();
        int[] as = a.getShapeUnsafe();
        int[] bs = b.getShapeUnsafe();
        int m = as[as.length - 2];
        int k = as[as.length - 1];
        int n = bs[bs.length - 1];
        if (!OpenBlasRuntime.isAvailable()) {
            fallbackToJava(a, b, node, context, "OpenBLAS FFM unavailable: " + OpenBlasRuntime.unavailableReason());
            return;
        }
        if (as.length != 2 || bs.length != 2) {
            fallbackToJava(a, b, node, context, "native OpenBLAS segment route supports only rank-2 matmul in this wave");
            return;
        }
        try {
            int aNodeId = context.inputNodeIds().get(0);
            int bNodeId = context.inputNodeIds().get(1);
            boolean aAlreadyNative = context.executionContext().residencyForNodeId(aNodeId).nativeCurrent();
            boolean bAlreadyNative = context.executionContext().residencyForNodeId(bNodeId).nativeCurrent();
            NativeTensorStorage aStorage = context.executionContext().requireNativeReadable(
                    aNodeId,
                    CpuMaterializationReason.CPU_CONSUMER
            );
            NativeTensorStorage bStorage = context.executionContext().requireNativeReadable(
                    bNodeId,
                    CpuMaterializationReason.CPU_CONSUMER
            );
            lastCopyInBytes = (aAlreadyNative ? 0L : logicalByteLength(a))
                    + (bAlreadyNative ? 0L : logicalByteLength(b));
            if (!(aStorage instanceof NativeFloat32Storage) || !(bStorage instanceof NativeFloat32Storage)) {
                fallbackToJava(a, b, node, context, "native OpenBLAS segment route requires FLOAT32 native input storage");
                return;
            }
            NativeTensorStorage outStorage = context.executionContext().allocateNativeStorage(
                    node.getDataType(),
                    node.getFlatDataSize(),
                    "node-" + context.nodeId() + ":" + node.getLabel() + ":openblas-f32"
            );
            OpenBlasSegmentGemm.sgemmRowMajorNoTransSegment(
                    m,
                    n,
                    k,
                    1.0f,
                    aStorage.segment(),
                    0L,
                    k,
                    bStorage.segment(),
                    0L,
                    n,
                    0.0f,
                    outStorage.segment(),
                    0L,
                    n
            );
            outStorage.markModified();
            context.executionContext().attachNativeStorage(context.nodeId(), outStorage, "openblas native segment wrote FLOAT32 output");
            lastRoute = MatMulExecutionRoute.OPENBLAS_NATIVE_SEGMENT;
        } catch (Throwable t) {
            fallbackToJava(a, b, node, context, "native OpenBLAS segment SGEMM failed: " + safeMessage(t));
        }
    }

    @Override
    public MatMulExecutionRoute lastExecutionRoute() {
        return lastRoute;
    }

    @Override
    public long lastCopyInBytes() {
        return lastCopyInBytes;
    }

    @Override
    public long lastCopyOutBytes() {
        return lastCopyOutBytes;
    }

    @Override
    public String lastFallbackReason() {
        return lastFallbackReason;
    }

    private void fallbackToJava(Tensor a, Tensor b, Tensor node, CpuKernelContext context, String reason) {
        lastRoute = MatMulExecutionRoute.JAVA_DIRECT;
        lastFallbackReason = reason == null ? "" : reason;
        lastCopyInBytes = -1L;
        lastCopyOutBytes = -1L;
        if (requiresNative(context)) {
            throw new IllegalStateException("Native CPU execution required but FLOAT32 matmul fell back to Java: " + lastFallbackReason);
        }
        requireCpuReadableInputs(context);
        float[] out = TensorInternalAccess.float32Data(node);
        Arrays.fill(out, 0.0f);
        F32MatMulJavaBackend.run(
                TensorInternalAccess.float32Data(a),
                a.getShapeUnsafe(),
                TensorInternalAccess.float32Data(b),
                b.getShapeUnsafe(),
                out,
                node.getShapeUnsafe(),
                hints
        );
    }

    private static void requireCpuReadableInputs(CpuKernelContext context) {
        for (int inputNodeId : context.inputNodeIds()) {
            context.executionContext().requireCpuReadable(inputNodeId, CpuMaterializationReason.CPU_CONSUMER);
        }
    }

    private static boolean requiresNative(CpuKernelContext context) {
        return context != null
                && context.executionContext().runtimeConfig() != null
                && context.executionContext().runtimeConfig().nativeCpuFailurePolicy() == NativeCpuFailurePolicy.REQUIRE_NATIVE;
    }

    private static long logicalByteLength(Tensor tensor) {
        return Math.multiplyExact((long) tensor.getFlatDataSize(), Float.BYTES);
    }

    private void resetTrace() {
        lastRoute = MatMulExecutionRoute.OPENBLAS_NATIVE_SEGMENT;
        lastCopyInBytes = 0L;
        lastCopyOutBytes = 0L;
        lastFallbackReason = "";
    }

    private static String safeMessage(Throwable t) {
        String message = t.getMessage();
        return message == null || message.isBlank() ? t.getClass().getSimpleName() : message;
    }
}
