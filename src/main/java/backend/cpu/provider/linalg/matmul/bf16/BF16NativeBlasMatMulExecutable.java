package backend.cpu.provider.linalg.matmul.bf16;

import tensor.TensorInternalAccess;

import backend.provider.blas.openblas.OpenBlasRuntime;
import backend.provider.blas.openblas.OpenBlasSegmentGemm;
import backend.cpu.execution.CpuKernelContext;
import backend.cpu.kernels.linalg.matmul.bf16.BF16MatMulJavaBackend;
import backend.cpu.provider.linalg.matmul.PreparedMatMulExecutable;
import backend.cpu.plan.linalg.matmul.MatMulExecutionRoute;
import backend.cpu.plan.linalg.matmul.ResolvedMatMulHints;
import runtime.contract.CpuMaterializationReason;
import tensor.storage.NativeBFloat16Storage;
import tensor.storage.NativeTensorStorage;
import tensor.Tensor;

import java.util.OptionalInt;

public final class BF16NativeBlasMatMulExecutable implements PreparedMatMulExecutable {
    private static final int NO_THREAD_RESTORE = -1;
    private static final short BF16_ONE = tensor.dtype.TensorDTypeOps.toBFloat16Bits(1.0f);
    private static final short BF16_ZERO = tensor.dtype.TensorDTypeOps.toBFloat16Bits(0.0f);

    private final ResolvedMatMulHints hints;
    private MatMulExecutionRoute lastRoute = MatMulExecutionRoute.OPENBLAS_NATIVE_SEGMENT;
    private long lastCopyInBytes;
    private long lastCopyOutBytes;
    private String lastFallbackReason = "";
    private String lastBlasSymbol = "";

    public BF16NativeBlasMatMulExecutable(ResolvedMatMulHints hints) {
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
        if (as.length != 2 || bs.length != 2) {
            fallbackToJava(a, b, node, context, "native OpenBLAS BF16 segment route supports only rank-2 matmul in this wave");
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
            if (!(aStorage instanceof NativeBFloat16Storage) || !(bStorage instanceof NativeBFloat16Storage)) {
                fallbackToJava(a, b, node, context, "native OpenBLAS segment route requires BFLOAT16 native input storage");
                return;
            }
            NativeTensorStorage outStorage = context.executionContext().allocateNativeStorage(
                    node.getDataType(),
                    node.getFlatDataSize(),
                    "node-" + context.nodeId() + ":" + node.getLabel() + ":openblas-bf16"
            );
            int previousThreads = applyPreparedThreads();
            try {
                OpenBlasSegmentGemm.bgemmRowMajorNoTransSegment(
                        m, n, k, BF16_ONE, aStorage.segment(), 0L, k,
                        bStorage.segment(), 0L, n, BF16_ZERO, outStorage.segment(), 0L, n
                );
            } finally {
                restoreThreads(previousThreads);
            }
            outStorage.markModified();
            context.executionContext().attachNativeStorage(context.nodeId(), outStorage, "openblas native segment wrote BFLOAT16 output");
            lastRoute = MatMulExecutionRoute.OPENBLAS_NATIVE_SEGMENT;
            lastBlasSymbol = "cblas_bgemm";
        } catch (Throwable t) {
            fallbackToJava(a, b, node, context, "native OpenBLAS segment BGEMM failed: " + safeMessage(t));
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

    @Override
    public String lastBlasSymbol() {
        return lastBlasSymbol;
    }

    private void fallbackToJava(Tensor a, Tensor b, Tensor node, CpuKernelContext context, String reason) {
        lastRoute = MatMulExecutionRoute.JAVA_DIRECT;
        lastFallbackReason = reason == null ? "" : reason;
        lastCopyInBytes = -1L;
        lastCopyOutBytes = -1L;
        lastBlasSymbol = "";
        if (requiresNative()) {
            throw new IllegalStateException("Native CPU execution required but BFLOAT16 matmul fell back to Java: " + lastFallbackReason);
        }
        requireCpuReadableInputs(context);
        BF16MatMulJavaBackend.run(
                TensorInternalAccess.bfloat16Data(a),
                a.getShapeUnsafe(),
                TensorInternalAccess.bfloat16Data(b),
                b.getShapeUnsafe(),
                TensorInternalAccess.bfloat16Data(node),
                node.getShapeUnsafe(),
                hints
        );
    }

    private static void requireCpuReadableInputs(CpuKernelContext context) {
        for (int inputNodeId : context.inputNodeIds()) {
            context.executionContext().requireCpuReadable(inputNodeId, CpuMaterializationReason.CPU_CONSUMER);
        }
    }

    private boolean requiresNative() {
        return "REQUIRE_NATIVE".equals(hints.nativeCpuFailurePolicy());
    }

    private static long logicalByteLength(Tensor tensor) {
        return Math.multiplyExact((long) tensor.getFlatDataSize(), Short.BYTES);
    }

    private int applyPreparedThreads() {
        int requestedThreads = hints.openBlasThreads();
        if (requestedThreads <= 0) {
            return NO_THREAD_RESTORE;
        }
        OptionalInt previousThreads = OpenBlasRuntime.getNumThreads();
        if (previousThreads.isEmpty()) {
            throw new IllegalStateException("OpenBLAS thread override requires openblas_get_num_threads.");
        }
        int previous = previousThreads.getAsInt();
        if (previous == requestedThreads) {
            return NO_THREAD_RESTORE;
        }
        if (!OpenBlasRuntime.setNumThreads(requestedThreads)) {
            throw new IllegalStateException("OpenBLAS thread override requires openblas_set_num_threads.");
        }
        return previous;
    }

    private static void restoreThreads(int previousThreads) {
        if (previousThreads > 0) {
            OpenBlasRuntime.setNumThreads(previousThreads);
        }
    }

    private void resetTrace() {
        lastRoute = MatMulExecutionRoute.OPENBLAS_NATIVE_SEGMENT;
        lastCopyInBytes = 0L;
        lastCopyOutBytes = 0L;
        lastFallbackReason = "";
        lastBlasSymbol = "";
    }

    private static String safeMessage(Throwable t) {
        String message = t.getMessage();
        return message == null || message.isBlank() ? t.getClass().getSimpleName() : message;
    }
}
