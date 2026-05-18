package backend.cpu.kernels.linalg.matmul.bf16;

import tensor.TensorInternalAccess;

import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.CpuNodeWorkspace;
import backend.cpu.kernels.linalg.matmul.blas.MatMulBlasBackend;
import backend.cpu.kernels.linalg.matmul.common.PackedLinearWeightCache;
import backend.cpu.kernels.linalg.matmul.exec.PreparedMatMulExecutable;
import backend.cpu.kernels.linalg.matmul.plan.ResolvedMatMulHints;
import tensor.Tensor;

abstract class AbstractBF16MatMulExecutable implements PreparedMatMulExecutable {
    protected final ResolvedMatMulHints hints;
    private final boolean publishFloatContinuation;
    private String lastBlasSymbol = "";

    protected AbstractBF16MatMulExecutable(ResolvedMatMulHints hints, boolean publishFloatContinuation) {
        this.hints = hints;
        this.publishFloatContinuation = publishFloatContinuation;
    }

    @Override
    public final void execute(Tensor a, Tensor b, Tensor node, CpuKernelContext context) {
        lastBlasSymbol = "";
        int[] as = a.getShapeUnsafe();
        int[] bs = b.getShapeUnsafe();
        short[] ad = TensorInternalAccess.bfloat16Data(a);
        short[] bd = TensorInternalAccess.bfloat16Data(b);
        float[] leftContinuation = context.inputFloatContinuation(0, a.getFlatDataSize());
        float[] rightContinuation = context.inputFloatContinuation(1, b.getFlatDataSize());
        if (publishFloatContinuation && tryExecuteToFloat(a, b, node, context, as, bs, ad, bd, leftContinuation, rightContinuation)) {
            return;
        }
        executeToBFloat16(a, b, node, context, as, bs, ad, bd, leftContinuation, rightContinuation);
    }

    private boolean tryExecuteToFloat(
            Tensor a,
            Tensor b,
            Tensor node,
            CpuKernelContext context,
            int[] as,
            int[] bs,
            short[] ad,
            short[] bd,
            float[] leftContinuation,
            float[] rightContinuation
    ) {
        if (allowPackedAndContinuationFastPaths()) {
            if (rightContinuation == null && tryPackedToFloat(a, b, node, context, leftContinuation)) {
                return true;
            }
            if (tryContinuationToFloat(node, context, as, bs, bd, leftContinuation, rightContinuation)) {
                return true;
            }
        }
        if (tryBackendToFloat(node, context, as, bs, ad, bd)) {
            return true;
        }
        CpuNodeWorkspace workspace = context.cpuWorkspace();
        if (workspace == null) {
            return false;
        }
        float[] continuation = workspace.requireFloatWorkspace();
        BF16MatMulJavaBackend.runToFloat(ad, as, bd, bs, continuation, node.getShapeUnsafe(), hints);
        workspace.publishFloatContinuation(node.getFlatDataSize());
        return true;
    }

    private void executeToBFloat16(
            Tensor a,
            Tensor b,
            Tensor node,
            CpuKernelContext context,
            int[] as,
            int[] bs,
            short[] ad,
            short[] bd,
            float[] leftContinuation,
            float[] rightContinuation
    ) {
        if (allowPackedAndContinuationFastPaths()) {
            if (rightContinuation == null && tryPackedToBFloat16(a, b, node, context, leftContinuation)) {
                return;
            }
            if (tryContinuationToBFloat16(node, as, bs, bd, leftContinuation, rightContinuation)) {
                return;
            }
        }
        if (tryBackendToBFloat16(node, context, as, bs, ad, bd)) {
            return;
        }
        BF16MatMulJavaBackend.run(ad, as, bd, bs, TensorInternalAccess.bfloat16Data(node), node.getShapeUnsafe(), hints);
    }

    private boolean tryPackedToFloat(
            Tensor a,
            Tensor b,
            Tensor node,
            CpuKernelContext context,
            float[] leftContinuation
    ) {
        CpuNodeWorkspace workspace = context.cpuWorkspace();
        if (workspace == null || workspace.packedLinearWeightCache() == null) {
            return false;
        }
        if (b.getShapeUnsafe().length != 2 || !b.isContiguous()) {
            return false;
        }
        PackedLinearWeightCache.BF16PackedWeights packed = workspace.packedLinearWeightCache().requireBF16(b, hints);
        if (packed == null) {
            return false;
        }
        float[] out = workspace.requireFloatWorkspace();
        if (leftContinuation != null) {
            backend.cpu.kernels.linalg.matmul.f32.F32MatMulJavaBackend.runPacked(leftContinuation, a.getShapeUnsafe(), packed, out, node.getShapeUnsafe(), hints);
        } else {
            BF16MatMulJavaBackend.runPackedToFloat(TensorInternalAccess.bfloat16Data(a), a.getShapeUnsafe(), packed, out, node.getShapeUnsafe(), hints);
        }
        workspace.publishFloatContinuation(node.getFlatDataSize());
        return true;
    }

    private boolean tryContinuationToFloat(
            Tensor node,
            CpuKernelContext context,
            int[] as,
            int[] bs,
            short[] bd,
            float[] leftContinuation,
            float[] rightContinuation
    ) {
        CpuNodeWorkspace workspace = context.cpuWorkspace();
        if (workspace == null) {
            return false;
        }
        float[] out = workspace.requireFloatWorkspace();
        if (leftContinuation != null && rightContinuation != null) {
            backend.cpu.kernels.linalg.matmul.f32.F32MatMulJavaBackend.run(leftContinuation, as, rightContinuation, bs, out, node.getShapeUnsafe(), hints);
            workspace.publishFloatContinuation(node.getFlatDataSize());
            return true;
        }
        if (leftContinuation != null) {
            BF16MatMulJavaBackend.runF32LeftBF16RightToFloat(leftContinuation, as, bd, bs, out, node.getShapeUnsafe(), hints);
            workspace.publishFloatContinuation(node.getFlatDataSize());
            return true;
        }
        return false;
    }

    private boolean tryPackedToBFloat16(
            Tensor a,
            Tensor b,
            Tensor node,
            CpuKernelContext context,
            float[] leftContinuation
    ) {
        CpuNodeWorkspace workspace = context.cpuWorkspace();
        if (workspace == null || workspace.packedLinearWeightCache() == null) {
            return false;
        }
        if (b.getShapeUnsafe().length != 2 || !b.isContiguous()) {
            return false;
        }
        PackedLinearWeightCache.BF16PackedWeights packed = workspace.packedLinearWeightCache().requireBF16(b, hints);
        if (packed == null) {
            return false;
        }
        if (leftContinuation != null) {
            BF16MatMulJavaBackend.runPackedF32ToBF16(leftContinuation, a.getShapeUnsafe(), packed, TensorInternalAccess.bfloat16Data(node), node.getShapeUnsafe(), hints);
            return true;
        }
        BF16MatMulJavaBackend.runPacked(TensorInternalAccess.bfloat16Data(a), a.getShapeUnsafe(), packed, TensorInternalAccess.bfloat16Data(node), node.getShapeUnsafe(), hints);
        return true;
    }

    private boolean tryContinuationToBFloat16(
            Tensor node,
            int[] as,
            int[] bs,
            short[] bd,
            float[] leftContinuation,
            float[] rightContinuation
    ) {
        if (leftContinuation != null && rightContinuation != null) {
            BF16MatMulJavaBackend.runF32ToBF16(leftContinuation, as, rightContinuation, bs, TensorInternalAccess.bfloat16Data(node), node.getShapeUnsafe(), hints);
            return true;
        }
        if (leftContinuation != null) {
            BF16MatMulJavaBackend.runF32LeftBF16RightToBF16(leftContinuation, as, bd, bs, TensorInternalAccess.bfloat16Data(node), node.getShapeUnsafe(), hints);
            return true;
        }
        return false;
    }

    protected abstract boolean allowPackedAndContinuationFastPaths();

    @Override
    public final String lastBlasSymbol() {
        return lastBlasSymbol;
    }

    protected final void recordBlasSymbol(String symbol) {
        lastBlasSymbol = symbol == null ? "" : symbol;
    }

    protected abstract boolean tryBackendToFloat(
            Tensor node,
            CpuKernelContext context,
            int[] as,
            int[] bs,
            short[] ad,
            short[] bd
    );

    protected abstract boolean tryBackendToBFloat16(
            Tensor node,
            CpuKernelContext context,
            int[] as,
            int[] bs,
            short[] ad,
            short[] bd
    );
}
