package backend.cpu.kernels.linalg.matmul.f32;

import tensor.TensorInternalAccess;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.provider.linalg.matmul.PreparedMatMulExecutable;
import backend.cpu.plan.linalg.matmul.ResolvedMatMulHints;
import tensor.Tensor;

import java.util.Arrays;

public final class F32JavaMatMulExecutable implements PreparedMatMulExecutable {
    private final ResolvedMatMulHints hints;

    public F32JavaMatMulExecutable(ResolvedMatMulHints hints) {
        this.hints = hints;
    }

    @Override
    public void execute(Tensor a, Tensor b, Tensor node, CpuKernelContext context) {
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
}
