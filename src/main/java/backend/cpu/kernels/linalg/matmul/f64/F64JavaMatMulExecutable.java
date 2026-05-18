package backend.cpu.kernels.linalg.matmul.f64;

import tensor.TensorInternalAccess;

import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.linalg.matmul.exec.PreparedMatMulExecutable;
import backend.cpu.kernels.linalg.matmul.plan.ResolvedMatMulHints;
import tensor.Tensor;

import java.util.Arrays;

public final class F64JavaMatMulExecutable implements PreparedMatMulExecutable {
    private final ResolvedMatMulHints hints;

    public F64JavaMatMulExecutable(ResolvedMatMulHints hints) {
        this.hints = hints;
    }

    @Override
    public void execute(Tensor a, Tensor b, Tensor node, CpuKernelContext context) {
        double[] out = TensorInternalAccess.float64Data(node);
        Arrays.fill(out, 0.0d);
        F64MatMulJavaBackend.run(
                TensorInternalAccess.float64Data(a),
                a.getShapeUnsafe(),
                TensorInternalAccess.float64Data(b),
                b.getShapeUnsafe(),
                out,
                node.getShapeUnsafe(),
                hints
        );
    }
}
