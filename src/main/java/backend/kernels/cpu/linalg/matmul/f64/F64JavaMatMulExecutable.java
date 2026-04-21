package backend.kernels.cpu.linalg.matmul.f64;

import backend.kernels.cpu.CpuKernelContext;
import backend.kernels.cpu.linalg.matmul.exec.PreparedMatMulExecutable;
import backend.kernels.cpu.linalg.matmul.plan.ResolvedMatMulHints;
import tensor.Tensor;

import java.util.Arrays;

public final class F64JavaMatMulExecutable implements PreparedMatMulExecutable {
    private final ResolvedMatMulHints hints;

    public F64JavaMatMulExecutable(ResolvedMatMulHints hints) {
        this.hints = hints;
    }

    @Override
    public void execute(Tensor a, Tensor b, Tensor node, CpuKernelContext context) {
        double[] out = node.getFloat64Data();
        Arrays.fill(out, 0.0d);
        F64MatMulJavaBackend.run(
                a.getFloat64Data(),
                a.getShapeUnsafe(),
                b.getFloat64Data(),
                b.getShapeUnsafe(),
                out,
                node.getShapeUnsafe(),
                hints
        );
    }
}
