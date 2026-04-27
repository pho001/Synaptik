package backend.cpu.kernels.linalg.matmul.f32;

import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.linalg.matmul.exec.PreparedMatMulExecutable;
import backend.cpu.kernels.linalg.matmul.plan.ResolvedMatMulHints;
import tensor.Tensor;

import java.util.Arrays;

public final class F32JavaMatMulExecutable implements PreparedMatMulExecutable {
    private final ResolvedMatMulHints hints;

    public F32JavaMatMulExecutable(ResolvedMatMulHints hints) {
        this.hints = hints;
    }

    @Override
    public void execute(Tensor a, Tensor b, Tensor node, CpuKernelContext context) {
        float[] out = node.getFloat32Data();
        Arrays.fill(out, 0.0f);
        F32MatMulJavaBackend.run(
                a.getFloat32Data(),
                a.getShapeUnsafe(),
                b.getFloat32Data(),
                b.getShapeUnsafe(),
                out,
                node.getShapeUnsafe(),
                hints
        );
    }
}
