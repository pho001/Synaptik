package backend.cpu.kernels.linalg.matmul.exec;

import backend.cpu.kernels.CpuKernelContext;
import tensor.Tensor;

public interface PreparedMatMulExecutable {
    void execute(Tensor a, Tensor b, Tensor node, CpuKernelContext context);
}
