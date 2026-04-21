package backend.kernels.cpu.linalg.matmul.exec;

import backend.kernels.cpu.CpuKernelContext;
import tensor.Tensor;

public interface PreparedMatMulExecutable {
    void execute(Tensor a, Tensor b, Tensor node, CpuKernelContext context);
}
