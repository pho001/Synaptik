package backend.cpu.kernels.linalg;

import backend.cpu.kernels.CpuKernel;
import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.CpuKernelCostClass;
import backend.cpu.kernels.linalg.matmul.exec.PreparedMatMulExecutable;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

public class CpuMatMulKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        runPrepared(inputs, node, context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        runPrepared(inputs, node, context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        runPrepared(inputs, node, context);
    }

    @Override
    public CpuKernelCostClass costClass(Operation op) {
        return CpuKernelCostClass.HIGH;
    }

    private static void runPrepared(List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        PreparedMatMulExecutable executable = context.matMulExecutable();
        if (executable == null) {
            throw new IllegalStateException("Missing PreparedMatMulExecutable for matmul execution.");
        }
        executable.execute(inputs.get(0), inputs.get(1), node, context);
    }
}
