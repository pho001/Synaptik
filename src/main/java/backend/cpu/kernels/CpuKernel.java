package backend.cpu.kernels;

import backend.cpu.plan.CpuKernelCostClass;
import operations.Operation;

public interface CpuKernel {
    CpuKernelResult execute(CpuKernelCall call);

    default CpuKernelCostClass costClass(Operation op) {
        return CpuKernelCostClass.MEDIUM;
    }
}
