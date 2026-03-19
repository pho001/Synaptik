package Backend;

import Config.backend.CpuKernelConfig;
import Backend.kernels.cpu.CpuKernel;
import Backend.kernels.cpu.CpuExecutionConfig;
import Backend.registry.CpuKernelRegistry;
import Tensor.Tensor;
import Operations.Operation;

import java.util.List;

public class CPUBackend {
    private CpuExecutionConfig executionConfig = CpuExecutionConfig.defaults();


    public void execute(Operation op, List<Tensor> inputs,Tensor node) {
        if (op == null) {
            return;
        }
        CpuKernel kernel = node.getResolvedCpuKernel();
        if (kernel == null) {
            kernel = CpuKernelRegistry.resolve(op.opType());
        }
        if (kernel == null) {
            throw new UnsupportedOperationException(
                    "Missing CPU kernel for opType=" + op.opType() +
                            " (operation class: " + op.getClass().getName() + ")"
            );
        }
        kernel.forward(op, inputs, node, executionConfig);
    }

    public void setExecutionConfig(CpuExecutionConfig executionConfig) {
        if (executionConfig == null) {
            throw new IllegalArgumentException("executionConfig cannot be null");
        }
        this.executionConfig = executionConfig;
    }

    public CpuExecutionConfig getExecutionConfig() {
        return executionConfig;
    }

    public void setKernelConfig(CpuKernelConfig cpuKernelConfig) {
        if (cpuKernelConfig == null) {
            throw new IllegalArgumentException("cpuKernelConfig cannot be null");
        }
        this.executionConfig = CpuExecutionConfig.fromKernelConfig(cpuKernelConfig);
    }


}
