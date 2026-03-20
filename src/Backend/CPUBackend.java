package Backend;

import Config.backend.CpuKernelConfig;
import Backend.kernels.cpu.CpuKernel;
import Backend.kernels.cpu.CpuExecutionConfig;
import Backend.kernels.cpu.CpuStridedElementWise;
import Backend.registry.CpuKernelRegistry;
import Tensor.Tensor;
import Tensor.TensorRemap;
import Operations.Operation;

import java.util.ArrayList;
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

        if (canUseStridedPath(op, inputs, node)) {
            CpuStridedElementWise.forward(op, inputs, node);
            return;
        }

        List<Tensor> preparedInputs = prepareInputs(op, inputs);
        kernel.forward(op, preparedInputs, node, executionConfig);
    }

    private boolean canUseStridedPath(Operation op, List<Tensor> inputs, Tensor node) {
        if (op == null || node == null || inputs == null || inputs.isEmpty()) {
            return false;
        }
        if (op.opType() == Operation.OpType.CONTIGUOUS) {
            return false;
        }
        if (!op.isElementWise() || !CpuStridedElementWise.supports(op)) {
            return false;
        }

        boolean hasNonContiguousInput = false;
        for (Tensor input : inputs) {
            if (input == null || input.getData() == null) {
                return false;
            }
            if (!input.isContiguous()) {
                hasNonContiguousInput = true;
            }
        }
        if (!hasNonContiguousInput) {
            return false;
        }

        int threshold = Math.max(0, executionConfig.contiguousMaterializeThreshold());
        int size = node.getFlatDataSize();
        return size < threshold;
    }

    private List<Tensor> prepareInputs(Operation op, List<Tensor> inputs) {
        if (inputs == null || inputs.isEmpty()) {
            return inputs;
        }
        // CONTIGUOUS kernel is already responsible for remapping layout.
        if (op != null && op.opType() == Operation.OpType.CONTIGUOUS) {
            return inputs;
        }

        List<Tensor> prepared = null;
        int materializeThreshold = Math.max(0, executionConfig.contiguousMaterializeThreshold());
        for (int i = 0; i < inputs.size(); i++) {
            Tensor input = inputs.get(i);
            boolean needsMaterialization = input != null
                    && input.getData() != null
                    && !input.isContiguous();

            if (!needsMaterialization) {
                if (prepared != null) {
                    prepared.add(input);
                }
                continue;
            }

            if (prepared == null) {
                prepared = new ArrayList<>(inputs.size());
                for (int j = 0; j < i; j++) {
                    prepared.add(inputs.get(j));
                }
            }

            Tensor contiguousInput = new Tensor(input.getShape(), null, input.getLabel() + "_contiguous_tmp");
            TensorRemap.apply(input, contiguousInput, materializeThreshold);
            prepared.add(contiguousInput);
        }

        return prepared == null ? inputs : prepared;
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
