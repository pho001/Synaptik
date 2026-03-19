package Backend.kernels.cpu;

import Backend.kernels.cpu.elementwise.AddExecutor;
import Operations.Operation;
import Tensor.Tensor;

import java.util.List;

public class CpuAddKernel implements CpuKernel {
    private static final AddExecutor EXECUTOR = new AddExecutor();

    @Override
    public void forward(Operation op, List<Tensor> inputs, Tensor node) {
        forward(op, inputs, node, CpuExecutionConfig.defaults());
    }

    @Override
    public void forward(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
        double[] a = inputs.get(0).getData();
        double[] b = inputs.get(1).getData();
        double[] out = node.getData();
        CpuExecutionMode mode = config.modeFor(op, node);
        EXECUTOR.execute(a, b, out, mode, config);
    }
}
