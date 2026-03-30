package operations;

import backend.ComputeBackend;
import tensor.Tensor;
import utils.FastExp;

import java.util.List;

public class fastTanh implements Operation {
    @Override
    public void apply(List<Tensor> inputs, Tensor node) {
        if (inputs.size() != 1) {
            throw new IllegalArgumentException("The input array must contain exactly 1 element");
        }
        double[] input = inputs.getFirst().getData();
        double[] output = node.getData();
        for (int i = 0; i < input.length; i++) {
            output[i] = FastExp.fastTanhF64(input[i]);
        }
    }

    @Override
    public OpType opType() {
        return OpType.FAST_TANH;
    }

    @Override
    public void gradient(List<Tensor> inputs, Tensor node) {
        if (inputs.size() != 1) {
            throw new IllegalArgumentException("The input array must contain exactly 1 element");
        }
        double[] output = node.getData();
        double[] inputGrad = inputs.getFirst().getGradient().getData();
        double[] outGrad = node.getGradient().getData();
        for (int i = 0; i < output.length; i++) {
            inputGrad[i] += outGrad[i] * (1.0 - output[i] * output[i]);
        }
    }

    @Override
    public ComputeBackend getPreferredBackend() {
        return ComputeBackend.CPU;
    }

    @Override
    public boolean supportsBackend(ComputeBackend backend) {
        return backend == ComputeBackend.GPU_CUDA || backend == ComputeBackend.CPU;
    }

    @Override
    public String getExpression() {
        return "fastTanh";
    }

    @Override
    public boolean isElementWise() {
        return true;
    }

    @Override
    public boolean requiresOutputForGradient() {
        return true;
    }
}

