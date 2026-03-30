package operations;

import backend.ComputeBackend;
import tensor.Tensor;
import utils.FastExp;

import java.util.List;

public class fastExp implements Operation {
    @Override
    public void apply(List<Tensor> inputs, Tensor node) {
        if (inputs.size() != 1) {
            throw new IllegalArgumentException("The input array must contain exactly 1 element");
        }
        double[] input = inputs.getFirst().getData();
        double[] out = node.getData();
        for (int i = 0; i < input.length; i++) {
            out[i] = FastExp.fastExpF64(input[i]);
        }
    }

    @Override
    public OpType opType() {
        return OpType.FAST_EXP;
    }

    @Override
    public void gradient(List<Tensor> inputs, Tensor node) {
        if (inputs.size() != 1) {
            throw new IllegalArgumentException("The input array must contain exactly 1 element");
        }
        double[] inputData = inputs.getFirst().getData();
        double[] inputGrad = inputs.getFirst().getGradient().getData();
        double[] outGrad = node.getGradient().getData();
        for (int i = 0; i < inputData.length; i++) {
            inputGrad[i] += outGrad[i] * FastExp.fastExpF64(inputData[i]);
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
        return "fastExp";
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

