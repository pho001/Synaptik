package Operations;
import Backend.ComputeBackend;

import Tensor.Tensor;

import java.util.List;

public class tanh implements Operation {



    //default implementation - CPU
    @Override
    public void apply(List<Tensor> inputs, Tensor node) {

        double[] input = inputs.getFirst().getData();
        double[] output = node.getData();

        for (int i = 0; i < input.length; i++) {
            output[i] = Math.tanh(input[i]);
        }

    }

    @Override
    public OpType opType() {
        return OpType.TANH;
    }

    @Override
    public void gradient(List<Tensor> inputs, Tensor node) {
        if (inputs.size() != 1) {
            throw new IllegalArgumentException("The input array must contain exactly 1 element");
        }

        double[] output = node.getData();                               // σ(x)
        double[] inputGrad = inputs.getFirst().getGradient().getData(); // ∇x
        double[] outGrad = node.getGradient().getData();                // ∇y

        for (int i = 0; i < output.length; i++) {
            inputGrad[i] += outGrad[i] * (1 - output[i] * output[i]);    // 1 - tanh^2(x)
        }
    }

    @Override
    public ComputeBackend getPreferredBackend() {
        return ComputeBackend.CPU;
    }

    @Override
    public boolean supportsBackend(ComputeBackend backend) {
        return backend == ComputeBackend.GPU_CUDA ||
                backend == ComputeBackend.CPU;
    }

    @Override
    public String getExpression() {
        return "tanh";
    }

    @Override
    public boolean isElementWise(){
        return true;
    }

    @Override
    public boolean requiresOutputForGradient() {
        return true;
    }


}
