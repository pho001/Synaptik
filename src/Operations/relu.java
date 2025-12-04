package Operations;
import Backend.ComputeBackend;

import Tensor.Tensor;

import java.util.List;

public class relu implements Operation {



    //default implementation - CPU
    @Override
    public void apply(List<Tensor> inputs, Tensor node) {

        double[] input = inputs.getFirst().getData();
        double[] output = node.getData();

        for (int i = 0; i < input.length; i++) {
            output[i] = Math.max(0, input[i]);
        }

    }

    @Override
    public void gradient(List<Tensor> inputs, Tensor node) {
        if (inputs.size() != 1) {
            throw new IllegalArgumentException("The input array must contain exactly 1 element");
        }

        double[] input = inputs.getFirst().getData();                   // x
        double[] inputGrad = inputs.getFirst().getGradient().getData(); // ∇x
        double[] outGrad = node.getGradient().getData();                // ∇y

        for (int i = 0; i < input.length; i++) {
            inputGrad[i] += input[i] > 0 ? outGrad[i] : 0;
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
        return "relu";
    }

    @Override
    public boolean isElementWise(){
        return true;
    }

    @Override
    public boolean requiresOutputForGradient() {
        return false;
    }


}