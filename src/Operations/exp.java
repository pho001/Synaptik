package Operations;

import java.util.List;

import Backend.ComputeBackend;
import Tensor.Tensor;

public class exp implements Operation {



    //default implementation - CPU
    @Override
    public void apply(List<Tensor> inputs, Tensor node) {

        if (inputs.size() != 1) {
            throw new IllegalArgumentException("The input array must contain exactly 1 elements");
        }
        double[] inputA=inputs.getFirst().getData();
        double[] result=node.getData();
        for (int i = 0; i < inputA.length; i++) {
            result[i]=Math.exp(inputA[i]);
        }

    }

    @Override
    public void gradient(List<Tensor> inputs, Tensor node) {
        if (inputs.size() != 1) {
            throw new IllegalArgumentException("The input array must contain exactly 1 element");
        }

        double[] inputData = inputs.getFirst().getData();               // x
        double[] inputGrad = inputs.getFirst().getGradient().getData(); // ∇x
        double[] outGrad = node.getGradient().getData();                // ∇y

        for (int i = 0; i < inputData.length; i++) {
            inputGrad[i] += outGrad[i] * Math.exp(inputData[i]);        // ∇x += ∇y * e^x
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
        return "exp";
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