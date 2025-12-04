package Operations;

import java.util.List;

import Backend.ComputeBackend;
import Tensor.Tensor;


public class log implements Operation {



    //default implementation - CPU
    @Override
    public void apply(List<Tensor> inputs, Tensor node) {

        if (inputs.size() != 1) {
            throw new IllegalArgumentException("The input array must contain exactly 1 element");
        }
        double[] inputA=inputs.getFirst().getData();
        double[] result=node.getData();

        for (int i = 0; i < inputA.length; i++) {
            result[i]=Math.log(inputA[i]);
        }


    }

    @Override
    public void gradient(List<Tensor> inputs, Tensor node){
        double epsilon=1e-10;
        if (inputs.size() != 1) {
            throw new IllegalArgumentException("The input array must contain exactly 1 element");
        }
        double[] inputA=inputs.getFirst().getData();
        double[] resultGrad=node.getGradient().getData();
        double[] inputAGrad=inputs.getFirst().getGradient().getData();

        for (int i = 0; i < inputA.length; i++) {
            inputAGrad[i]+=resultGrad[i]/(inputA[i]+epsilon);
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
        return "log";
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