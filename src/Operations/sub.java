package Operations;
import java.util.Arrays;
import java.util.List;

import Backend.ComputeBackend;
import Tensor.Tensor;


public class sub implements Operation {

    //default implementation - CPU
    @Override
    public void apply(List<Tensor> inputs, Tensor node) {

        if (inputs.size() != 2) {
            throw new IllegalArgumentException("The input array must contain exactly 2 elements");
        }
        double[] inputA=inputs.getFirst().getData();
        double[] inputB=inputs.getLast().getData();
        double[] result=node.getData();
        for (int i = 0; i < result.length; i++) {
            result[i] = inputA[i] - inputB[i];
        }
        node.setData(result);
    }

    @Override
    public OpType opType() {
        return OpType.SUB;
    }

    @Override
    public void gradient(List<Tensor> inputs, Tensor node) {

        if (inputs.size() != 2) {
            throw new IllegalArgumentException("The input array must contain exactly 2 elements");
        }
        if(!Arrays.equals(inputs.getFirst().getShape(),inputs.getLast().getShape())){
            throw new IllegalArgumentException("Input shapes must match");
        }

        double[] inputAgrad=inputs.getFirst().getGradient().getData();
        double[] inputBgrad=inputs.getLast().getGradient().getData();
        double[] resultGrad=node.getGradient().getData();
        for (int i = 0; i < resultGrad.length; i++) {
            inputAgrad[i]+=resultGrad[i];
            inputBgrad[i]-=resultGrad[i];
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
        return "-";
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
