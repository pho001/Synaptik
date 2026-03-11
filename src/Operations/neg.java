package Operations;

import Backend.ComputeBackend;
import Tensor.Tensor;

import java.util.List;


public class neg implements Operation{
    @Override
    public void apply(List<Tensor> inputs, Tensor node) {

        if (inputs.size() != 1) {
            throw new IllegalArgumentException("The input array must contain exactly 1 element");
        }

        double[] inputA=inputs.getFirst().getData();
        double[] result=node.getData();
        for (int i=0;i<inputA.length;i++){
            result[i]=-inputA[i];
        }
    }

    @Override
    public OpType opType() {
        return OpType.NEG;
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
        return "*";
    }

    @Override
    public boolean isElementWise(){
        return true;
    }
    @Override
    public boolean isCheap() { return true;}
}
