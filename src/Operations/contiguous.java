package Operations;

import Backend.ComputeBackend;
import Tensor.Tensor;

import java.util.List;
import Utils.*;


public class contiguous implements Operation{
    @Override
    public void apply(List<Tensor> inputs, Tensor node) {
        remap.apply(inputs.getFirst(),node,10000);
    }

    @Override
    public void gradient(List<Tensor> inputs, Tensor node){
        remap.apply(node.getGradient(),inputs.getFirst().getGradient(),10000);
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
