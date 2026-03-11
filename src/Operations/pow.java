package Operations;

import java.util.List;
import Tensor.Tensor;
import Backend.ComputeBackend;

public class pow implements Operation {

    double exponent;

    //default implementation - CPU
    public pow(double exponent) {
        this.exponent = exponent;
    }


    @Override
    public void apply(List<Tensor> inputs,Tensor node) {

        if (inputs.size() != 1) {
            throw new IllegalArgumentException("The input array must contain exactly 1 element");
        }
        double[] inputA=inputs.getFirst().getData();
        double[] result=node.getData();
        for (int i=0;i<inputA.length;i++){
            if (exponent==0){
                result[i]=1;
            }
            if (exponent==1){
                result[i]=inputA[i];
            }
            else if (exponent==2){
                result[i]=inputA[i]*inputA[i];
            }
            else{
                result[i]=Math.pow(inputA[i],exponent);
            }
        }
    }

    @Override
    public OpType opType() {
        return OpType.POW;
    }

    @Override
    public void gradient(List<Tensor> inputs,Tensor node) {
        if (inputs.size() != 1) {
            throw new IllegalArgumentException("The input array must contain exactly 1 element");
        }
        double[] outGrad=node.getGradient().getData();
        double[] inputGrad=inputs.getFirst().getGradient().getData();
        double[] inputVals=inputs.getFirst().getData();
        for (int i=0;i<inputGrad.length;i++){
            if (exponent == 0) {
                break;
                //inputGrad[i] += 0;
            }
            else if (exponent==1){
                inputGrad[i]+=outGrad[i];
            }
            else if (exponent==2){
                inputGrad[i]+=outGrad[i]*2*inputVals[i];
            }
            else{
                inputGrad[i]+=outGrad[i]*exponent*Math.pow(inputVals[i],exponent-1);
            }
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
        return "pow(" + exponent + ")";
    }

    @Override
    public boolean isElementWise(){
        return true;
    }

    public double getExponent() {
        return exponent;
    }

    @Override
    public boolean requiresOutputForGradient() {
        return exponent != 1 && exponent != 0;
    }


}
