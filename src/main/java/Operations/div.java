package Operations;

import java.util.Arrays;
import java.util.List;


import Backend.ComputeBackend;
import Tensor.BroadcastPlan;
import Tensor.Tensor;

public class div implements Operation {
    private final BroadcastPlan broadcastPlan;

    public div() {
        this(null);
    }

    public div(BroadcastPlan broadcastPlan) {
        this.broadcastPlan = broadcastPlan;
    }

    public BroadcastPlan getBroadcastPlan() {
        return broadcastPlan;
    }



    //default implementation - CPU
    @Override
    public void apply(List<Tensor> inputs,Tensor node) {

        if (inputs.size() != 2) {
            throw new IllegalArgumentException("The input array must contain exactly 2 elements");
        }

        if ((broadcastPlan == null || broadcastPlan.isNoBroadcast())
                && !Arrays.equals(inputs.getFirst().getShape(), inputs.getLast().getShape())) {
            throw new IllegalArgumentException("Input shapes must match");
        }
        double[] inputA=inputs.getFirst().getData();
        double[] inputB=inputs.getLast().getData();
        double[] result=node.getData();
        if (broadcastPlan == null || broadcastPlan.isNoBroadcast()) {
            for (int i = 0; i < inputA.length; i++) {
                result[i] = inputA[i] / inputB[i];
            }
        } else {
            int[] outStrides = broadcastPlan.outStrides();
            int[] aEffStrides = broadcastPlan.aEffStrides();
            int[] bEffStrides = broadcastPlan.bEffStrides();
            for (int i = 0; i < result.length; i++) {
                int aIdx = remap(i, outStrides, aEffStrides);
                int bIdx = remap(i, outStrides, bEffStrides);
                result[i] = inputA[aIdx] / inputB[bIdx];
            }
        }

    }

    private static int remap(int flatOut, int[] outStrides, int[] inEffStrides) {
        int idx = flatOut;
        int inFlat = 0;
        for (int d = 0; d < outStrides.length; d++) {
            int coord = idx / outStrides[d];
            idx %= outStrides[d];
            inFlat += coord * inEffStrides[d];
        }
        return inFlat;
    }

    @Override
    public OpType opType() {
        return OpType.DIV;
    }

    @Override
    public void gradient(List<Tensor> inputs,Tensor node) {
        if (inputs.size() != 2) {
            throw new IllegalArgumentException("The input array must contain exactly 2 elements");
        }
        double[] inputAgrad=inputs.getFirst().getGradient().getData();
        double[] inputBgrad=inputs.getLast().getGradient().getData();
        double[] resultGrad=node.getGradient().getData();
        double[] valueA=inputs.getFirst().getData();
        double[] valueB=inputs.getLast().getData();
        for(int i=0;i<inputAgrad.length;i++){
            inputAgrad[i]+=resultGrad[i]/valueB[i];
            inputBgrad[i]+=-resultGrad[i]*(valueA[i]/(valueB[i]*valueB[i]));

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
        return "/";
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
