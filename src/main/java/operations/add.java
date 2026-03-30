package operations;

import backend.ComputeBackend;
import tensor.BroadcastPlan;
import tensor.Tensor;

import java.util.Arrays;
import java.util.List;

public class add implements Operation {
    private final BroadcastPlan broadcastPlan;

    public add() {
        this(null);
    }

    public add(BroadcastPlan broadcastPlan) {
        this.broadcastPlan = broadcastPlan;
    }

    public BroadcastPlan getBroadcastPlan() {
        return broadcastPlan;
    }


    //default implementation - CPU
    @Override
    public void apply(List<Tensor> inputs, Tensor node) {

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
            for (int i = 0; i < result.length; i++) {
                result[i] = inputA[i] + inputB[i];
            }
        } else {
            int[] outStrides = broadcastPlan.outStrides();
            int[] aEffStrides = broadcastPlan.aEffStrides();
            int[] bEffStrides = broadcastPlan.bEffStrides();
            for (int i = 0; i < result.length; i++) {
                int aIdx = remap(i, outStrides, aEffStrides);
                int bIdx = remap(i, outStrides, bEffStrides);
                result[i] = inputA[aIdx] + inputB[bIdx];
            }
        }
        node.setData(result);
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
        return OpType.ADD;
    }

    @Override
    public void gradient(List<Tensor> inputs, Tensor node) {

        if (inputs.size() != 2) {
            throw new IllegalArgumentException("The input array must contain exactly 2 elements");
        }

        double[] inputAgrad=inputs.getFirst().getGradient().getData();
        double[] inputBgrad=inputs.getLast().getGradient().getData();
        double[] resultGrad=node.getGradient().getData();
        for (int i = 0; i < resultGrad.length; i++) {
            inputAgrad[i]+=resultGrad[i];
            inputBgrad[i]+=resultGrad[i];
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
        return "+";
    }

    @Override
    public boolean requiresOutputForGradient() {
        return false;
    }

    @Override
    public boolean isElementWise(){
        return true;
    }


}
