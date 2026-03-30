package operations;

import backend.ComputeBackend;
import tensor.BroadcastPlan;
import tensor.Tensor;

import java.util.Arrays;
import java.util.List;

public class max implements Operation {
    private final BroadcastPlan broadcastPlan;

    public max() {
        this(null);
    }

    public max(BroadcastPlan broadcastPlan) {
        this.broadcastPlan = broadcastPlan;
    }

    public BroadcastPlan getBroadcastPlan() {
        return broadcastPlan;
    }

    @Override
    public void apply(List<Tensor> inputs, Tensor node) {
        if (inputs.size() != 2) {
            throw new IllegalArgumentException("The input array must contain exactly 2 elements");
        }
        if ((broadcastPlan == null || broadcastPlan.isNoBroadcast())
                && !Arrays.equals(inputs.getFirst().getShape(), inputs.getLast().getShape())) {
            throw new IllegalArgumentException("Input shapes must match");
        }
        double[] a = inputs.getFirst().getData();
        double[] b = inputs.getLast().getData();
        double[] out = node.getData();
        if (broadcastPlan == null || broadcastPlan.isNoBroadcast()) {
            for (int i = 0; i < out.length; i++) {
                out[i] = Math.max(a[i], b[i]);
            }
        } else {
            int[] outStrides = broadcastPlan.outStrides();
            int[] aEffStrides = broadcastPlan.aEffStrides();
            int[] bEffStrides = broadcastPlan.bEffStrides();
            for (int i = 0; i < out.length; i++) {
                int aIdx = remap(i, outStrides, aEffStrides);
                int bIdx = remap(i, outStrides, bEffStrides);
                out[i] = Math.max(a[aIdx], b[bIdx]);
            }
        }
        node.setData(out);
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
        return OpType.MAX;
    }

    @Override
    public ComputeBackend getPreferredBackend() {
        return ComputeBackend.CPU;
    }

    @Override
    public boolean supportsBackend(ComputeBackend backend) {
        return backend == ComputeBackend.GPU_CUDA || backend == ComputeBackend.CPU;
    }

    @Override
    public String getExpression() {
        return "max";
    }

    @Override
    public boolean isElementWise() {
        return true;
    }
}
