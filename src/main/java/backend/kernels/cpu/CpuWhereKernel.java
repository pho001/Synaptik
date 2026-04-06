package backend.kernels.cpu;

import operations.Operation;
import tensor.Tensor;

import java.util.List;

public final class CpuWhereKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        byte[] cond = inputs.get(0).getBoolData();
        double[] ifTrue = inputs.get(1).getFloat64Data();
        double[] ifFalse = inputs.get(2).getFloat64Data();
        double[] out = node.getFloat64Data();
        runF64(cond, ifTrue, ifFalse, out, context.whereBroadcastPlan());
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        byte[] cond = inputs.get(0).getBoolData();
        float[] ifTrue = inputs.get(1).getFloat32Data();
        float[] ifFalse = inputs.get(2).getFloat32Data();
        float[] out = node.getFloat32Data();
        runF32(cond, ifTrue, ifFalse, out, context.whereBroadcastPlan());
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        byte[] cond = inputs.get(0).getBoolData();
        short[] ifTrue = inputs.get(1).getBFloat16Data();
        short[] ifFalse = inputs.get(2).getBFloat16Data();
        short[] out = node.getBFloat16Data();
        runBF16(cond, ifTrue, ifFalse, out, context.whereBroadcastPlan());
    }

    private static void runF64(byte[] cond, double[] ifTrue, double[] ifFalse, double[] out, ResolvedWhereBroadcastPlan plan) {
        if (plan == null || plan.isNoBroadcast()) {
            for (int i = 0; i < out.length; i++) {
                out[i] = cond[i] != 0 ? ifTrue[i] : ifFalse[i];
            }
            return;
        }
        int[] outStrides = plan.outStrides();
        int[] outShape = plan.outShape();
        int[] cEff = plan.condEffStrides();
        int[] tEff = plan.trueEffStrides();
        int[] fEff = plan.falseEffStrides();
        int[] cResets = plan.condResets();
        int[] tResets = plan.trueResets();
        int[] fResets = plan.falseResets();
        int rank = outStrides.length;
        int[] coords = new int[rank];
        int cIdx = 0;
        int tIdx = 0;
        int fIdx = 0;
        for (int i = 0; i < out.length; i++) {
            out[i] = cond[cIdx] != 0 ? ifTrue[tIdx] : ifFalse[fIdx];
            for (int d = rank - 1; d >= 0; d--) {
                coords[d]++;
                cIdx += cEff[d];
                tIdx += tEff[d];
                fIdx += fEff[d];
                if (coords[d] < outShape[d]) {
                    break;
                }
                coords[d] = 0;
                cIdx -= cResets[d];
                tIdx -= tResets[d];
                fIdx -= fResets[d];
            }
        }
    }

    private static void runF32(byte[] cond, float[] ifTrue, float[] ifFalse, float[] out, ResolvedWhereBroadcastPlan plan) {
        if (plan == null || plan.isNoBroadcast()) {
            for (int i = 0; i < out.length; i++) {
                out[i] = cond[i] != 0 ? ifTrue[i] : ifFalse[i];
            }
            return;
        }
        int[] outStrides = plan.outStrides();
        int[] outShape = plan.outShape();
        int[] cEff = plan.condEffStrides();
        int[] tEff = plan.trueEffStrides();
        int[] fEff = plan.falseEffStrides();
        int[] cResets = plan.condResets();
        int[] tResets = plan.trueResets();
        int[] fResets = plan.falseResets();
        int rank = outStrides.length;
        int[] coords = new int[rank];
        int cIdx = 0;
        int tIdx = 0;
        int fIdx = 0;
        for (int i = 0; i < out.length; i++) {
            out[i] = cond[cIdx] != 0 ? ifTrue[tIdx] : ifFalse[fIdx];
            for (int d = rank - 1; d >= 0; d--) {
                coords[d]++;
                cIdx += cEff[d];
                tIdx += tEff[d];
                fIdx += fEff[d];
                if (coords[d] < outShape[d]) {
                    break;
                }
                coords[d] = 0;
                cIdx -= cResets[d];
                tIdx -= tResets[d];
                fIdx -= fResets[d];
            }
        }
    }

    private static void runBF16(byte[] cond, short[] ifTrue, short[] ifFalse, short[] out, ResolvedWhereBroadcastPlan plan) {
        if (plan == null || plan.isNoBroadcast()) {
            for (int i = 0; i < out.length; i++) {
                out[i] = cond[i] != 0 ? ifTrue[i] : ifFalse[i];
            }
            return;
        }
        int[] outStrides = plan.outStrides();
        int[] outShape = plan.outShape();
        int[] cEff = plan.condEffStrides();
        int[] tEff = plan.trueEffStrides();
        int[] fEff = plan.falseEffStrides();
        int[] cResets = plan.condResets();
        int[] tResets = plan.trueResets();
        int[] fResets = plan.falseResets();
        int rank = outStrides.length;
        int[] coords = new int[rank];
        int cIdx = 0;
        int tIdx = 0;
        int fIdx = 0;
        for (int i = 0; i < out.length; i++) {
            out[i] = cond[cIdx] != 0 ? ifTrue[tIdx] : ifFalse[fIdx];
            for (int d = rank - 1; d >= 0; d--) {
                coords[d]++;
                cIdx += cEff[d];
                tIdx += tEff[d];
                fIdx += fEff[d];
                if (coords[d] < outShape[d]) {
                    break;
                }
                coords[d] = 0;
                cIdx -= cResets[d];
                tIdx -= tResets[d];
                fIdx -= fResets[d];
            }
        }
    }
}
