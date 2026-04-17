package tensor;

import backend.kernels.cpu.CpuDTypeOps;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public final class TensorBroadcastOps {
    private TensorBroadcastOps() {}

    public static BroadcastPlan planBinary(Tensor first, Tensor second) {
        return BroadcastPlanner.plan(first, second);
    }

    public static Tensor sumToShape(Tensor gradOut, int[] targetShape) {
        int[] outShape = gradOut.getShape();
        int[] normalizedTarget = targetShape.length == 0 ? new int[]{1} : targetShape.clone();
        if (Arrays.equals(outShape, normalizedTarget)) {
            return gradOut;
        }

        int outRank = outShape.length;
        int inRank = normalizedTarget.length;
        if (inRank > outRank) {
            throw new IllegalArgumentException("Target rank cannot exceed grad rank. target="
                    + Arrays.toString(normalizedTarget) + ", grad=" + Arrays.toString(outShape));
        }

        int[] alignedTarget = new int[outRank];
        int offset = outRank - inRank;
        for (int d = 0; d < outRank; d++) {
            alignedTarget[d] = d < offset ? 1 : normalizedTarget[d - offset];
        }

        List<Integer> reduceAxes = new ArrayList<>();
        for (int d = 0; d < outRank; d++) {
            int td = alignedTarget[d];
            int od = outShape[d];
            if (td != od) {
                if (td != 1) {
                    throw new IllegalArgumentException("Incompatible target shape for broadcast reduction. target="
                            + Arrays.toString(normalizedTarget) + ", grad=" + Arrays.toString(outShape));
                }
                reduceAxes.add(d);
            }
        }

        Tensor reduced = gradOut;
        for (int i = reduceAxes.size() - 1; i >= 0; i--) {
            reduced = reduced.sum(reduceAxes.get(i));
        }

        if (reduced.getDataType() != gradOut.getDataType()) {
            reduced.setDataType(gradOut.getDataType());
        }
        if (!Arrays.equals(reduced.getShape(), normalizedTarget)) {
            reduced = reduced.reshape(normalizedTarget);
        }
        return reduced;
    }

    static Tensor minMaxGradForInput(Tensor first, Tensor second, Tensor outGrad, BroadcastPlan plan, boolean forFirst, boolean isMax) {
        if (outGrad == null) {
            return null;
        }
        DataType dtype = outGrad.getDataType();
        return switch (dtype) {
            case FLOAT64 -> minMaxGradF64(first, second, outGrad, plan, forFirst, isMax);
            case FLOAT32 -> minMaxGradF32(first, second, outGrad, plan, forFirst, isMax);
            case BFLOAT16 -> minMaxGradF16(first, second, outGrad, plan, forFirst, isMax);
            case INT32, BOOL -> throw new UnsupportedOperationException("INT32/BOOL are not supported by min/max numeric gradient helpers.");
        };
    }

    private static Tensor minMaxGradF64(Tensor first, Tensor second, Tensor outGrad, BroadcastPlan plan, boolean forFirst, boolean isMax) {
        double[] a = first.toDoubleArrayCopy();
        double[] b = second.toDoubleArrayCopy();
        double[] og = outGrad.toDoubleArrayCopy();
        double[] grad = new double[(forFirst ? first : second).getFlatDataSize()];
        accumulateMinMaxGrad(og.length, plan, isMax, forFirst,
                i -> a[i], i -> b[i], i -> og[i], (idx, val) -> grad[idx] += val);
        return new Tensor(grad, (forFirst ? first : second).getShape().clone(), null,
                (isMax ? "max" : "min") + "_grad_" + (forFirst ? "a" : "b"), DataType.FLOAT64);
    }

    private static Tensor minMaxGradF32(Tensor first, Tensor second, Tensor outGrad, BroadcastPlan plan, boolean forFirst, boolean isMax) {
        float[] a = first.getFloat32Data();
        float[] b = second.getFloat32Data();
        float[] og = outGrad.getFloat32Data();
        float[] grad = new float[(forFirst ? first : second).getFlatDataSize()];
        accumulateMinMaxGrad(og.length, plan, isMax, forFirst,
                i -> a[i], i -> b[i], i -> og[i], (idx, val) -> grad[idx] += (float) val);
        return new Tensor(grad, (forFirst ? first : second).getShape().clone(), null,
                (isMax ? "max" : "min") + "_grad_" + (forFirst ? "a" : "b"), DataType.FLOAT32);
    }

    private static Tensor minMaxGradF16(Tensor first, Tensor second, Tensor outGrad, BroadcastPlan plan, boolean forFirst, boolean isMax) {
        short[] a = first.getBFloat16Data();
        short[] b = second.getBFloat16Data();
        short[] og = outGrad.getBFloat16Data();
        float[] gradAcc = new float[(forFirst ? first : second).getFlatDataSize()];
        accumulateMinMaxGrad(og.length, plan, isMax, forFirst,
                i -> CpuDTypeOps.fromBFloat16Bits(a[i]),
                i -> CpuDTypeOps.fromBFloat16Bits(b[i]),
                i -> CpuDTypeOps.fromBFloat16Bits(og[i]),
                (idx, val) -> gradAcc[idx] += (float) val);

        short[] grad = new short[gradAcc.length];
        for (int i = 0; i < gradAcc.length; i++) {
            grad[i] = CpuDTypeOps.toBFloat16Bits(gradAcc[i]);
        }
        return new Tensor(grad, (forFirst ? first : second).getShape().clone(), null,
                (isMax ? "max" : "min") + "_grad_" + (forFirst ? "a" : "b"), DataType.BFLOAT16);
    }

    private static void accumulateMinMaxGrad(
            int outSize,
            BroadcastPlan plan,
            boolean isMax,
            boolean forFirst,
            IndexToDouble aAt,
            IndexToDouble bAt,
            IndexToDouble gAt,
            Accumulator addGrad
    ) {
        int[] outStrides = plan.outStrides();
        int[] outShape = plan.outShape();
        int[] aEff = plan.aEffStrides();
        int[] bEff = plan.bEffStrides();
        int rank = outStrides.length;

        int[] coords = new int[rank];
        int aIdx = 0;
        int bIdx = 0;

        for (int i = 0; i < outSize; i++) {
            double av = aAt.get(aIdx);
            double bv = bAt.get(bIdx);
            double gv = gAt.get(i);

            if (av == bv) {
                double half = 0.5d * gv;
                if (forFirst) {
                    addGrad.add(aIdx, half);
                } else {
                    addGrad.add(bIdx, half);
                }
            } else {
                boolean firstWins = isMax ? (av > bv) : (av < bv);
                if (forFirst == firstWins) {
                    addGrad.add(forFirst ? aIdx : bIdx, gv);
                }
            }

            for (int d = rank - 1; d >= 0; d--) {
                coords[d]++;
                aIdx += aEff[d];
                bIdx += bEff[d];
                if (coords[d] < outShape[d]) {
                    break;
                }
                coords[d] = 0;
                aIdx -= outShape[d] * aEff[d];
                bIdx -= outShape[d] * bEff[d];
            }
        }
    }

    @FunctionalInterface
    private interface IndexToDouble {
        double get(int index);
    }

    @FunctionalInterface
    private interface Accumulator {
        void add(int index, double value);
    }
}
