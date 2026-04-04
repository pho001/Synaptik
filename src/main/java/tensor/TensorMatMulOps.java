package tensor;

import operations.matmul;

import java.util.List;

final class TensorMatMulOps {
    private TensorMatMulOps() {}

    static Tensor matmul(Tensor first, Tensor second) {
        int[] as = first.getShapeUnsafe();
        int[] bs = second.getShapeUnsafe();
        if (as.length < 2 || bs.length < 2) {
            throw new IllegalArgumentException("matmul requires rank >= 2 tensors. got " + as.length + "D and " + bs.length + "D");
        }
        int m = as[as.length - 2];
        int k = as[as.length - 1];
        int rhsK = bs[bs.length - 2];
        if (rhsK != k) {
            throw new IllegalArgumentException("matmul shape mismatch on reduction axis: " + k + " vs " + rhsK);
        }
        int n = bs[bs.length - 1];
        int[] outShape = resolveOutputShape(as, bs, m, n);

        Tensor out = new Tensor(outShape, List.of(first, second), new matmul(), "matmul");
        out.setDataType(TensorDataTypeUtil.binary(first, second));
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) return;

            if (first.getRequiresGrad()) {
                Tensor gradRaw = outGrad.matmul(swapLastTwoAxes(second));
                Tensor gradForFirst = TensorBroadcastOps.sumToShape(gradRaw, first.getShapeUnsafe());
                accumulateGradient(first, gradForFirst);
            }
            if (second.getRequiresGrad()) {
                Tensor gradRaw = swapLastTwoAxes(first).matmul(outGrad);
                Tensor gradForSecond = TensorBroadcastOps.sumToShape(gradRaw, second.getShapeUnsafe());
                accumulateGradient(second, gradForSecond);
            }
        });
        return out;
    }

    private static int[] resolveOutputShape(int[] as, int[] bs, int m, int n) {
        int[] aBatch = java.util.Arrays.copyOf(as, as.length - 2);
        int[] bBatch = java.util.Arrays.copyOf(bs, bs.length - 2);
        int[] outBatch = broadcastLeadingShape(aBatch, bBatch);
        int[] outShape = java.util.Arrays.copyOf(outBatch, outBatch.length + 2);
        outShape[outBatch.length] = m;
        outShape[outBatch.length + 1] = n;
        return outShape;
    }

    private static int[] broadcastLeadingShape(int[] first, int[] second) {
        int rank = Math.max(first.length, second.length);
        int[] out = new int[rank];
        for (int i = 0; i < rank; i++) {
            int a = i < rank - first.length ? 1 : first[i - (rank - first.length)];
            int b = i < rank - second.length ? 1 : second[i - (rank - second.length)];
            if (a != b && a != 1 && b != 1) {
                throw new IllegalArgumentException("matmul batch dimensions are not broadcast-compatible.");
            }
            out[i] = Math.max(a, b);
        }
        return out;
    }

    private static Tensor swapLastTwoAxes(Tensor tensor) {
        int rank = tensor.getShapeUnsafe().length;
        if (rank == 2) {
            return tensor.transpose();
        }
        int[] axes = new int[rank];
        for (int i = 0; i < rank; i++) {
            axes[i] = i;
        }
        int tmp = axes[rank - 1];
        axes[rank - 1] = axes[rank - 2];
        axes[rank - 2] = tmp;
        return tensor.permute(axes);
    }

    private static void accumulateGradient(Tensor input, Tensor gradientDelta) {
        if (input.getGradient() == null) {
            input.setGradient(gradientDelta);
        } else {
            input.setGradient(input.getGradient().add(gradientDelta));
        }
    }
}
