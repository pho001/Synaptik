package Tensor;

import Operations.matmul;

import java.util.List;

final class TensorMatMulOps {
    private TensorMatMulOps() {}

    static Tensor matmul(Tensor first, Tensor second) {
        int[] as = first.getShape();
        int[] bs = second.getShape();
        if (as.length != 2 || bs.length != 2) {
            throw new IllegalArgumentException("matmul currently supports rank-2 tensors only. got " + as.length + "D and " + bs.length + "D");
        }
        int m = as[0];
        int k = as[1];
        if (bs[0] != k) {
            throw new IllegalArgumentException("matmul shape mismatch: (" + m + "," + k + ") x (" + bs[0] + "," + bs[1] + ")");
        }
        int n = bs[1];

        Tensor out = new Tensor(new int[]{m, n}, List.of(first, second), new matmul(), "matmul");
        out.setDataType(TensorDataTypeUtil.binary(first, second));
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) return;

            if (first.getRequiresGrad()) {
                Tensor gradForFirst = outGrad.matmul(second.transpose());
                accumulateGradient(first, gradForFirst);
            }
            if (second.getRequiresGrad()) {
                Tensor gradForSecond = first.transpose().matmul(outGrad);
                accumulateGradient(second, gradForSecond);
            }
        });
        return out;
    }

    private static void accumulateGradient(Tensor input, Tensor gradientDelta) {
        if (input.getGradient() == null) {
            input.setGradient(gradientDelta);
        } else {
            input.setGradient(input.getGradient().add(gradientDelta));
        }
    }
}

