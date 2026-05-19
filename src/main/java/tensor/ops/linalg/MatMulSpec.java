package tensor.ops.linalg;

import operations.linalg.matmul;
import tensor.DataType;
import tensor.Tensor;
import tensor.dtype.TensorDataTypeUtil;

record MatMulSpec(
        int[] outShape,
        DataType outputType
) {
    static MatMulSpec resolve(Tensor first, Tensor second) {
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
        int[] aBatch = java.util.Arrays.copyOf(as, as.length - 2);
        int[] bBatch = java.util.Arrays.copyOf(bs, bs.length - 2);
        int[] outBatch = LinalgSupport.broadcastLeadingShape(aBatch, bBatch, "matmul batch dimensions are not broadcast-compatible.");
        int[] outShape = java.util.Arrays.copyOf(outBatch, outBatch.length + 2);
        outShape[outBatch.length] = m;
        outShape[outBatch.length + 1] = n;
        return new MatMulSpec(outShape, TensorDataTypeUtil.binary(first, second));
    }

    matmul operation() {
        return new matmul();
    }
}
