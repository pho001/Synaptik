package operations;

import backend.ComputeBackend;
import tensor.Tensor;

import java.util.List;

public class matmul implements Operation {
    @Override
    public OpType opType() {
        return OpType.MATMUL;
    }

    @Override
    public boolean isElementWise() {
        return false;
    }

    @Override
    public void apply(List<Tensor> inputs, Tensor out) {
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("matmul expects exactly 2 inputs");
        }
        Tensor a = inputs.get(0);
        Tensor b = inputs.get(1);
        int[] as = a.getShape();
        int[] bs = b.getShape();
        if (as.length != 2 || bs.length != 2) {
            throw new IllegalArgumentException("matmul currently supports rank-2 tensors only");
        }
        int m = as[0];
        int k = as[1];
        if (bs[0] != k) {
            throw new IllegalArgumentException("matmul shape mismatch: " + as[0] + "x" + as[1] + " vs " + bs[0] + "x" + bs[1]);
        }
        int n = bs[1];
        // Fallback apply path for FLOAT64 only; runtime CPU kernels handle all dtypes.
        double[] ad = a.toDoubleArrayCopy();
        double[] bd = b.toDoubleArrayCopy();
        double[] od = new double[m * n];
        for (int i = 0; i < m; i++) {
            int aRow = i * k;
            int oRow = i * n;
            for (int p = 0; p < k; p++) {
                double av = ad[aRow + p];
                int bRow = p * n;
                for (int j = 0; j < n; j++) {
                    od[oRow + j] += av * bd[bRow + j];
                }
            }
        }
        out.setData(od);
    }

    @Override
    public ComputeBackend getPreferredBackend() {
        return ComputeBackend.CPU;
    }

    @Override
    public boolean supportsBackend(ComputeBackend backend) {
        return backend == ComputeBackend.CPU;
    }

    @Override
    public String getExpression() {
        return "matmul";
    }
}
