package Backend.kernels.cpu;

import Operations.Operation;
import Tensor.Tensor;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

import java.util.Arrays;
import java.util.List;

public class CpuMatMulKernel implements CpuKernel {
    private static final VectorSpecies<Double> F64 = DoubleVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Float> F32 = FloatVector.SPECIES_PREFERRED;

    @Override
    public void forward(Operation op, List<Tensor> inputs, Tensor node) {
        forwardF64(op, inputs, node, CpuExecutionConfig.defaults());
    }

    @Override
    public void forward(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
        forwardF64(op, inputs, node, config);
    }

    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
        Tensor a = inputs.get(0);
        Tensor b = inputs.get(1);
        int[] as = a.getShape();
        int[] bs = b.getShape();
        int m = as[0];
        int k = as[1];
        int n = bs[1];

        double[] ad = a.getFloat64Data();
        double[] bd = b.getFloat64Data();
        double[] out = node.getFloat64Data();
        Arrays.fill(out, 0.0d);
        runF64(ad, bd, out, m, n, k, config);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
        Tensor a = inputs.get(0);
        Tensor b = inputs.get(1);
        int[] as = a.getShape();
        int[] bs = b.getShape();
        int m = as[0];
        int k = as[1];
        int n = bs[1];

        float[] ad = a.getFloat32Data();
        float[] bd = b.getFloat32Data();
        float[] out = node.getFloat32Data();
        Arrays.fill(out, 0.0f);
        runF32(ad, bd, out, m, n, k, config);
    }

    @Override
    public void forwardF16(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
        Tensor a = inputs.get(0);
        Tensor b = inputs.get(1);
        int[] as = a.getShape();
        int[] bs = b.getShape();
        int m = as[0];
        int k = as[1];
        int n = bs[1];

        short[] ad = a.getFloat16Data();
        short[] bd = b.getFloat16Data();
        short[] out = node.getFloat16Data();
        Arrays.fill(out, (short) 0);
        runF16(ad, bd, out, m, n, k, config);
    }

    private static void runF64(double[] a, double[] b, double[] out, int m, int n, int k, CpuExecutionConfig config) {
        int tm = positiveTile(config.matMulTileM(), 32);
        int tn = positiveTile(config.matMulTileN(), 64);
        int tk = positiveTile(config.matMulTileK(), 64);
        long work = (long) m * n * k;
        boolean parallel = work >= config.matMulParallelMinSize() && config.plannedWorkers() > 1;

        int blockRows = (m + tm - 1) / tm;
        if (parallel && blockRows > 1) {
            CpuThreadPool.runChunks(blockRows, config.plannedWorkers(), block -> {
                int i0 = block * tm;
                int i1 = Math.min(i0 + tm, m);
                computeBlockF64(a, b, out, i0, i1, 0, n, 0, k, n, k, tn, tk);
            });
            return;
        }
        computeBlockF64(a, b, out, 0, m, 0, n, 0, k, n, k, tn, tk);
    }

    private static void computeBlockF64(
            double[] a, double[] b, double[] out,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F64.length();
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                for (int i = iStart; i < iEnd; i++) {
                    int aRow = i * k;
                    int oRow = i * n;
                    for (int p = kk; p < kkEnd; p++) {
                        double av = a[aRow + p];
                        int bRow = p * n;
                        DoubleVector avv = DoubleVector.broadcast(F64, av);
                        int j = jj;
                        int upper = jjEnd - ((jjEnd - j) % width);
                        for (; j < upper; j += width) {
                            DoubleVector cv = DoubleVector.fromArray(F64, out, oRow + j);
                            DoubleVector bv = DoubleVector.fromArray(F64, b, bRow + j);
                            cv.add(avv.mul(bv)).intoArray(out, oRow + j);
                        }
                        for (; j < jjEnd; j++) {
                            out[oRow + j] += av * b[bRow + j];
                        }
                    }
                }
            }
        }
    }

    private static void runF32(float[] a, float[] b, float[] out, int m, int n, int k, CpuExecutionConfig config) {
        int tm = positiveTile(config.matMulTileM(), 32);
        int tn = positiveTile(config.matMulTileN(), 64);
        int tk = positiveTile(config.matMulTileK(), 64);
        long work = (long) m * n * k;
        boolean parallel = work >= config.matMulParallelMinSize() && config.plannedWorkers() > 1;

        int blockRows = (m + tm - 1) / tm;
        if (parallel && blockRows > 1) {
            CpuThreadPool.runChunks(blockRows, config.plannedWorkers(), block -> {
                int i0 = block * tm;
                int i1 = Math.min(i0 + tm, m);
                computeBlockF32(a, b, out, i0, i1, 0, n, 0, k, n, k, tn, tk);
            });
            return;
        }
        computeBlockF32(a, b, out, 0, m, 0, n, 0, k, n, k, tn, tk);
    }

    private static void computeBlockF32(
            float[] a, float[] b, float[] out,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        int width = F32.length();
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                for (int i = iStart; i < iEnd; i++) {
                    int aRow = i * k;
                    int oRow = i * n;
                    for (int p = kk; p < kkEnd; p++) {
                        float av = a[aRow + p];
                        int bRow = p * n;
                        FloatVector avv = FloatVector.broadcast(F32, av);
                        int j = jj;
                        int upper = jjEnd - ((jjEnd - j) % width);
                        for (; j < upper; j += width) {
                            FloatVector cv = FloatVector.fromArray(F32, out, oRow + j);
                            FloatVector bv = FloatVector.fromArray(F32, b, bRow + j);
                            cv.add(avv.mul(bv)).intoArray(out, oRow + j);
                        }
                        for (; j < jjEnd; j++) {
                            out[oRow + j] += av * b[bRow + j];
                        }
                    }
                }
            }
        }
    }

    private static void runF16(short[] a, short[] b, short[] out, int m, int n, int k, CpuExecutionConfig config) {
        int tm = positiveTile(config.matMulTileM(), 32);
        int tn = positiveTile(config.matMulTileN(), 64);
        int tk = positiveTile(config.matMulTileK(), 64);
        long work = (long) m * n * k;
        boolean parallel = work >= config.matMulParallelMinSize() && config.plannedWorkers() > 1;

        int blockRows = (m + tm - 1) / tm;
        if (parallel && blockRows > 1) {
            CpuThreadPool.runChunks(blockRows, config.plannedWorkers(), block -> {
                int i0 = block * tm;
                int i1 = Math.min(i0 + tm, m);
                computeBlockF16(a, b, out, i0, i1, 0, n, 0, k, n, k, tn, tk);
            });
            return;
        }
        computeBlockF16(a, b, out, 0, m, 0, n, 0, k, n, k, tn, tk);
    }

    private static void computeBlockF16(
            short[] a, short[] b, short[] out,
            int iStart, int iEnd,
            int jStart, int jEnd,
            int kStart, int kEnd,
            int n, int k,
            int tn, int tk
    ) {
        for (int kk = kStart; kk < kEnd; kk += tk) {
            int kkEnd = Math.min(kk + tk, kEnd);
            for (int jj = jStart; jj < jEnd; jj += tn) {
                int jjEnd = Math.min(jj + tn, jEnd);
                for (int i = iStart; i < iEnd; i++) {
                    int aRow = i * k;
                    int oRow = i * n;
                    for (int p = kk; p < kkEnd; p++) {
                        float av = CpuDTypeOps.fromHalfBits(a[aRow + p]);
                        int bRow = p * n;
                        for (int j = jj; j < jjEnd; j++) {
                            float cur = CpuDTypeOps.fromHalfBits(out[oRow + j]);
                            float bv = CpuDTypeOps.fromHalfBits(b[bRow + j]);
                            out[oRow + j] = CpuDTypeOps.toHalfBits(cur + av * bv);
                        }
                    }
                }
            }
        }
    }

    private static int positiveTile(int value, int fallback) {
        return value > 0 ? value : fallback;
    }

    @Override
    public CpuKernelCostClass costClass(Operation op) {
        return CpuKernelCostClass.HIGH;
    }
}
