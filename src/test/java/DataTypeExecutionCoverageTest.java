import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.optimizer.OptimizerStage;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.codegen.FusedDTypeOps;
import tensor.DataType;
import tensor.Tensor;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class DataTypeExecutionCoverageTest {

    @ParameterizedTest
    @EnumSource(value = DataType.class, names = {"FLOAT64", "FLOAT32", "BFLOAT16"})
    void nonFusedElementWiseAcrossAllDataTypes(DataType dataType) {
        int mode = modeFor(dataType);
        double eps = epsFor(dataType);

        Tensor a = tensor(new double[]{0.25, 0.75, 1.25, 2.0}, dataType, "a");
        Tensor b = tensor(new double[]{0.5, 0.1, 0.75, 1.5}, dataType, "b");

        Tensor add = a.add(b);
        CompiledGraph.compile(add, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        assertArrayEquals(expectedBinary(a.toDoubleArrayCopy(), b.toDoubleArrayCopy(), mode, "add"), add.toDoubleArrayCopy(), eps);

        Tensor sub = a.sub(b);
        CompiledGraph.compile(sub, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        assertArrayEquals(expectedBinary(a.toDoubleArrayCopy(), b.toDoubleArrayCopy(), mode, "sub"), sub.toDoubleArrayCopy(), eps);

        Tensor mul = a.mul(b);
        CompiledGraph.compile(mul, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        assertArrayEquals(expectedBinary(a.toDoubleArrayCopy(), b.toDoubleArrayCopy(), mode, "mul"), mul.toDoubleArrayCopy(), eps);

        Tensor div = a.div(b);
        CompiledGraph.compile(div, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        assertArrayEquals(expectedBinary(a.toDoubleArrayCopy(), b.toDoubleArrayCopy(), mode, "div"), div.toDoubleArrayCopy(), eps);

        Tensor min = a.min(b);
        CompiledGraph.compile(min, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        assertArrayEquals(expectedBinary(a.toDoubleArrayCopy(), b.toDoubleArrayCopy(), mode, "min"), min.toDoubleArrayCopy(), eps);

        Tensor max = a.max(b);
        CompiledGraph.compile(max, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        assertArrayEquals(expectedBinary(a.toDoubleArrayCopy(), b.toDoubleArrayCopy(), mode, "max"), max.toDoubleArrayCopy(), eps);

        Tensor chain = a.mul(0.5).exp().log().pow(2.0).sqrt().sigmoid().inv().neg();
        CompiledGraph.compile(chain, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        assertArrayEquals(expectedUnaryChain(a.toDoubleArrayCopy(), mode), chain.toDoubleArrayCopy(), epsForChain(dataType));
        assertEquals(dataType, chain.getDataType());
    }

    @ParameterizedTest
    @EnumSource(value = DataType.class, names = {"FLOAT64", "FLOAT32", "BFLOAT16"})
    void fusedElementWiseAcrossAllDataTypes(DataType dataType) {
        int mode = modeFor(dataType);
        double eps = epsForChain(dataType);

        Tensor a = tensor(buildInput(4096, 0.05), dataType, "af");
        Tensor b = tensor(buildInput(4096, -0.03), dataType, "bf");
        Tensor c = tensor(buildInput(4096, 0.01), dataType, "cf");

        Tensor out = a.add(b).mul(c).add(a.mul(0.25)).max(b).min(c).sigmoid();
        CompiledGraph.compile(out, fuseOnlyInferenceConfig()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(expectedFused(a.toDoubleArrayCopy(), b.toDoubleArrayCopy(), c.toDoubleArrayCopy(), mode), out.toDoubleArrayCopy(), eps);
        assertEquals(dataType, out.getDataType());
    }

    @ParameterizedTest
    @EnumSource(value = DataType.class, names = {"FLOAT64", "FLOAT32", "BFLOAT16"})
    void sumAndContiguousAcrossAllDataTypes(DataType dataType) {
        int mode = modeFor(dataType);
        double eps = epsFor(dataType);

        Tensor base = tensor(new double[]{1.0, 2.0, 3.0, 4.0}, dataType, "base");
        Tensor sum = base.sum();
        CompiledGraph.compile(sum, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        double expectedSum = 0.0;
        for (double v : base.toDoubleArrayCopy()) {
            expectedSum = FusedDTypeOps.add(expectedSum, v, mode);
        }
        assertArrayEquals(new double[]{expectedSum}, sum.toDoubleArrayCopy(), eps);
        assertEquals(dataType, sum.getDataType());

        Tensor view = new Tensor(base.toDoubleArrayCopy().clone(), new int[]{2, 2}, new int[]{1, 2}, null, "view");
        view.setDataType(dataType);
        Tensor contiguous = view.contiguous();
        CompiledGraph.compile(contiguous, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        double[] expectedContiguous = expectedContiguousView(base.toDoubleArrayCopy(), new int[]{2, 2}, new int[]{1, 2}, mode);
        assertArrayEquals(expectedContiguous, contiguous.toDoubleArrayCopy(), eps);
        assertEquals(dataType, contiguous.getDataType());
    }

    private static Tensor tensor(double[] values, DataType dataType, String label) {
        Tensor t = new Tensor(values.clone(), new int[]{values.length}, null, label);
        t.setDataType(dataType);
        return t;
    }

    private static int modeFor(DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> FusedDTypeOps.MODE_F64;
            case FLOAT32 -> FusedDTypeOps.MODE_F32;
            case BFLOAT16 -> FusedDTypeOps.MODE_BF16;
            case INT32, BOOL -> throw new UnsupportedOperationException("INT32/BOOL are not part of DataTypeExecutionCoverageTest.");
        };
    }

    private static double epsFor(DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> 1e-9;
            case FLOAT32 -> 1e-6;
            case BFLOAT16 -> 2e-3;
            case INT32, BOOL -> throw new UnsupportedOperationException("INT32/BOOL are not part of DataTypeExecutionCoverageTest.");
        };
    }

    private static double epsForChain(DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> 1e-9;
            case FLOAT32 -> 1e-6;
            case BFLOAT16 -> 6e-3;
            case INT32, BOOL -> throw new UnsupportedOperationException("INT32/BOOL are not part of DataTypeExecutionCoverageTest.");
        };
    }

    private static double[] expectedBinary(double[] a, double[] b, int mode, String op) {
        double[] out = new double[a.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = switch (op) {
                case "add" -> FusedDTypeOps.add(a[i], b[i], mode);
                case "sub" -> FusedDTypeOps.sub(a[i], b[i], mode);
                case "mul" -> FusedDTypeOps.mul(a[i], b[i], mode);
                case "div" -> FusedDTypeOps.div(a[i], b[i], mode);
                case "min" -> FusedDTypeOps.min(a[i], b[i], mode);
                case "max" -> FusedDTypeOps.max(a[i], b[i], mode);
                default -> throw new IllegalArgumentException("Unsupported op: " + op);
            };
        }
        return out;
    }

    private static double[] expectedUnaryChain(double[] in, int mode) {
        double[] out = new double[in.length];
        for (int i = 0; i < out.length; i++) {
            double v = in[i];
            v = FusedDTypeOps.mulScalar(v, 0.5, mode);
            v = FusedDTypeOps.exp(v, mode);
            v = FusedDTypeOps.log(v, mode);
            v = FusedDTypeOps.pow(v, 2.0, mode);
            v = FusedDTypeOps.sqrt(v, mode);
            v = FusedDTypeOps.sigmoid(v, mode);
            v = FusedDTypeOps.inv(v, mode);
            v = FusedDTypeOps.neg(v, mode);
            out[i] = v;
        }
        return out;
    }

    private static double[] buildInput(int size, double scale) {
        double[] out = new double[size];
        for (int i = 0; i < size; i++) {
            out[i] = Math.sin(i * 0.1) + (i % 13) * scale + 1.0;
        }
        return out;
    }

    private static double[] expectedFused(double[] a, double[] b, double[] c, int mode) {
        double[] out = new double[a.length];
        for (int i = 0; i < out.length; i++) {
            double v1 = FusedDTypeOps.add(a[i], b[i], mode);
            double v2 = FusedDTypeOps.mul(v1, c[i], mode);
            double v3 = FusedDTypeOps.mulScalar(a[i], 0.25, mode);
            double v4 = FusedDTypeOps.add(v2, v3, mode);
            double v5 = FusedDTypeOps.max(v4, b[i], mode);
            double v6 = FusedDTypeOps.min(v5, c[i], mode);
            out[i] = FusedDTypeOps.sigmoid(v6, mode);
        }
        return out;
    }

    private static double[] expectedContiguousView(double[] data, int[] shape, int[] strides, int mode) {
        int size = shape[0] * shape[1];
        double[] out = new double[size];
        int idx = 0;
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                int offset = i * strides[0] + j * strides[1];
                out[idx++] = FusedDTypeOps.cast(data[offset], mode);
            }
        }
        return out;
    }

    private static OptimizerConfig fuseOnlyInferenceConfig() {
        return OptimizerConfig.inferenceDefaults().withStageOrder(java.util.List.of(OptimizerStage.FUSE));
    }
}
