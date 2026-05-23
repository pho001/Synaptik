import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.compile.GraphOptimizationConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import tensor.DataType;
import tensor.Tensor;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static testsupport.NumericPrecisionOracle.add;
import static testsupport.NumericPrecisionOracle.cast;
import static testsupport.NumericPrecisionOracle.div;
import static testsupport.NumericPrecisionOracle.exp;
import static testsupport.NumericPrecisionOracle.inv;
import static testsupport.NumericPrecisionOracle.log;
import static testsupport.NumericPrecisionOracle.max;
import static testsupport.NumericPrecisionOracle.min;
import static testsupport.NumericPrecisionOracle.mul;
import static testsupport.NumericPrecisionOracle.mulScalar;
import static testsupport.NumericPrecisionOracle.neg;
import static testsupport.NumericPrecisionOracle.pow;
import static testsupport.NumericPrecisionOracle.sigmoid;
import static testsupport.NumericPrecisionOracle.sqrt;
import static testsupport.NumericPrecisionOracle.sub;

public class DataTypeExecutionCoverageTest {

    @ParameterizedTest
    @EnumSource(value = DataType.class, names = {"FLOAT64", "FLOAT32", "BFLOAT16"})
    void nonFusedElementWiseAcrossAllDataTypes(DataType dataType) {
        double eps = epsFor(dataType);

        Tensor a = tensor(new double[]{0.25, 0.75, 1.25, 2.0}, dataType, "a");
        Tensor b = tensor(new double[]{0.5, 0.1, 0.75, 1.5}, dataType, "b");

        Tensor add = a.add(b);
        CompiledGraph.compile(add, CompileConfig.noGraphOptimizationBaseline()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        assertArrayEquals(expectedBinary(a.toDoubleArrayCopy(), b.toDoubleArrayCopy(), dataType, "add"), add.toDoubleArrayCopy(), eps);

        Tensor sub = a.sub(b);
        CompiledGraph.compile(sub, CompileConfig.noGraphOptimizationBaseline()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        assertArrayEquals(expectedBinary(a.toDoubleArrayCopy(), b.toDoubleArrayCopy(), dataType, "sub"), sub.toDoubleArrayCopy(), eps);

        Tensor mul = a.mul(b);
        CompiledGraph.compile(mul, CompileConfig.noGraphOptimizationBaseline()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        assertArrayEquals(expectedBinary(a.toDoubleArrayCopy(), b.toDoubleArrayCopy(), dataType, "mul"), mul.toDoubleArrayCopy(), eps);

        Tensor div = a.div(b);
        CompiledGraph.compile(div, CompileConfig.noGraphOptimizationBaseline()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        assertArrayEquals(expectedBinary(a.toDoubleArrayCopy(), b.toDoubleArrayCopy(), dataType, "div"), div.toDoubleArrayCopy(), eps);

        Tensor min = a.min(b);
        CompiledGraph.compile(min, CompileConfig.noGraphOptimizationBaseline()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        assertArrayEquals(expectedBinary(a.toDoubleArrayCopy(), b.toDoubleArrayCopy(), dataType, "min"), min.toDoubleArrayCopy(), eps);

        Tensor max = a.max(b);
        CompiledGraph.compile(max, CompileConfig.noGraphOptimizationBaseline()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        assertArrayEquals(expectedBinary(a.toDoubleArrayCopy(), b.toDoubleArrayCopy(), dataType, "max"), max.toDoubleArrayCopy(), eps);

        Tensor chain = a.mul(0.5).exp().log().pow(2.0).sqrt().sigmoid().inv().neg();
        CompiledGraph.compile(chain, CompileConfig.noGraphOptimizationBaseline()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        assertArrayEquals(expectedUnaryChain(a.toDoubleArrayCopy(), dataType), chain.toDoubleArrayCopy(), epsForChain(dataType));
        assertEquals(dataType, chain.getDataType());
    }

    @ParameterizedTest
    @EnumSource(value = DataType.class, names = {"FLOAT64", "FLOAT32", "BFLOAT16"})
    void fusedElementWiseAcrossAllDataTypes(DataType dataType) {
        double eps = epsForChain(dataType);

        Tensor a = tensor(buildInput(4096, 0.05), dataType, "af");
        Tensor b = tensor(buildInput(4096, -0.03), dataType, "bf");
        Tensor c = tensor(buildInput(4096, 0.01), dataType, "cf");

        Tensor out = a.add(b).mul(c).add(a.mul(0.25)).max(b).min(c).sigmoid();
        CompiledGraph.compile(out, fuseOnlyInferenceConfig()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(expectedFused(a.toDoubleArrayCopy(), b.toDoubleArrayCopy(), c.toDoubleArrayCopy(), dataType), out.toDoubleArrayCopy(), eps);
        assertEquals(dataType, out.getDataType());
    }

    @ParameterizedTest
    @EnumSource(value = DataType.class, names = {"FLOAT64", "FLOAT32", "BFLOAT16"})
    void sumAndContiguousAcrossAllDataTypes(DataType dataType) {
        double eps = epsFor(dataType);

        Tensor base = tensor(new double[]{1.0, 2.0, 3.0, 4.0}, dataType, "base");
        Tensor sum = base.sum();
        CompiledGraph.compile(sum, CompileConfig.noGraphOptimizationBaseline()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        double expectedSum = 0.0;
        for (double v : base.toDoubleArrayCopy()) {
            expectedSum = add(expectedSum, v, dataType);
        }
        assertArrayEquals(new double[]{expectedSum}, sum.toDoubleArrayCopy(), eps);
        assertEquals(dataType, sum.getDataType());

        Tensor view = new Tensor(base.toDoubleArrayCopy().clone(), new int[]{2, 2}, new int[]{1, 2}, null, "view");
        view.setDataType(dataType);
        Tensor contiguous = view.contiguous();
        CompiledGraph.compile(contiguous, CompileConfig.noGraphOptimizationBaseline()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        double[] expectedContiguous = expectedContiguousView(base.toDoubleArrayCopy(), new int[]{2, 2}, new int[]{1, 2}, dataType);
        assertArrayEquals(expectedContiguous, contiguous.toDoubleArrayCopy(), eps);
        assertEquals(dataType, contiguous.getDataType());
    }

    private static Tensor tensor(double[] values, DataType dataType, String label) {
        Tensor t = new Tensor(values.clone(), new int[]{values.length}, null, label);
        t.setDataType(dataType);
        return t;
    }

    private static double epsFor(DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> 1e-9;
            case FLOAT32 -> 1e-6;
            case BFLOAT16 -> 2e-3;
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException("INT32/INT64/BOOL are not part of DataTypeExecutionCoverageTest.");
        };
    }

    private static double epsForChain(DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> 1e-9;
            case FLOAT32 -> 1e-6;
            case BFLOAT16 -> 8e-3;
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException("INT32/INT64/BOOL are not part of DataTypeExecutionCoverageTest.");
        };
    }

    private static double[] expectedBinary(double[] a, double[] b, DataType dataType, String op) {
        double[] out = new double[a.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = switch (op) {
                case "add" -> add(a[i], b[i], dataType);
                case "sub" -> sub(a[i], b[i], dataType);
                case "mul" -> mul(a[i], b[i], dataType);
                case "div" -> div(a[i], b[i], dataType);
                case "min" -> min(a[i], b[i], dataType);
                case "max" -> max(a[i], b[i], dataType);
                default -> throw new IllegalArgumentException("Unsupported op: " + op);
            };
        }
        return out;
    }

    private static double[] expectedUnaryChain(double[] in, DataType dataType) {
        double[] out = new double[in.length];
        for (int i = 0; i < out.length; i++) {
            double v = in[i];
            v = mulScalar(v, 0.5, dataType);
            v = exp(v, dataType);
            v = log(v, dataType);
            v = pow(v, 2.0, dataType);
            v = sqrt(v, dataType);
            v = sigmoid(v, dataType);
            v = inv(v, dataType);
            v = neg(v, dataType);
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

    private static double[] expectedFused(double[] a, double[] b, double[] c, DataType dataType) {
        double[] out = new double[a.length];
        for (int i = 0; i < out.length; i++) {
            double v1 = add(a[i], b[i], dataType);
            double v2 = mul(v1, c[i], dataType);
            double v3 = mulScalar(a[i], 0.25, dataType);
            double v4 = add(v2, v3, dataType);
            double v5 = max(v4, b[i], dataType);
            double v6 = min(v5, c[i], dataType);
            out[i] = sigmoid(v6, dataType);
        }
        return out;
    }

    private static double[] expectedContiguousView(double[] data, int[] shape, int[] strides, DataType dataType) {
        int size = shape[0] * shape[1];
        double[] out = new double[size];
        int idx = 0;
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                int offset = i * strides[0] + j * strides[1];
                out[idx++] = cast(data[offset], dataType);
            }
        }
        return out;
    }

    private static CompileConfig fuseOnlyInferenceConfig() {
        return CompileConfig.inference().withGraphOptimization(GraphOptimizationConfig.noGraphOptimization());
    }
}
