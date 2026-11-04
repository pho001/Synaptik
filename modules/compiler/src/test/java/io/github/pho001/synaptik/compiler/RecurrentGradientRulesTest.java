package io.github.pho001.synaptik.compiler;

import static io.github.pho001.synaptik.compiler.ScanBodyInferenceTest.tensor;
import static org.junit.jupiter.api.Assertions.assertEquals;
import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.recurrent.*;
import io.github.pho001.synaptik.model.tensor.*;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Independent finite differences of the public fixed-cell VJP, including masked padding. */
final class RecurrentGradientRulesTest {
    @Test void jointSeedsDifferentiateEveryFloatingRoleWithMixedLengthsInBothDirections() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            for (RecurrentScanKind kind : RecurrentScanKind.values()) {
                for (RecurrentDirection direction : RecurrentDirection.values()) {
                    check(type, kind, direction);
                }
            }
        }
    }

    @Test void lowPrecisionFixedRecipesCompileEverySelectedOutputAndFloatingRole() {
        for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT16,
                DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2))
            for (RecurrentScanKind kind : RecurrentScanKind.values())
                for (RecurrentDirection direction : RecurrentDirection.values()) {
                    int gates = kind == RecurrentScanKind.RNN_TANH ? 1
                            : kind == RecurrentScanKind.GRU_RESET_AFTER ? 3 : 4;
                    Tensor x = tensor(type, true, 3, 2, 1);
                    Tensor lengths = tensor(DataType.INT64, false, 2);
                    Tensor h = tensor(type, true, 2, 1), c = tensor(type, true, 2, 1);
                    Tensor w = tensor(type, true, gates, 1), u = tensor(type, true, gates, 1);
                    Tensor bias = tensor(type, true, gates);
                    List<Tensor> outputs;
                    if (kind == RecurrentScanKind.LSTM) {
                        var result = RecurrentScan.lstm(x, lengths, h, c, w, u, bias, direction);
                        outputs = List.of(result.outputs(), result.finalHidden(), result.finalCell());
                    } else {
                        var result = kind == RecurrentScanKind.RNN_TANH
                                ? RecurrentScan.rnn(x, lengths, h, w, u, bias, direction)
                                : RecurrentScan.gru(x, lengths, h, w, u, bias, direction);
                        outputs = List.of(result.outputs(), result.finalHidden());
                    }
                    var targets = new ArrayList<>(List.of(x, h));
                    if (kind == RecurrentScanKind.LSTM) targets.add(c);
                    targets.addAll(List.of(w, u, bias));
                    var seeds = outputs.stream().map(output -> TensorFactory.create(
                            new TensorDescriptor(type, output.descriptor().shape(),
                                    output.descriptor().layout(), false))).toList();
                    var compilation = GraphCompiler.compile(CompileMode.FORWARD_AND_BACKWARD,
                            outputs, java.util.Optional.of(ScanGradientRulesTest.request(
                                    outputs, seeds, targets)), CompileTimeConstantGraph.Ingress.empty(),
                            io.github.pho001.synaptik.config.compile.GraphOptimizationConfig.disabled());
                    assertEquals(targets.size(), compilation.gradientResults().size());
                    for (int i = 0; i < targets.size(); i++)
                        assertEquals(targets.get(i).id(), compilation.gradientResults().get(i).target());
                }
    }

    private static void check(DataType type, RecurrentScanKind kind, RecurrentDirection direction) {
        int gates = kind == RecurrentScanKind.RNN_TANH ? 1 : kind == RecurrentScanKind.GRU_RESET_AFTER ? 3 : 4;
        Tensor x = tensor(type, true, 3, 3, 1), lengths = tensor(DataType.INT64, false, 3);
        Tensor h = tensor(type, true, 3, 1), c = tensor(type, true, 3, 1);
        Tensor w = tensor(type, true, gates, 1), u = tensor(type, true, gates, 1);
        Tensor bias = tensor(type, true, gates);
        List<Tensor> outputs;
        if (kind == RecurrentScanKind.LSTM) {
            var result = RecurrentScan.lstm(x, lengths, h, c, w, u, bias, direction);
            outputs = List.of(result.outputs(), result.finalHidden(), result.finalCell());
        } else {
            var result = kind == RecurrentScanKind.RNN_TANH
                    ? RecurrentScan.rnn(x, lengths, h, w, u, bias, direction)
                    : RecurrentScan.gru(x, lengths, h, w, u, bias, direction);
            outputs = List.of(result.outputs(), result.finalHidden());
        }
        var targets = new ArrayList<>(List.of(x, h));
        if (kind == RecurrentScanKind.LSTM) targets.add(c);
        targets.addAll(List.of(w, u, bias));
        var data = new ArrayList<double[]>();
        // Row zero never executes; row one executes only t=0. Padded NaNs must never enter math.
        data.add(new double[] {Double.NaN, .125, -.25, Double.NaN, Double.NaN, .375,
                Double.NaN, Double.NaN, -.125});
        data.add(new double[] {.25, -.125, .375});
        if (kind == RecurrentScanKind.LSTM) data.add(new double[] {-.25, .125, .25});
        double[] wd = new double[gates], ud = new double[gates], bd = new double[gates];
        for (int g = 0; g < gates; g++) { wd[g] = (g + 1) * .125; ud[g] = (g - 1) * .125; bd[g] = (2 - g) * .0625; }
        data.add(wd); data.add(ud); data.add(bd);
        var seeds = outputs.stream().map(t -> TensorFactory.create(new TensorDescriptor(type,
                t.descriptor().shape(), t.descriptor().layout(), false))).toList();
        double[] sequenceSeed = {.25, -.5, .125, -.25, .375, .5, .125, -.125, .25};
        double[] hiddenSeed = {.5, -.25, .375}, cellSeed = {-.125, .25, -.5};
        var plan = AutogradPreflight.preflight(CompileMode.FORWARD_AND_BACKWARD, outputs,
                ScanGradientRulesTest.request(outputs, seeds, targets), CompileTimeConstantGraph.Ingress.empty());
        var expansion = FirstOrderAutograd.expand(plan.stageOne(), CompileTimeConstantGraph.Ingress.empty());
        var oracle = new ScanGradientRulesTest.Oracle();
        for (int i = 0; i < targets.size(); i++) oracle.bind(targets.get(i), data.get(i));
        oracle.bind(lengths, new double[] {0, 1, 3});
        oracle.bind(seeds.get(0), sequenceSeed); oracle.bind(seeds.get(1), hiddenSeed);
        if (kind == RecurrentScanKind.LSTM) oracle.bind(seeds.get(2), cellSeed);
        oracle.constants(expansion);
        double epsilon = 1e-5;
        for (int role = 0; role < targets.size(); role++) {
            double[] actual = oracle.eval(expansion.targetGradients().get(role).gradient());
            for (int index = 0; index < actual.length; index++) {
                double original = data.get(role)[index];
                double expected;
                if (Double.isNaN(original)) expected = 0;
                else {
                    data.get(role)[index] = original + epsilon;
                    double positive = loss(kind, direction, data, sequenceSeed, hiddenSeed, cellSeed);
                    data.get(role)[index] = original - epsilon;
                    double negative = loss(kind, direction, data, sequenceSeed, hiddenSeed, cellSeed);
                    data.get(role)[index] = original;
                    expected = (positive - negative) / (2 * epsilon);
                }
                assertEquals(expected, actual[index], type == DataType.FLOAT64 ? 2e-9 : 2e-6,
                        type + " " + kind + " " + direction + " role=" + role + " index=" + index);
            }
        }
    }

    /** Scalar forward-only reference, deliberately containing no derivative formula. */
    private static double loss(RecurrentScanKind kind, RecurrentDirection direction, List<double[]> values,
            double[] sequenceSeed, double[] hiddenSeed, double[] cellSeed) {
        boolean lstm = kind == RecurrentScanKind.LSTM;
        int parameter = lstm ? 3 : 2;
        double[] x = values.get(0), initial = values.get(1);
        double[] w = values.get(parameter), u = values.get(parameter + 1), bias = values.get(parameter + 2);
        int[] lengths = {0, 1, 3};
        double loss = 0;
        for (int row = 0; row < 3; row++) {
            double h = initial[row], c = lstm ? values.get(2)[row] : 0;
            for (int n = 0; n < lengths[row]; n++) {
                int t = direction == RecurrentDirection.REVERSE ? lengths[row] - 1 - n : n;
                double[] a = new double[w.length], b = new double[w.length];
                for (int g = 0; g < w.length; g++) { a[g] = x[t * 3 + row] * w[g] + bias[g]; b[g] = h * u[g]; }
                if (kind == RecurrentScanKind.RNN_TANH) h = Math.tanh(a[0] + b[0]);
                else if (kind == RecurrentScanKind.GRU_RESET_AFTER) {
                    double reset = sigmoid(a[0] + b[0]), update = sigmoid(a[1] + b[1]);
                    double candidate = Math.tanh(a[2] + reset * b[2]);
                    h = candidate + update * (h - candidate);
                } else {
                    c = sigmoid(a[1] + b[1]) * c + sigmoid(a[0] + b[0]) * Math.tanh(a[2] + b[2]);
                    h = sigmoid(a[3] + b[3]) * Math.tanh(c);
                }
                loss += sequenceSeed[t * 3 + row] * h;
            }
            loss += hiddenSeed[row] * h;
            if (lstm) loss += cellSeed[row] * c;
        }
        return loss;
    }

    private static double sigmoid(double value) {
        double exponential = Math.exp(value >= 0 ? -value : value);
        return value >= 0 ? 1 / (1 + exponential) : exponential / (1 + exponential);
    }
}
