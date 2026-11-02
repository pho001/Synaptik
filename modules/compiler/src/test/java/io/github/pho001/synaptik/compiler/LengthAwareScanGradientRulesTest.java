package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.*;
import static io.github.pho001.synaptik.compiler.ScanBodyInferenceTest.tensor;
import static io.github.pho001.synaptik.compiler.ScanGradientRulesTest.request;
import static io.github.pho001.synaptik.compiler.SymbolicScanCompilerTest.*;
import io.github.pho001.synaptik.config.compile.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.*;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Numerical checks of actual selected reverse expressions against ordinary active-prefix AD. */
final class LengthAwareScanGradientRulesTest {
    private static TensorProducer producer(Tensor t) { return t.provenance().orElseThrow().producer(); }
    private static Tensor seed(Tensor t) { return fresh(t, false); }

    private static void validateAndCompile(ScanResult source, List<Tensor> targets) {
        var roots = new ArrayList<>(source.finalCarry()); roots.addAll(source.outputs());
        var req = request(roots, roots.stream().map(LengthAwareScanGradientRulesTest::seed).toList(), targets);
        for (var opt : List.of(GraphOptimizationConfig.disabled(), GraphOptimizationConfig.standard())) {
            assertDoesNotThrow(() -> compile(roots, Optional.of(req), opt));
        }
    }

    @Test void symbolicRow1FeatureSumRestorationAndEqualAxisSumToShapeRetainTheirCoordinates() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) for (boolean reverse : List.of(false, true))
            for (int features : new int[] {0, 1, 2}) for (int mode = 0; mode < 5; mode++) {
                Tensor h = symbolic(type, false, B, fixed(features)), x = fresh(h, true);
                Tensor transposed = x.permute(1, 0);
                Tensor reduced = switch (mode) {
                    case 0 -> transposed.sum(0, false);
                    case 1 -> transposed.sum(0, true);
                    case 2 -> transposed.sumToShape(shape(B));
                    case 3 -> transposed.sumToShape(shape(fixed(1), B));
                    default -> transposed.sumToShape(transposed.descriptor().shape());
                };
                Tensor leading = mode == 0 || mode == 2 ? reduced.expandDims(1) : reduced.permute(1, 0);
                Tensor next = h.add(leading.expand(h.descriptor().shape()));
                var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(next), List.of());
                Tensor initial = fresh(h, false), seq = symbolic(type, true, fixed(3), B, fixed(features));
                Tensor lengths = symbolic(DataType.INT64, false, B);
                var source = Scan.apply(List.of(initial), List.of(seq), List.of(), lengths,
                        1, List.of(OptionalInt.of(0)), body, reverse);
                var constants = new FirstOrderAutograd.DerivativeConstants();
                Tensor cotangent = seed(source.finalCarry().getFirst());
                Tensor[] gradients = ScanGradientRulesTest.applySelected(producer(source.finalCarry().getFirst()),
                        Map.of(0, cotangent), List.of(1), constants);
                validateAndCompile(source, List.of(seq));
                for (int batch : new int[] {0, 1, 2, 4}) {
                    var oracle = new ScanGradientRulesTest.Oracle(batch);
                    double[] ls = new double[batch]; for (int b = 0; b < batch; b++) ls[b] = (b+1)%4;
                    oracle.bind(lengths, ls); oracle.bind(initial, data(batch*features, .1));
                    oracle.bind(seq, data(3*batch*features, -.1)); oracle.bind(cotangent, data(batch*features, .3));
                    for (var constant : constants.bindings()) oracle.bind(constant.tensor(), new double[] {ScanGradientRulesTest.Oracle.scalar(constant.splat().value())});
                    double[] expected = new double[3*batch*features];
                    for (int b = 0; b < batch; b++) {
                        int valid = (int) ls[b]; if (valid == 0) continue;
                        var reference = new ScanGradientRulesTest.Oracle();
                        Tensor state = tensor(type, false, 1, features), seed = tensor(type, false, 1, features);
                        reference.bind(state, Arrays.copyOfRange(oracle.eval(initial), b*features, (b+1)*features));
                        reference.bind(seed, Arrays.copyOfRange(oracle.eval(cotangent), b*features, (b+1)*features));
                        var steps = new ArrayList<Tensor>();
                        for (int t = 0; t < valid; t++) {
                            Tensor step = tensor(type, true, 1, features); steps.add(step);
                            reference.bind(step, Arrays.copyOfRange(oracle.eval(seq), (t*batch+b)*features, (t*batch+b+1)*features));
                        }
                        for (int iteration = 0; iteration < valid; iteration++) {
                            int t = reverse ? valid-1-iteration : iteration;
                            Tensor contribution = mode == 4 ? steps.get(t) : steps.get(t).sum(1, true).expand(Shape.of(1, features));
                            state = state.add(contribution);
                        }
                        Tensor loss = state.mul(seed).sum();
                        var ordinary = FirstOrderAutograd.expand(AutogradPreflight.preflight(CompileMode.FORWARD_AND_BACKWARD,
                                List.of(loss), FunctionalGradientTestSupport.stage(loss, steps), CompileTimeConstantGraph.Ingress.empty()),
                                CompileTimeConstantGraph.Ingress.empty());
                        reference.constants(ordinary);
                        for (int t = 0; t < valid; t++) System.arraycopy(reference.eval(ordinary.targetGradients().get(t).gradient()),
                                0, expected, (t*batch+b)*features, features);
                    }
                    assertArrayEquals(expected, oracle.eval(gradients[1]), type == DataType.FLOAT32 ? 1e-6 : 1e-14);
                }
            }
    }

    @Test void symbolicLengthSiblingLoadsOverriddenExactTanhAndSigmoidSitesWithoutReplay() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) for (boolean reverse : List.of(false, true)) {
            Tensor h = symbolic(type, true, B, fixed(2)), x = fresh(h, true);
            Tensor tanh = h.add(x).permute(1, 0).tanh(), sigmoid = tanh.sigmoid(), next = sigmoid.permute(1, 0);
            var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(next), List.of(next));
            Tensor initial = fresh(h, true), seq = symbolic(type, true, fixed(1), B, fixed(2));
            Tensor lengths = symbolic(DataType.INT64, false, B);
            var scan = Scan.apply(List.of(initial), List.of(seq), List.of(), lengths,
                    1, List.of(OptionalInt.of(0)), body, reverse);
            Tensor fs = seed(initial), es = seed(scan.outputs().getFirst());
            var constants = new FirstOrderAutograd.DerivativeConstants();
            Tensor[] gradients = ScanGradientRulesTest.applySelected(producer(scan.finalCarry().getFirst()),
                    Map.of(0, fs, 1, es), List.of(0, 1), constants);
            Tensor savedTanh = scan.residuals().get(body.retainedValues().indexOf(tanh));
            Tensor savedSigmoid = scan.residuals().get(body.retainedValues().indexOf(sigmoid));
            for (int batch : new int[] {1, 2, 4}) {
                var oracle = new ScanGradientRulesTest.Oracle(batch);
                double[] ls = new double[batch]; for (int row = 0; row < batch; row++) ls[row] = row % 2 == 0 ? 1 : 0;
                oracle.bind(lengths, ls); oracle.bind(initial, data(batch*2, .1)); oracle.bind(seq, data(batch*2, .2));
                oracle.bind(fs, data(batch*2, .3)); oracle.bind(es, data(batch*2, -.2));
                for (var constant : constants.bindings()) oracle.bind(constant.tensor(), new double[] {ScanGradientRulesTest.Oracle.scalar(constant.splat().value())});
                oracle.eval(scan.finalCarry().getFirst());
                double[] st = new double[batch*2], ss = new double[batch*2];
                Arrays.fill(st, Double.NaN); Arrays.fill(ss, Double.NaN);
                for (int row = 0; row < batch; row++) if (ls[row] == 1) {
                    st[row] = 1; st[batch+row] = -0.0;
                    ss[row] = 0; ss[batch+row] = .25;
                }
                oracle.bind(savedTanh, st); oracle.bind(savedSigmoid, ss);
                double[] dh = new double[batch*2], dx = new double[batch*2];
                for (int row = 0; row < batch; row++) {
                    if (ls[row] == 0) { System.arraycopy(oracle.eval(fs), row*2, dh, row*2, 2); continue; }
                    var reference = new ScanGradientRulesTest.Oracle();
                    Tensor ih = tensor(type, true, 1, 2), ix = tensor(type, true, 1, 2);
                    Tensor rt = ih.add(ix).permute(1, 0).tanh(), rs = rt.sigmoid(), result = rs.permute(1, 0);
                    Tensor finalSeed = tensor(type, false, 1, 2), emissionSeed = tensor(type, false, 1, 2);
                    reference.bind(ih, Arrays.copyOfRange(oracle.eval(initial), row*2, row*2+2));
                    reference.bind(ix, Arrays.copyOfRange(oracle.eval(seq), row*2, row*2+2));
                    reference.bind(finalSeed, Arrays.copyOfRange(oracle.eval(fs), row*2, row*2+2));
                    reference.bind(emissionSeed, Arrays.copyOfRange(oracle.eval(es), row*2, row*2+2));
                    Tensor loss = result.mul(finalSeed).sum().add(result.mul(emissionSeed).sum());
                    var ordinary = FirstOrderAutograd.expand(AutogradPreflight.preflight(CompileMode.FORWARD_AND_BACKWARD,
                            List.of(loss), FunctionalGradientTestSupport.stage(loss, List.of(ih, ix)), CompileTimeConstantGraph.Ingress.empty()),
                            CompileTimeConstantGraph.Ingress.empty());
                    reference.constants(ordinary); reference.eval(result);
                    // Override the exact ordinary saved sites too: no hand-written reverse rule.
                    reference.bind(rt, new double[] {1, -0.0}); reference.bind(rs, new double[] {0, .25});
                    System.arraycopy(reference.eval(ordinary.targetGradients().get(0).gradient()), 0, dh, row*2, 2);
                    System.arraycopy(reference.eval(ordinary.targetGradients().get(1).gradient()), 0, dx, row*2, 2);
                }
                assertArrayEquals(dh, oracle.eval(gradients[0]), type == DataType.FLOAT32 ? 1e-7 : 1e-15);
                assertArrayEquals(dx, oracle.eval(gradients[1]), type == DataType.FLOAT32 ? 1e-7 : 1e-15);
            }
        }
    }

}
