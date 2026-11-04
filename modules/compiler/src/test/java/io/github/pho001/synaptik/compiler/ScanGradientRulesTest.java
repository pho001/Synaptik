package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.*;
import static io.github.pho001.synaptik.compiler.ScanBodyInferenceTest.tensor;
import io.github.pho001.synaptik.config.compile.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.*;
import io.github.pho001.synaptik.model.operation.elementwise.binary.*;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.*;
import io.github.pho001.synaptik.model.operation.elementwise.unary.*;
import io.github.pho001.synaptik.model.operation.elementwise.cast.*;
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.operation.layout.*;
import io.github.pho001.synaptik.model.operation.linalg.*;
import io.github.pho001.synaptik.model.operation.reduction.*;
import io.github.pho001.synaptik.model.operation.scan.ScanAttrs;
import io.github.pho001.synaptik.model.tensor.*;
import java.util.*;
import org.junit.jupiter.api.Test;

final class ScanGradientRulesTest {
    @Test void deepLengthBodyPreservesSelectedRoutesAndOriginalRowCorrelation() {
        Tensor h = tensor(DataType.FLOAT64, true, 2), x = tensor(DataType.FLOAT64, false, 2);
        Tensor next = h;
        for (int i = 0; i < 20_000; i++) next = next.neg();
        var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(next), List.of());
        Tensor initial = tensor(DataType.FLOAT64, true, 2);
        var scan = Scan.apply(List.of(initial), List.of(tensor(DataType.FLOAT64, false, 1, 2)), List.of(),
                tensor(DataType.INT64, false, 2), 1, List.of(OptionalInt.of(0)), body, false);
        Tensor output = scan.finalCarry().getFirst();
        assertTrue(ScanGradientRules.sameRow(initial, 0, output, 0,
                new io.github.pho001.synaptik.model.shape.StaticDimension(2)));
        assertDoesNotThrow(() -> GraphCompiler.compile(CompileMode.FORWARD_AND_BACKWARD, List.of(output),
                Optional.of(request(List.of(output), List.of(tensor(DataType.FLOAT64, false, 2)), List.of(initial))),
                CompileTimeConstantGraph.Ingress.empty(), GraphOptimizationConfig.disabled()));
    }

    @Test void nonzeroAndSharedOriginalSequenceRolesRejectAd() {
        for (var axis : List.of(OptionalInt.of(1), OptionalInt.empty())) {
            Tensor h = tensor(DataType.FLOAT64, true, 2, 2), x = tensor(DataType.FLOAT64, false, 2, 2);
            var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h), List.of());
            var scan = Scan.apply(List.of(tensor(DataType.FLOAT64, true, 2, 2)), List.of(tensor(DataType.FLOAT64, false, 0, 2, 2)),
                    List.of(), tensor(DataType.INT64, false, 2), 1, List.of(axis), body, false);
            Tensor output = scan.finalCarry().getFirst();
            assertThrows(IllegalArgumentException.class, () -> GraphCompiler.compile(
                    CompileMode.FORWARD_AND_BACKWARD, List.of(output),
                    Optional.of(request(List.of(output), List.of(tensor(DataType.FLOAT64, false, 2, 2)),
                            List.of(output.provenance().orElseThrow().inputs().getFirst()))),
                    CompileTimeConstantGraph.Ingress.empty(), GraphOptimizationConfig.disabled()));
        }
    }
    static FunctionalGradientRequest request(List<Tensor> outputs, List<Tensor> seeds, List<Tensor> targets) {
        return new FunctionalGradientRequest(List.of(new FunctionalGradientRequest.Stage(
                outputs.stream().map(t -> (FunctionalGradientRequest.OutputReference)
                        new FunctionalGradientRequest.ForwardTensorReference(t)).toList(),
                seeds.stream().map(Optional::of).toList(), targets, false,
                FunctionalGradientRequest.DisconnectedPolicy.ZERO)));
    }


    @Test void matchesUnrolledExistingGradientsWithTwoCarriesBothSeedsAndBroadcastParameters() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            for (boolean reverse : List.of(false, true)) compareUnrolled(type, reverse, false);
        }
    }

    @Test void matchesUnrolledRankTwoMatmulTransposeClosureInBothDirections() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            for (boolean reverse : List.of(false, true)) compareUnrolled(type, reverse, true);
        }
    }


    @Test void emptyBroadcastReductionProducesNonemptyTypedZeroParameterGradient() {
        Tensor h = tensor(DataType.FLOAT32, false, 0, 2), x = tensor(DataType.FLOAT32, true, 0, 2), w = tensor(DataType.FLOAT32, true, 2);
        var body = new ScanBody(List.of(h), List.of(x), List.of(w), List.of(h.add(x.mul(w))), List.of());
        Tensor initial = tensor(DataType.FLOAT32, false, 0, 2), seq = tensor(DataType.FLOAT32, true, 3, 0, 2), weight = tensor(DataType.FLOAT32, true, 2);
        var scan = Scan.apply(List.of(initial), List.of(seq), List.of(weight), body, false);
        Tensor seed = tensor(DataType.FLOAT32, false, 0, 2);
        var constants = new FirstOrderAutograd.DerivativeConstants();
        Tensor[] gradients = applySelected(scan.finalCarry().getFirst().provenance().orElseThrow().producer(),
                Map.of(0, seed), List.of(2), constants);
        var oracle = new Oracle(); oracle.bind(initial, new double[0]); oracle.bind(seq, new double[0]);
        oracle.bind(weight, new double[] {Double.NaN, Double.POSITIVE_INFINITY}); oracle.bind(seed, new double[0]);
        for (var binding : constants.bindings()) oracle.bind(binding.tensor(), new double[] {Oracle.scalar(binding.splat().value())});
        assertArrayEquals(new double[] {0, 0}, oracle.eval(gradients[2]), 0);
    }

    @Test void emissionOnlySeedsUsePreviousCarryBeforeOriginalTimeStep() {
        for (boolean reverse : List.of(false, true)) {
            Tensor h = tensor(DataType.FLOAT32, false), x = tensor(DataType.FLOAT32, true);
            var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h.add(x)), List.of(h));
            Tensor initial = tensor(DataType.FLOAT32, false), seq = tensor(DataType.FLOAT32, true, 3), seed = tensor(DataType.FLOAT32, false, 3);
            var scan = Scan.apply(List.of(initial), List.of(seq), List.of(), body, reverse);
            Tensor output = scan.outputs().getFirst();
            var expansion = FirstOrderAutograd.expand(AutogradPreflight.preflight(CompileMode.FORWARD_AND_BACKWARD,
                    List.of(output), request(List.of(output), List.of(seed), List.of(seq)),
                    CompileTimeConstantGraph.Ingress.empty()).stageOne(), CompileTimeConstantGraph.Ingress.empty());
            var oracle = new Oracle(); oracle.bind(initial, new double[] {0}); oracle.bind(seq, new double[] {1, 2, 3}); oracle.bind(seed, new double[] {1, 1, 1}); oracle.constants(expansion);
            assertArrayEquals(reverse ? new double[] {5, 3, 0} : new double[] {0, 1, 3}, oracle.eval(output), 0);
            assertArrayEquals(reverse ? new double[] {0, 1, 2} : new double[] {2, 1, 0}, oracle.eval(expansion.targetGradients().getFirst().gradient()), 0);
        }
    }

    private static List<Tensor> recipe(Tensor h, Tensor k, Tensor x, Tensor w, Tensor u, boolean matrix) {
        Tensor nextH = (matrix ? h.matmul(w).add(x.matmul(u)) : h.mul(w).add(x.mul(u))).tanh();
        Tensor nextK = k.add(x).sub(h).sigmoid();
        return List.of(nextH, nextK);
    }

    private static void compareUnrolled(DataType type, boolean reverse, boolean matrix) {
        int time = 3;
        long[] dims = matrix ? new long[] {1, 2} : new long[] {2, 2};
        Tensor h = tensor(type, false, dims), k = tensor(type, false, dims), x = tensor(type, true, dims);
        Tensor w = tensor(type, true, matrix ? new long[] {2, 2} : new long[] {2});
        Tensor u = tensor(type, true, matrix ? new long[] {2, 2} : new long[0]);
        var next = recipe(h, k, x, w, u, matrix);
        var body = new ScanBody(List.of(h, k), List.of(x), List.of(w, u), next, List.of(h, next.getLast()));
        Tensor h0 = tensor(type, false, dims), k0 = tensor(type, false, dims);
        long[] stacked = new long[dims.length + 1];
        stacked[0] = time;
        System.arraycopy(dims, 0, stacked, 1, dims.length);
        Tensor sequence = tensor(type, true, stacked), weight = tensor(type, true, w.descriptor().shape().dimensions().stream().mapToLong(d -> d.staticSize().orElseThrow()).toArray());
        Tensor inputWeight = tensor(type, true, u.descriptor().shape().dimensions().stream().mapToLong(d -> d.staticSize().orElseThrow()).toArray());
        var scan = Scan.apply(List.of(h0, k0), List.of(sequence), List.of(weight, inputWeight), body, reverse);
        var outputs = new ArrayList<>(scan.finalCarry());
        outputs.addAll(scan.outputs());
        var seeds = outputs.stream().map(t -> tensor(type, false,
                t.descriptor().shape().dimensions().stream().mapToLong(d -> d.staticSize().orElseThrow()).toArray())).toList();
        var targets = List.of(sequence, weight, inputWeight);
        var req = request(outputs, seeds, targets);
        var expanded = FirstOrderAutograd.expand(AutogradPreflight.preflight(CompileMode.FORWARD_AND_BACKWARD,
                outputs, req, CompileTimeConstantGraph.Ingress.empty()).stageOne(), CompileTimeConstantGraph.Ingress.empty());
        var oracle = new Oracle();
        oracle.bind(h0, new double[Oracle.count(h0)]);
        oracle.bind(k0, new double[Oracle.count(k0)]);
        oracle.bind(sequence, values(Oracle.count(sequence), .12));
        oracle.bind(weight, values(Oracle.count(weight), .16));
        oracle.bind(inputWeight, values(Oracle.count(inputWeight), .21));
        for (int i = 0; i < seeds.size(); i++) oracle.bind(seeds.get(i), values(Oracle.count(seeds.get(i)), .3 + i * .1));
        oracle.constants(expanded);

        // Independent Tensor unroll: only this test grows with T. It reuses the established AD
        // owners, never ScanGradientRules or an analytic second derivative implementation.
        var steps = new ArrayList<Tensor>();
        var emittedH = new Tensor[time];
        var emittedK = new Tensor[time];
        for (int t = 0; t < time; t++) {
            Tensor step = tensor(type, true, dims);
            steps.add(step);
            oracle.bind(step, Arrays.copyOfRange(oracle.eval(sequence), t * Oracle.count(step), (t + 1) * Oracle.count(step)));
        }
        Tensor uh = h0, uk = k0;
        for (int iteration = 0; iteration < time; iteration++) {
            int t = reverse ? time - 1 - iteration : iteration;
            emittedH[t] = uh;
            var values = recipe(uh, uk, steps.get(t), weight, inputWeight, matrix);
            uh = values.getFirst(); uk = values.getLast(); emittedK[t] = uk;
        }
        Tensor objective = uh.mul(seeds.get(0)).sum().add(uk.mul(seeds.get(1)).sum());
        for (int t = 0; t < time; t++) {
            for (int port = 0; port < 2; port++) {
                Tensor localSeed = tensor(type, false, dims);
                int width = Oracle.count(localSeed);
                oracle.bind(localSeed, Arrays.copyOfRange(oracle.eval(seeds.get(2 + port)), t * width, (t + 1) * width));
                objective = objective.add((port == 0 ? emittedH[t] : emittedK[t]).mul(localSeed).sum());
            }
        }
        var referenceTargets = new ArrayList<>(List.of(weight, inputWeight)); referenceTargets.addAll(steps);
        var reference = FirstOrderAutograd.expand(AutogradPreflight.preflight(CompileMode.FORWARD_AND_BACKWARD,
                List.of(objective), FunctionalGradientTestSupport.stage(objective, referenceTargets),
                CompileTimeConstantGraph.Ingress.empty()), CompileTimeConstantGraph.Ingress.empty());
        oracle.constants(reference);
        double tolerance = type == DataType.FLOAT32 ? 2e-6 : 2e-14;
        assertArrayEquals(oracle.eval(uh), oracle.eval(scan.finalCarry().getFirst()), 0);
        assertArrayEquals(oracle.eval(uk), oracle.eval(scan.finalCarry().getLast()), 0);
        assertArrayEquals(oracle.eval(reference.targetGradients().get(0).gradient()), oracle.eval(expanded.targetGradients().get(1).gradient()), tolerance);
        assertArrayEquals(oracle.eval(reference.targetGradients().get(1).gradient()), oracle.eval(expanded.targetGradients().get(2).gradient()), tolerance);
        double[] dx = oracle.eval(expanded.targetGradients().getFirst().gradient());
        for (int t = 0; t < time; t++) {
            int width = Oracle.count(steps.get(t));
            assertArrayEquals(oracle.eval(reference.targetGradients().get(2 + t).gradient()), Arrays.copyOfRange(dx, t * width, (t + 1) * width), tolerance);
        }
        for (var optimization : List.of(GraphOptimizationConfig.disabled(), GraphOptimizationConfig.standard())) {
            assertDoesNotThrow(() -> GraphCompiler.compile(CompileMode.FORWARD_AND_BACKWARD, outputs, Optional.of(req),
                    CompileTimeConstantGraph.Ingress.empty(), optimization));
        }
    }

    private static double[] values(int count, double base) {
        double[] result = new double[count];
        for (int i = 0; i < count; i++) result[i] = base + (i % 7 - 3) * .03;
        return result;
    }

    @Test void emptyTimeKeepsRawInitialCarryRoutesFinalSeedAndZeroParameterSums() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            for (boolean reverse : List.of(false, true)) {
                Tensor h = tensor(type, true, 2), x = tensor(type, true, 2), w = tensor(type, true, 2);
                var body = new ScanBody(List.of(h), List.of(x), List.of(w), List.of(h.mul(w).add(x).tanh()), List.of(h));
                Tensor initial = tensor(type, true, 2), seq = tensor(type, true, 0, 2), weight = tensor(type, true, 2);
                var scan = Scan.apply(List.of(initial), List.of(seq), List.of(weight), body, reverse);
                Tensor seed = tensor(type, false, 2);
                var expansion = FirstOrderAutograd.expand(AutogradPreflight.preflight(CompileMode.FORWARD_AND_BACKWARD,
                        List.of(scan.finalCarry().getFirst()), request(List.of(scan.finalCarry().getFirst()), List.of(seed),
                                List.of(initial, seq, weight)), CompileTimeConstantGraph.Ingress.empty()).stageOne(), CompileTimeConstantGraph.Ingress.empty());
                var oracle = new Oracle();
                oracle.bind(initial, new double[] {-0.0, Double.POSITIVE_INFINITY});
                oracle.bind(seq, new double[0]); oracle.bind(weight, new double[] {Double.NaN, 2});
                oracle.bind(seed, new double[] {-0.0, 7}); oracle.constants(expansion);
                assertEquals(Double.doubleToRawLongBits(-0.0), Double.doubleToRawLongBits(oracle.eval(scan.finalCarry().getFirst())[0]));
                assertArrayEquals(oracle.eval(initial), oracle.eval(scan.finalCarry().getFirst()), 0);
                assertArrayEquals(oracle.eval(seed), oracle.eval(expansion.targetGradients().get(0).gradient()), 0);
                assertEquals(0, oracle.eval(expansion.targetGradients().get(1).gradient()).length);
                assertArrayEquals(new double[] {0, 0}, oracle.eval(expansion.targetGradients().get(2).gradient()), 0);
                assertEquals(0, oracle.eval(scan.outputs().getFirst()).length);
                for (Tensor residual : scan.residuals()) assertEquals(0, oracle.eval(residual).length);
            }
        }
    }

    @Test void retainedPreviousCarryHistoryIsLegalBehindWhereConditionCut() {
        for (boolean reverse : List.of(false, true)) {
            Tensor h = tensor(DataType.FLOAT64, true), x = tensor(DataType.FLOAT64, false);
            var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h), List.of());
            Tensor p = tensor(DataType.FLOAT64, true), unusedSequence = tensor(DataType.FLOAT64, false, 1);
            var scan = Scan.apply(List.of(p), List.of(unusedSequence), List.of(), body, reverse);
            Tensor history = scan.residuals().get(body.retainedValues().indexOf(h));
            Tensor objective = Tensor.where(history.sum().greaterThan(p), p, p.neg());
            var req = request(List.of(objective), List.of(tensor(DataType.FLOAT64, false)), List.of(p));
            var preflight = assertDoesNotThrow(() -> AutogradPreflight.preflight(
                    CompileMode.FORWARD_AND_BACKWARD, List.of(objective), req,
                    CompileTimeConstantGraph.Ingress.empty()));
            assertDoesNotThrow(() -> FirstOrderAutograd.expand(
                    preflight.stageOne(), CompileTimeConstantGraph.Ingress.empty()));
            for (var optimization : List.of(GraphOptimizationConfig.disabled(), GraphOptimizationConfig.standard())) {
                assertDoesNotThrow(() -> GraphCompiler.compile(
                        CompileMode.FORWARD_AND_BACKWARD, List.of(objective), Optional.of(req),
                        CompileTimeConstantGraph.Ingress.empty(), optimization));
            }
        }
    }

    @Test void savedActivationOutputIsTheExactResidualOperandEvenForSpecialValues() {
        for (boolean sigmoid : List.of(false, true)) {
            Tensor h = tensor(DataType.FLOAT64, true, 4), x = tensor(DataType.FLOAT64, true, 4);
            Tensor activation = sigmoid ? h.add(x).sigmoid() : h.add(x).tanh();
            var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(activation), List.of());
            Tensor initial = tensor(DataType.FLOAT64, true, 4), seq = tensor(DataType.FLOAT64, true, 1, 4), seed = tensor(DataType.FLOAT64, false, 4);
            var scan = Scan.apply(List.of(initial), List.of(seq), List.of(), body, false);
            var constants = new FirstOrderAutograd.DerivativeConstants();
            Tensor[] gradients = applySelected(scan.finalCarry().getFirst().provenance().orElseThrow().producer(),
                    Map.of(0, seed), List.of(0), constants);
            var oracle = new Oracle();
            oracle.bind(initial, new double[] {1, 1, 1, 1}); oracle.bind(seq, new double[] {2, 2, 2, 2});
            oracle.bind(seed, new double[] {1, 1, 1, 1});
            oracle.eval(scan.finalCarry().getFirst());
            // Replace only the test's saved-value supply, not source primals. A replay would
            // produce tanh(3)/sigmoid(3) instead of these exact stored values.
            double[] saved = {.25, -0.0, Double.NaN, Double.POSITIVE_INFINITY};
            int slot = body.retainedValues().indexOf(activation);
            oracle.bind(scan.residuals().get(slot), saved);
            for (var binding : constants.bindings()) oracle.bind(binding.tensor(), new double[] {Oracle.scalar(binding.splat().value())});
            double[] actual = oracle.eval(gradients[0]);
            for (int i = 0; i < actual.length; i++) {
                double expected = sigmoid ? saved[i] * (1 - saved[i]) : 1 - saved[i] * saved[i];
                if (Double.isNaN(expected)) assertTrue(Double.isNaN(actual[i]));
                else assertEquals(Double.doubleToRawLongBits(expected), Double.doubleToRawLongBits(actual[i]));
            }
        }
    }

    @Test void divUsesExistingUnmaskedFormulaForZeroNaNInfinityAndSignedZero() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            Tensor h = tensor(type, true, 6), x = tensor(type, true, 6), w = tensor(type, true, 6);
            var body = new ScanBody(List.of(h), List.of(x), List.of(w), List.of(h.add(x).div(w)), List.of());
            Tensor initial = tensor(type, true, 6), seq = tensor(type, true, 1, 6), weight = tensor(type, true, 6), seed = tensor(type, false, 6);
            var scan = Scan.apply(List.of(initial), List.of(seq), List.of(weight), body, false);
            var constants = new FirstOrderAutograd.DerivativeConstants();
            Tensor[] gradients = applySelected(scan.finalCarry().getFirst().provenance().orElseThrow().producer(),
                    Map.of(0, seed), List.of(0, 2), constants);
            var oracle = new Oracle();
            double[] numerator = {1, -1, 0, Double.NaN, Double.POSITIVE_INFINITY, -0.0};
            double[] denominator = {0, -0.0, 0, 1, Double.POSITIVE_INFINITY, 2};
            oracle.bind(initial, numerator); oracle.bind(seq, new double[6]); oracle.bind(weight, denominator);
            oracle.bind(seed, new double[] {1, 1, 1, 1, 1, 1});
            for (var binding : constants.bindings()) oracle.bind(binding.tensor(), new double[] {Oracle.scalar(binding.splat().value())});
            double[] dh = oracle.eval(gradients[0]);
            double[] dw = oracle.eval(gradients[2]);
            for (int i = 0; i < dh.length; i++) {
                double a = Oracle.round(initial, numerator[i] + 0.0);
                double expectedH = Oracle.round(initial, 1 / denominator[i]);
                double product = Oracle.round(initial, denominator[i] * denominator[i]);
                double expectedW = Oracle.round(initial, -a / product);
                assertSpecial(expectedH, dh[i]);
                // Parameter sums start at exact typed +0 and add the contribution once.
                assertSpecial(Oracle.round(initial, 0.0 + expectedW), dw[i]);
            }
        }
    }

    private static void assertSpecial(double expected, double actual) {
        if (Double.isNaN(expected)) assertTrue(Double.isNaN(actual));
        else assertEquals(Double.doubleToRawLongBits(expected), Double.doubleToRawLongBits(actual));
    }

    @Test void repeatedRootsTiedActualRolesAndReusedOccurrencesAccumulatePositionally() {
        Tensor h = tensor(DataType.FLOAT64, true), x = tensor(DataType.FLOAT64, true), w = tensor(DataType.FLOAT64, true), u = tensor(DataType.FLOAT64, true);
        Tensor next = h.mul(w).add(x.mul(u));
        var body = new ScanBody(List.of(h), List.of(x), List.of(w, u), List.of(next), List.of(next, next));
        Tensor shared = tensor(DataType.FLOAT64, true), seq = tensor(DataType.FLOAT64, true, 3);
        var first = Scan.apply(List.of(shared), List.of(seq), List.of(shared, shared), body, false);
        var second = Scan.apply(List.of(shared), List.of(seq), List.of(shared, shared), body, false);
        Tensor objective = first.finalCarry().getFirst().add(first.outputs().get(0).sum())
                .add(first.outputs().get(1).sum()).add(second.finalCarry().getFirst());
        var expansion = FirstOrderAutograd.expand(AutogradPreflight.preflight(CompileMode.FORWARD_AND_BACKWARD,
                List.of(objective), FunctionalGradientTestSupport.stage(objective, List.of(shared)),
                CompileTimeConstantGraph.Ingress.empty()), CompileTimeConstantGraph.Ingress.empty());
        Tensor state = shared, sum = shared.mul(ScalarValue.float64(0));
        var steps = new ArrayList<Tensor>();
        for (int t = 0; t < 3; t++) {
            Tensor step = tensor(DataType.FLOAT64, true); steps.add(step);
            state = state.mul(shared).add(step.mul(shared)); sum = sum.add(state).add(state);
        }
        Tensor referenceObjective = state.add(sum).add(state);
        var reference = FirstOrderAutograd.expand(AutogradPreflight.preflight(CompileMode.FORWARD_AND_BACKWARD,
                List.of(referenceObjective), FunctionalGradientTestSupport.stage(referenceObjective, List.of(shared)),
                CompileTimeConstantGraph.Ingress.empty()), CompileTimeConstantGraph.Ingress.empty());
        var oracle = new Oracle(); oracle.bind(shared, new double[] {.7}); oracle.bind(seq, new double[] {.2, .4, -.1});
        for (int t = 0; t < 3; t++) oracle.bind(steps.get(t), new double[] {oracle.eval(seq)[t]});
        oracle.constants(expansion); oracle.constants(reference);
        assertArrayEquals(oracle.eval(reference.targetGradients().getFirst().gradient()), oracle.eval(expansion.targetGradients().getFirst().gradient()), 1e-14);
        assertDoesNotThrow(() -> GraphCompiler.compile(CompileMode.FORWARD_AND_BACKWARD, List.of(objective),
                Optional.of(FunctionalGradientTestSupport.request(objective, List.of(shared))), CompileTimeConstantGraph.Ingress.empty(), GraphOptimizationConfig.standard()));
    }

    /**
     * Selects qualified differentiable pairs for this construction-only numerical fixture.
     * Missing seed ports remain absent; public ND-only and disconnected policy belongs to preflight.
     *
     * @param producer exact source occurrence
     * @param seeds actual cotangents keyed by selected canonical public output ports
     * @param inputs original floating input declaration positions needed by the assertion
     * @param constants explicit derivative literal owner
     * @return original-input-aligned nullable selected derivatives
     */
    static Tensor[] applySelected(TensorProducer producer, Map<Integer, Tensor> seeds,
            List<Integer> inputs, FirstOrderAutograd.DerivativeConstants constants) {
        var source = ScanGradientRules.preflightSource(producer);
        var pairs = new ArrayList<ScanGradientRules.Pair>();
        for (int port : new TreeSet<>(seeds.keySet())) for (int input : inputs) {
            var q = ScanGradientRules.qualify(source, port, input);
            if (q.differentiableRoute() && !q.failClosed()) {
                pairs.add(new ScanGradientRules.Pair(port, input));
            }
        }
        var plan = ScanGradientRules.plan(source, new ScanGradientRules.Demand(pairs));
        return ScanGradientRules.apply(plan, plan.selectedSeedPorts().stream().map(seeds::get).toList(), constants);
    }

    /**
     * Bounded test-fixture evaluator of actual Tensor expressions and generated siblings.
     * Length-mode local evaluation gates row arithmetic and batch contributors by explicit roles;
     * it is deliberately not a production runner or a handwritten backward implementation.
     */
    static final class Oracle {
        final IdentityHashMap<Tensor, double[]> values = new IdentityHashMap<>();
        final Set<Tensor> reads = Collections.newSetFromMap(new IdentityHashMap<>());
        final IdentityHashMap<Tensor, java.util.function.Supplier<double[]>> declarations = new IdentityHashMap<>();
        final IdentityHashMap<Tensor, Integer> rows = new IdentityHashMap<>();
        final IdentityHashMap<Tensor, Scan.ValueSemantics> semantics = new IdentityHashMap<>();
        io.github.pho001.synaptik.model.shape.Dimension batchDimension;
        boolean[] active;
        int bodySteps;
        final long batch;
        Oracle() { this(-1); }
        Oracle(long batch) { this.batch = batch; }
        int[] actualShape(Tensor tensor) {
            return tensor.descriptor().shape().dimensions().stream().mapToInt(d ->
                    Math.toIntExact(d.staticSize().orElseGet(() -> {
                        if (!(d instanceof io.github.pho001.synaptik.model.shape.DynamicDimension) || batch < 0)
                            throw new AssertionError("unbound oracle dimension " + d);
                        return batch;
                    }))).toArray();
        }
        int actualCount(Tensor tensor) {
            int result = 1;
            for (int extent : actualShape(tensor)) result = Math.multiplyExact(result, extent);
            return result;
        }
        void bind(Tensor tensor, double[] value) {
            assertEquals(actualCount(tensor), value.length);
            double[] copy = value.clone();
            for (int i = 0; i < copy.length; i++) copy[i] = round(tensor, copy[i]);
            values.put(tensor, copy);
        }
        void constants(FirstOrderAutograd.Expansion expansion) {
            for (var binding : expansion.ingress().bindings()) {
                double[] data = new double[actualCount(binding.tensor())];
                Arrays.fill(data, scalar(binding.splat().value())); bind(binding.tensor(), data);
            }
        }
        private int row(Tensor tensor) {
            if (active == null) return -1;
            return fact(tensor).rowAxis().orElse(-1);
        }
        private Scan.ValueSemantics fact(Tensor tensor) {
            Scan.ValueSemantics known = semantics.get(tensor);
            if (known != null) return known;
            Integer declaration = rows.get(tensor);
            if (declaration != null) {
                var result = Scan.declaredSemantics(tensor.descriptor(),
                        declaration < 0 ? OptionalInt.empty() : OptionalInt.of(declaration), batchDimension);
                semantics.put(tensor, result);
                return result;
            }
            var p = tensor.provenance().orElseThrow().producer();
            var results = Scan.ordinarySemantics(p.operation(),
                    p.inputs().stream().map(Tensor::descriptor).toList(),
                    p.inputs().stream().map(this::fact).toList(), p.outputDescriptors(), batchDimension);
            for (int slot = 0; slot < p.outputCount(); slot++) semantics.put(p.output(slot), results.get(slot));
            return semantics.get(tensor);
        }
        private boolean activeCell(Tensor tensor, int index) {
            int axis = row(tensor);
            return axis < 0 || active[coordinates(index, actualShape(tensor))[axis]];
        }
        double[] eval(Tensor tensor) {
            reads.add(tensor);
            if (values.containsKey(tensor)) return values.get(tensor);
            var declaration = declarations.remove(tensor);
            if (declaration != null) {
                bind(tensor, declaration.get());
                return values.get(tensor);
            }
            var p = tensor.provenance().orElseThrow(() -> new AssertionError("unbound test leaf: " + tensor.id())).producer();
            if (p.scanBody().isPresent()) { scan(p); return values.get(tensor); }
            List<double[]> inputs = p.inputs().stream().map(this::eval).toList();
            var kind = p.operation().kind();
            var attrs = p.operation().attrs();
            int[] outShape = actualShape(tensor);
            double[] result = new double[actualCount(tensor)];
            if (kind instanceof BinaryArithmeticKind k) {
                for (int i = 0; i < result.length; i++) {
                    if (!activeCell(tensor, i)) continue;
                    double a = inputs.get(0)[broadcast(i, outShape, actualShape(p.inputs().get(0)))];
                    double b = inputs.get(1)[broadcast(i, outShape, actualShape(p.inputs().get(1)))];
                    result[i] = round(tensor, switch (k) { case ADD -> a+b; case SUB -> a-b; case MUL -> a*b; case DIV -> a/b; default -> throw new AssertionError(k); });
                }
            } else if (kind instanceof ScalarElementwiseKind k) {
                double b = scalar(((ScalarValueAttrs) attrs).value());
                for (int i = 0; i < result.length; i++) {
                    if (!activeCell(tensor, i)) continue;
                    double a = inputs.getFirst()[i];
                    result[i] = round(tensor, switch(k) { case ADD -> a+b; case SUB -> a-b; case MUL -> a*b; case DIV -> a/b; default -> throw new AssertionError(k); });
                }
            } else if (kind instanceof UnaryElementwiseKind k) {
                for (int i = 0; i < result.length; i++) {
                    if (!activeCell(tensor, i)) continue;
                    double a = inputs.getFirst()[i];
                    result[i] = round(tensor, switch(k) {
                        case NEG -> -a;
                        case TANH -> Math.tanh(a);
                        case SIGMOID -> {
                            double e = round(tensor, Math.exp(a >= 0 ? -a : a));
                            double denominator = round(tensor, 1 + e);
                            yield (a >= 0 ? 1 : e) / denominator;
                        }
                        default -> throw new AssertionError(k);
                    });
                }
            } else if (kind == MatmulKind.MATMUL) {
                int m = outShape[0], n = outShape[1], contraction = actualShape(p.inputs().getFirst())[1];
                for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) {
                    if (!activeCell(tensor, i*n+j)) continue;
                    double sum = 0;
                    for (int l = 0; l < contraction; l++) {
                        if (active != null && row(p.inputs().getFirst()) == 1 && !active[l]) continue;
                        sum = round(tensor, sum + round(tensor, inputs.get(0)[i*contraction+l] * inputs.get(1)[l*n+j]));
                    }
                    result[i*n+j] = sum;
                }
            } else if (kind == AxisTransformKind.PERMUTE) {
                var axes = ((PermutationAttrs) attrs).axes();
                int[] inShape = actualShape(p.inputs().getFirst());
                for (int i = 0; i < result.length; i++) {
                    if (!activeCell(tensor, i)) continue;
                    int[] outCoord = coordinates(i, outShape), inCoord = new int[inShape.length];
                    for (int a = 0; a < axes.size(); a++) inCoord[axes.get(a)] = outCoord[a];
                    result[i] = inputs.getFirst()[index(inCoord, inShape)];
                }
            } else if (kind == ShapeTransformKind.EXPAND) {
                for (int i = 0; i < result.length; i++) if (activeCell(tensor, i))
                    result[i] = inputs.getFirst()[broadcast(i, outShape, actualShape(p.inputs().getFirst()))];
            } else if (kind instanceof SliceKind && attrs instanceof SliceAttrs slice) {
                boolean update = kind == SliceKind.SLICE_UPDATE;
                int[] selectedShape = update ? actualShape(p.inputs().get(1)) : outShape;
                int[] baseShape = update ? outShape : actualShape(p.inputs().getFirst());
                if (update) result = inputs.getFirst().clone();
                int count = update ? inputs.get(1).length : result.length;
                for (int i = 0; i < count; i++) {
                    int[] coordinate = coordinates(i, selectedShape);
                    for (int axis = 0; axis < slice.axes().size(); axis++) {
                        int a = slice.axes().get(axis);
                        coordinate[a] = Math.toIntExact(slice.starts().get(axis)
                                + coordinate[a] * slice.steps().get(axis));
                    }
                    int base = index(coordinate, baseShape);
                    int destination = update ? base : i;
                    if (activeCell(tensor, destination)) {
                        result[destination] = update ? inputs.get(1)[i] : inputs.getFirst()[base];
                    }
                }
            } else if (kind == AggregateReductionKind.SUM || kind == AggregateReductionKind.MEAN) {
                int[] inShape = actualShape(p.inputs().getFirst());
                List<Integer> axes;
                boolean keep;
                if (attrs instanceof SumToShapeAttrs) {
                    // An equal-shape SUM-to-shape is raw movement, including singleton rows.
                    if (Arrays.equals(inShape, outShape)) {
                        for (int i = 0; i < result.length; i++) if (activeCell(tensor, i)) result[i] = inputs.getFirst()[i];
                        values.put(tensor, result); return result;
                    }
                    for (int i = 0; i < inputs.getFirst().length; i++) {
                        if (!activeCell(p.inputs().getFirst(), i)) continue;
                        int dest = broadcast(i, inShape, outShape);
                        result[dest] = round(tensor, result[dest] + inputs.getFirst()[i]);
                    }
                    values.put(tensor, result); return result;
                } else if (attrs instanceof AxisReductionAttrs a) { axes = List.of(a.axis()); keep = a.keepDimensions(); }
                else if (attrs instanceof MultiAxisReductionAttrs a) { axes = a.axes(); keep = a.keepDimensions(); }
                else { axes = java.util.stream.IntStream.range(0, inShape.length).boxed().toList(); keep = false; }
                for (int i = 0; i < inputs.getFirst().length; i++) {
                    if (!activeCell(p.inputs().getFirst(), i)) continue;
                    int[] coord = coordinates(i, inShape), dest = new int[outShape.length];
                    int pos = 0;
                    for (int a = 0; a < inShape.length; a++) {
                        if (!axes.contains(a)) dest[pos++] = coord[a]; else if (keep) dest[pos++] = 0;
                    }
                    int offset = index(dest, outShape);
                    result[offset] = round(tensor, result[offset] + inputs.getFirst()[i]);
                }
                if (kind == AggregateReductionKind.MEAN) {
                    int contributors = 1;
                    for (int axis : axes) contributors *= inShape[axis];
                    for (int i = 0; i < result.length; i++) result[i] = round(tensor, result[i] / contributors);
                }
            } else if (kind == WhereSelectionKind.WHERE) {
                for (int i = 0; i < result.length; i++) if (activeCell(tensor, i)) {
                    boolean condition = inputs.get(0)[broadcast(i, outShape, actualShape(p.inputs().get(0)))] != 0;
                    int branch = condition ? 1 : 2;
                    result[i] = inputs.get(branch)[broadcast(i, outShape, actualShape(p.inputs().get(branch)))];
                }
            } else if (kind == ShapeTransformKind.RESHAPE || kind == AxisTransformKind.EXPAND_DIMS
                    || kind == AxisTransformKind.SQUEEZE || kind == ContiguousKind.CONTIGUOUS || kind == CastKind.CAST) {
                for (int i = 0; i < result.length; i++) if (activeCell(tensor, i)) {
                    double value = inputs.getFirst()[i];
                    result[i] = kind == CastKind.CAST && tensor.descriptor().dataType() == DataType.INT64
                            ? (long) value : round(tensor, value);
                }
            } else throw new AssertionError("test oracle closure missing " + p.operation());
            values.put(tensor, result); return result;
        }
        private void scan(TensorProducer producer) {
            var attrs = (ScanAttrs) producer.operation().attrs();
            var body = producer.scanBody().orElseThrow();
            int c = attrs.carryCount(), s = attrs.sequenceCount();
            boolean length = attrs.lengthAware();
            double[] lengths = length ? eval(producer.inputs().getLast()) : null;
            if (length) for (double l : lengths) if (l < 0 || l > attrs.timeCount() || l != Math.rint(l)) {
                throw new IllegalArgumentException("test oracle invalid length");
            }
            var carried = new ArrayList<double[]>();
            for (int i = 0; i < c; i++) carried.add(eval(producer.inputs().get(i)).clone());
            for (int i = c; i < producer.outputCount(); i++) values.put(producer.output(i), new double[actualCount(producer.output(i))]);
            for (int iteration = 0; iteration < attrs.timeCount(); iteration++) {
                int t = attrs.reverse() ? Math.toIntExact(attrs.timeCount()) - 1 - iteration : iteration;
                var local = new Oracle(batch);
                if (length) {
                    local.batchDimension = producer.inputs().getLast().descriptor().shape().dimension(0);
                    local.active = new boolean[lengths.length];
                    boolean any = false;
                    for (int b = 0; b < lengths.length; b++) any |= local.active[b] = t < lengths[b];
                    if (!any) continue;
                }
                bodySteps++;
                for (int i = 0; i < c; i++) {
                    local.bind(body.carry().get(i), carried.get(i));
                    if (length) local.rows.put(body.carry().get(i), i < attrs.rowCarryCount() ? 0 : -1);
                }
                for (int i = 0; i < s; i++) {
                    Tensor formal = body.step().get(i), actual = producer.inputs().get(c+i);
                    int width = actualCount(formal), start = t * width;
                    // A mandatory time-domain declaration need not have a consumed payload.
                    // Keep the resolver inside this iteration's lexical scope and memoize on use.
                    local.declarations.put(formal, () -> Arrays.copyOfRange(eval(actual), start, start + width));
                    if (length) local.rows.put(body.step().get(i), attrs.sequenceRowAxes().get(i).orElse(-1));
                }
                for (int i = 0; i < attrs.invariantCount(); i++) {
                    Tensor formal = body.invariants().get(i), actual = producer.inputs().get(c+s+i);
                    local.declarations.put(formal, () -> eval(actual));
                    if (length) local.rows.put(body.invariants().get(i), -1);
                }
                var roots = new ArrayList<>(body.emissions()); roots.addAll(body.retainedValues());
                for (int i = 0; i < roots.size(); i++) {
                    double[] data = local.eval(roots.get(i));
                    double[] destination = values.get(producer.output(c+i));
                    for (int cell = 0; cell < data.length; cell++) if (local.activeCell(roots.get(i), cell))
                        destination[t*data.length+cell] = data[cell];
                }
                var next = new ArrayList<double[]>();
                for (int i = 0; i < c; i++) {
                    Tensor root = body.nextCarry().get(i);
                    double[] data = local.eval(root).clone();
                    for (int cell = 0; cell < data.length; cell++) if (!local.activeCell(root, cell)) data[cell] = carried.get(i)[cell];
                    next.add(data);
                }
                carried = next;
            }
            for (int i = 0; i < c; i++) values.put(producer.output(i), carried.get(i).clone());
        }
        static int count(Tensor t) { return Math.toIntExact(t.descriptor().shape().knownElementCount().orElseThrow()); }
        static int[] shape(Tensor t) { return t.descriptor().shape().dimensions().stream().mapToInt(d -> Math.toIntExact(d.staticSize().orElseThrow())).toArray(); }
        static double round(Tensor t, double value) { return t.descriptor().dataType() == DataType.FLOAT32 ? (double) (float) value : value; }
        static double scalar(ScalarValue value) { return value.dataType() == DataType.FLOAT32 ? value.float32Value() : value.float64Value(); }
        static int[] coordinates(int i, int[] dims) {
            int[] result = new int[dims.length];
            for (int a = dims.length - 1; a >= 0; a--) { result[a] = i % dims[a]; i /= dims[a]; }
            return result;
        }
        static int index(int[] coords, int[] dims) { int result = 0; for (int a = 0; a < dims.length; a++) result = result*dims[a]+coords[a]; return result; }
        static int broadcast(int i, int[] out, int[] in) {
            int[] coords = coordinates(i, out), local = new int[in.length];
            for (int a = 0; a < in.length; a++) local[a] = in[a] == 1 ? 0 : coords[out.length-in.length+a];
            return index(local, in);
        }
    }
}
