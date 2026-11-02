package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.*;
import static io.github.pho001.synaptik.compiler.ScanBodyInferenceTest.tensor;

import io.github.pho001.synaptik.config.compile.*;
import io.github.pho001.synaptik.model.datatype.*;
import io.github.pho001.synaptik.model.tensor.*;
import java.util.*;
import org.junit.jupiter.api.Test;

/**
 * Actual finite selected VJPs, including complete-route cuts and independently present
 * materialized cotangents at successive Scan boundaries.
 */
final class SelectedScanDemandTest {
    @Test void zeroTimeSelectedIdentityPreservesTheActualSeedAndUnrequestedPositionsStayAbsent() {
        for (boolean reverse : List.of(false, true)) for (boolean length : List.of(false, true)) {
            Tensor h = tensor(DataType.FLOAT64, true, 2, 1), k = tensor(DataType.FLOAT32, true, 2, 1);
            Tensor x = tensor(DataType.FLOAT64, true, 2, 1), w = tensor(DataType.FLOAT64, true, 1);
            var body = new ScanBody(List.of(h, k), List.of(x), List.of(w),
                    List.of(h.add(x).mul(w), k), List.of(h));
            Tensor initial = tensor(DataType.FLOAT64, true, 2, 1), other = tensor(DataType.FLOAT32, true, 2, 1);
            Tensor sequence = tensor(DataType.FLOAT64, true, 0, 2, 1), weight = tensor(DataType.FLOAT64, true, 1);
            var scan = length ? Scan.apply(List.of(initial, other), List.of(sequence), List.of(weight),
                    tensor(DataType.INT64, false, 2), 2, List.of(OptionalInt.of(0)), body, reverse)
                    : Scan.apply(List.of(initial, other), List.of(sequence), List.of(weight), body, reverse);
            var source = ScanGradientRules.preflightSource(producer(scan.finalCarry().getFirst()));
            var plan = ScanGradientRules.plan(source, new ScanGradientRules.Demand(
                    List.of(new ScanGradientRules.Pair(0, 0))));
            Tensor seed = tensor(DataType.FLOAT64, false, 2, 1);
            Tensor[] gradients = ScanGradientRules.apply(plan, List.of(seed), new FirstOrderAutograd.DerivativeConstants());
            assertSame(seed, gradients[0], "T=0 is the selected actual cotangent, not a manufactured carry");
            for (int i = 1; i < gradients.length; i++) assertNull(gradients[i]);
            var oracle = new ScanGradientRulesTest.Oracle(); oracle.bind(seed, new double[] {-0.0, Double.POSITIVE_INFINITY});
            assertEquals(Double.doubleToRawLongBits(-0.0), Double.doubleToRawLongBits(oracle.eval(gradients[0])[0]));
            assertArrayEquals(new double[] {-0.0, Double.POSITIVE_INFINITY}, oracle.eval(gradients[0]), 0);
        }
    }

    @Test void finiteZeroOneAndTwoTransitionCyclesMatchOrdinaryUnrollingForEveryLengthPrefix() {
        for (boolean reverse : List.of(false, true)) for (boolean length : List.of(false, true))
            for (int time : new int[] {0, 1, 2}) {
                int batch = 3;
                Tensor h = tensor(DataType.FLOAT64, true, batch, 1), k = tensor(DataType.FLOAT64, true, batch, 1);
                Tensor x = tensor(DataType.FLOAT64, true, batch, 1), w = tensor(DataType.FLOAT64, true, 1);
                var body = new ScanBody(List.of(h, k), List.of(x), List.of(w), List.of(k.mul(w), h.add(x)), List.of());
                Tensor ih = tensor(DataType.FLOAT64, true, batch, 1), ik = tensor(DataType.FLOAT64, true, batch, 1);
                Tensor sequence = tensor(DataType.FLOAT64, true, time, batch, 1), weight = tensor(DataType.FLOAT64, true, 1);
                Tensor lengths = tensor(DataType.INT64, false, batch), seed = tensor(DataType.FLOAT64, false, batch, 1);
                var scan = length ? Scan.apply(List.of(ih, ik), List.of(sequence), List.of(weight), lengths,
                        2, List.of(OptionalInt.of(0)), body, reverse)
                        : Scan.apply(List.of(ih, ik), List.of(sequence), List.of(weight), body, reverse);
                Tensor root = scan.finalCarry().getFirst();
                var targets = List.of(ih, ik, sequence, weight);
                var expansion = expand(List.of(root), ScanGradientRulesTest.request(List.of(root), List.of(seed), targets));
                var oracle = new ScanGradientRulesTest.Oracle();
                double[] ls = {0, Math.min(1, time), time};
                oracle.bind(lengths, ls); oracle.bind(ih, new double[] {.7, 1.7, 2.7});
                oracle.bind(ik, new double[] {-.4, -.2, .1}); oracle.bind(weight, new double[] {1.3});
                oracle.bind(sequence, SymbolicScanCompilerTest.data(time * batch, .2));
                oracle.bind(seed, new double[] {.3, -.7, 1.1}); oracle.constants(expansion);
                double[] expectedH = new double[batch], expectedK = new double[batch], expectedX = new double[time * batch];
                double expectedW = 0;
                for (int row = 0; row < batch; row++) {
                    int valid = length ? (int) ls[row] : time;
                    var reference = new ScanGradientRulesTest.Oracle();
                    Tensor rh = tensor(DataType.FLOAT64, true, 1, 1), rk = tensor(DataType.FLOAT64, true, 1, 1);
                    Tensor rw = tensor(DataType.FLOAT64, true, 1), rs = tensor(DataType.FLOAT64, false, 1, 1);
                    reference.bind(rh, new double[] {oracle.eval(ih)[row]}); reference.bind(rk, new double[] {oracle.eval(ik)[row]});
                    reference.bind(rw, oracle.eval(weight)); reference.bind(rs, new double[] {oracle.eval(seed)[row]});
                    var steps = new ArrayList<Tensor>();
                    for (int t = 0; t < valid; t++) {
                        Tensor step = tensor(DataType.FLOAT64, true, 1, 1); steps.add(step);
                        reference.bind(step, new double[] {oracle.eval(sequence)[t * batch + row]});
                    }
                    Tensor a = rh, b = rk;
                    for (int iteration = 0; iteration < valid; iteration++) {
                        int t = reverse ? valid - 1 - iteration : iteration;
                        Tensor next = b.mul(rw); b = a.add(steps.get(t)); a = next;
                    }
                    Tensor objective = a.mul(rs).sum();
                    var ordinaryTargets = new ArrayList<Tensor>(); ordinaryTargets.add(rh);
                    if (valid > 0) { ordinaryTargets.add(rk); ordinaryTargets.add(rw); ordinaryTargets.addAll(steps); }
                    var ordinaryRoots = List.of(objective, b);
                    var strictStage = FunctionalGradientTestSupport.stage(objective, ordinaryTargets);
                    if (valid > 0) {
                        // H1=k*w excludes h and x0; H2=(h+x0)*w excludes k and x1.
                        // Keep every requested role and verify ERROR before requesting zeros.
                        assertThrows(IllegalArgumentException.class, () -> AutogradPreflight.preflight(
                                CompileMode.FORWARD_AND_BACKWARD, ordinaryRoots, strictStage,
                                CompileTimeConstantGraph.Ingress.empty()));
                    }
                    var ordinaryStage = valid == 0 ? strictStage : FunctionalGradientTestSupport.stage(
                            objective, ordinaryTargets, FunctionalGradientRequest.DisconnectedPolicy.ZERO);
                    var ordinary = FirstOrderAutograd.expand(AutogradPreflight.preflight(CompileMode.FORWARD_AND_BACKWARD,
                            ordinaryRoots, ordinaryStage,
                            CompileTimeConstantGraph.Ingress.empty()), CompileTimeConstantGraph.Ingress.empty());
                    reference.constants(ordinary);
                    assertEquals(reference.eval(a)[0], oracle.eval(root)[row], 0);
                    expectedH[row] = reference.eval(ordinary.targetGradients().getFirst().gradient())[0];
                    if (valid > 0) {
                        expectedK[row] = reference.eval(ordinary.targetGradients().get(1).gradient())[0];
                        expectedW += reference.eval(ordinary.targetGradients().get(2).gradient())[0];
                        for (int t = 0; t < valid; t++) expectedX[t * batch + row] = reference.eval(ordinary.targetGradients().get(3 + t).gradient())[0];
                    }
                }
                assertArrayEquals(expectedH, oracle.eval(expansion.targetGradients().get(0).gradient()), 1e-14);
                assertArrayEquals(expectedK, oracle.eval(expansion.targetGradients().get(1).gradient()), 1e-14);
                assertArrayEquals(expectedX, oracle.eval(expansion.targetGradients().get(2).gradient()), 1e-14);
                assertArrayEquals(new double[] {expectedW}, oracle.eval(expansion.targetGradients().get(3).gradient()), 1e-14);
            }
    }

    @Test void nonDifferentiableOnlyIsNotDisconnectedUnderEitherPolicy() {
        for (boolean reverse : List.of(false, true)) for (boolean length : List.of(false, true)) {
            Tensor h = tensor(DataType.FLOAT64, true, 2, 1), x = tensor(DataType.FLOAT64, true, 2, 1);
            Tensor unused = tensor(DataType.FLOAT64, true, 2, 1);
            var body = new ScanBody(List.of(h), List.of(x, unused), List.of(),
                    List.of(x.cast(DataType.INT64).cast(DataType.FLOAT64)), List.of());
            Tensor initial = tensor(DataType.FLOAT64, true, 2, 1), sequence = tensor(DataType.FLOAT64, true, 1, 2, 1);
            Tensor disconnected = tensor(DataType.FLOAT64, true, 1, 2, 1), lengths = tensor(DataType.INT64, false, 2);
            var scan = length ? Scan.apply(List.of(initial), List.of(sequence, disconnected), List.of(), lengths,
                    1, List.of(OptionalInt.of(0), OptionalInt.of(0)), body, reverse)
                    : Scan.apply(List.of(initial), List.of(sequence, disconnected), List.of(), body, reverse);
            Tensor root = scan.finalCarry().getFirst(), seed = tensor(DataType.FLOAT64, false, 2, 1);
            for (var policy : FunctionalGradientRequest.DisconnectedPolicy.values()) {
                var nd = request(root, seed, sequence, policy);
                assertThrows(IllegalArgumentException.class, () -> SymbolicScanCompilerTest.compile(
                        List.of(root), Optional.of(nd), GraphOptimizationConfig.disabled()));
            }
            var error = request(root, seed, disconnected, FunctionalGradientRequest.DisconnectedPolicy.ERROR);
            assertThrows(IllegalArgumentException.class, () -> SymbolicScanCompilerTest.compile(
                    List.of(root), Optional.of(error), GraphOptimizationConfig.disabled()));
            var zero = expand(List.of(root), request(root, seed, disconnected, FunctionalGradientRequest.DisconnectedPolicy.ZERO));
            var oracle = new ScanGradientRulesTest.Oracle(); oracle.constants(zero);
            assertArrayEquals(new double[] {0, 0}, oracle.eval(zero.targetGradients().getFirst().gradient()), 0);
        }
    }

    @Test void selectedDifferentiableRootsAndTiedActualPositionsOutweighNonDifferentiableRoutes() {
        for (boolean reverse : List.of(false, true)) for (boolean length : List.of(false, true))
            for (boolean tied : List.of(false, true)) {
                Tensor h = tensor(DataType.FLOAT64, true, 2, 1), x = tensor(DataType.FLOAT64, true, 2, 1);
                Tensor y = tensor(DataType.FLOAT64, true, 2, 1);
                var body = new ScanBody(List.of(h), tied ? List.of(x, y) : List.of(x), List.of(),
                        List.of(x.cast(DataType.INT64).cast(DataType.FLOAT64)), List.of(tied ? y : x));
                Tensor initial = tensor(DataType.FLOAT64, true, 2, 1);
                Tensor sequence = tensor(DataType.FLOAT64, true, 1, 2, 1), lengths = tensor(DataType.INT64, false, 2);
                var actuals = tied ? List.of(sequence, sequence) : List.of(sequence);
                var scan = length ? Scan.apply(List.of(initial), actuals, List.of(), lengths, 1,
                        tied ? List.of(OptionalInt.of(0), OptionalInt.of(0)) : List.of(OptionalInt.of(0)), body, reverse)
                        : Scan.apply(List.of(initial), actuals, List.of(), body, reverse);
                var roots = List.of(scan.finalCarry().getFirst(), scan.outputs().getFirst());
                Tensor carrySeed = tensor(DataType.FLOAT64, false, 2, 1);
                Tensor emissionSeed = tensor(DataType.FLOAT64, false, 1, 2, 1);
                for (var policy : FunctionalGradientRequest.DisconnectedPolicy.values()) {
                    var req = request(roots, List.of(carrySeed, emissionSeed), sequence, policy);
                    assertDoesNotThrow(() -> SymbolicScanCompilerTest.compile(roots, Optional.of(req), GraphOptimizationConfig.disabled()));
                    var expansion = expand(roots, req);
                    var oracle = new ScanGradientRulesTest.Oracle();
                    oracle.bind(initial, new double[] {.2, -.4}); oracle.bind(sequence, new double[] {1.3, -2.7});
                    oracle.bind(lengths, new double[] {0, 1}); oracle.bind(carrySeed, new double[] {.3, -.7});
                    oracle.bind(emissionSeed, new double[] {-.2, 1.1}); oracle.constants(expansion);
                    assertArrayEquals(length ? new double[] {0, 1.1} : new double[] {-.2, 1.1},
                            oracle.eval(expansion.targetGradients().getFirst().gradient()), 0);
                }
            }
    }

    @Test void lengthPolicyAggregatesSelectedRoutesWithinEachPrefixNotAcrossDifferentLengths() {
        for (boolean reverse : List.of(false, true)) {
            Tensor h = tensor(DataType.FLOAT64, true, 2, 1), k = tensor(DataType.FLOAT64, true, 2, 1);
            Tensor step = tensor(DataType.FLOAT64, false, 2, 1), x = tensor(DataType.FLOAT64, true, 1);
            var body = new ScanBody(List.of(h, k), List.of(step), List.of(x),
                    List.of(k.add(x.cast(DataType.INT64).cast(DataType.FLOAT64)), h.add(x)), List.of());
            Tensor initialH = tensor(DataType.FLOAT64, true, 2, 1), initialK = tensor(DataType.FLOAT64, true, 2, 1);
            Tensor sequence = tensor(DataType.FLOAT64, false, 2, 2, 1), actual = tensor(DataType.FLOAT64, true, 1);
            Tensor lengths = tensor(DataType.INT64, false, 2), seed = tensor(DataType.FLOAT64, false, 2, 1);
            var full = Scan.apply(List.of(initialH, initialK), List.of(sequence), List.of(actual), body, reverse);
            var masked = Scan.apply(List.of(initialH, initialK), List.of(sequence), List.of(actual), lengths, 2,
                    List.of(OptionalInt.of(0)), body, reverse);
            for (var policy : FunctionalGradientRequest.DisconnectedPolicy.values()) {
                // At L=2 the second carry contributes D to the same target as the direct ND cast.
                Tensor fullRoot = full.finalCarry().getFirst();
                var req = request(fullRoot, seed, actual, policy);
                assertDoesNotThrow(() -> SymbolicScanCompilerTest.compile(List.of(fullRoot), Optional.of(req), GraphOptimizationConfig.disabled()));
                var expansion = expand(List.of(fullRoot), req);
                var oracle = new ScanGradientRulesTest.Oracle();
                oracle.bind(initialH, new double[] {.2, -.4}); oracle.bind(initialK, new double[] {1.3, -.8});
                oracle.bind(actual, new double[] {2.7}); oracle.bind(seed, new double[] {.25, -.75});
                oracle.constants(expansion);
                assertArrayEquals(new double[] {-.5}, oracle.eval(expansion.targetGradients().getFirst().gradient()), 0);
                // L=1 has only the direct ND cast: a D route at L=2 must not hide that legal prefix.
                Tensor maskedRoot = masked.finalCarry().getFirst();
                assertRejectedAs(List.of(maskedRoot), request(maskedRoot, seed, actual, policy), "non-differentiable");
                // An ordinary D route outside Scan rescues the ND-only L=1 prefix as well as L=0.
                Tensor rescuedObjective = maskedRoot.add(actual);
                var rescuedRequest = request(rescuedObjective, seed, actual, policy);
                assertDoesNotThrow(() -> SymbolicScanCompilerTest.compile(List.of(rescuedObjective),
                        Optional.of(rescuedRequest), GraphOptimizationConfig.disabled()));
                var rescued = expand(List.of(rescuedObjective), rescuedRequest);
                for (int firstLength = 0; firstLength <= 2; firstLength++)
                    for (int secondLength = 0; secondLength <= 2; secondLength++) {
                        var rescuedOracle = new ScanGradientRulesTest.Oracle();
                        rescuedOracle.bind(initialH, new double[] {.2, -.4});
                        rescuedOracle.bind(initialK, new double[] {1.3, -.8});
                        rescuedOracle.bind(actual, new double[] {2.7});
                        rescuedOracle.bind(lengths, new double[] {firstLength, secondLength});
                        rescuedOracle.bind(seed, new double[] {.25, -.75});
                        rescuedOracle.constants(rescued);
                        double expected = -.5 + (firstLength == 2 ? .25 : 0) + (secondLength == 2 ? -.75 : 0);
                        assertArrayEquals(new double[] {expected},
                                rescuedOracle.eval(rescued.targetGradients().getFirst().gradient()), 0);
                    }
            }
        }
    }

    @Test void unusedFloatingInvariantOfDiscreteIdentityCarryIsDisconnectedButAUsedOneIsNdOnly() {
        for (boolean reverse : List.of(false, true)) for (boolean length : List.of(false, true))
            for (boolean used : List.of(false, true)) {
                long[] carryShape = length ? new long[] {2, 1} : new long[0];
                Tensor h = tensor(DataType.INT64, false, carryShape), step = tensor(DataType.FLOAT64, false, carryShape);
                Tensor formal = tensor(DataType.FLOAT64, true);
                var body = new ScanBody(List.of(h), List.of(step), List.of(formal),
                        List.of(used ? h.add(formal.cast(DataType.INT64)) : h), List.of());
                Tensor initial = tensor(DataType.INT64, false, carryShape);
                Tensor sequence = tensor(DataType.FLOAT64, false, length ? new long[] {1, 2, 1} : new long[] {1});
                Tensor x = tensor(DataType.FLOAT64, true), u = tensor(DataType.FLOAT64, true);
                Tensor lengths = tensor(DataType.INT64, false, 2), seed = tensor(DataType.FLOAT64, false);
                var scan = length ? Scan.apply(List.of(initial), List.of(sequence), List.of(x), lengths, 1,
                        List.of(OptionalInt.of(0)), body, reverse)
                        : Scan.apply(List.of(initial), List.of(sequence), List.of(x), body, reverse);
                Tensor finalValue = scan.finalCarry().getFirst().cast(DataType.FLOAT64);
                Tensor objective = u.add(length ? finalValue.sum() : finalValue);
                for (var policy : FunctionalGradientRequest.DisconnectedPolicy.values()) {
                    var req = request(objective, seed, x, policy);
                    if (used) {
                        assertRejectedAs(List.of(objective), req, "differentiable");
                    } else if (policy == FunctionalGradientRequest.DisconnectedPolicy.ERROR) {
                        assertRejectedAs(List.of(objective), req, "disconnected");
                    } else {
                        assertDoesNotThrow(() -> SymbolicScanCompilerTest.compile(List.of(objective), Optional.of(req), GraphOptimizationConfig.disabled()));
                        var expansion = expand(List.of(objective), req);
                        var oracle = new ScanGradientRulesTest.Oracle(); oracle.bind(seed, new double[] {.7});
                        oracle.constants(expansion);
                        assertArrayEquals(new double[] {0}, oracle.eval(expansion.targetGradients().getFirst().gradient()), 0);
                    }
                }
            }
    }

    @Test void unusedComputedDiscreteActualDoesNotMakeFloatingIdentityCarryNdConnected() {
        for (boolean reverse : List.of(false, true)) for (boolean length : List.of(false, true))
            for (boolean used : List.of(false, true)) {
                Tensor h = tensor(DataType.FLOAT64, true, 2, 1), step = tensor(DataType.FLOAT64, false, 2, 1);
                Tensor index = tensor(DataType.INT64, false);
                var body = new ScanBody(List.of(h), List.of(step), List.of(index),
                        List.of(used ? h.add(index.cast(DataType.FLOAT64)) : h), List.of());
                Tensor initial = tensor(DataType.FLOAT64, true, 2, 1), sequence = tensor(DataType.FLOAT64, false, 1, 2, 1);
                Tensor x = tensor(DataType.FLOAT64, true), lengths = tensor(DataType.INT64, false, 2);
                Tensor discreteActual = x.cast(DataType.INT64), seed = tensor(DataType.FLOAT64, false, 2, 1);
                var scan = length ? Scan.apply(List.of(initial), List.of(sequence), List.of(discreteActual), lengths, 1,
                        List.of(OptionalInt.of(0)), body, reverse)
                        : Scan.apply(List.of(initial), List.of(sequence), List.of(discreteActual), body, reverse);
                Tensor root = scan.finalCarry().getFirst();
                for (var policy : FunctionalGradientRequest.DisconnectedPolicy.values()) {
                    var req = request(root, seed, x, policy);
                    if (used) {
                        assertRejectedAs(List.of(root), req, "differentiable");
                    } else if (policy == FunctionalGradientRequest.DisconnectedPolicy.ERROR) {
                        assertRejectedAs(List.of(root), req, "disconnected");
                    } else {
                        assertDoesNotThrow(() -> SymbolicScanCompilerTest.compile(List.of(root), Optional.of(req), GraphOptimizationConfig.disabled()));
                        var expansion = expand(List.of(root), req);
                        var oracle = new ScanGradientRulesTest.Oracle(); oracle.bind(seed, new double[] {.3, -.7});
                        oracle.constants(expansion);
                        assertArrayEquals(new double[] {0}, oracle.eval(expansion.targetGradients().getFirst().gradient()), 0);
                    }
                }
            }
    }

    @Test void anUnsupportedSavedAuxiliaryRouteRejectsOnlyWhenSelected() {
        for (boolean reverse : List.of(false, true)) for (int time : new int[] {0, 1, 2}) {
            Tensor x = tensor(DataType.FLOAT64, true, 2, 2), scale = tensor(DataType.FLOAT64, true, 2);
            Tensor bias = tensor(DataType.FLOAT64, true, 2), mean = tensor(DataType.FLOAT64, true, 2), variance = tensor(DataType.FLOAT64, true, 2);
            var normalization = x.batchNormTraining(1, scale, bias, mean, variance,
                    ScalarValue.float64(.1), ScalarValue.float64(1e-5));
            Tensor savedMean = producer(normalization.output()).output(3);
            var body = new ScanBody(List.of(), List.of(x), List.of(scale, bias, mean, variance),
                    List.of(), List.of(x.mul(x), savedMean));
            Tensor sequence = tensor(DataType.FLOAT64, true, time, 2, 2);
            var scan = Scan.apply(List.of(), List.of(sequence), List.of(
                    tensor(DataType.FLOAT64, true, 2), tensor(DataType.FLOAT64, true, 2),
                    tensor(DataType.FLOAT64, true, 2), tensor(DataType.FLOAT64, true, 2)), body, reverse);
            Tensor selected = scan.outputs().getFirst(), unsupported = scan.outputs().getLast();
            var supportedRequest = ScanGradientRulesTest.request(List.of(selected), List.of(SymbolicScanCompilerTest.fresh(selected, false)), List.of(sequence));
            assertDoesNotThrow(() -> SymbolicScanCompilerTest.compile(scan.outputs(), Optional.of(supportedRequest), GraphOptimizationConfig.disabled()));
            var badRequest = ScanGradientRulesTest.request(List.of(unsupported), List.of(SymbolicScanCompilerTest.fresh(unsupported, false)), List.of(sequence));
            if (time > 0) assertThrows(IllegalArgumentException.class, () -> SymbolicScanCompilerTest.compile(
                    scan.outputs(), Optional.of(badRequest), GraphOptimizationConfig.disabled()));
            else assertDoesNotThrow(() -> SymbolicScanCompilerTest.compile(scan.outputs(), Optional.of(badRequest), GraphOptimizationConfig.disabled()));
        }
    }

    @Test void exactMixedTypePairsCastBackWithoutCrossSeedTermsOrDiscreteCotangents() {
        for (boolean reverse : List.of(false, true)) for (boolean length : List.of(false, true)) {
            Tensor h = tensor(DataType.FLOAT32, false, 2, 1), k = tensor(DataType.FLOAT64, false, 2, 1);
            Tensor index = tensor(DataType.INT64, false, 2, 1), flag = tensor(DataType.BOOL, false, 2, 1);
            Tensor x = tensor(DataType.FLOAT32, true, 2, 1), y = tensor(DataType.FLOAT64, true, 2, 1);
            Tensor promoted = h.cast(DataType.FLOAT64).add(y).mul(index.cast(DataType.FLOAT64));
            Tensor nextH = Tensor.where(flag, promoted, promoted.neg()).cast(DataType.FLOAT32).add(x);
            Tensor nextK = k.add(h.cast(DataType.FLOAT64)).add(x.cast(DataType.FLOAT64)).add(y);
            var body = new ScanBody(List.of(h, k, index, flag), List.of(x, y), List.of(),
                    List.of(nextH, nextK, index, flag), List.of());
            Tensor initialH = tensor(DataType.FLOAT32, false, 2, 1), initialK = tensor(DataType.FLOAT64, false, 2, 1);
            Tensor initialIndex = tensor(DataType.INT64, false, 2, 1), initialFlag = tensor(DataType.BOOL, false, 2, 1);
            Tensor sequenceX = tensor(DataType.FLOAT32, true, 1, 2, 1), sequenceY = tensor(DataType.FLOAT64, true, 1, 2, 1);
            Tensor lengths = tensor(DataType.INT64, false, 2);
            var carries = List.of(initialH, initialK, initialIndex, initialFlag);
            var scan = length ? Scan.apply(carries, List.of(sequenceX, sequenceY), List.of(), lengths, 4,
                    List.of(OptionalInt.of(0), OptionalInt.of(0)), body, reverse)
                    : Scan.apply(carries, List.of(sequenceX, sequenceY), List.of(), body, reverse);
            var source = ScanGradientRules.preflightSource(producer(scan.finalCarry().getFirst()));
            var plan = ScanGradientRules.plan(source, new ScanGradientRules.Demand(List.of(
                    new ScanGradientRules.Pair(0, 5), new ScanGradientRules.Pair(1, 4))));
            Tensor seedH = tensor(DataType.FLOAT32, false, 2, 1), seedK = tensor(DataType.FLOAT64, false, 2, 1);
            var seeds = Map.of(0, seedH, 1, seedK);
            var constants = new FirstOrderAutograd.DerivativeConstants();
            Tensor[] gradients = ScanGradientRules.apply(plan, plan.selectedSeedPorts().stream().map(seeds::get).toList(), constants);
            for (int position = 0; position < 4; position++) assertNull(gradients[position]);
            if (length) assertNull(gradients[6]);
            assertEquals(DataType.FLOAT32, gradients[4].descriptor().dataType());
            assertEquals(DataType.FLOAT64, gradients[5].descriptor().dataType());
            var oracle = new ScanGradientRulesTest.Oracle();
            oracle.bind(initialH, new double[] {.7, -.3}); oracle.bind(initialK, new double[] {1.2, .9});
            oracle.bind(initialIndex, new double[] {3, 5}); oracle.bind(initialFlag, new double[] {1, 0});
            oracle.bind(sequenceX, new double[] {.2, .4}); oracle.bind(sequenceY, new double[] {.3, -.6});
            oracle.bind(lengths, new double[] {0, 1}); oracle.bind(seedH, new double[] {.25, -.5});
            oracle.bind(seedK, new double[] {.123456789123, 1.987654321});
            for (var binding : constants.bindings()) oracle.bind(binding.tensor(),
                    new double[] {ScanGradientRulesTest.Oracle.scalar(binding.splat().value())});
            assertArrayEquals(length ? new double[] {0, (double) (float) 1.987654321}
                    : new double[] {(double) (float) .123456789123, (double) (float) 1.987654321},
                    oracle.eval(gradients[4]), 0);
            assertArrayEquals(length ? new double[] {0, 2.5} : new double[] {.75, 2.5}, oracle.eval(gradients[5]), 0);
        }
    }

    @Test void sourceRealUnusedHistoriesAreNotLoadedByDescriptorOnlyDerivativeRules() {
        for (boolean reverse : List.of(false, true)) for (boolean length : List.of(false, true)) {
            Tensor h = tensor(DataType.FLOAT64, false, 3, 1), x = tensor(DataType.FLOAT64, true, 3, 1);
            Tensor edited = x.expandDims(2).squeeze(2).contiguous();
            var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h.add(edited)), List.of());
            Tensor initial = tensor(DataType.FLOAT64, false, 3, 1), sequence = tensor(DataType.FLOAT64, true, 2, 3, 1);
            Tensor lengths = tensor(DataType.INT64, false, 3), seed = tensor(DataType.FLOAT64, false, 3, 1);
            var scan = length ? Scan.apply(List.of(initial), List.of(sequence), List.of(), lengths,
                    1, List.of(OptionalInt.of(0)), body, reverse)
                    : Scan.apply(List.of(initial), List.of(sequence), List.of(), body, reverse);
            var source = ScanGradientRules.preflightSource(producer(scan.finalCarry().getFirst()));
            var plan = ScanGradientRules.plan(source, new ScanGradientRules.Demand(List.of(new ScanGradientRules.Pair(0, 1))));
            var constants = new FirstOrderAutograd.DerivativeConstants();
            Tensor[] gradients = ScanGradientRules.apply(plan, List.of(seed), constants);
            var oracle = new ScanGradientRulesTest.Oracle();
            oracle.bind(lengths, new double[] {0, 1, 2}); oracle.bind(seed, new double[] {2, 3, 5});
            for (var binding : constants.bindings()) oracle.bind(binding.tensor(),
                    new double[] {ScanGradientRulesTest.Oracle.scalar(binding.splat().value())});
            // Original actuals and all source histories are deliberately unbound. A real load
            // fails the fixture evaluator, unlike a membership assertion about reverse inputs.
            assertArrayEquals(length ? new double[] {0, 3, 5, 0, 0, 5} : new double[] {2, 3, 5, 2, 3, 5},
                    oracle.eval(gradients[1]), 0);
            assertFalse(oracle.reads.contains(sequence), "an unused mandatory time anchor must not load its payload");
            assertFalse(oracle.reads.contains(initial), "descriptor-only derivatives must not load initial primals");
            for (Tensor history : scan.residuals()) assertFalse(oracle.reads.contains(history), "unused history payload");
            assertNull(gradients[0]);
        }
    }


    @Test void carryFreeLengthReverseUsesLengthsBatchAndOnlyTheSelectedEmissionSeed() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) for (boolean reverse : List.of(false, true)) {
            Tensor x = tensor(type, true, 3, 1);
            var body = new ScanBody(List.of(), List.of(x), List.of(), List.of(), List.of(x.mul(x), x));
            Tensor sequence = tensor(type, true, 2, 3, 1), lengths = tensor(DataType.INT64, false, 3);
            var scan = Scan.apply(List.of(), List.of(sequence), List.of(), lengths, 0,
                    List.of(OptionalInt.of(0)), body, reverse);
            var source = ScanGradientRules.preflightSource(producer(scan.outputs().getFirst()));
            var plan = ScanGradientRules.plan(source, new ScanGradientRules.Demand(List.of(new ScanGradientRules.Pair(0, 0))));
            Tensor seed = tensor(type, false, 2, 3, 1);
            var constants = new FirstOrderAutograd.DerivativeConstants();
            Tensor[] gradients = ScanGradientRules.apply(plan, List.of(seed), constants);
            assertNull(gradients[1]);
            var oracle = new ScanGradientRulesTest.Oracle();
            oracle.bind(lengths, new double[] {0, 1, 2});
            oracle.bind(sequence, new double[] {Double.NaN, .3, -.7, Double.NaN, Double.NaN, 1.1});
            oracle.bind(seed, new double[] {Double.NaN, -.2, .9, Double.NaN, Double.NaN, -.4});
            for (var binding : constants.bindings()) oracle.bind(binding.tensor(),
                    new double[] {ScanGradientRulesTest.Oracle.scalar(binding.splat().value())});
            double[] expected = new double[6];
            for (int position : new int[] {1, 2, 5}) {
                double term = ScanGradientRulesTest.Oracle.round(sequence, oracle.eval(seed)[position] * oracle.eval(sequence)[position]);
                expected[position] = ScanGradientRulesTest.Oracle.round(sequence, term + term);
            }
            assertArrayEquals(expected, oracle.eval(gradients[0]), 0);
        }
    }

    @Test void sameRowSameLengthsOneStepCutsOnlyTheImpossibleTargetPair() {
        for (boolean reverse : List.of(false, true)) for (boolean layout : List.of(false, true))
            for (boolean otherTarget : List.of(false, true)) {
                Tensor h = tensor(DataType.FLOAT64, true, 3, 1, 1);
                Tensor step = tensor(DataType.FLOAT64, true, 3, 1, 1);
                var aBody = new ScanBody(List.of(h), List.of(step), List.of(),
                        List.of(step.scaledDotProductAttention(step, step)), List.of());
                Tensor initial = tensor(DataType.FLOAT64, true, 3, 1, 1);
                Tensor x = tensor(DataType.FLOAT64, true, 1, 3, 1, 1);
                Tensor lengths = tensor(DataType.INT64, false, 3);
                var a = rowScan(List.of(initial), x, lengths, aBody, reverse);
                Tensor between = a.finalCarry().getFirst();
                if (layout) between = between.expandDims(3).squeeze(3).contiguous();
                Tensor u = tensor(DataType.FLOAT64, true, 3, 1, 1);
                Tensor zStep = tensor(DataType.FLOAT64, false, 3, 1, 1);
                var bBody = new ScanBody(List.of(u), List.of(zStep), List.of(), List.of(zStep), List.of());
                Tensor z = tensor(DataType.FLOAT64, false, 1, 3, 1, 1);
                var b = rowScan(List.of(between), z, lengths, bBody, reverse);
                Tensor objective = b.finalCarry().getFirst().add(x.sum(0, false));
                Tensor seed = tensor(DataType.FLOAT64, false, 3, 1, 1);
                var targets = otherTarget ? List.of(x, initial) : List.of(x);
                var req = ScanGradientRulesTest.request(List.of(objective), List.of(seed), targets);
                assertDoesNotThrow(() -> SymbolicScanCompilerTest.compile(List.of(objective),
                        Optional.of(req), GraphOptimizationConfig.disabled()));
                var expansion = expand(List.of(objective), req);
                for (int mask = 0; mask < 8; mask++) {
                    var oracle = new ScanGradientRulesTest.Oracle();
                    oracle.bind(lengths, new double[] {mask & 1, (mask >> 1) & 1, (mask >> 2) & 1});
                    oracle.bind(seed, new double[] {.25, -.75, 1.5}); oracle.constants(expansion);
                    // No A/B primal or residual is supplied: selecting another target may keep
                    // the shared channel, but must not reconstruct the impossible x pair.
                    assertArrayEquals(new double[] {.25, -.75, 1.5},
                            oracle.eval(expansion.targetGradients().getFirst().gradient()), 0);
                    if (otherTarget) assertArrayEquals(new double[] {
                            (mask & 1) == 0 ? .25 : 0, (mask & 2) == 0 ? -.75 : 0,
                            (mask & 4) == 0 ? 1.5 : 0},
                            oracle.eval(expansion.targetGradients().get(1).gradient()), 0);
                }
            }
    }

    @Test void retainedTwoStepChannelChecksConsumerFcEvenWhereProducerPublishesZero() {
        for (boolean reverse : List.of(false, true)) {
            Tensor h = tensor(DataType.FLOAT64, false, 3, 1, 1);
            Tensor k = tensor(DataType.FLOAT64, false, 3, 1, 1);
            Tensor step = tensor(DataType.FLOAT64, true, 3, 1, 1);
            var aBody = new ScanBody(List.of(h, k), List.of(step), List.of(),
                    List.of(k.add(step.sqrt()), step.scaledDotProductAttention(step, step)), List.of());
            Tensor x = tensor(DataType.FLOAT64, true, 2, 3, 1, 1);
            Tensor lengths = tensor(DataType.INT64, false, 3);
            var a = rowScan(List.of(tensor(DataType.FLOAT64, false, 3, 1, 1),
                    tensor(DataType.FLOAT64, false, 3, 1, 1)), x, lengths, aBody, reverse);
            Tensor u = tensor(DataType.FLOAT64, true, 3, 1, 1);
            Tensor v = tensor(DataType.FLOAT64, true, 3, 1, 1);
            Tensor zStep = tensor(DataType.FLOAT64, false, 3, 1, 1);
            var bBody = new ScanBody(List.of(u, v), List.of(zStep), List.of(), List.of(v, zStep), List.of());
            Tensor aFinal = a.finalCarry().getFirst();
            var b = rowScan(List.of(aFinal, aFinal), tensor(DataType.FLOAT64, false, 2, 3, 1, 1),
                    lengths, bBody, reverse);
            Tensor objective = b.finalCarry().getFirst().add(x.sum(0, false));
            Tensor seed = tensor(DataType.FLOAT64, false, 3, 1, 1);
            for (var policy : FunctionalGradientRequest.DisconnectedPolicy.values()) {
                assertRejectedAs(List.of(objective), request(objective, seed, x, policy), "unsupported");
                // An explicitly represented zero also remains a present seed, not a cut.
                assertRejectedAs(List.of(objective), request(objective, seed.mul(0), x, policy), "unsupported");
            }
        }
    }

    @Test void retainedTwoStepNdObligationNeedsAnOrdinaryBypassAtTheSamePrefix() {
        for (boolean reverse : List.of(false, true)) {
            Tensor a0 = tensor(DataType.FLOAT64, false, 3, 1), a1 = tensor(DataType.FLOAT64, true, 3, 1);
            Tensor unusedStep = tensor(DataType.FLOAT64, false, 3, 1);
            var aBody = new ScanBody(List.of(a0, a1), List.of(unusedStep), List.of(),
                    List.of(a1, a1.cast(DataType.INT64).cast(DataType.FLOAT64)), List.of());
            Tensor independent = tensor(DataType.FLOAT64, false, 3, 1), x = tensor(DataType.FLOAT64, true, 3, 1);
            Tensor sequence = tensor(DataType.FLOAT64, false, 2, 3, 1), lengths = tensor(DataType.INT64, false, 3);
            var a = rowScan(List.of(independent, x), sequence, lengths, aBody, reverse);
            Tensor u = tensor(DataType.FLOAT64, false, 3, 1), v = tensor(DataType.FLOAT64, true, 3, 1);
            Tensor zStep = tensor(DataType.FLOAT64, false, 3, 1);
            var bBody = new ScanBody(List.of(u, v), List.of(zStep), List.of(), List.of(v, zStep), List.of());
            Tensor bInitial = tensor(DataType.FLOAT64, false, 3, 1), z = tensor(DataType.FLOAT64, false, 2, 3, 1);
            var b = rowScan(List.of(bInitial, a.finalCarry().getFirst()), z, lengths, bBody, reverse);
            Tensor selected = b.finalCarry().getFirst(), seed = tensor(DataType.FLOAT64, false, 3, 1);
            Tensor cu = tensor(DataType.FLOAT64, true, 3, 1), cv = tensor(DataType.FLOAT64, true, 3, 1);
            Tensor wStep = tensor(DataType.FLOAT64, false, 3, 1);
            var cBody = new ScanBody(List.of(cu, cv), List.of(wStep), List.of(), List.of(cv, wStep), List.of());
            Tensor derived = x.mul(2);
            var c = rowScan(List.of(derived, derived), sequence, lengths, cBody, reverse);
            Tensor falseRescue = selected.add(c.finalCarry().getFirst()), rescued = selected.add(x);
            for (var policy : FunctionalGradientRequest.DisconnectedPolicy.values()) {
                // A is needed at L=1. At L=2 its bound root is present and reaches only ND.
                assertRejectedAs(List.of(selected), request(selected, seed, x, policy), "non-differentiable");
                // C has a genuine D route at L=1, but its L=2 published zero cannot rescue A.
                assertRejectedAs(List.of(falseRescue), request(falseRescue, seed, x, policy), "non-differentiable");
                var req = request(rescued, seed, x, policy);
                assertDoesNotThrow(() -> SymbolicScanCompilerTest.compile(List.of(rescued),
                        Optional.of(req), GraphOptimizationConfig.disabled()));
                var expansion = expand(List.of(rescued), req);
                for (int l0 = 0; l0 <= 2; l0++) for (int l1 = 0; l1 <= 2; l1++)
                    for (int l2 = 0; l2 <= 2; l2++) {
                        var oracle = new ScanGradientRulesTest.Oracle();
                        oracle.bind(independent, new double[] {.1, .2, .3});
                        oracle.bind(x, new double[] {1.2, -2.7, 3.4});
                        oracle.bind(bInitial, new double[] {.4, .5, .6});
                        oracle.bind(sequence, new double[] {1, 2, 3, 4, 5, 6});
                        oracle.bind(z, new double[] {-1, -2, -3, -4, -5, -6});
                        oracle.bind(lengths, new double[] {l0, l1, l2});
                        oracle.bind(seed, new double[] {.25, -.75, 1.5}); oracle.constants(expansion);
                        assertArrayEquals(new double[] {.25 * (l0 == 1 ? 2 : 1),
                                -.75 * (l1 == 1 ? 2 : 1), 1.5 * (l2 == 1 ? 2 : 1)},
                                oracle.eval(expansion.targetGradients().getFirst().gradient()), 0);
                    }
            }
        }
    }

    @Test void retainedLocalDifferentiablePairNeedsATargetSuffixAtTheSamePrefix() {
        for (boolean reverse : List.of(false, true)) for (boolean excludedSuffix : List.of(false, true))
            for (boolean nonDifferentiable : List.of(false, true)) {
            Tensor x = tensor(DataType.FLOAT64, true, 3, 1);
            Tensor lengths = tensor(DataType.INT64, false, 3);
            Tensor c0 = tensor(DataType.FLOAT64, !excludedSuffix, 3, 1);
            Tensor c1 = tensor(DataType.FLOAT64, true, 3, 1);
            Tensor wStep = tensor(DataType.FLOAT64, false, 3, 1);
            Tensor independentC = tensor(DataType.FLOAT64, false, 3, 1);
            ScanBody cBody;
            List<Tensor> cInitial;
            if (excludedSuffix) {
                Tensor c2 = tensor(DataType.FLOAT64, true, 3, 1);
                cBody = new ScanBody(List.of(c0, c1, c2), List.of(wStep), List.of(),
                        List.of(c1, c2, wStep), List.of());
                cInitial = List.of(independentC, x, x.mul(2));
            } else {
                cBody = new ScanBody(List.of(c0, c1), List.of(wStep), List.of(), List.of(c1, wStep), List.of());
                cInitial = List.of(x, x);
            }
            Tensor w = tensor(DataType.FLOAT64, false, 2, 3, 1);
            var c = rowScan(cInitial, w, lengths, cBody, reverse);
            Tensor p = c.finalCarry().getFirst();
            Tensor a0 = tensor(DataType.FLOAT64, false, 3, 1), a1 = tensor(DataType.FLOAT64, true, 3, 1);
            Tensor a2 = tensor(DataType.FLOAT64, true, 3, 1), unusedStep = tensor(DataType.FLOAT64, false, 3, 1);
            Tensor nextA1 = nonDifferentiable ? a1.cast(DataType.INT64).cast(DataType.FLOAT64) : unusedStep;
            var aBody = new ScanBody(List.of(a0, a1, a2), List.of(unusedStep), List.of(),
                    List.of(a1.add(a2), nextA1, a2), List.of());
            Tensor independent = tensor(DataType.FLOAT64, false, 3, 1);
            Tensor sequence = tensor(DataType.FLOAT64, false, 2, 3, 1);
            var a = rowScan(List.of(independent, x, p), sequence, lengths, aBody, reverse);
            Tensor u = tensor(DataType.FLOAT64, false, 3, 1), v = tensor(DataType.FLOAT64, true, 3, 1);
            Tensor zStep = tensor(DataType.FLOAT64, false, 3, 1);
            var bBody = new ScanBody(List.of(u, v), List.of(zStep), List.of(), List.of(v, zStep), List.of());
            Tensor bInitial = tensor(DataType.FLOAT64, false, 3, 1), z = tensor(DataType.FLOAT64, false, 2, 3, 1);
            var b = rowScan(List.of(bInitial, a.finalCarry().getFirst()), z, lengths, bBody, reverse);
            Tensor selected = b.finalCarry().getFirst(), seed = tensor(DataType.FLOAT64, false, 3, 1);
            Tensor rescued = selected.add(x);
            for (var policy : FunctionalGradientRequest.DisconnectedPolicy.values())
                for (boolean bypass : List.of(false, true)) {
                // L=1 retains both A inputs. At L=2 A's materialized seed remains present;
                // its p suffix is either disconnected or uses frozen-out C/derived→x edges.
                // Only a genuine ND route can create an ND obligation, not an excluded D route.
                if (nonDifferentiable && !bypass) {
                    assertRejectedAs(List.of(selected), request(selected, seed, x, policy), "non-differentiable");
                    assertRejectedAs(List.of(selected), request(selected, seed.mul(0), x, policy), "non-differentiable");
                    continue;
                }
                Tensor objective = bypass ? rescued : selected;
                var req = request(objective, seed, x, policy);
                assertDoesNotThrow(() -> SymbolicScanCompilerTest.compile(List.of(objective),
                        Optional.of(req), GraphOptimizationConfig.disabled()));
                var expansion = expand(List.of(objective), req);
                double ordinaryContribution = bypass ? 1 : 0;
                for (int l0 = 0; l0 <= 2; l0++) for (int l1 = 0; l1 <= 2; l1++)
                    for (int l2 = 0; l2 <= 2; l2++) {
                        var oracle = new ScanGradientRulesTest.Oracle();
                        oracle.bind(x, new double[] {1.2, -2.7, 3.4});
                        oracle.bind(independent, new double[] {.1, .2, .3});
                        oracle.bind(independentC, new double[] {.7, .8, .9});
                        oracle.bind(bInitial, new double[] {.4, .5, .6});
                        oracle.bind(w, new double[] {2, 3, 4, 5, 6, 7});
                        oracle.bind(sequence, new double[] {1, 2, 3, 4, 5, 6});
                        oracle.bind(z, new double[] {-1, -2, -3, -4, -5, -6});
                        oracle.bind(lengths, new double[] {l0, l1, l2});
                        oracle.bind(seed, new double[] {.25, -.75, 1.5}); oracle.constants(expansion);
                        // Both retained inputs contribute at L=1. Otherwise the result is
                        // disconnected zero, with only the optional ordinary bypass remaining.
                        assertArrayEquals(new double[] {.25 * ((l0 == 1 ? 2 : 0) + ordinaryContribution),
                                -.75 * ((l1 == 1 ? 2 : 0) + ordinaryContribution),
                                1.5 * ((l2 == 1 ? 2 : 0) + ordinaryContribution)},
                                oracle.eval(expansion.targetGradients().getFirst().gradient()), 0);
                    }
            }
        }
    }

    @Test void independentLengthsAndCrossRowOrdinaryMappingsCannotProveAUniformCut() {
        for (boolean reverse : List.of(false, true)) for (String mapping : List.of("direct", "gather", "reduce", "matmul"))
            for (boolean supported : List.of(false, true)) {
                Tensor h = tensor(DataType.FLOAT64, false, 3, 1, 1);
                Tensor step = tensor(DataType.FLOAT64, true, 3, 1, 1);
                Tensor next = supported ? step.mul(step) : step.scaledDotProductAttention(step, step);
                var aBody = new ScanBody(List.of(h), List.of(step), List.of(), List.of(next), List.of());
                Tensor x = tensor(DataType.FLOAT64, true, 1, 3, 1, 1);
                Tensor aLengths = tensor(DataType.INT64, false, 3);
                var a = rowScan(List.of(tensor(DataType.FLOAT64, false, 3, 1, 1)), x, aLengths, aBody, reverse);
                Tensor between = a.finalCarry().getFirst();
                between = switch (mapping) {
                    case "direct" -> between;
                    case "gather" -> between.gather(tensor(DataType.INT64, false, 3), 0);
                    case "reduce" -> between.sum(0, true).expand(3, 1, 1);
                    case "matmul" -> tensor(DataType.FLOAT64, false, 3, 3)
                            .matmul(between.reshape(3, 1)).reshape(3, 1, 1);
                    default -> throw new AssertionError(mapping);
                };
                Tensor u = tensor(DataType.FLOAT64, true, 3, 1, 1);
                Tensor zStep = tensor(DataType.FLOAT64, false, 3, 1, 1);
                var bBody = new ScanBody(List.of(u), List.of(zStep), List.of(), List.of(zStep), List.of());
                // Direct rows remain compatible when their two length declarations differ.
                // With shared lengths, each mixing expression permits A-row != B-row.
                Tensor bLengths = mapping.equals("direct") ? tensor(DataType.INT64, false, 3) : aLengths;
                var b = rowScan(List.of(between), tensor(DataType.FLOAT64, false, 1, 3, 1, 1),
                        bLengths, bBody, reverse);
                Tensor objective = b.finalCarry().getFirst().add(x.sum(0, false));
                Tensor seed = tensor(DataType.FLOAT64, false, 3, 1, 1);
                var req = request(objective, seed, x, FunctionalGradientRequest.DisconnectedPolicy.ERROR);
                if (supported) {
                    assertDoesNotThrow(() -> SymbolicScanCompilerTest.compile(List.of(objective),
                            Optional.of(req), GraphOptimizationConfig.disabled()));
                    assertDoesNotThrow(() -> expand(List.of(objective), req));
                } else {
                    // The paired supported expressions ensure this is an ordinary VJP failure,
                    // not rejection of gather, reduction, matmul, or unknown row correspondence.
                    assertRejectedAs(List.of(objective), req, "unsupported");
                }
            }
    }

    @Test void publishedZeroAndExplicitZeroBothExecuteASingularConsumingReverse() {
        for (boolean reverse : List.of(false, true)) {
            Tensor h = tensor(DataType.FLOAT64, false, 3, 1), k = tensor(DataType.FLOAT64, false, 3, 1);
            Tensor step = tensor(DataType.FLOAT64, true, 3, 1), quotient = step.div(step);
            var aBody = new ScanBody(List.of(h, k), List.of(step), List.of(),
                    List.of(k.add(quotient), quotient), List.of());
            Tensor initialH = tensor(DataType.FLOAT64, false, 3, 1), initialK = tensor(DataType.FLOAT64, false, 3, 1);
            Tensor x = tensor(DataType.FLOAT64, true, 2, 3, 1), lengths = tensor(DataType.INT64, false, 3);
            var a = rowScan(List.of(initialH, initialK), x, lengths, aBody, reverse);
            Tensor u = tensor(DataType.FLOAT64, true, 3, 1), v = tensor(DataType.FLOAT64, true, 3, 1);
            Tensor zStep = tensor(DataType.FLOAT64, false, 3, 1);
            var bBody = new ScanBody(List.of(u, v), List.of(zStep), List.of(), List.of(v, zStep), List.of());
            Tensor aFinal = a.finalCarry().getFirst(), z = tensor(DataType.FLOAT64, false, 2, 3, 1);
            var b = rowScan(List.of(aFinal, aFinal), z, lengths, bBody, reverse);
            Tensor seed = tensor(DataType.FLOAT64, false, 3, 1);
            Tensor explicitZero = tensor(DataType.FLOAT64, false, 3, 1);
            var publishedRequest = request(b.finalCarry().getFirst(), seed, x,
                    FunctionalGradientRequest.DisconnectedPolicy.ERROR);
            var explicitRequest = request(aFinal, explicitZero, x, FunctionalGradientRequest.DisconnectedPolicy.ERROR);
            assertDoesNotThrow(() -> SymbolicScanCompilerTest.compile(List.of(b.finalCarry().getFirst()),
                    Optional.of(publishedRequest), GraphOptimizationConfig.disabled()));
            assertDoesNotThrow(() -> SymbolicScanCompilerTest.compile(List.of(aFinal),
                    Optional.of(explicitRequest), GraphOptimizationConfig.disabled()));
            var published = expand(List.of(b.finalCarry().getFirst()), publishedRequest);
            var explicit = expand(List.of(aFinal), explicitRequest);
            for (var expansion : List.of(published, explicit)) {
                var oracle = new ScanGradientRulesTest.Oracle();
                oracle.bind(initialH, new double[] {1, 2, 3}); oracle.bind(initialK, new double[] {4, 5, 6});
                oracle.bind(x, new double[6]); oracle.bind(z, new double[6]);
                oracle.bind(lengths, new double[] {2, 2, 2});
                oracle.bind(seed, new double[] {.25, -.75, 1.5}); oracle.bind(explicitZero, new double[3]);
                oracle.constants(expansion);
                // B's selected initial gradient is numerically zero at L=2. It is nevertheless
                // a bound A root, just like the explicit zero seed: 0/0 must not become absence.
                for (double value : oracle.eval(expansion.targetGradients().getFirst().gradient()))
                    assertTrue(Double.isNaN(value));
            }
        }
    }

    /**
     * Constructs a genuine length-aware occurrence with ROW(0) carries and one ROW(0) sequence.
     *
     * @param initial ordered actual carries, matching every formal's dtype, shape and gradient flag
     * @param sequence actual time-stacked sequence matching the sole step formal
     * @param lengths independent or shared non-gradient INT64 batch-length declaration
     * @param body ordinary source body with only row carries
     * @param reverse whether traversal decreases original time
     * @return the actual occurrence's canonical ports
     */
    private static ScanResult rowScan(List<Tensor> initial, Tensor sequence, Tensor lengths,
            ScanBody body, boolean reverse) {
        return Scan.apply(initial, List.of(sequence), List.of(), lengths, initial.size(),
                List.of(OptionalInt.of(0)), body, reverse);
    }

    private static TensorProducer producer(Tensor value) { return value.provenance().orElseThrow().producer(); }

    private static FirstOrderAutograd.Expansion expand(List<Tensor> roots, FunctionalGradientRequest request) {
        return FirstOrderAutograd.expand(AutogradPreflight.preflight(CompileMode.FORWARD_AND_BACKWARD,
                roots, request, CompileTimeConstantGraph.Ingress.empty()).stageOne(), CompileTimeConstantGraph.Ingress.empty());
    }

    private static void assertRejectedAs(List<Tensor> roots, FunctionalGradientRequest request, String reason) {
        var compileFailure = assertThrows(IllegalArgumentException.class, () -> SymbolicScanCompilerTest.compile(
                roots, Optional.of(request), GraphOptimizationConfig.disabled()));
        assertTrue(compileFailure.getMessage().contains(reason), compileFailure::getMessage);
        var expansionFailure = assertThrows(IllegalArgumentException.class, () -> expand(roots, request));
        assertTrue(expansionFailure.getMessage().contains(reason), expansionFailure::getMessage);
    }

    private static FunctionalGradientRequest request(Tensor root, Tensor seed, Tensor target,
            FunctionalGradientRequest.DisconnectedPolicy policy) {
        return request(List.of(root), List.of(seed), target, policy);
    }

    private static FunctionalGradientRequest request(List<Tensor> roots, List<Tensor> seeds, Tensor target,
            FunctionalGradientRequest.DisconnectedPolicy policy) {
        return new FunctionalGradientRequest(List.of(new FunctionalGradientRequest.Stage(
                roots.stream().map(t -> (FunctionalGradientRequest.OutputReference)
                        new FunctionalGradientRequest.ForwardTensorReference(t)).toList(),
                seeds.stream().map(Optional::of).toList(), List.of(target), false, policy)));
    }
}
