package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.*;
import static io.github.pho001.synaptik.compiler.ScanBodyInferenceTest.tensor;
import static io.github.pho001.synaptik.compiler.SymbolicScanCompilerTest.*;

import io.github.pho001.synaptik.config.compile.*;
import io.github.pho001.synaptik.model.datatype.*;
import io.github.pho001.synaptik.model.tensor.*;
import java.util.*;
import org.junit.jupiter.api.Test;

final class SymbolicScanGradientRulesTest {
    @Test void tensorScalarArithmeticAndTransposedSumRestorationCloseAtActualBatch() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) for (boolean reverse : List.of(false, true)) {
            Tensor h = symbolic(type, false, B, fixed(2)), x = symbolic(type, true, B, fixed(2));
            Tensor w = tensor(type, true, 2);
            Tensor next = algebra(h, x, w);
            var body = new ScanBody(List.of(h), List.of(x), List.of(w), List.of(next), List.of(next.sum(0, false)));
            Tensor initial = fresh(h, false), seq = symbolic(type, true, fixed(3), B, fixed(2)), weight = fresh(w, true);
            var scan = Scan.apply(List.of(initial), List.of(seq), List.of(weight), body, reverse);
            Tensor objective = scan.finalCarry().getFirst().sum().add(scan.outputs().getFirst().sum());
            var request = FunctionalGradientTestSupport.request(objective, List.of(seq, weight));
            var expansion = FirstOrderAutograd.expand(AutogradPreflight.preflight(CompileMode.FORWARD_AND_BACKWARD,
                    List.of(objective), request, CompileTimeConstantGraph.Ingress.empty()).stageOne(), CompileTimeConstantGraph.Ingress.empty());
            var artifacts = compile(List.of(objective), Optional.of(request), GraphOptimizationConfig.standard());
            var plan = BatchBindingPlan.prepare(artifacts);
            for (int batch : new int[] {0, 1, 3}) {
                plan.bind(artifacts.constants().bindableInputs().stream().map(id -> actual(
                        artifacts.graph().values().get(index(artifacts, id)).descriptor(), batch, false)).toList());
                var oracle = new ScanGradientRulesTest.Oracle(batch);
                oracle.bind(initial, data(batch * 2, .05)); oracle.bind(seq, data(3 * batch * 2, -.07));
                oracle.bind(weight, data(2, .1)); oracle.constants(expansion);
                var reference = new ScanGradientRulesTest.Oracle();
                Tensor state = tensor(type, false, batch, 2), rw = tensor(type, true, 2);
                reference.bind(state, oracle.eval(initial)); reference.bind(rw, oracle.eval(weight));
                var steps = new ArrayList<Tensor>();
                for (int t = 0; t < 3; t++) {
                    Tensor step = tensor(type, true, batch, 2); steps.add(step);
                    reference.bind(step, Arrays.copyOfRange(oracle.eval(seq), t * batch * 2, (t + 1) * batch * 2));
                }
                Tensor loss = null;
                for (int iteration = 0; iteration < 3; iteration++) {
                    int t = reverse ? 2 - iteration : iteration;
                    state = algebra(state, steps.get(t), rw);
                    loss = loss == null ? state.sum(0, false).sum() : loss.add(state.sum(0, false).sum());
                }
                loss = state.sum().add(loss);
                var targets = new ArrayList<>(steps); targets.add(rw);
                var ordinary = FirstOrderAutograd.expand(AutogradPreflight.preflight(CompileMode.FORWARD_AND_BACKWARD,
                        List.of(loss), FunctionalGradientTestSupport.stage(loss, targets), CompileTimeConstantGraph.Ingress.empty()), CompileTimeConstantGraph.Ingress.empty());
                reference.constants(ordinary);
                double tolerance = type == DataType.FLOAT32 ? 2e-5 : 1e-12;
                assertArrayEquals(reference.eval(loss), oracle.eval(objective), tolerance);
                assertArrayEquals(reference.eval(ordinary.targetGradients().getLast().gradient()),
                        oracle.eval(expansion.targetGradients().getLast().gradient()), tolerance);
                double[] dx = oracle.eval(expansion.targetGradients().getFirst().gradient());
                for (int t = 0; t < 3; t++) assertArrayEquals(reference.eval(ordinary.targetGradients().get(t).gradient()),
                        Arrays.copyOfRange(dx, t * batch * 2, (t + 1) * batch * 2), tolerance);
            }
        }
    }

    private static Tensor algebra(Tensor h, Tensor x, Tensor w) {
        DataType type = h.descriptor().dataType();
        ScalarValue two = type == DataType.FLOAT32 ? ScalarValue.float32(2) : ScalarValue.float64(2);
        Tensor a = h.add(x).sub(w).mul(w.add(two)).div(w.add(two)).neg().add(two).sub(two).mul(two).div(two);
        Tensor shared = a.permute(1, 0).sum(1, false);
        return a.add(shared.expandDims(0).expand(h.descriptor().shape()).contiguous());
    }

    @Test void repeatedInvariantRolesReusedTemplatesAndCrossOccurrenceSumsKeepFalseCarryFlow() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            Tensor h = symbolic(type, false, B, fixed(2)), x = symbolic(type, true, B, fixed(2));
            Tensor w = tensor(type, true, 2, 2), u = tensor(type, true, 2, 2);
            Tensor transpose = u.permute(1, 0).contiguous();
            Tensor next = h.matmul(w).add(x.matmul(transpose)).tanh();
            var body = new ScanBody(List.of(h), List.of(x), List.of(w, u), List.of(next), List.of(h, next, next));
            Tensor initial = fresh(h, false), seq = symbolic(type, true, fixed(3), B, fixed(2)), weight = fresh(w, true);
            var a = Scan.apply(List.of(initial), List.of(seq), List.of(weight, weight), body, false);
            var b = Scan.apply(List.of(initial), List.of(seq), List.of(weight, weight), body, false);
            Tensor objective = a.finalCarry().getFirst().sum().add(a.outputs().get(0).sum())
                    .add(a.outputs().get(1).sum()).add(a.outputs().get(2).sum()).add(b.finalCarry().getFirst().sum());
            var req = FunctionalGradientTestSupport.request(objective, List.of(seq, weight));
            var expansion = FirstOrderAutograd.expand(AutogradPreflight.preflight(CompileMode.FORWARD_AND_BACKWARD,
                    List.of(objective), req, CompileTimeConstantGraph.Ingress.empty()).stageOne(), CompileTimeConstantGraph.Ingress.empty());
            for (var optimization : List.of(GraphOptimizationConfig.disabled(), GraphOptimizationConfig.standard())) {
                var artifacts = compile(List.of(objective), Optional.of(req), optimization);
                var plan = BatchBindingPlan.prepare(artifacts);
                for (int batch : new int[] {0, 1, 2, 5}) {
                    plan.bind(artifacts.constants().bindableInputs().stream().map(id -> actual(
                            artifacts.graph().values().get(index(artifacts, id)).descriptor(), batch, false)).toList());
                }
            }
            for (int batch : new int[] {0, 1, 2, 5}) {
                var oracle = new ScanGradientRulesTest.Oracle(batch);
                oracle.bind(initial, data(batch * 2, .17)); oracle.bind(seq, data(3 * batch * 2, -.12));
                oracle.bind(weight, data(4, .09)); oracle.constants(expansion);
                var reference = new ScanGradientRulesTest.Oracle();
                Tensor ih = tensor(type, false, batch, 2), rw = tensor(type, true, 2, 2);
                reference.bind(ih, oracle.eval(initial)); reference.bind(rw, oracle.eval(weight));
                var steps = new ArrayList<Tensor>(); Tensor state = ih, loss = null;
                for (int t = 0; t < 3; t++) {
                    Tensor step = tensor(type, true, batch, 2); steps.add(step);
                    reference.bind(step, Arrays.copyOfRange(oracle.eval(seq), t * batch * 2, (t + 1) * batch * 2));
                    Tensor update = state.matmul(rw).add(step.matmul(rw.permute(1, 0).contiguous())).tanh();
                    Tensor term = state.sum().add(update.sum()).add(update.sum());
                    loss = loss == null ? term : loss.add(term); state = update;
                }
                loss = state.sum().add(loss).add(state.sum());
                var targets = new ArrayList<>(steps); targets.add(rw);
                var unrolled = FirstOrderAutograd.expand(AutogradPreflight.preflight(CompileMode.FORWARD_AND_BACKWARD,
                        List.of(loss), FunctionalGradientTestSupport.stage(loss, targets), CompileTimeConstantGraph.Ingress.empty()), CompileTimeConstantGraph.Ingress.empty());
                reference.constants(unrolled);
                double tolerance = type == DataType.FLOAT32 ? 2e-5 : 1e-12;
                assertArrayEquals(reference.eval(loss), oracle.eval(objective), tolerance);
                assertArrayEquals(reference.eval(unrolled.targetGradients().getLast().gradient()),
                        oracle.eval(expansion.targetGradients().getLast().gradient()), tolerance);
                double[] dx = oracle.eval(expansion.targetGradients().getFirst().gradient());
                for (int t = 0; t < 3; t++) assertArrayEquals(reference.eval(unrolled.targetGradients().get(t).gradient()),
                        Arrays.copyOfRange(dx, t * batch * 2, (t + 1) * batch * 2), tolerance);
            }
        }
    }

    @Test void zeroBatchDoesNotSkipNonemptySharedCarryEmissionAndParameterWork() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (int time : new int[] {0, 3}) for (boolean reverse : List.of(false, true)) {
                Tensor h = symbolic(type, true, B, fixed(2)), x = fresh(h, true);
                Tensor q = tensor(type, true, 2, 2), w = fresh(q, true);
                var body = new ScanBody(List.of(h, q), List.of(x), List.of(w), List.of(h.add(x), q.add(w)), List.of(q));
                Tensor initial = fresh(h, true), shared = fresh(q, true), weight = fresh(w, true);
                Tensor seq = symbolic(type, true, fixed(time), B, fixed(2));
                var scan = Scan.apply(List.of(initial, shared), List.of(seq), List.of(weight), body, reverse);
                Tensor objective = scan.finalCarry().get(0).sum().add(scan.finalCarry().get(1).sum());
                var targets = List.of(initial, shared, seq, weight);
                var strictRequest = FunctionalGradientTestSupport.request(objective, targets);
                if (time == 0) {
                    // SUM(initial)+SUM(shared) has no sequence or weight route.
                    assertThrows(IllegalArgumentException.class, () -> AutogradPreflight.preflight(
                            CompileMode.FORWARD_AND_BACKWARD, List.of(objective), strictRequest,
                            CompileTimeConstantGraph.Ingress.empty()));
                }
                var req = time == 0 ? FunctionalGradientTestSupport.request(objective, targets,
                        FunctionalGradientRequest.DisconnectedPolicy.ZERO) : strictRequest;
                var expansion = FirstOrderAutograd.expand(AutogradPreflight.preflight(CompileMode.FORWARD_AND_BACKWARD,
                        List.of(objective), req, CompileTimeConstantGraph.Ingress.empty()).stageOne(), CompileTimeConstantGraph.Ingress.empty());
                var oracle = new ScanGradientRulesTest.Oracle(0);
                oracle.bind(initial, new double[0]); oracle.bind(seq, new double[0]);
                oracle.bind(shared, new double[] {-0., 1, 2, 3}); oracle.bind(weight, new double[] {1, 1, 1, 1}); oracle.constants(expansion);
                double[] expected = {-0., 1, 2, 3};
                for (int t = 0; t < time; t++) for (int i = 0; i < 4; i++) expected[i] += 1;
                assertArrayEquals(expected, oracle.eval(scan.finalCarry().get(1)), 0);
                if (time == 0) assertEquals(Double.doubleToRawLongBits(-0.),
                        Double.doubleToRawLongBits(oracle.eval(scan.finalCarry().get(1))[0]));
                assertEquals(time * 4, oracle.eval(scan.outputs().getFirst()).length);
                assertArrayEquals(new double[] {1, 1, 1, 1}, oracle.eval(expansion.targetGradients().get(1).gradient()), 0);
                assertArrayEquals(new double[] {time, time, time, time}, oracle.eval(expansion.targetGradients().getLast().gradient()), 0);
                assertEquals(0, oracle.eval(expansion.targetGradients().get(2).gradient()).length);
            }
    }

    @Test void residualAndHigherOrderRequestsReject() {
        Tensor h = symbolic(DataType.FLOAT64, true, B, fixed(2)), x = fresh(h, true);
        Tensor initial = fresh(h, true), seq = symbolic(DataType.FLOAT64, true, fixed(0), B, fixed(2));
        var scan = Scan.apply(List.of(initial), List.of(seq), List.of(),
                new ScanBody(List.of(h), List.of(x), List.of(), List.of(h.add(x).tanh()), List.of()), false);
        Tensor residual = scan.residuals().getFirst().sum();
        assertThrows(IllegalArgumentException.class, () -> GraphCompiler.compile(CompileMode.FORWARD_AND_BACKWARD, List.of(residual),
                Optional.of(FunctionalGradientTestSupport.request(residual, List.of(seq))), CompileTimeConstantGraph.Ingress.empty(), GraphOptimizationConfig.disabled()));
        var first = new FunctionalGradientRequest.Stage(List.of(new FunctionalGradientRequest.ForwardTensorReference(scan.finalCarry().getFirst())),
                List.of(Optional.of(fresh(h, false))), List.of(seq), true, FunctionalGradientRequest.DisconnectedPolicy.ZERO);
        var second = new FunctionalGradientRequest.Stage(List.of(new FunctionalGradientRequest.FirstStageGradientReference(0)),
                List.of(Optional.of(fresh(seq, false))), List.of(seq), false, FunctionalGradientRequest.DisconnectedPolicy.ZERO);
        assertThrows(IllegalArgumentException.class, () -> GraphCompiler.compile(CompileMode.FORWARD_AND_BACKWARD, scan.finalCarry(), Optional.of(new FunctionalGradientRequest(List.of(first, second))),
                CompileTimeConstantGraph.Ingress.empty(), GraphOptimizationConfig.disabled()));
    }

}
