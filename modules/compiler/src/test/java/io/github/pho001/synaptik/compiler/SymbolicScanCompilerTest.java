package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.*;
import static io.github.pho001.synaptik.compiler.ScanBodyInferenceTest.tensor;

import io.github.pho001.synaptik.backend.contract.*;
import io.github.pho001.synaptik.config.compile.*;
import io.github.pho001.synaptik.model.datatype.*;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.*;
import io.github.pho001.synaptik.model.tensor.*;
import io.github.pho001.synaptik.planning.capability.*;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Tests metadata and the actual symbolic Tensor derivative expressions, never a backend runner. */
final class SymbolicScanCompilerTest {
    static final DynamicDimension B = new DynamicDimension("Rows");

    static Shape shape(Dimension... dims) { return Shape.ofDimensions(dims); }
    static StaticDimension fixed(long size) { return new StaticDimension(size); }
    static Tensor symbolic(DataType type, boolean grad, Dimension... dims) {
        return TensorFactory.create(new TensorDescriptor(type, shape(dims), Optional.empty(), grad));
    }
    static Tensor fresh(Tensor tensor, boolean grad) {
        var d = tensor.descriptor();
        return TensorFactory.create(new TensorDescriptor(d.dataType(), d.shape(), d.layout(), grad));
    }

    static TensorDescriptor actual(TensorDescriptor template, long batch, boolean view) {
        if (template.shape().isFullyStatic()) return template;
        Shape shape = Shape.of(template.shape().dimensions().stream()
                .mapToLong(d -> d.staticSize().orElse(batch)).toArray());
        var canonical = LayoutDescriptor.contiguous(shape);
        return new TensorDescriptor(template.dataType(), shape,
                Optional.of(LayoutDescriptor.of(shape, canonical.strides(), 0, view)), template.requiresGrad());
    }

    static CompileArtifacts compile(List<Tensor> roots, Optional<FunctionalGradientRequest> request,
            GraphOptimizationConfig optimization) {
        return compile(roots, request, optimization, CompileTimeConstantGraph.Ingress.empty());
    }

    static CompileArtifacts compile(List<Tensor> roots, Optional<FunctionalGradientRequest> request,
            GraphOptimizationConfig optimization, CompileTimeConstantGraph.Ingress ingress) {
        BackendId id = new BackendId("symbolic-scan-metadata-test-only");
        BackendCapabilityProvider provider = new BackendCapabilityProvider() {
            public BackendId backendId() { return id; }
            public boolean supports(OperationCapabilityQuery query) { return true; }
        };
        return GraphCompiler.compile(request.isPresent() ? CompileMode.FORWARD_AND_BACKWARD : CompileMode.FORWARD_ONLY,
                roots, request, ingress, optimization, BackendIntent.unconstrained(), PartitionScoringConfig.neutral(),
                List.of(provider), List.of(new BackendAvailabilitySnapshot(id,
                        Map.of(new BackendDeviceId(id, "metadata-only"), DeviceClass.CPU))));
    }

    @Test void actualReverseMatchesIndependentNonuniformTensorUnrollingAcrossBatchReuse() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (boolean reverse : List.of(false, true))
                for (boolean emissionOnly : List.of(false, true)) compare(type, reverse, emissionOnly, false, 3);
    }

    @Test void actualMeanCountAndSeparateZeroTimePlansMatchOrdinaryUnrolling() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (boolean reverse : List.of(false, true)) {
                compare(type, reverse, false, true, 3);
                compare(type, reverse, false, false, 0);
                compare(type, reverse, false, true, 0);
            }
    }

    @Test void outerRankThreePointwiseLayoutClosureAndExactSavedActivationRemainSymbolic() {
        Tensor h = symbolic(DataType.FLOAT64, true, B, fixed(2)), x = fresh(h, true);
        Tensor activated = h.add(x).tanh();
        var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(activated), List.of(activated));
        Tensor initial = fresh(h, true), sequence = symbolic(DataType.FLOAT64, true, fixed(2), B, fixed(2));
        var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(), body, false);
        Tensor seedBase = fresh(scan.outputs().getFirst(), false);
        Tensor seed = seedBase.add(ScalarValue.float64(.1)).expandDims(3).squeeze(3)
                .permute(1, 0, 2).contiguous().reshape(shape(B, fixed(4)))
                .reshape(shape(B, fixed(2), fixed(2))).permute(1, 0, 2).contiguous();
        var request = ScanGradientRulesTest.request(scan.outputs(), List.of(seed), List.of(initial, sequence));
        var artifacts = compile(scan.outputs(), Optional.of(request), GraphOptimizationConfig.standard());
        var plan = BatchBindingPlan.prepare(artifacts);
        assertEquals(2, plan.bind(artifacts.constants().bindableInputs().stream().map(id -> actual(
                artifacts.graph().values().get(index(artifacts, id)).descriptor(), 2, false)).toList()).batchSize());
        var constants = new FirstOrderAutograd.DerivativeConstants();
        Tensor[] gradients = ScanGradientRulesTest.applySelected(scan.finalCarry().getFirst().provenance().orElseThrow().producer(),
                Map.of(1, seed), List.of(0, 1), constants);
        var oracle = new ScanGradientRulesTest.Oracle(2);
        oracle.bind(initial, data(4, .2)); oracle.bind(sequence, data(8, -.1)); oracle.bind(seedBase, data(8, .3));
        oracle.eval(scan.finalCarry().getFirst());
        double[] saved = {.25, -.0, .5, -.25, .1, .2, -.3, .4};
        oracle.bind(scan.residuals().get(body.retainedValues().indexOf(activated)), saved);
        for (var constant : constants.bindings()) oracle.bind(constant.tensor(), new double[] {ScanGradientRulesTest.Oracle.scalar(constant.splat().value())});
        double[] expected = new double[4], s = oracle.eval(seed);
        for (int i = 0; i < 4; i++) expected[i] = (s[i] + s[i + 4] * (1 - saved[i + 4] * saved[i + 4])) * (1 - saved[i] * saved[i]);
        assertArrayEquals(expected, oracle.eval(gradients[0]), 1e-14);
    }

    private static void compare(DataType type, boolean reverse, boolean emissionOnly, boolean mean, int time) {
        int f = 2, width = 3;
        Tensor h = symbolic(type, true, B, fixed(width)), x = symbolic(type, true, B, fixed(f));
        Tensor wx = tensor(type, true, f, width), wh = tensor(type, true, width, width), bias = tensor(type, true, width);
        // Feature edits deliberately keep B's coordinate while exercising both activations and
        // the ordinary existing rank/SUM/contiguous/same-type-cast rules.
        Tensor edited = x.expandDims(2).squeeze(2).contiguous().cast(type);
        Tensor next = edited.matmul(wx).add(h.matmul(wh)).add(bias).tanh();
        Tensor emit = next.sigmoid().add(h);
        var body = new ScanBody(List.of(h), List.of(x), List.of(wx, wh, bias), List.of(next), List.of(emit));
        Tensor initial = fresh(h, true), sequence = symbolic(type, true, fixed(time), B, fixed(f));
        Tensor wxa = fresh(wx, true), wha = fresh(wh, true), ba = fresh(bias, true);
        var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(wxa, wha, ba), body, reverse);
        Tensor finalSeed = fresh(scan.finalCarry().getFirst(), false), emissionSeed = fresh(scan.outputs().getFirst(), false);
        List<Tensor> targets = List.of(initial, sequence, wxa, wha, ba);
        List<Tensor> roots = mean ? List.of(scan.finalCarry().getFirst().mean())
                : emissionOnly ? scan.outputs() : List.of(scan.finalCarry().getFirst(), scan.outputs().getFirst());
        var strictRequest = mean ? FunctionalGradientTestSupport.request(roots.getFirst(), targets)
                : ScanGradientRulesTest.request(roots, emissionOnly ? List.of(emissionSeed) : List.of(finalSeed, emissionSeed), targets);
        if (time == 0 && mean) {
            // MEAN(final) is MEAN(initial); no sequence or invariant role is connected.
            assertThrows(IllegalArgumentException.class, () -> AutogradPreflight.preflight(
                    CompileMode.FORWARD_AND_BACKWARD, roots, strictRequest, CompileTimeConstantGraph.Ingress.empty()));
        }
        var request = time == 0 && mean ? FunctionalGradientTestSupport.request(roots.getFirst(), targets,
                FunctionalGradientRequest.DisconnectedPolicy.ZERO) : strictRequest;
        var expansion = FirstOrderAutograd.expand(AutogradPreflight.preflight(CompileMode.FORWARD_AND_BACKWARD,
                roots, request, CompileTimeConstantGraph.Ingress.empty()).stageOne(), CompileTimeConstantGraph.Ingress.empty());
        var artifacts = compile(roots, Optional.of(request), GraphOptimizationConfig.standard());
        var plan = BatchBindingPlan.prepare(artifacts);
        var scans = artifacts.graph().nodes().stream().filter(n -> n.scanBody().isPresent()).toList();
        for (int batch : new int[] {0, 1, f, width, 64, 40, 1, 64}) {
            var supplied = artifacts.constants().bindableInputs().stream().map(id ->
                    actual(artifacts.graph().values().get(index(artifacts, id)).descriptor(), batch, batch == 1)).toList();
            var binding = plan.bind(supplied);
            assertEquals(batch, binding.batchSize());
            for (var node : scans) for (ValueId id : node.outputs()) {
                int index = index(artifacts, id);
                var descriptor = binding.descriptor(index);
                assertTrue(descriptor.shape().isFullyStatic());
                assertEquals(LayoutDescriptor.contiguous(descriptor.shape()), descriptor.layout().orElseThrow());
                assertFalse(descriptor.layout().orElseThrow().isView());
                var template = artifacts.graph().values().get(index).descriptor();
                if (!template.shape().isFullyStatic()) assertTrue(template.layout().isEmpty());
            }
            var oracle = new ScanGradientRulesTest.Oracle(batch);
            oracle.bind(initial, data(batch * width, .13)); oracle.bind(sequence, data(time * batch * f, -.17));
            oracle.bind(wxa, data(f * width, .11)); oracle.bind(wha, data(width * width, -.07)); oracle.bind(ba, data(width, .04));
            oracle.bind(finalSeed, data(batch * width, .31)); oracle.bind(emissionSeed, data(time * batch * width, -.23));
            oracle.constants(expansion);

            // Independent ordinary Tensor unrolling at the actual geometry. Only this reference
            // grows with T. It does not reconstruct, clone, bind or differentiate a static Scan.
            var reference = new ScanGradientRulesTest.Oracle();
            Tensor ih = tensor(type, true, batch, width), rwx = tensor(type, true, f, width);
            Tensor rwh = tensor(type, true, width, width), rb = tensor(type, true, width);
            reference.bind(ih, oracle.eval(initial)); reference.bind(rwx, oracle.eval(wxa));
            reference.bind(rwh, oracle.eval(wha)); reference.bind(rb, oracle.eval(ba));
            var steps = new ArrayList<Tensor>(); var emitted = new Tensor[time];
            for (int t = 0; t < time; t++) {
                Tensor step = tensor(type, true, batch, f); steps.add(step);
                reference.bind(step, Arrays.copyOfRange(oracle.eval(sequence), t * batch * f, (t + 1) * batch * f));
            }
            Tensor state = ih;
            for (int iteration = 0; iteration < time; iteration++) {
                int t = reverse ? time - 1 - iteration : iteration;
                Tensor update = steps.get(t).matmul(rwx).add(state.matmul(rwh)).add(rb).tanh();
                emitted[t] = update.sigmoid().add(state); state = update;
            }
            assertArrayEquals(reference.eval(state), oracle.eval(scan.finalCarry().getFirst()), 0);
            for (int t = 0; t < time; t++) assertArrayEquals(reference.eval(emitted[t]),
                    Arrays.copyOfRange(oracle.eval(scan.outputs().getFirst()), t * batch * width, (t + 1) * batch * width), 0);
            Tensor objective;
            if (mean) objective = state.mean();
            else {
                Tensor fs = tensor(type, false, batch, width); reference.bind(fs, oracle.eval(finalSeed));
                objective = emissionOnly ? null : state.mul(fs).sum();
                for (int t = 0; t < time; t++) {
                    Tensor es = tensor(type, false, batch, width);
                    reference.bind(es, Arrays.copyOfRange(oracle.eval(emissionSeed), t * batch * width, (t + 1) * batch * width));
                    Tensor term = emitted[t].mul(es).sum();
                    objective = objective == null ? term : objective.add(term);
                }
            }
            var referenceTargets = new ArrayList<>(List.of(ih, rwx, rwh, rb)); referenceTargets.addAll(steps);
            // T=0 leaves parameters/steps disconnected and outside the original inventory.
            var active = time == 0 ? List.of(ih) : referenceTargets;
            var ordinary = FirstOrderAutograd.expand(AutogradPreflight.preflight(CompileMode.FORWARD_AND_BACKWARD,
                    List.of(objective), FunctionalGradientTestSupport.stage(objective, active),
                    CompileTimeConstantGraph.Ingress.empty()), CompileTimeConstantGraph.Ingress.empty());
            reference.constants(ordinary);
            double tolerance = type == DataType.FLOAT32 ? 4e-5 : 2e-12;
            assertArrayEquals(reference.eval(ordinary.targetGradients().getFirst().gradient()),
                    oracle.eval(expansion.targetGradients().getFirst().gradient()), tolerance);
            if (mean) {
                double[] value = oracle.eval(roots.getFirst());
                if (batch == 0) assertTrue(Double.isNaN(value[0])); else assertArrayEquals(reference.eval(objective), value, tolerance);
            }
            for (int i = 0; i < 3; i++) {
                double[] expected = time == 0 ? new double[oracle.actualCount(targets.get(i + 2))]
                        : reference.eval(ordinary.targetGradients().get(i + 1).gradient());
                assertArrayEquals(expected, oracle.eval(expansion.targetGradients().get(i + 2).gradient()), tolerance);
            }
            double[] dx = oracle.eval(expansion.targetGradients().get(1).gradient());
            assertEquals(time * batch * f, dx.length);
            for (int t = 0; t < time; t++) assertArrayEquals(reference.eval(ordinary.targetGradients().get(t + 4).gradient()),
                    Arrays.copyOfRange(dx, t * batch * f, (t + 1) * batch * f), tolerance);
        }
    }

    static int index(CompileArtifacts artifacts, ValueId id) {
        for (int i = 0; i < artifacts.graph().values().size(); i++) if (artifacts.graph().values().get(i).id().equals(id)) return i;
        throw new AssertionError(id);
    }
    static double[] data(int size, double base) {
        double[] result = new double[size];
        for (int i = 0; i < size; i++) result[i] = base + ((i * 17) % 23 - 11) * .013;
        return result;
    }
}
