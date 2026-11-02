package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.*;
import static io.github.pho001.synaptik.compiler.ScanBodyInferenceTest.tensor;
import static io.github.pho001.synaptik.compiler.SymbolicScanCompilerTest.*;

import io.github.pho001.synaptik.config.compile.*;
import io.github.pho001.synaptik.model.datatype.*;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.*;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.layout.*;
import io.github.pho001.synaptik.model.operation.scan.*;
import io.github.pho001.synaptik.model.shape.*;
import io.github.pho001.synaptik.model.tensor.*;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Compiler-only metadata and actual sibling evidence; only the test Oracle reads lengths. */
final class SymbolicLengthScanCompilerTest {
    @Test void perValueTypedSymbolicSourceInfersAndDoesNotPoisonUnrelatedAd() {
        for (DataType type : DataType.values())
            for (long time : new long[] {0, 3}) for (int role = -2; role < 3; role++) {
                var source = LengthAwareScanBodyInferenceTest.typedLengthSource(type, B, time, role);
                var artifacts = compile(source.finalCarry(), Optional.empty(), GraphOptimizationConfig.disabled());
                var binding = BatchBindingPlan.prepare(artifacts);
                for (int batch : new int[] {0, 1, 3})
                    assertPorts(artifacts, binding.bind(inputs(artifacts, batch, false)));
                Tensor target = tensor(DataType.FLOAT32, true), objective = target.mul(target);
                var request = FunctionalGradientTestSupport.request(objective, List.of(target));
                assertDoesNotThrow(() -> GraphCompiler.compile(CompileMode.FORWARD_AND_BACKWARD,
                        List.of(objective, source.finalCarry().getFirst()), Optional.of(request),
                        CompileTimeConstantGraph.Ingress.empty(), GraphOptimizationConfig.disabled()));
            }
    }

    @Test void emptyAndSharedOnlyCarriesBindBatchFromLengthsRatherThanCarryGeometry() {
        for (boolean shared : List.of(false, true)) for (boolean reverse : List.of(false, true))
            for (long time : new long[] {0, 3}) {
                Tensor x = symbolic(DataType.FLOAT64, false, B, fixed(2));
                Tensor old = tensor(DataType.FLOAT64, false, 2);
                var carry = shared ? List.of(old) : List.<Tensor>of();
                var next = shared ? List.of(old.add(x.sum(0, false))) : List.<Tensor>of();
                var body = new ScanBody(carry, List.of(x), List.of(), next, List.of(x));
                var source = Scan.apply(carry.stream().map(t -> fresh(t, false)).toList(),
                        List.of(symbolic(DataType.FLOAT64, false, fixed(time), B, fixed(2))), List.of(),
                        symbolic(DataType.INT64, false, B), 0, List.of(OptionalInt.of(0)), body, reverse);
                var artifacts = compile(source.outputs(), Optional.empty(), GraphOptimizationConfig.disabled());
                var plan = BatchBindingPlan.prepare(artifacts);
                for (int batch : new int[] {0, 1, 3, 1}) {
                    var bound = plan.bind(inputs(artifacts, batch, false));
                    assertEquals(batch, bound.batchSize());
                    assertPorts(artifacts, bound);
                    var output = artifacts.publication().forwardBindings().getFirst().valueId();
                    assertEquals(Shape.of(time, batch, 2), bound.descriptor(index(artifacts, output)).shape());
                }
            }
    }

    @Test void actualSiblingsMatchActivePrefixOrdinaryUnrollingWithFreshBatchAndLengthPayloads() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (boolean reverse : List.of(false, true)) for (int seeds = 0; seeds < 3; seeds++) {
                compare(type, reverse, seeds, false, 3);
            }
    }

    @Test void outerMeanUsesActualDenseBatchCountAndZeroTimeHasItsOwnPlan() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) for (boolean reverse : List.of(false, true)) {
            compare(type, reverse, 0, true, 3);
            compare(type, reverse, 2, false, 0);
            compare(type, reverse, 0, true, 0);
        }
    }

    @Test void broaderForwardMixedRolesAndDistinctTerminalCarriesInferAndBindWithoutSampling() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) for (boolean reverse : List.of(false, true)) {
            Tensor h = symbolic(type, true, B, fixed(3)), x = symbolic(type, false, B, fixed(2));
            Tensor y = symbolic(type, false, fixed(2), new DynamicDimension("Rows"));
            Tensor shared = tensor(type, false, 2, 3), a = tensor(type, false, 3), b = tensor(type, false, 2, 3);
            Tensor next = h.add(x.matmul(shared)).add(y.permute(1, 0).matmul(shared));
            Tensor contribution = y.matmul(h);
            var body = new ScanBody(List.of(h, a, b), List.of(x, y, shared), List.of(),
                    List.of(next, a.add(h.sum(0, false)), b.add(contribution)), List.of(h));
            Tensor initial = fresh(h, true), sx = symbolic(type, false, fixed(3), B, fixed(2));
            Tensor sy = symbolic(type, false, fixed(3), fixed(2), B), ss = tensor(type, false, 3, 2, 3);
            Tensor lengths = symbolic(DataType.INT64, false, B);
            List<OptionalInt> axes = List.of(OptionalInt.of(0), OptionalInt.of(1), OptionalInt.empty());
            // Capture an all-valid source then add the length header without Model row admission.
            var source = Scan.apply(List.of(initial, fresh(a, false), fresh(b, false)), List.of(sx, sy, ss), List.of(), body, reverse);
            var raw = LengthAwareScanBodyInferenceTest.lengthIR(GraphCapture.capture(source.finalCarry()), 1, axes, lengths.descriptor());
            assertTrue(CapturedGraphInference.inferAndValidate(raw).constraints().isEmpty());
            var admitted = Scan.apply(List.of(initial, fresh(a, false), fresh(b, false)), List.of(sx, sy, ss), List.of(),
                    lengths, 1, axes, body, reverse);
            for (var optimization : List.of(GraphOptimizationConfig.disabled(), GraphOptimizationConfig.standard())) {
                var artifacts = compile(List.of(initial, admitted.finalCarry().getFirst()), Optional.empty(), optimization);
                var plan = BatchBindingPlan.prepare(artifacts);
                var past = new ArrayList<BatchBindingPlan.Binding>();
                int round = 0;
                for (int batch : new int[] {0, 1, 2, 3, 5, 32, 32, 7, 1, 32}) {
                    var supplied = inputs(artifacts, batch, round++ % 2 == 0);
                    var binding = plan.bind(supplied); past.add(binding); assertPorts(artifacts, binding);
                    var published = artifacts.publication().forwardBindings().getFirst().valueId();
                    assertSame(supplied.get(artifacts.constants().bindableInputs().indexOf(published)), binding.descriptor(index(artifacts, published)));
                    for (int i = 0; i < supplied.size(); i++) {
                        TensorDescriptor d = supplied.get(i);
                        if (d.shape().rank() == 0) continue;
                        var bad = new ArrayList<>(supplied);
                        var dims = d.shape().toLongArray(); dims[0]++;
                        Shape wrong = Shape.of(dims);
                        bad.set(i, new TensorDescriptor(d.dataType(), wrong, Optional.of(LayoutDescriptor.contiguous(wrong)), d.requiresGrad()));
                        assertThrows(IllegalArgumentException.class, () -> plan.bind(bad), "including unused SHARED sequence and L");
                    }
                }
                assertEquals(32, past.get(5).batchSize());
                var published = artifacts.publication().forwardBindings().getFirst().valueId();
                assertFalse(past.get(5).descriptor(index(artifacts, published)).layout().orElseThrow().isView());
                assertTrue(past.get(6).descriptor(index(artifacts, published)).layout().orElseThrow().isView());
            }
        }
    }

    @Test void rawLengthsUnusedDeclarationsAttributesAndLatentSiblingsRejectDeliberately() {
        for (long time : new long[] {0, 3}) {
            Tensor h = symbolic(DataType.FLOAT64, false, B, fixed(2)), x = fresh(h, false), unused = fresh(h, false);
            Tensor next = h.add(x).reshape(shape(B, fixed(1), fixed(2))).squeeze(1);
            var allValid = Scan.apply(List.of(fresh(h, false)), List.of(symbolic(DataType.FLOAT64, false, fixed(time), B, fixed(2))),
                    List.of(fresh(unused, false)), new ScanBody(List.of(h), List.of(x), List.of(unused), List.of(next), List.of(h)), false);
            var good = LengthAwareScanBodyInferenceTest.lengthIR(GraphCapture.capture(allValid.finalCarry()), 1,
                    List.of(OptionalInt.of(0)), symbolic(DataType.INT64, false, B).descriptor());
            // The unused invariant itself makes this row-mode header invalid, despite matching B.
            assertThrows(IllegalArgumentException.class, () -> CapturedGraphInference.inferAndValidate(good));
            Tensor shared = tensor(DataType.FLOAT64, false, 2);
            var validSource = Scan.apply(List.of(fresh(h, false)), List.of(symbolic(DataType.FLOAT64, false, fixed(time), B, fixed(2))),
                    List.of(fresh(shared, false)), new ScanBody(List.of(h), List.of(x), List.of(shared), List.of(next), List.of(h)), false);
            var valid = LengthAwareScanBodyInferenceTest.lengthIR(GraphCapture.capture(validSource.finalCarry()), 1,
                    List.of(OptionalInt.of(0)), symbolic(DataType.INT64, false, B).descriptor());
            assertDoesNotThrow(() -> CapturedGraphInference.inferAndValidate(valid));
            var node = valid.nodes().getLast(); var b = node.scanBody().orElseThrow();
            for (TensorDescriptor badLengths : List.of(symbolic(DataType.INT64, false, new DynamicDimension("Other")).descriptor(),
                    symbolic(DataType.INT32, false, B).descriptor(), symbolic(DataType.FLOAT64, true, B).descriptor(),
                    symbolic(DataType.INT64, false, B, fixed(1)).descriptor(), tensor(DataType.INT64, false, 1).descriptor())) {
                var values = valid.values().stream().map(v -> v.id().equals(node.inputs().getLast())
                        ? new GraphValue(v.id(), badLengths) : v).toList();
                assertThrows(IllegalArgumentException.class, () -> CapturedGraphInference.inferAndValidate(
                        new CompiledGraphModel(values, valid.nodes(), valid.inputs(), valid.outputs(), valid.nodePhases())));
            }
            ValueId id = b.graph().inputs().getLast();
            for (Shape malformed : List.of(shape(new DynamicDimension("Other"), fixed(2)), shape(B, B),
                    shape(DimensionExpressions.addConstant(B, 1), fixed(2)),
                    shape(DimensionExpressions.unknown(0, Optional.empty()), fixed(2)))) {
                var values = b.graph().values().stream().map(v -> v.id().equals(id)
                        ? new GraphValue(v.id(), new TensorDescriptor(DataType.FLOAT64, malformed, Optional.empty(), false)) : v).toList();
                assertBadBody(valid, new CompiledGraphModel(values, b.graph().nodes(), b.graph().inputs(), b.graph().outputs(), b.graph().nodePhases()));
            }
            // A dead canonical outer residual must be independently derived, not trusted.
            var ports = new ArrayList<>(valid.values());
            ValueId dead = node.outputs().getLast();
            ports.replaceAll(v -> v.id().equals(dead) ? new GraphValue(v.id(), symbolic(DataType.FLOAT64, false,
                    fixed(time), B, fixed(9)).descriptor()) : v);
            assertThrows(IllegalArgumentException.class, () -> CapturedGraphInference.inferAndValidate(
                    new CompiledGraphModel(ports, valid.nodes(), valid.inputs(), valid.outputs(), valid.nodePhases())));
            var nodes = new ArrayList<>(b.graph().nodes());
            int reshape = java.util.stream.IntStream.range(0, nodes.size()).filter(i -> nodes.get(i).operation().kind()
                    == ShapeTransformKind.RESHAPE).findFirst().orElseThrow();
            var r = nodes.get(reshape);
            nodes.set(reshape, new CompiledNode(r.id(), new Operation(ShapeTransformKind.RESHAPE,
                    new TargetShapeAttrs(shape(new DynamicDimension("Other"), fixed(1), fixed(2)))), r.inputs(), r.outputs()));
            assertBadBody(valid, new CompiledGraphModel(b.graph().values(), nodes, b.graph().inputs(), b.graph().outputs(), b.graph().nodePhases()));
        }
    }

    private static void assertBadBody(CompiledGraphModel outer, CompiledGraphModel local) {
        var node = outer.nodes().getLast(); var b = node.scanBody().orElseThrow();
        var badBody = new CompiledScanBody(local, b.nextCarry(), b.emissions(), b.residuals(), b.demandedOutputIndices());
        var nodes = new ArrayList<>(outer.nodes());
        nodes.set(nodes.size()-1, new CompiledNode(node.id(), node.operation(), node.inputs(), node.outputs(), Optional.of(badBody)));
        assertThrows(IllegalArgumentException.class, () -> CapturedGraphInference.inferAndValidate(
                new CompiledGraphModel(outer.values(), nodes, outer.inputs(), outer.outputs(), outer.nodePhases())));
    }

    @Test void originalMixedRolesResidualRootsAndHigherOrderRejectBeforeDerivativeIds() throws Exception {
        for (long time : new long[] {0, 3}) for (int mode = 0; mode < 3; mode++) {
            Tensor h = symbolic(DataType.FLOAT64, false, B, fixed(2));
            Tensor x = mode == 0 ? symbolic(DataType.FLOAT64, false, fixed(2), B)
                    : mode == 2 ? fresh(h, false) : tensor(DataType.FLOAT64, false, 2, 2);
            Tensor q = tensor(DataType.FLOAT64, false, 2);
            List<Tensor> carries = mode == 2 ? List.of(h, q) : List.of(h);
            List<Tensor> next = mode == 2 ? List.of(h, q.add(h.sum(0, false))) : List.of(h);
            var body = new ScanBody(carries, List.of(x), List.of(), next, List.of());
            Tensor seq = mode == 0 ? symbolic(DataType.FLOAT64, false, fixed(time), fixed(2), B)
                    : mode == 2 ? symbolic(DataType.FLOAT64, false, fixed(time), B, fixed(2)) : tensor(DataType.FLOAT64, false, time, 2, 2);
            var scan = Scan.apply(carries.stream().map(t -> fresh(t, false)).toList(), List.of(seq), List.of(),
                    symbolic(DataType.INT64, false, B), 1,
                    List.of(mode == 0 ? OptionalInt.of(1) : mode == 2 ? OptionalInt.of(0) : OptionalInt.empty()), body, false);
            var artifacts = compile(scan.finalCarry(), Optional.empty(), GraphOptimizationConfig.standard());
            BatchBindingPlan.prepare(artifacts).bind(inputs(artifacts, 0, false));
            Tensor target = tensor(DataType.FLOAT64, true), objective = target.mul(target);
            for (var optimization : List.of(GraphOptimizationConfig.disabled(), GraphOptimizationConfig.standard()))
                noIds(() -> GraphCompiler.compile(CompileMode.FORWARD_AND_BACKWARD,
                        List.of(objective, scan.finalCarry().getFirst()),
                        Optional.of(FunctionalGradientTestSupport.request(objective, List.of(target))),
                        CompileTimeConstantGraph.Ingress.empty(), optimization));
        }
        Tensor h = symbolic(DataType.FLOAT64, true, B, fixed(2)), x = fresh(h, true), w = tensor(DataType.FLOAT64, true, 2);
        Tensor initial = fresh(h, true), seq = symbolic(DataType.FLOAT64, true, fixed(0), B, fixed(2)), weight = fresh(w, true);
        var scan = Scan.apply(List.of(initial), List.of(seq), List.of(weight), symbolic(DataType.INT64, false, B),
                1, List.of(OptionalInt.of(0)), new ScanBody(List.of(h), List.of(x), List.of(w), List.of(h.add(x).mul(w).tanh()), List.of(h)), false);
        Tensor residual = scan.residuals().getFirst().sum();
        noIds(() -> GraphCompiler.compile(CompileMode.FORWARD_AND_BACKWARD, List.of(residual),
                Optional.of(FunctionalGradientTestSupport.request(residual, List.of(seq))),
                CompileTimeConstantGraph.Ingress.empty(), GraphOptimizationConfig.disabled()));
        var first = new FunctionalGradientRequest.Stage(List.of(new FunctionalGradientRequest.ForwardTensorReference(scan.finalCarry().getFirst())),
                List.of(Optional.of(fresh(h, false))), List.of(seq), true, FunctionalGradientRequest.DisconnectedPolicy.ZERO);
        var second = new FunctionalGradientRequest.Stage(List.of(new FunctionalGradientRequest.FirstStageGradientReference(0)),
                List.of(Optional.of(fresh(seq, false))), List.of(seq), false, FunctionalGradientRequest.DisconnectedPolicy.ZERO);
        noIds(() -> GraphCompiler.compile(CompileMode.FORWARD_AND_BACKWARD, scan.finalCarry(), Optional.of(new FunctionalGradientRequest(List.of(first, second))),
                CompileTimeConstantGraph.Ingress.empty(), GraphOptimizationConfig.disabled()));
        // Model still validates before it allocates canonical output identities.
        Tensor badLengths = tensor(DataType.INT64, false, 1);
        noIds(() -> Scan.apply(List.of(initial), List.of(seq), List.of(weight), badLengths,
                1, List.of(OptionalInt.of(0)), scan.finalCarry().getFirst().provenance().orElseThrow().producer().scanBody().orElseThrow(), false));
    }

    private static void noIds(Runnable action) {
        Tensor marker = tensor(DataType.BOOL, false);
        assertThrows(IllegalArgumentException.class, action::run);
        assertEquals(marker.id().value() + 1, tensor(DataType.BOOL, false).id().value());
    }

    @Test void neighboringEqualLocalIdsCannotErasePrivateObligationsBesideLengthScopes() {
        Tensor h = symbolic(DataType.FLOAT64, false, B, fixed(2)), x = fresh(h, false);
        Tensor initial = fresh(h, false), seq = symbolic(DataType.FLOAT64, false, fixed(0), B, fixed(2));
        var length = Scan.apply(List.of(initial), List.of(seq), List.of(), symbolic(DataType.INT64, false, B),
                1, List.of(OptionalInt.of(0)), new ScanBody(List.of(h), List.of(x), List.of(), List.of(h), List.of()), false);
        Tensor a = fresh(h, false), b = fresh(h, false);
        var other = Scan.apply(List.of(initial), List.of(seq), List.of(),
                new ScanBody(List.of(a), List.of(b), List.of(), List.of(a), List.of(b.topK(3, 0).values())), true);
        var artifacts = compile(List.of(length.finalCarry().getFirst(), other.finalCarry().getFirst()), Optional.empty(), GraphOptimizationConfig.standard());
        var plan = BatchBindingPlan.prepare(artifacts);
        for (int i = 0; i < 2; i++) {
            assertEquals(3, plan.bind(inputs(artifacts, 3, false)).batchSize());
            assertThrows(IllegalArgumentException.class, () -> plan.bind(inputs(artifacts, 2, true)));
        }
    }

    @Test void rawUnusedFixedFeatureOverflowCannotHideBehindZeroTimeOrBatch() {
        for (long time : new long[] {0, 3}) for (Dimension batch : List.of(B, fixed(0), fixed(1))) {
            Shape row = shape(batch, fixed(1));
            Shape huge = shape(batch, fixed(Long.MAX_VALUE), fixed(2));
            var d = new TensorDescriptor(DataType.FLOAT64, row, row.isFullyStatic()
                    ? Optional.of(LayoutDescriptor.contiguous(row)) : Optional.empty(), false);
            var large = new TensorDescriptor(DataType.FLOAT64, huge, Optional.empty(), false);
            var ids = List.of(new ValueId(0), new ValueId(1), new ValueId(2));
            var local = new CompiledGraphModel(List.of(new GraphValue(ids.get(0), d), new GraphValue(ids.get(1), d),
                    new GraphValue(ids.get(2), large)), List.of(), ids, List.of(ids.getFirst()), Map.of());
            var body = new CompiledScanBody(local, List.of(ids.getFirst()), List.of(), List.of(ids.getFirst()), List.of());
            var node = new CompiledNode(new NodeId(0), new Operation(ScanKind.SCAN,
                    new ScanAttrs(time, 1, 2, 0, 0, 1, false, true, 1, List.of(OptionalInt.of(0), OptionalInt.of(0)))),
                    List.of(new ValueId(0), new ValueId(1), new ValueId(2), new ValueId(3)),
                    List.of(new ValueId(4), new ValueId(5)), Optional.of(body));
            var actuals = List.of(d, descriptor(DataType.FLOAT64, shape(fixed(time), batch, fixed(1))),
                    new TensorDescriptor(DataType.FLOAT64, shape(fixed(time), batch, fixed(Long.MAX_VALUE), fixed(2)), Optional.empty(), false),
                    descriptor(DataType.INT64, shape(batch)));
            // Calls the independent descriptor seam directly: no Model constructor or layout
            // materialization is allowed to become the reason this malformed domain rejects.
            assertThrows(ArithmeticException.class, () -> ScanBodyInference.infer(node, actuals));
        }
        // Zero features are legitimate and distinct from an overflowing nonzero feature bank.
        Tensor h = symbolic(DataType.FLOAT64, true, B, fixed(0)), x = fresh(h, true);
        var scan = Scan.apply(List.of(fresh(h, true)), List.of(symbolic(DataType.FLOAT64, true, fixed(0), B, fixed(0))), List.of(),
                symbolic(DataType.INT64, false, B), 1, List.of(OptionalInt.of(0)),
                new ScanBody(List.of(h), List.of(x), List.of(), List.of(h.add(x)), List.of(h)), false);
        var artifacts = compile(scan.finalCarry(), Optional.empty(), GraphOptimizationConfig.standard());
        var plan = BatchBindingPlan.prepare(artifacts);
        for (int batch : new int[] {0, 1, 4}) assertPorts(artifacts, plan.bind(inputs(artifacts, batch, false)));
    }

    private static TensorDescriptor descriptor(DataType type, Shape shape) {
        return new TensorDescriptor(type, shape, shape.isFullyStatic() ? Optional.of(LayoutDescriptor.contiguous(shape)) : Optional.empty(), false);
    }

    @Test void tiedTwoCarriesFalseFlagFlowRepeatedRootsAndCrossOccurrenceSumsMatchOrdinaryUnrolling() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            Tensor h = symbolic(type, false, B, fixed(2)), k = fresh(h, false);
            Tensor x = fresh(h, true), y = fresh(h, true), w = tensor(type, true, 2), u = fresh(w, true);
            Tensor next = h.mul(w).add(k.mul(u)).add(x).add(y);
            var body = new ScanBody(List.of(h, k), List.of(x, y), List.of(w, u), List.of(next, next), List.of(next, next));
            Tensor initial = fresh(h, false), seq = symbolic(type, true, fixed(3), B, fixed(2)), weight = fresh(w, true);
            Tensor lengths = symbolic(DataType.INT64, false, B);
            var sources = new ArrayList<ScanResult>();
            Tensor objective = null;
            for (boolean reverse : List.of(false, true)) {
                var source = Scan.apply(List.of(initial, initial), List.of(seq, seq), List.of(weight, weight), lengths,
                        2, List.of(OptionalInt.of(0), OptionalInt.of(0)), body, reverse);
                sources.add(source);
                for (Tensor root : List.of(source.finalCarry().get(0), source.finalCarry().get(1), source.outputs().get(0), source.outputs().get(1)))
                    objective = objective == null ? root.sum() : objective.add(root.sum());
            }
            var request = FunctionalGradientTestSupport.request(objective, List.of(seq, weight));
            var expansion = FirstOrderAutograd.expand(AutogradPreflight.preflight(CompileMode.FORWARD_AND_BACKWARD,
                    List.of(objective), request, CompileTimeConstantGraph.Ingress.empty()).stageOne(), CompileTimeConstantGraph.Ingress.empty());
            for (var gradient : expansion.targetGradients()) assertEquals(BinaryArithmeticKind.ADD,
                    gradient.gradient().provenance().orElseThrow().producer().operation().kind());
            for (var optimization : List.of(GraphOptimizationConfig.disabled(), GraphOptimizationConfig.standard())) {
                var artifacts = compile(List.of(objective), Optional.of(request), optimization);
                assertEquals(4, artifacts.graph().nodes().stream().filter(n -> n.scanBody().isPresent()).count());
                var plan = BatchBindingPlan.prepare(artifacts);
                for (int batch : new int[] {0, 1, 2, 5}) {
                    assertPorts(artifacts, plan.bind(inputs(artifacts, batch, true)));
                    var oracle = new ScanGradientRulesTest.Oracle(batch);
                    oracle.bind(initial, data(batch*2, .1)); oracle.bind(seq, data(3*batch*2, -.1));
                    oracle.bind(weight, data(2, .2));
                    double[] ls = new double[batch]; for (int b = 0; b < batch; b++) ls[b] = (b+1)%4;
                    oracle.bind(lengths, ls); oracle.constants(expansion);
                    double[] expectedX = new double[3*batch*2], expectedW = new double[2];
                    for (int b = 0; b < batch; b++) {
                        int valid = (int) ls[b]; if (valid == 0) continue;
                        var reference = new ScanGradientRulesTest.Oracle();
                        Tensor ih = tensor(type, false, 1, 2), rw = tensor(type, true, 2);
                        reference.bind(ih, Arrays.copyOfRange(oracle.eval(initial), b*2, (b+1)*2)); reference.bind(rw, oracle.eval(weight));
                        var steps = new ArrayList<Tensor>();
                        for (int t = 0; t < valid; t++) {
                            Tensor rt = tensor(type, true, 1, 2); steps.add(rt);
                            reference.bind(rt, Arrays.copyOfRange(oracle.eval(seq), (t*batch+b)*2, (t*batch+b+1)*2));
                        }
                        Tensor loss = null;
                        for (boolean reverse : List.of(false, true)) {
                            Tensor state = ih;
                            for (int iteration = 0; iteration < valid; iteration++) {
                                int t = reverse ? valid-1-iteration : iteration;
                                state = state.mul(rw).add(state.mul(rw)).add(steps.get(t)).add(steps.get(t));
                                Tensor term = state.sum().add(state.sum());
                                loss = loss == null ? term : loss.add(term);
                            }
                            loss = loss.add(state.sum()).add(state.sum());
                        }
                        var targets = new ArrayList<>(steps); targets.add(rw);
                        var unrolled = FirstOrderAutograd.expand(AutogradPreflight.preflight(CompileMode.FORWARD_AND_BACKWARD,
                                List.of(loss), FunctionalGradientTestSupport.stage(loss, targets), CompileTimeConstantGraph.Ingress.empty()),
                                CompileTimeConstantGraph.Ingress.empty());
                        reference.constants(unrolled);
                        for (int t = 0; t < valid; t++) System.arraycopy(reference.eval(unrolled.targetGradients().get(t).gradient()),
                                0, expectedX, (t*batch+b)*2, 2);
                        double[] dw = reference.eval(unrolled.targetGradients().getLast().gradient());
                        for (int j = 0; j < 2; j++) expectedW[j] += dw[j];
                    }
                    double tolerance = type == DataType.FLOAT32 ? 2e-5 : 1e-12;
                    assertArrayEquals(expectedX, oracle.eval(expansion.targetGradients().getFirst().gradient()), tolerance);
                    assertArrayEquals(expectedW, oracle.eval(expansion.targetGradients().getLast().gradient()), tolerance);
                    assertFalse(initial.descriptor().requiresGrad());
                    assertTrue(sources.getFirst().outputs().getFirst().descriptor().requiresGrad());
                }
            }
        }
    }

    @Test void disconnectedZerosAndEmptyRowsPreserveRawSeedsWithoutPrimalOrBodyWork() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) for (boolean reverse : List.of(false, true))
            for (int time : new int[] {0, 3}) {
                Tensor h = symbolic(type, true, B, fixed(2)), x = fresh(h, true), unused = fresh(h, false);
                var body = new ScanBody(List.of(h), List.of(x, unused), List.of(), List.of(x), List.of(h));
                Tensor initial = fresh(h, true), seq = symbolic(type, true, fixed(time), B, fixed(2));
                Tensor dead = fresh(seq, false), lengths = symbolic(DataType.INT64, false, B);
                var source = Scan.apply(List.of(initial), List.of(seq, dead), List.of(), lengths, 1,
                        List.of(OptionalInt.of(0), OptionalInt.of(0)), body, reverse);
                Tensor seed = fresh(initial, false);
                var constants = new FirstOrderAutograd.DerivativeConstants();
                var checkedSource = ScanGradientRules.preflightSource(
                        source.finalCarry().getFirst().provenance().orElseThrow().producer());
                var pairs = new ArrayList<ScanGradientRules.Pair>();
                pairs.add(new ScanGradientRules.Pair(0, 0));
                if (time > 0) pairs.add(new ScanGradientRules.Pair(0, 1));
                var checked = ScanGradientRules.plan(checkedSource, new ScanGradientRules.Demand(pairs));
                Tensor[] gradients = ScanGradientRules.apply(checked, List.of(seed), constants);
                assertNull(gradients[2]);
                if (time == 0) {
                    assertSame(seed, gradients[0]);
                    assertNull(gradients[1]);
                }
                for (int batch : new int[] {0, 1, 2, 3}) {
                    var oracle = new ScanGradientRulesTest.Oracle(batch);
                    double[] bits = new double[batch*2];
                    for (int i = 0; i < bits.length; i++) bits[i] = i % 2 == 0 ? -0.0 : type == DataType.FLOAT32
                            ? Float.intBitsToFloat(0x7fc00123) : Double.longBitsToDouble(0x7ff8000000000123L);
                    oracle.bind(initial, bits); oracle.bind(seed, bits);
                    double[] poison = new double[time*batch*2]; Arrays.fill(poison, Double.NaN);
                    oracle.bind(seq, poison); oracle.bind(dead, poison); oracle.bind(lengths, new double[batch]);
                    for (var constant : constants.bindings()) oracle.bind(constant.tensor(), new double[] {ScanGradientRulesTest.Oracle.scalar(constant.splat().value())});
                    assertRawEquals(oracle.eval(initial), oracle.eval(source.finalCarry().getFirst()));
                    assertRawEquals(oracle.eval(seed), oracle.eval(gradients[0]));
                    for (Tensor t : source.outputs()) for (double v : oracle.eval(t)) assertEquals(0L, Double.doubleToRawLongBits(v));
                    for (Tensor t : source.residuals()) for (double v : oracle.eval(t)) assertEquals(0L, Double.doubleToRawLongBits(v));
                    if (time > 0) for (double v : oracle.eval(gradients[1]))
                        assertEquals(0L, Double.doubleToRawLongBits(v));
                    if (batch > 0) for (double invalid : new double[] {-1, time+1}) {
                        var bad = new ScanGradientRulesTest.Oracle(batch);
                        double[] ls = new double[batch]; ls[batch-1] = invalid;
                        bad.bind(lengths, ls);
                        assertThrows(IllegalArgumentException.class, () -> bad.eval(source.finalCarry().getFirst()));
                    }
                }
            }
    }

    private static void assertRawEquals(double[] expected, double[] actual) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) assertEquals(Double.doubleToRawLongBits(expected[i]), Double.doubleToRawLongBits(actual[i]));
    }

    @Test void explicitRow1TypedZeroSequencesAndProducedIntegralLengthsRemainMetadataOnly() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) for (int time : new int[] {0, 3}) {
            Tensor h = symbolic(type, false, B, fixed(2)), x = symbolic(type, false, fixed(2), B);
            var constants = new FirstOrderAutograd.DerivativeConstants();
            Tensor zeroSequence = constants.zeroBase(type).expand(shape(fixed(time), fixed(2), B));
            Tensor floatLengths = symbolic(type, false, B), producedLengths = floatLengths.cast(DataType.INT64);
            Tensor initial = fresh(h, false);
            var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h.add(x.permute(1, 0))), List.of(h));
            var scan = Scan.apply(List.of(initial), List.of(zeroSequence), List.of(), producedLengths,
                    1, List.of(OptionalInt.of(1)), body, false);
            for (var optimization : List.of(GraphOptimizationConfig.disabled(), GraphOptimizationConfig.standard())) {
                var artifacts = compile(scan.finalCarry(), Optional.empty(), optimization,
                        new CompileTimeConstantGraph.Ingress(constants.bindings()));
                var plan = BatchBindingPlan.prepare(artifacts);
                for (int batch : new int[] {0, 1, 2, 5}) {
                    assertPorts(artifacts, plan.bind(inputs(artifacts, batch, true)));
                    var oracle = new ScanGradientRulesTest.Oracle(batch);
                    oracle.bind(initial, data(batch*2, .1));
                    double[] ls = new double[batch]; for (int b = 0; b < batch; b++) ls[b] = (time == 0 ? 0 : (b+1)%4) + .75;
                    oracle.bind(floatLengths, ls); oracle.bind(constants.bindings().getFirst().tensor(), new double[] {0});
                    assertArrayEquals(oracle.eval(initial), oracle.eval(scan.finalCarry().getFirst()), 0);
                }
                var n = artifacts.graph().nodes().stream().filter(p -> p.scanBody().isPresent()).findFirst().orElseThrow();
                assertEquals(List.of(OptionalInt.of(1)), ((ScanAttrs) n.operation().attrs()).sequenceRowAxes());
                TensorDescriptor lengths = artifacts.graph().values().get(index(artifacts, n.inputs().getLast())).descriptor();
                assertEquals(DataType.INT64, lengths.dataType()); assertFalse(lengths.requiresGrad());
            }
        }
    }

    /**
     * Checks actual symbolic siblings against independently unrolled active-prefix Tensor AD.
     * Repeated batch bindings also check current descriptors without admitting runtime execution.
     * @param type non-null homogeneous FLOAT32 or FLOAT64 fixture type
     * @param reverse whether the original scan traverses decreasing original time
     * @param seeds seed mode: 0 final-only, 1 emission-only, or 2 joint; mean overrides root selection
     * @param mean whether the objective is the dense final-carry MEAN, with count B times H
     * @param time fixed non-negative time count; these fixtures use separate zero- and three-step plans
     * @throws AssertionError if metadata, saved-site structure or numerical results disagree
     */
    private static void compare(DataType type, boolean reverse, int seeds, boolean mean, int time) {
        int f = 2, h = 3;
        Tensor state = symbolic(type, true, B, fixed(h)), step = symbolic(type, true, B, fixed(f));
        Tensor wx = tensor(type, true, h, f), wh = tensor(type, true, h, h), bias = tensor(type, true, h);
        Tensor edited = step.expandDims(2).squeeze(2).contiguous().cast(type);
        Tensor sharedTranspose = wx.permute(1, 0);
        Tensor preactivation = edited.matmul(sharedTranspose).add(state.matmul(wh)).add(bias);
        Tensor activation = seeds == 2 ? preactivation.permute(1, 0).tanh() : preactivation.tanh();
        Tensor next = seeds == 2 ? activation.permute(1, 0) : activation;
        Tensor emit = next.sigmoid().add(state);
        var body = new ScanBody(List.of(state), List.of(step), List.of(wx, wh, bias), List.of(next), List.of(emit));
        Tensor initial = fresh(state, true), sequence = symbolic(type, true, fixed(time), new DynamicDimension("Rows"), fixed(f));
        Tensor wxa = fresh(wx, true), wha = fresh(wh, true), ba = fresh(bias, true);
        Tensor lengths = symbolic(DataType.INT64, false, new DynamicDimension("Rows"));
        var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(wxa, wha, ba), lengths,
                1, List.of(OptionalInt.of(0)), body, reverse);
        Tensor finalSeed = fresh(scan.finalCarry().getFirst(), false), emissionSeed = fresh(scan.outputs().getFirst(), false);
        List<Tensor> targets = List.of(initial, sequence, wxa, wha, ba);
        List<Tensor> roots = mean ? List.of(scan.finalCarry().getFirst().mean()) : seeds == 0 ? scan.finalCarry()
                : seeds == 1 ? scan.outputs() : List.of(scan.finalCarry().getFirst(), scan.outputs().getFirst());
        var strictRequest = mean ? FunctionalGradientTestSupport.request(roots.getFirst(), targets)
                : ScanGradientRulesTest.request(roots, seeds == 0 ? List.of(finalSeed) : seeds == 1
                        ? List.of(emissionSeed) : List.of(finalSeed, emissionSeed), targets);
        if (time == 0 && mean) {
            // Dense MEAN at T=0 depends only on initial state, not on cell inputs/parameters.
            assertThrows(IllegalArgumentException.class, () -> AutogradPreflight.preflight(
                    CompileMode.FORWARD_AND_BACKWARD, roots, strictRequest, CompileTimeConstantGraph.Ingress.empty()));
        }
        var request = time == 0 && mean ? FunctionalGradientTestSupport.request(roots.getFirst(), targets,
                FunctionalGradientRequest.DisconnectedPolicy.ZERO) : strictRequest;
        var expansion = FirstOrderAutograd.expand(AutogradPreflight.preflight(CompileMode.FORWARD_AND_BACKWARD,
                roots, request, CompileTimeConstantGraph.Ingress.empty()).stageOne(), CompileTimeConstantGraph.Ingress.empty());
        if (time == 0) {
            if (!mean && seeds != 1) {
                // No cell or reverse history exists: the final-carry cotangent is a raw identity.
                assertSame(finalSeed, expansion.targetGradients().getFirst().gradient());
            }
        } else {
            TensorProducer sibling = expansion.targetGradients().getFirst().gradient().provenance().orElseThrow().producer();
            var sa = (ScanAttrs) sibling.operation().attrs();
            assertSame(lengths, sibling.inputs().getLast());
            var reverseBody = sibling.scanBody().orElseThrow();
            for (int i = sa.rowCarryCount(); i < sa.carryCount(); i++) {
                assertTrue(reverseBody.carry().get(i).descriptor().shape().isFullyStatic(), "no B-by-parameter banks");
                var addition = reverseBody.nextCarry().get(i).provenance().orElseThrow().producer();
                assertEquals(BinaryArithmeticKind.ADD, addition.operation().kind());
                assertSame(reverseBody.carry().get(i), addition.inputs().getFirst(), "distinct old.add(contribution) site");
            }
            Tensor savedTranspose = scan.residuals().get(body.retainedValues().indexOf(sharedTranspose));
            int sharedPosition = sibling.inputs().subList(sa.carryCount(), sa.carryCount()+sa.sequenceCount()).indexOf(savedTranspose);
            assertTrue(sharedPosition >= 0);
            assertEquals(OptionalInt.empty(), sa.sequenceRowAxes().get(sharedPosition));
            assertEquals(Shape.of(time, f, h), savedTranspose.descriptor().shape());
            if (seeds == 2) {
                Tensor saved = scan.residuals().get(body.retainedValues().indexOf(activation));
                int position = sibling.inputs().subList(sa.carryCount(), sa.carryCount()+sa.sequenceCount()).indexOf(saved);
                assertTrue(position >= 0);
                assertEquals(OptionalInt.of(1), sa.sequenceRowAxes().get(position));
                assertEquals(shape(fixed(time), fixed(h), B), saved.descriptor().shape());
                assertTrue(java.util.stream.IntStream.range(0, sa.sequenceCount()).anyMatch(i ->
                        sa.sequenceRowAxes().get(i).equals(OptionalInt.of(1))
                        && sibling.inputs().get(sa.carryCount()+i).provenance().map(p ->
                                p.producer().operation().kind() == ShapeTransformKind.EXPAND).orElse(false)),
                        "ROW(1) activation literals must be explicit [T,H,B] sequences");
            }
        }
        for (var optimization : List.of(GraphOptimizationConfig.disabled(), GraphOptimizationConfig.standard())) {
            var artifacts = compile(roots, Optional.of(request), optimization);
            var scans = artifacts.graph().nodes().stream().filter(n -> n.scanBody().isPresent()).toList();
            if (time > 0) {
                assertEquals(2, scans.size());
                assertNotEquals(((ScanAttrs) scans.getFirst().operation().attrs()).reverse(),
                        ((ScanAttrs) scans.getLast().operation().attrs()).reverse());
                assertTrue(scans.getLast().scanBody().orElseThrow().graph().nodes().stream().noneMatch(n ->
                        n.operation().kind() == UnaryElementwiseKind.TANH || n.operation().kind() == UnaryElementwiseKind.SIGMOID));
            }
            var plan = BatchBindingPlan.prepare(artifacts);
            var oldBindings = new ArrayList<BatchBindingPlan.Binding>();
            int invocation = 0;
            for (int batch : new int[] {0, 1, f, h, 5, 32, 32, 7, 1, 32}) {
                boolean view = invocation++ % 2 == 0;
                var supplied = inputs(artifacts, batch, view);
                var binding = plan.bind(supplied); oldBindings.add(binding);
                assertEquals(batch, binding.batchSize());
                assertPorts(artifacts, binding);
                for (int i = 0; i < supplied.size(); i++) assertSame(supplied.get(i), binding.descriptor(
                        index(artifacts, artifacts.constants().bindableInputs().get(i))));
                if (oldBindings.size() > 1) assertNotSame(binding, oldBindings.get(oldBindings.size() - 2));
                // At repeated B these are fresh storage/length payloads, not binder observations.
                for (int pattern = 0; pattern < 3; pattern++) {
                    double[] ls = new double[batch];
                    for (int b = 0; b < batch; b++) ls[b] = pattern == 0 ? 0 : pattern == 1 ? time : (b + invocation) % (time + 1);
                    var oracle = new ScanGradientRulesTest.Oracle(batch);
                    oracle.bind(initial, data(batch*h, .13 + invocation*.001));
                    double[] xs = data(time*batch*f, -.17), es = data(time*batch*h, -.23);
                    for (int t = 0; t < time; t++) for (int b = 0; b < batch; b++) if (t >= ls[b]) {
                        Arrays.fill(xs, (t*batch+b)*f, (t*batch+b+1)*f, Double.NaN);
                        Arrays.fill(es, (t*batch+b)*h, (t*batch+b+1)*h, Double.NaN);
                    }
                    oracle.bind(sequence, xs); oracle.bind(emissionSeed, es); oracle.bind(lengths, ls);
                    oracle.bind(wxa, data(f*h, .11)); oracle.bind(wha, data(h*h, -.07)); oracle.bind(ba, data(h, .04));
                    oracle.bind(finalSeed, data(batch*h, .31)); oracle.constants(expansion);
                    double[] expectedFinal = new double[batch*h], expectedEmit = new double[time*batch*h];
                    double[] expectedInitial = new double[batch*h], expectedSequence = new double[time*batch*f];
                    double[][] expectedWeights = {new double[f*h], new double[h*h], new double[h]};
                    for (int b = 0; b < batch; b++) {
                        var reference = new ScanGradientRulesTest.Oracle();
                        Tensor ih = tensor(type, true, 1, h), rwx = tensor(type, true, h, f);
                        Tensor rwh = tensor(type, true, h, h), rb = tensor(type, true, h);
                        reference.bind(ih, Arrays.copyOfRange(oracle.eval(initial), b*h, (b+1)*h));
                        reference.bind(rwx, oracle.eval(wxa)); reference.bind(rwh, oracle.eval(wha)); reference.bind(rb, oracle.eval(ba));
                        int valid = (int) ls[b];
                        var steps = new ArrayList<Tensor>();
                        var emitted = new Tensor[valid];
                        for (int t = 0; t < valid; t++) {
                            Tensor x = tensor(type, true, 1, f); steps.add(x);
                            reference.bind(x, Arrays.copyOfRange(oracle.eval(sequence), (t*batch+b)*f, (t*batch+b+1)*f));
                        }
                        Tensor value = ih;
                        for (int iteration = 0; iteration < valid; iteration++) {
                            int t = reverse ? valid - 1 - iteration : iteration;
                            Tensor update = steps.get(t).matmul(rwx.permute(1, 0)).add(value.matmul(rwh)).add(rb).tanh();
                            emitted[t] = update.sigmoid().add(value); value = update;
                        }
                        System.arraycopy(reference.eval(value), 0, expectedFinal, b*h, h);
                        for (int t = 0; t < valid; t++) System.arraycopy(reference.eval(emitted[t]), 0, expectedEmit, (t*batch+b)*h, h);
                        Tensor fs = tensor(type, false, 1, h);
                        double[] finalCotangent = new double[h];
                        for (int j = 0; j < h; j++) finalCotangent[j] = mean ? 1.0 / (batch*h)
                                : seeds == 1 ? 0 : oracle.eval(finalSeed)[b*h+j];
                        reference.bind(fs, finalCotangent);
                        Tensor objective = value.mul(fs).sum();
                        if (!mean && seeds != 0) for (int t = 0; t < valid; t++) {
                            Tensor seed = tensor(type, false, 1, h);
                            reference.bind(seed, Arrays.copyOfRange(oracle.eval(emissionSeed), (t*batch+b)*h, (t*batch+b+1)*h));
                            objective = objective.add(emitted[t].mul(seed).sum());
                        }
                        var rt = new ArrayList<>(List.of(ih));
                        if (valid > 0) { rt.addAll(List.of(rwx, rwh, rb)); rt.addAll(steps); }
                        var ordinary = FirstOrderAutograd.expand(AutogradPreflight.preflight(CompileMode.FORWARD_AND_BACKWARD,
                                List.of(objective), FunctionalGradientTestSupport.stage(objective, rt), CompileTimeConstantGraph.Ingress.empty()),
                                CompileTimeConstantGraph.Ingress.empty());
                        reference.constants(ordinary);
                        System.arraycopy(reference.eval(ordinary.targetGradients().getFirst().gradient()), 0, expectedInitial, b*h, h);
                        if (valid > 0) {
                            for (int w = 0; w < 3; w++) {
                                double[] gradient = reference.eval(ordinary.targetGradients().get(w+1).gradient());
                                for (int j = 0; j < gradient.length; j++) expectedWeights[w][j] += gradient[j];
                            }
                            for (int t = 0; t < valid; t++) System.arraycopy(reference.eval(ordinary.targetGradients().get(t+4).gradient()),
                                    0, expectedSequence, (t*batch+b)*f, f);
                        }
                    }
                    double tolerance = type == DataType.FLOAT32 ? 8e-5 : 3e-12;
                    assertArrayEquals(expectedFinal, oracle.eval(scan.finalCarry().getFirst()), tolerance);
                    assertArrayEquals(expectedEmit, oracle.eval(scan.outputs().getFirst()), tolerance);
                    assertArrayEquals(expectedInitial, oracle.eval(expansion.targetGradients().getFirst().gradient()), tolerance);
                    double[] dx = oracle.eval(expansion.targetGradients().get(1).gradient());
                    assertArrayEquals(expectedSequence, dx, tolerance);
                    for (int w = 0; w < 3; w++) assertArrayEquals(expectedWeights[w],
                            oracle.eval(expansion.targetGradients().get(w+2).gradient()), tolerance);
                    for (int t = 0; t < time; t++) for (int b = 0; b < batch; b++) if (t >= ls[b]) {
                        for (int j = 0; j < f; j++) assertEquals(0L, Double.doubleToRawLongBits(dx[(t*batch+b)*f+j]));
                        for (int j = 0; j < h; j++) assertEquals(0L,
                                Double.doubleToRawLongBits(oracle.eval(scan.outputs().getFirst())[(t*batch+b)*h+j]));
                    }
                    if (mean) {
                        double[] result = oracle.eval(roots.getFirst());
                        if (batch == 0) assertTrue(Double.isNaN(result[0]));
                        else assertEquals(Arrays.stream(expectedFinal).sum()/(batch*h), result[0], tolerance);
                    }
                    if (pattern == 0 || time == 0 || batch == 0) assertEquals(0, oracle.bodySteps);
                }
            }
            assertEquals(0, oldBindings.getFirst().batchSize());
            // Both directions' body size stays O(body), never proportional to fixed T.
            assertTrue(scans.stream().mapToInt(n -> n.scanBody().orElseThrow().graph().nodes().size()).sum() < 100);
        }
    }

    static List<TensorDescriptor> inputs(CompileArtifacts artifacts, long batch, boolean view) {
        return artifacts.constants().bindableInputs().stream().map(id -> actual(
                artifacts.graph().values().get(index(artifacts, id)).descriptor(), batch, view)).toList();
    }

    static void assertPorts(CompileArtifacts artifacts, BatchBindingPlan.Binding binding) {
        for (var node : artifacts.graph().nodes()) if (node.scanBody().isPresent()) {
            var a = (ScanAttrs) node.operation().attrs();
            assertEquals(a.inputCount()-1, node.scanBody().orElseThrow().graph().inputs().size());
            for (ValueId id : node.outputs()) {
                TensorDescriptor d = binding.descriptor(index(artifacts, id));
                assertEquals(LayoutDescriptor.contiguous(d.shape()), d.layout().orElseThrow());
                assertFalse(d.layout().orElseThrow().isView());
                if (!artifacts.graph().values().get(index(artifacts, id)).descriptor().shape().isFullyStatic())
                    assertTrue(artifacts.graph().values().get(index(artifacts, id)).descriptor().layout().isEmpty());
            }
        }
        for (var gradient : artifacts.publication().gradientBindings())
            assertTrue(binding.descriptor(index(artifacts, gradient.valueId())).shape().isFullyStatic());
    }
}
