package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.*;
import static io.github.pho001.synaptik.compiler.ScanBodyInferenceTest.tensor;
import static io.github.pho001.synaptik.compiler.SymbolicScanCompilerTest.*;

import io.github.pho001.synaptik.config.compile.*;
import io.github.pho001.synaptik.model.datatype.*;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.*;
import io.github.pho001.synaptik.model.operation.layout.*;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.operation.normalization.*;
import io.github.pho001.synaptik.model.operation.random.*;
import io.github.pho001.synaptik.model.operation.scan.ScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.ScanKind;
import io.github.pho001.synaptik.model.shape.*;
import io.github.pho001.synaptik.model.tensor.*;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Raw-IR and all-port witnesses independent of Model's Scan constructors. */
final class SymbolicScanInferenceTest {
    @Test void rawSymbolicLengthHeaderInfersIndependentlyOfModelSourceAdmission() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) for (long time : new long[] {0, 3}) {
            Tensor h = symbolic(type, false, B, fixed(2)), x = fresh(h, false);
            var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h.add(x)), List.of(h));
            var allValid = Scan.apply(List.of(fresh(h, false)), List.of(symbolic(type, false, fixed(time), B, fixed(2))),
                    List.of(), body, false);
            var raw = LengthAwareScanBodyInferenceTest.lengthIR(GraphCapture.capture(allValid.finalCarry()), 1,
                    List.of(OptionalInt.of(0)), symbolic(DataType.INT64, false, B).descriptor());
            assertTrue(CapturedGraphInference.inferAndValidate(raw).constraints().isEmpty());
        }
    }

    @Test void forwardKeepsLowOrdinaryAndDiscreteCanonicalSiblings() {
        for (DataType type : List.of(DataType.FLOAT16, DataType.BFLOAT16, DataType.FLOAT32)) {
            Tensor h = symbolic(type, true, B, fixed(3)), x = fresh(h, true);
            Tensor next = type == DataType.FLOAT32 ? h.add(x).relu() : h.add(x).tanh();
            var top = x.topK(2, 1);
            Tensor unused = symbolic(DataType.BOOL, false, B);
            var body = new ScanBody(List.of(h), List.of(x), List.of(unused), List.of(next), List.of(top.values()));
            var scan = Scan.apply(List.of(fresh(h, true)), List.of(symbolic(type, true, fixed(3), B, fixed(3))),
                    List.of(fresh(unused, false)), body, true);
            var artifacts = compile(scan.finalCarry(), Optional.empty(), GraphOptimizationConfig.standard());
            var plan = BatchBindingPlan.prepare(artifacts);
            var binding = plan.bind(inputs(artifacts, 2));
            var n = artifacts.graph().nodes().getFirst();
            for (ValueId id : n.outputs()) {
                var d = binding.descriptor(index(artifacts, id));
                assertFalse(d.layout().orElseThrow().isView());
                assertEquals(d.dataType().isDifferentiable(), d.requiresGrad());
            }
            assertDoesNotThrow(() -> ScanGradientRules.preflightSource(
                    scan.finalCarry().getFirst().provenance().orElseThrow().producer()));
        }
    }

    @Test void numericDomainDisjunctionRemainsFreshScopedBindingNotCoordinateGuess() {
        Tensor h = symbolic(DataType.FLOAT32, false, B, fixed(6)), x = fresh(h, false);
        var parameters = List.of(tensor(DataType.FLOAT32, false, 6), tensor(DataType.FLOAT32, false, 6),
                tensor(DataType.FLOAT32, false, 6), tensor(DataType.FLOAT32, false, 6));
        var trained = x.batchNormTraining(1, parameters.get(0), parameters.get(1), parameters.get(2), parameters.get(3),
                ScalarValue.float32(.1f), ScalarValue.float32(.00001f));
        var body = new ScanBody(List.of(h), List.of(x), parameters, List.of(h), List.of(trained.output()));
        var scan = Scan.apply(List.of(fresh(h, false)), List.of(symbolic(DataType.FLOAT32, false, fixed(0), B, fixed(6))),
                parameters.stream().map(t -> fresh(t, false)).toList(), body, false);
        var artifacts = compile(scan.finalCarry(), Optional.empty(), GraphOptimizationConfig.standard());
        var plan = BatchBindingPlan.prepare(artifacts);
        assertEquals(2, plan.bind(inputs(artifacts, 2)).batchSize());
        assertThrows(IllegalArgumentException.class, () -> plan.bind(inputs(artifacts, 1)));
        assertEquals(2, plan.bind(inputs(artifacts, 2)).batchSize());
        assertThrows(IllegalArgumentException.class, () -> plan.bind(inputs(artifacts, 0)));
    }

    @Test void deadCanonicalSiblingsInIndependentEqualIdScopesImposeEveryFreshObligation() {
        Tensor h = symbolic(DataType.FLOAT64, false, B, fixed(2)), x = fresh(h, false);
        var firstBody = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h), List.of(x.topK(2, 0).values()));
        Tensor h2 = fresh(h, false), x2 = fresh(h, false);
        var secondBody = new ScanBody(List.of(h2), List.of(x2), List.of(), List.of(h2), List.of(x2.topK(3, 0).values()));
        Tensor initial = fresh(h, false), seq = symbolic(DataType.FLOAT64, false, fixed(0), B, fixed(2));
        var first = Scan.apply(List.of(initial), List.of(seq), List.of(), firstBody, false);
        var second = Scan.apply(List.of(initial), List.of(seq), List.of(), secondBody, true);
        var artifacts = compile(List.of(first.finalCarry().getFirst(), second.finalCarry().getFirst()), Optional.empty(), GraphOptimizationConfig.standard());
        var plan = BatchBindingPlan.prepare(artifacts);
        for (int i = 0; i < 2; i++) {
            assertEquals(3, plan.bind(inputs(artifacts, 3)).batchSize());
            assertThrows(IllegalArgumentException.class, () -> plan.bind(inputs(artifacts, 2)));
        }
        var wrong = new ArrayList<>(inputs(artifacts, 3));
        int seqIndex = artifacts.constants().bindableInputBindings().stream().map(b -> b.tensorId()).toList().indexOf(seq.id());
        wrong.set(seqIndex, actual(seq.descriptor(), 4, false));
        assertThrows(IllegalArgumentException.class, () -> plan.bind(wrong));
    }

    @Test void deadOverflowRejectsDuringBindEvenForZeroBatchAndSeparateZeroTime() {
        for (long time : new long[] {0, 3}) {
            Tensor h = symbolic(DataType.FLOAT32, false, B, fixed(1)), x = fresh(h, false);
            Tensor scalar = tensor(DataType.FLOAT32, false);
            Tensor huge = scalar.expand(shape(B, fixed(0), fixed(Long.MAX_VALUE), fixed(2)));
            var body = new ScanBody(List.of(h), List.of(x), List.of(scalar), List.of(h), List.of(huge));
            var scan = Scan.apply(List.of(fresh(h, false)), List.of(symbolic(DataType.FLOAT32, false, fixed(time), B, fixed(1))),
                    List.of(fresh(scalar, false)), body, false);
            var artifacts = compile(scan.finalCarry(), Optional.empty(), GraphOptimizationConfig.disabled());
            var plan = BatchBindingPlan.prepare(artifacts);
            for (long batch : new long[] {0, 1, 0})
                assertThrows(ArithmeticException.class, () -> plan.bind(inputs(artifacts, batch)));
        }
    }

    @Test void inputBackedPublicationUsesThisCallsDescriptorAndViewEvenAtRepeatedBatch() {
        Tensor h = symbolic(DataType.FLOAT32, false, B, fixed(2)), x = fresh(h, false);
        Tensor initial = fresh(h, false), seq = symbolic(DataType.FLOAT32, false, fixed(3), B, fixed(2));
        var scan = Scan.apply(List.of(initial), List.of(seq), List.of(),
                new ScanBody(List.of(h), List.of(x), List.of(), List.of(h.add(x)), List.of()), false);
        var artifacts = compile(List.of(initial, scan.finalCarry().getFirst()), Optional.empty(), GraphOptimizationConfig.standard());
        var plan = BatchBindingPlan.prepare(artifacts);
        int published = index(artifacts, artifacts.publication().forwardBindings().getFirst().valueId());
        var bindings = new ArrayList<BatchBindingPlan.Binding>();
        for (boolean view : List.of(false, true, false)) {
            var supplied = artifacts.constants().bindableInputs().stream().map(id -> actual(
                    artifacts.graph().values().get(index(artifacts, id)).descriptor(), 4, view)).toList();
            var binding = plan.bind(supplied); bindings.add(binding);
            int input = artifacts.constants().bindableInputs().indexOf(artifacts.publication().forwardBindings().getFirst().valueId());
            assertSame(supplied.get(input), binding.descriptor(published));
            assertEquals(view, binding.descriptor(published).layout().orElseThrow().isView());
        }
        assertFalse(bindings.get(0).descriptor(published).layout().orElseThrow().isView());
        assertTrue(bindings.get(1).descriptor(published).layout().orElseThrow().isView());
    }

    @Test void rawIrRejectsNonPureFamilyThoughOrdinaryInferenceAcceptsIt() {
        TensorDescriptor state = tensor(DataType.INT64, false, 2).descriptor();
        var values = List.of(new GraphValue(new ValueId(0), state), new GraphValue(new ValueId(1), state),
                new GraphValue(new ValueId(2), new TensorDescriptor(DataType.INT64, state.shape(), Optional.empty(), false)));
        var rng = new CompiledNode(new NodeId(0), new Operation(GraphRngKind.INITIAL_STATE, new GraphRngStateAttrs(1, 2)), List.of(), List.of(new ValueId(2)));
        var local = new CompiledGraphModel(values, List.of(rng), List.of(new ValueId(0), new ValueId(1)), List.of(new ValueId(2), new ValueId(0)), Map.of(rng.id(), GraphPhase.FORWARD));
        assertDoesNotThrow(() -> CapturedGraphInference.inferAndValidate(local));
        var body = new CompiledScanBody(local, List.of(new ValueId(2)), List.of(), List.of(new ValueId(2), new ValueId(0)), List.of());
        var outer = raw(body, List.of(state, tensor(DataType.INT64, false, 0, 2).descriptor()), 0);
        assertThrows(IllegalArgumentException.class, () -> CapturedGraphInference.inferAndValidate(outer));
    }

    @Test void rawIrChecksUnusedForeignRepeatedAnonymousAndResolvedSymbolicDeclarations() {
        Tensor h = symbolic(DataType.FLOAT64, false, B, fixed(2)), x = fresh(h, false), unused = fresh(h, false);
        var source = Scan.apply(List.of(fresh(h, false)), List.of(symbolic(DataType.FLOAT64, false, fixed(0), B, fixed(2))),
                List.of(fresh(unused, false)), new ScanBody(List.of(h), List.of(x), List.of(unused), List.of(h), List.of()), false);
        var graph = GraphCapture.capture(source.finalCarry()); var node = graph.nodes().getFirst();
        var body = node.scanBody().orElseThrow(); ValueId unusedId = body.graph().inputs().getLast();
        for (Shape malformed : List.of(shape(new DynamicDimension("Foreign"), fixed(2)), shape(B, B),
                shape(DimensionExpressions.addConstant(B, 1), fixed(2)),
                shape(DimensionExpressions.unknown(0, Optional.empty()), fixed(2)))) {
            var locals = body.graph().values().stream().map(v -> v.id().equals(unusedId)
                    ? new GraphValue(v.id(), new TensorDescriptor(DataType.FLOAT64, malformed, Optional.empty(), false)) : v).toList();
            var altered = new CompiledScanBody(new CompiledGraphModel(locals, body.graph().nodes(), body.graph().inputs(), body.graph().outputs(), body.graph().nodePhases()),
                    body.nextCarry(), body.emissions(), body.residuals(), List.of());
            var badNode = new CompiledNode(node.id(), node.operation(), node.inputs(), node.outputs(), Optional.of(altered));
            var bad = new CompiledGraphModel(graph.values(), List.of(badNode), graph.inputs(), graph.outputs(), graph.nodePhases());
            assertThrows(IllegalArgumentException.class, () -> CapturedGraphInference.inferAndValidate(bad));
        }
    }

    @Test void rawMatmulCannotEquateNamedContractionToFixedFeatureAtOneBatch() {
        TensorDescriptor row = symbolic(DataType.FLOAT64, false, fixed(2), B).descriptor();
        TensorDescriptor weight = tensor(DataType.FLOAT64, false, 3, 2).descriptor();
        TensorDescriptor output = new TensorDescriptor(DataType.FLOAT64, shape(fixed(2), fixed(2)), Optional.empty(), false);
        var values = List.of(new GraphValue(new ValueId(0), row), new GraphValue(new ValueId(1), row),
                new GraphValue(new ValueId(2), weight), new GraphValue(new ValueId(3), output));
        var matmul = new CompiledNode(new NodeId(0), new Operation(MatmulKind.MATMUL, NoOperationAttrs.INSTANCE),
                List.of(new ValueId(1), new ValueId(2)), List.of(new ValueId(3)));
        var local = new CompiledGraphModel(values, List.of(matmul), List.of(new ValueId(0), new ValueId(1), new ValueId(2)),
                List.of(new ValueId(0), new ValueId(3)), Map.of(matmul.id(), GraphPhase.FORWARD));
        assertFalse(CapturedGraphInference.inferAndValidate(local).constraints().isEmpty());
        var body = new CompiledScanBody(local, List.of(new ValueId(0)), List.of(new ValueId(3)), List.of(new ValueId(0), new ValueId(3)), List.of());
        assertThrows(IllegalArgumentException.class, () -> CapturedGraphInference.inferAndValidate(raw(body,
                List.of(row, symbolic(DataType.FLOAT64, false, fixed(0), fixed(2), B).descriptor(), weight), 0)));
    }

    @Test void rawAttentionRejectsAmbiguousBatchBroadcastWhileNumericDomainDisjunctionBinds() {
        Tensor q = symbolic(DataType.FLOAT32, false, B, fixed(2), fixed(6));
        Tensor key = tensor(DataType.FLOAT32, false, 32, 2, 6), value = tensor(DataType.FLOAT32, false, 32, 2, 4);
        Tensor attention = q.scaledDotProductAttention(key, value);
        var local = GraphCapture.captureBody(List.of(q, attention), List.of(q, key, value));
        assertFalse(CapturedGraphInference.inferAndValidate(local).constraints().isEmpty());
        var body = new CompiledScanBody(local, List.of(local.inputs().getFirst()), List.of(local.outputs().getLast()),
                local.outputs(), List.of());
        assertThrows(IllegalArgumentException.class, () -> CapturedGraphInference.inferAndValidate(raw(body,
                List.of(q.descriptor(), tensor(DataType.FLOAT32, false, 0, 32, 2, 6).descriptor(), value.descriptor()), 0)));
    }

    @Test void rawShapeAttributeCannotHideForeignSymbolBehindUnchangedOutputShape() {
        Tensor h = symbolic(DataType.FLOAT32, false, B, fixed(2)), x = fresh(h, false);
        Tensor normalized = x.layerNorm(shape(fixed(2)), ScalarValue.float32(.001f));
        var local = GraphCapture.captureBody(List.of(h, normalized), List.of(h, x)); var n = local.nodes().getFirst();
        var altered = new CompiledNode(n.id(), new Operation(LayerNormKind.LAYER_NORM,
                new LayerNormAttrs(shape(new DynamicDimension("Foreign")), ScalarValue.float32(.001f))), n.inputs(), n.outputs());
        var graph = new CompiledGraphModel(local.values(), List.of(altered), local.inputs(), local.outputs(), local.nodePhases());
        assertFalse(CapturedGraphInference.inferAndValidate(graph).constraints().isEmpty());
        var body = new CompiledScanBody(graph, List.of(local.outputs().getFirst()), List.of(local.outputs().getLast()), local.outputs(), List.of());
        assertThrows(IllegalArgumentException.class, () -> CapturedGraphInference.inferAndValidate(raw(body,
                List.of(h.descriptor(), symbolic(DataType.FLOAT32, false, fixed(0), B, fixed(2)).descriptor()), 0)));
    }


    private static List<TensorDescriptor> inputs(CompileArtifacts artifacts, long batch) {
        return artifacts.constants().bindableInputs().stream().map(id -> actual(
                artifacts.graph().values().get(index(artifacts, id)).descriptor(), batch, false)).toList();
    }

    /** Builds raw canonical outer ports without Model Scan construction or source validation. */
    private static CompiledGraphModel raw(CompiledScanBody body, List<TensorDescriptor> inputs, long time) {
        var local = new HashMap<ValueId, TensorDescriptor>(); body.graph().values().forEach(v -> local.put(v.id(), v.descriptor()));
        var values = new ArrayList<GraphValue>(); var in = new ArrayList<ValueId>(); var out = new ArrayList<ValueId>();
        for (TensorDescriptor d : inputs) { var id = new ValueId(values.size()); values.add(new GraphValue(id, d)); in.add(id); }
        var roles = new ArrayList<>(body.nextCarry()); roles.addAll(body.emissions()); roles.addAll(body.residuals());
        for (int i = 0; i < roles.size(); i++) {
            var d = local.get(roles.get(i)); var dims = new ArrayList<>(d.shape().dimensions());
            if (i >= body.nextCarry().size()) dims.addFirst(fixed(time));
            Shape shape = shape(dims.toArray(Dimension[]::new)); var id = new ValueId(values.size());
            values.add(new GraphValue(id, new TensorDescriptor(d.dataType(), shape, shape.isFullyStatic()
                    ? Optional.of(LayoutDescriptor.contiguous(shape)) : Optional.empty(), false))); out.add(id);
        }
        var attrs = new ScanAttrs(time, body.nextCarry().size(), 1, inputs.size() - body.nextCarry().size() - 1,
                body.emissions().size(), body.residuals().size(), false);
        return new CompiledGraphModel(values, List.of(new CompiledNode(new NodeId(0), new Operation(ScanKind.SCAN, attrs), in, out, Optional.of(body))), in, List.of(out.getFirst()), Map.of(new NodeId(0), GraphPhase.FORWARD));
    }
}
