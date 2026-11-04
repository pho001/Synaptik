package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.operation.scan.ScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.ScanKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.planning.capability.ScanActivityPlan;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;
import org.junit.jupiter.api.Test;

/** Binding/resource behavior; actual arithmetic and Compiler reverse execute in the native test. */
class MetalScanPlanTest {
    @Test void stridedTypedControlsRejectColdWhileProvedViewsAndNumericMappingsRemainAdmitted() {
        var numeric = additiveQuery(3, 2, true, false);
        var inputs = new java.util.ArrayList<>(numeric.inputs());
        inputs.set(inputs.size() - 1, mapped(inputs.getLast(), new long[] {2}, 1));
        assertFalse(MetalScanLowering.supports(new OperationCapabilityQuery(numeric.operation(),
                inputs, numeric.outputs(), numeric.scanBody(), numeric.scanActivityPlan())));
        for (var type : List.of(DataType.INT32, DataType.INT64, DataType.BOOL)) {
            var original = sourceOnlyQuery(type, 2, 2, false, false);
            var controls = new java.util.ArrayList<>(original.inputs());
            controls.set(1, mapped(controls.get(1), new long[] {5, 2, 1}, 1));
            assertFalse(MetalScanLowering.supports(new OperationCapabilityQuery(original.operation(),
                    controls, original.outputs(), original.scanBody(), original.scanActivityPlan())));
            var physical = new java.util.ArrayList<MetalMpsGraphProgram.ValueDescriptor>();
            original.inputs().forEach(d -> physical.add(MetalMpsGraphProgram.ValueDescriptor.from(d)));
            original.outputs().forEach(d -> physical.add(MetalMpsGraphProgram.ValueDescriptor.from(d)));
            var node = MetalScanLowering.lower(original, new int[] {0, 1, 2}, new int[] {3, 4, 5});
            physical.set(1, MetalMpsGraphProgram.ValueDescriptor.from(controls.get(1),
                    MetalMpsGraphProgram.ValueState.AFFINE_VIEW));
            assertDoesNotThrow(() -> MetalScanLowering.validate(node, physical),
                    "an independently proved produced representation keeps its dense physical binding");
        }
        assertTrue(MetalScanLowering.supports(stridedAdditiveQuery(DataType.FLOAT32, true, false)));
    }

    @Test void failurePrefixIsPartOfExactColdWorkspaceAuthentication() {
        var plan = MetalScanLowering.plan(sourceOnlyQuery(DataType.BOOL, 2, 2, false, false));
        assertEquals(15, plan.controlWords());
        long[] old = MetalScanEncoding.encode(plan);
        old[31] -= 5; old[22] -= 40;
        assertThrows(IllegalArgumentException.class, () -> MetalScanEncoding.validate(old));
    }

    @Test void bindingsRetainPreviousCarryAndRepeatedDistinctRolesWithoutTimeExpansion() {
        var small = MetalScanLowering.plan(additiveQuery(2, 3, false, false));
        var large = MetalScanLowering.plan(additiveQuery(200, 3, false, false));
        assertEquals(List.of(0, 1), small.formals());
        assertEquals(List.of(2, 2, 0), small.roles());
        assertEquals(List.of(0, 1, 2), small.demanded());
        assertEquals(small.workspaceBytes(), large.workspaceBytes(), "only outer demanded histories grow with T");
        assertEquals(-1, small.values().get(1).byteOffset(), "sequence is an offset binding, not a local copy");
        assertNotEquals(small.values().get(0).byteOffset(), small.values().get(2).byteOffset(), "old carry survives next-carry production");
    }

    @Test void demandDoesNotTurnLatentResidualsIntoPhysicalHistories() {
        var query = additiveQuery(4, 2, true, true);
        var original = query.scanBody().orElseThrow();
        var selected = new CompiledScanBody(original.graph(), original.nextCarry(), original.emissions(), original.residuals(), List.of(0));
        var plan = MetalScanLowering.plan(new OperationCapabilityQuery(query.operation(), query.inputs(), query.outputs(), Optional.of(selected), Optional.empty()));
        assertSame(selected, plan.body());
        assertEquals(List.of(0), plan.demanded());
        assertEquals(List.of(2, 2, 0), plan.roles(), "latent canonical identity is retained independently of allocation");
    }

    @Test void zeroTimeAndZeroBatchDoNotEraseBindingsOrStructuralValidation() {
        for (var type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16))
            for (boolean lengths : List.of(false, true)) for (boolean reverse : List.of(false, true)) {
            for (long time : new long[] {0, 3}) for (long batch : new long[] {0, 1}) {
                var query = additiveQuery(type, time, batch, lengths, reverse);
                assertTrue(MetalScanLowering.supports(query));
                var plan = MetalScanLowering.plan(query);
                assertEquals(reverse, plan.header().reverse());
                assertEquals(batch * type.byteWidth(), plan.values().getFirst().byteSize());
                assertDoesNotThrow(() -> MetalScanEncoding.validate(MetalScanEncoding.encode(plan)));
            }
        }
        var valid = additiveQuery(0, 0, true, false);
        var wrong = new OperationCapabilityQuery(valid.operation(), valid.inputs(),
                List.of(descriptor(DataType.FLOAT32, 1, 1), valid.outputs().get(1), valid.outputs().get(2)), valid.scanBody(), Optional.empty());
        assertFalse(MetalScanLowering.supports(wrong), "T=B=0 does not hide a malformed carry result");
    }

    @Test void mismatchedCarriersMissingBodiesAndSymbolicDomainsAreNotGrantedByTheBinder() {
        var valid = additiveQuery(3, 2, false, false);
        var inputs = new java.util.ArrayList<>(valid.inputs());
        inputs.set(0, descriptor(DataType.FLOAT16, 2, 1));
        assertFalse(MetalScanLowering.supports(new OperationCapabilityQuery(valid.operation(), inputs, valid.outputs(), valid.scanBody(), Optional.empty())));
        assertFalse(MetalScanLowering.supports(new OperationCapabilityQuery(valid.operation(), valid.inputs(), valid.outputs())));
        assertFalse(MetalScanLowering.supports(symbolicQuery()), "CPU named-B qualification is not Metal admission");
        assertFalse(MetalScanLowering.supports(additiveQuery(DataType.FLOAT64, 3, 2, false, false)),
                "mapped numeric ingress does not admit F64 arithmetic");
    }

    @Test void rawConstructorRejectsStaleGrammarOverlappingWritersAndRepeatedDemand() {
        var plan = MetalScanLowering.plan(additiveQuery(3, 2, false, false));
        long[] valid = MetalScanEncoding.encode(plan);
        assertDoesNotThrow(() -> MetalScanEncoding.validate(valid));
        long[] stale = valid.clone(); stale[1] = 1; // Prior control prefix has no failure record.
        assertThrows(IllegalArgumentException.class, () -> MetalScanEncoding.validate(stale));
        long[] repeated = valid.clone(); repeated[Math.toIntExact(repeated[27]) + 1] = 0;
        assertThrows(IllegalArgumentException.class, () -> MetalScanEncoding.validate(repeated));
        long[] overlap = valid.clone(); int local = Math.toIntExact(overlap[23]);
        for (int i = 0; i < 2; i++) local += Math.toIntExact(overlap[local]);
        overlap[local + 7] = plan.values().getFirst().byteOffset();
        assertThrows(IllegalArgumentException.class, () -> MetalScanEncoding.validate(overlap));
        long[] wideType = valid.clone(); wideType[Math.toIntExact(wideType[23]) + 1] = 0x1_0000_0001L;
        assertThrows(IllegalArgumentException.class, () -> MetalScanEncoding.validate(wideType));
    }

    @Test void lowCarrierWidthsAndOriginalGradientFlagsSurviveTheIndependentCodec() {
        for (var type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
            var query = additiveQuery(type, 3, 2, false, false);
            var original = query.scanBody().orElseThrow();
            var graph = original.graph();
            var values = graph.values().stream().map(v -> new GraphValue(v.id(), gradient(v.descriptor()))).toList();
            var differentiated = new CompiledGraphModel(values, graph.nodes(), graph.inputs(), graph.outputs(),
                    Map.of(graph.nodes().getFirst().id(), GraphPhase.FORWARD));
            var body = new CompiledScanBody(differentiated, original.nextCarry(), original.emissions(),
                    original.residuals(), original.demandedOutputIndices());
            var typed = new OperationCapabilityQuery(query.operation(), query.inputs().stream().map(MetalScanPlanTest::gradient).toList(),
                    query.outputs().stream().map(MetalScanPlanTest::gradient).toList(), Optional.of(body), Optional.empty());
            var plan = MetalScanLowering.plan(typed);
            var decoded = MetalScanDecoding.decode(MetalScanEncoding.encode(plan));
            assertTrue(decoded.values().stream().allMatch(v -> v.descriptor().requiresGrad()));
            assertEquals(2L * type.byteWidth(), decoded.values().getFirst().byteSize());
            long[] forged = MetalScanEncoding.encode(plan);
            forged[Math.toIntExact(forged[23]) + 8] = 2L * (type.byteWidth() == 4 ? 2 : 4);
            assertThrows(IllegalArgumentException.class, () -> MetalScanEncoding.validate(forged),
                    "physical byte count must use the original carrier");
        }
    }

    private static TensorDescriptor gradient(TensorDescriptor descriptor) {
        return new TensorDescriptor(descriptor.dataType(), descriptor.shape(), descriptor.layout(), true);
    }

    @Test void rawOuterReferencesContainOnlyDemandedCanonicalPorts() {
        var query = additiveQuery(3, 2, true, false);
        var old = query.scanBody().orElseThrow();
        var body = new CompiledScanBody(old.graph(), old.nextCarry(), old.emissions(), old.residuals(), List.of(0));
        var selected = new OperationCapabilityQuery(query.operation(), query.inputs(), query.outputs(), Optional.of(body), Optional.empty());
        var values = new java.util.ArrayList<MetalMpsGraphProgram.ValueDescriptor>();
        query.inputs().forEach(d -> values.add(MetalMpsGraphProgram.ValueDescriptor.from(d)));
        values.add(MetalMpsGraphProgram.ValueDescriptor.from(query.outputs().getFirst()));
        var node = MetalScanLowering.lower(selected, new int[] {0, 1, 2}, new int[] {3});
        assertDoesNotThrow(() -> MetalScanLowering.validate(node, values));
    }

    @Test void producedLogicalViewsAreNotRemappedButExternalNumericViewsAreMapped() {
        var original = additiveQuery(3, 2, true, false);
        var carryShape = original.inputs().getFirst().shape();
        var expanded = new TensorDescriptor(DataType.FLOAT32, carryShape,
                Optional.of(LayoutDescriptor.of(carryShape, new long[] {0, 0}, 0, true)), false);
        var sequenceShape = original.inputs().get(1).shape();
        var transposed = new TensorDescriptor(DataType.FLOAT32, sequenceShape,
                Optional.of(LayoutDescriptor.of(sequenceShape, new long[] {1, 3, 1}, 0, true)), false);
        var query = new OperationCapabilityQuery(original.operation(),
                List.of(expanded, transposed, original.inputs().getLast()), original.outputs(),
                original.scanBody(), original.scanActivityPlan());
        assertDoesNotThrow(() -> MetalScanLowering.plan(query),
                "logical view descriptors are not the represented storage proof");
        var values = new java.util.ArrayList<MetalMpsGraphProgram.ValueDescriptor>();
        values.add(MetalMpsGraphProgram.ValueDescriptor.from(expanded, MetalMpsGraphProgram.ValueState.AFFINE_VIEW));
        values.add(MetalMpsGraphProgram.ValueDescriptor.from(transposed, MetalMpsGraphProgram.ValueState.AFFINE_VIEW));
        values.add(MetalMpsGraphProgram.ValueDescriptor.from(query.inputs().getLast()));
        query.outputs().forEach(d -> values.add(MetalMpsGraphProgram.ValueDescriptor.from(d)));
        var node = MetalScanLowering.lower(query, new int[] {0, 1, 2}, new int[] {3, 4, 5});
        assertDoesNotThrow(() -> MetalScanLowering.validate(node, values));
        values.set(0, MetalMpsGraphProgram.ValueDescriptor.from(expanded));
        assertDoesNotThrow(() -> MetalScanLowering.validate(node, values));
        long ingressBytes = expanded.dataType().byteWidth() * expanded.shape().knownElementCount().orElseThrow();
        assertEquals(planWorkspace(query) + ingressBytes, MetalProgramWorkspace.scratchBytes(node, values),
                "only the external carry requires canonical ingress storage");
    }

    @Test void stridedOffsetIngressRetainsBoundedDenseBodyAndCheckedOuterWorkspace() {
        for (var type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16))
            for (boolean lengths : List.of(false, true)) for (boolean reverse : List.of(false, true)) {
                var query = stridedAdditiveQuery(type, lengths, reverse);
                assertTrue(MetalScanLowering.supports(query));
                var values = new java.util.ArrayList<MetalMpsGraphProgram.ValueDescriptor>();
                query.inputs().forEach(d -> values.add(MetalMpsGraphProgram.ValueDescriptor.from(d)));
                query.outputs().forEach(d -> values.add(MetalMpsGraphProgram.ValueDescriptor.from(d)));
                int[] inputs = java.util.stream.IntStream.range(0, query.inputs().size()).toArray();
                int[] outputs = java.util.stream.IntStream.range(query.inputs().size(), values.size()).toArray();
                var node = MetalScanLowering.lower(query, inputs, outputs);
                assertDoesNotThrow(() -> MetalScanLowering.validate(node, values));
                long base = planWorkspace(query);
                long afterCarry = base + 2L * type.byteWidth();
                long expected = ((afterCarry + 7L) & ~7L) + 6L * type.byteWidth();
                assertEquals(expected, MetalProgramWorkspace.scratchBytes(node, values));
                var plan = MetalScanLowering.plan(query);
                assertEquals(-1, plan.values().get(1).byteOffset(),
                        "the dense body still slices its canonical ingress, never the external stride");
                assertDoesNotThrow(() -> MetalScanEncoding.validate(MetalScanEncoding.encode(plan)));
                if (lengths) {
                    var malformed = new java.util.ArrayList<>(values);
                    malformed.set(inputs.length - 1, MetalMpsGraphProgram.ValueDescriptor.from(
                            mapped(query.inputs().getLast(), new long[] {2}, 1)));
                    assertThrows(IllegalArgumentException.class, () -> MetalScanLowering.validate(node, malformed),
                            "lengths are typed controls, not numeric mapped ingress");
                }
            }
    }

    private static long planWorkspace(OperationCapabilityQuery query) {
        return MetalScanLowering.plan(query).workspaceBytes();
    }

    static OperationCapabilityQuery stridedAdditiveQuery(DataType type, boolean lengths, boolean reverse) {
        var original = additiveQuery(type, 3, 2, lengths, reverse);
        var inputs = new java.util.ArrayList<>(original.inputs());
        inputs.set(0, mapped(inputs.get(0), new long[] {2, 1}, 1));
        inputs.set(1, mapped(inputs.get(1), new long[] {5, 2, 1}, 1));
        return new OperationCapabilityQuery(original.operation(), inputs, original.outputs(),
                original.scanBody(), original.scanActivityPlan());
    }

    static TensorDescriptor mapped(TensorDescriptor descriptor, long[] strides, long offset) {
        return new TensorDescriptor(descriptor.dataType(), descriptor.shape(),
                Optional.of(LayoutDescriptor.of(descriptor.shape(), strides, offset, true)),
                descriptor.requiresGrad());
    }

    @Test void rawCoordinatesAreProvedFromOperationsDespiteEqualBatchAndFeatureExtents() {
        for (long batch : new long[] {0, 1, 2}) for (long time : new long[] {0, 2}) {
            var plan = MetalScanLowering.plan(transposedQuery(time, batch, false));
            assertDoesNotThrow(() -> MetalScanEncoding.validate(MetalScanEncoding.encode(plan)));
            long[] forged = forgedTransposedCoordinates(plan);
            assertThrows(IllegalArgumentException.class, () -> MetalScanEncoding.validate(forged));
            assertThrows(IllegalArgumentException.class, () -> MetalMpsGraphProgram.Node.generic(
                    MetalMpsGraphProgram.NodeKind.GENERAL_SCAN, new int[] {0, 1, 2}, new int[] {3, 4, 5},
                    MetalMpsGraphProgram.AttributeKind.GENERAL_SCAN, forged));
        }
    }

    static OperationCapabilityQuery transposedQuery(long time, long batch, boolean reverse) {
        var step = descriptor(DataType.FLOAT32, batch, batch);
        var computed = new TensorDescriptor(DataType.FLOAT32, step.shape(), Optional.empty(), false);
        var h = new ValueId(0); var x = new ValueId(1); var middle = new ValueId(2); var next = new ValueId(3);
        var operation = new Operation(AxisTransformKind.PERMUTE, new PermutationAttrs(List.of(1, 0)));
        var first = new CompiledNode(new NodeId(0), operation, List.of(x), List.of(middle));
        var second = new CompiledNode(new NodeId(1), operation, List.of(middle), List.of(next));
        var graph = new CompiledGraphModel(List.of(new GraphValue(h, step), new GraphValue(x, step),
                new GraphValue(middle, computed), new GraphValue(next, computed)), List.of(first, second), List.of(h, x),
                List.of(next, middle), Map.of(first.id(), GraphPhase.FORWARD, second.id(), GraphPhase.FORWARD));
        var body = new CompiledScanBody(graph, List.of(next), List.of(next), List.of(middle), List.of(0, 1, 2));
        var attrs = new ScanAttrs(time, 1, 1, 0, 1, 1, reverse, true, 1, List.of(OptionalInt.of(0)));
        var history = descriptor(DataType.FLOAT32, time, batch, batch);
        return new OperationCapabilityQuery(new Operation(ScanKind.SCAN, attrs),
                List.of(step, history, descriptor(DataType.INT64, batch)), List.of(step, history, history),
                Optional.of(body), Optional.empty());
    }

    /** Mutates a consistent claim, not its shape: both equal extents remain legitimate B. */
    static long[] forgedTransposedCoordinates(MetalScanPlan plan) {
        long[] words = MetalScanEncoding.encode(plan);
        int middle = plan.units().getFirst().node().outputs()[0];
        int record = Math.toIntExact(words[23]);
        for (int v = 0; v < middle; v++) record += Math.toIntExact(words[record]);
        words[record + 9] = 0;
        words[record + 11] = plan.batch();
        words[Math.toIntExact(words[24]) + 9] = 0;
        return words;
    }

    @Test void qualificationBoundsDenseSerializedMembershipsNotJustWorkspaceBitsets() {
        var within = denseActivityQuery(480);
        assertTrue(MetalScanLowering.supports(within));
        var plan = MetalScanLowering.plan(within);
        long[] image = MetalScanEncoding.encode(plan);
        assertEquals(MetalScanEncoding.wordCount(plan), image.length);
        assertTrue(image.length <= MetalScanEncoding.MAX_WORDS);
        assertTrue(plan.controlWords() < MetalScanEncoding.MAX_WORDS);
        var oversized = denseActivityQuery(512);
        assertFalse(MetalScanLowering.supports(oversized));
        assertThrows(IllegalArgumentException.class, () -> MetalScanLowering.plan(oversized));
    }

    private static OperationCapabilityQuery denseActivityQuery(int denseScopes) {
        int seeds = 2048;
        var scalar = descriptor(DataType.FLOAT32);
        var values = new java.util.ArrayList<GraphValue>();
        var inputs = new java.util.ArrayList<ValueId>();
        var actuals = new java.util.ArrayList<TensorDescriptor>();
        for (int f = 0; f < seeds + 2; f++) {
            var id = new ValueId(f); inputs.add(id); values.add(new GraphValue(id, scalar));
            actuals.add(f == 0 ? scalar : descriptor(DataType.FLOAT32, 1));
        }
        var graph = new CompiledGraphModel(values, List.of(), inputs, List.of(inputs.getFirst()), Map.of());
        var body = new CompiledScanBody(graph, List.of(inputs.getFirst()), List.of(), List.of(), List.of(0));
        var memberships = new java.util.ArrayList<List<Integer>>();
        memberships.add(List.of(0));
        var dense = java.util.stream.IntStream.range(0, seeds + 1).boxed().toList();
        for (int s = 0; s < denseScopes; s++) memberships.add(dense);
        var activity = new ScanActivityPlan(graph, List.of(new ScanActivityPlan.Carry(0, true, 0)),
                java.util.stream.IntStream.range(2, seeds + 2).boxed().toList(), memberships, Map.of(), Map.of(), List.of());
        return new OperationCapabilityQuery(new Operation(ScanKind.SCAN, new ScanAttrs(1, 1, seeds + 1, 0, 0, 0, false)),
                actuals, List.of(scalar), Optional.of(body), Optional.of(activity));
    }

    private static OperationCapabilityQuery symbolicQuery() {
        var batch = new io.github.pho001.synaptik.model.shape.DynamicDimension("Rows");
        var feature = new io.github.pho001.synaptik.model.shape.StaticDimension(1);
        var time = new io.github.pho001.synaptik.model.shape.StaticDimension(3);
        var step = new TensorDescriptor(DataType.FLOAT32, Shape.ofDimensions(batch, feature), Optional.empty(), false);
        var stacked = new TensorDescriptor(DataType.FLOAT32, Shape.ofDimensions(time, batch, feature), Optional.empty(), false);
        var h = new ValueId(0); var x = new ValueId(1); var next = new ValueId(2);
        var add = new CompiledNode(new NodeId(0), new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE), List.of(h, x), List.of(next));
        var graph = new CompiledGraphModel(List.of(new GraphValue(h, step), new GraphValue(x, step), new GraphValue(next, step)),
                List.of(add), List.of(h, x), List.of(next), Map.of(add.id(), GraphPhase.FORWARD));
        var body = new CompiledScanBody(graph, List.of(next), List.of(), List.of(), List.of(0));
        return new OperationCapabilityQuery(new Operation(ScanKind.SCAN, new ScanAttrs(3, 1, 1, 0, 0, 0, false)),
                List.of(step, stacked), List.of(step), Optional.of(body), Optional.empty());
    }

    static OperationCapabilityQuery additiveQuery(long time, long batch, boolean lengths, boolean reverse) {
        return additiveQuery(DataType.FLOAT32, time, batch, lengths, reverse);
    }

    static OperationCapabilityQuery additiveQuery(DataType type, long time, long batch, boolean lengths, boolean reverse) {
        var step = descriptor(type, batch, 1);
        var h = new ValueId(0); var x = new ValueId(1); var next = new ValueId(2);
        var add = new CompiledNode(new NodeId(0), new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE), List.of(h, x), List.of(next));
        var graph = new CompiledGraphModel(List.of(new GraphValue(h, step), new GraphValue(x, step), new GraphValue(next, step)),
                List.of(add), List.of(h, x), List.of(next, h), Map.of(add.id(), GraphPhase.FORWARD));
        var body = new CompiledScanBody(graph, List.of(next), List.of(next), List.of(h), List.of(0, 1, 2));
        var attrs = lengths ? new ScanAttrs(time, 1, 1, 0, 1, 1, reverse, true, 1, List.of(OptionalInt.of(0)))
                : new ScanAttrs(time, 1, 1, 0, 1, 1, reverse);
        var inputs = new java.util.ArrayList<>(List.of(step, descriptor(type, time, batch, 1)));
        if (lengths) inputs.add(descriptor(DataType.INT64, batch));
        return new OperationCapabilityQuery(new Operation(ScanKind.SCAN, attrs), inputs,
                List.of(step, descriptor(type, time, batch, 1), descriptor(type, time, batch, 1)),
                Optional.of(body), Optional.empty());
    }

    @Test void variadicBodyKeepsEveryRepeatedReferenceBeyondSixteen() {
        for (var type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16))
            for (int arity : new int[] {33, 257}) for (boolean stack : List.of(false, true))
                for (boolean lengths : List.of(false, true)) {
                    var plan = MetalScanLowering.plan(variadicQuery(type, arity, stack, lengths, false));
                    var decoded = MetalScanDecoding.decode(MetalScanEncoding.encode(plan));
                    assertEquals(arity, decoded.units().getFirst().node().inputs().length);
                    assertTrue(java.util.Arrays.stream(decoded.units().getFirst().node().inputs()).allMatch(i -> i == 0));
                }
    }

    static OperationCapabilityQuery variadicQuery(DataType type, int arity, boolean stack, boolean lengths, boolean reverse) {
        var input = descriptor(type, 2, 1);
        long[] shape = stack ? new long[] {2, arity, 1} : new long[] {2, arity};
        var output = descriptor(type, shape);
        var x = new ValueId(0); var y = new ValueId(1);
        var operation = new Operation(stack
                ? io.github.pho001.synaptik.model.operation.layout.TensorCompositionKind.STACK
                : io.github.pho001.synaptik.model.operation.layout.TensorCompositionKind.CONCAT,
                new io.github.pho001.synaptik.model.operation.layout.CompositionAxisAttrs(1));
        var node = new CompiledNode(new NodeId(0), operation, java.util.Collections.nCopies(arity, x), List.of(y));
        var graph = new CompiledGraphModel(List.of(new GraphValue(x, input), new GraphValue(y, output)), List.of(node),
                List.of(x), List.of(y), Map.of(node.id(), GraphPhase.FORWARD));
        var body = new CompiledScanBody(graph, List.of(), List.of(y), List.of(), List.of(0));
        var attrs = lengths ? new ScanAttrs(2, 0, 1, 0, 1, 0, reverse, true, 0, List.of(OptionalInt.of(0)))
                : new ScanAttrs(2, 0, 1, 0, 1, 0, reverse);
        var inputs = new java.util.ArrayList<>(List.of(descriptor(type, 2, 2, 1)));
        if (lengths) inputs.add(descriptor(DataType.INT64, 2));
        long[] history = new long[shape.length + 1]; history[0] = 2;
        System.arraycopy(shape, 0, history, 1, shape.length);
        return new OperationCapabilityQuery(new Operation(ScanKind.SCAN, attrs), inputs,
                List.of(descriptor(type, history)), Optional.of(body), Optional.empty());
    }

    static OperationCapabilityQuery sourceOnlyQuery(DataType type, long time, long batch, boolean lengths, boolean reverse) {
        var step = descriptor(type, batch, 1);
        var invariant = descriptor(type);
        var h = new ValueId(0); var x = new ValueId(1); var k = new ValueId(2);
        var graph = new CompiledGraphModel(List.of(new GraphValue(h, step), new GraphValue(x, step), new GraphValue(k, invariant)),
                List.of(), List.of(h, x, k), List.of(h, x, k), Map.of());
        var body = new CompiledScanBody(graph, List.of(h), List.of(x), List.of(k), List.of(0, 1, 2));
        var attrs = lengths ? new ScanAttrs(time, 1, 1, 1, 1, 1, reverse, true, 1, List.of(OptionalInt.of(0)))
                : new ScanAttrs(time, 1, 1, 1, 1, 1, reverse);
        var inputs = new java.util.ArrayList<>(List.of(step, descriptor(type, time, batch, 1), invariant));
        if (lengths) inputs.add(descriptor(DataType.INT64, batch));
        return new OperationCapabilityQuery(new Operation(ScanKind.SCAN, attrs), inputs,
                List.of(step, descriptor(type, time, batch, 1), descriptor(type, time)), Optional.of(body), Optional.empty());
    }

    @Test void sourceOnlyPlansKeepTypedInvariantsAndRealEmptyDomains() {
        for (var type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16, DataType.FLOAT64, DataType.BOOL, DataType.INT32, DataType.INT64))
            for (long time : new long[] {0, 3}) for (long batch : new long[] {0, 2})
                for (boolean lengths : List.of(false, true)) for (boolean reverse : List.of(false, true)) {
                    var plan = MetalScanLowering.plan(sourceOnlyQuery(type, time, batch, lengths, reverse));
                    assertTrue(plan.units().isEmpty());
                    assertEquals(3, plan.formals().size());
                    assertEquals(type.byteWidth(), plan.values().get(2).byteSize());
                    assertDoesNotThrow(() -> MetalScanEncoding.validate(MetalScanEncoding.encode(plan)));
                }
    }

    @Test void comparisonsRetainBothOriginalNumericCarriersAndTheBoolResultRole() {
        var types = List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16);
        for (var left : types) for (var right : types)
            for (boolean lengths : List.of(false, true)) for (boolean reverse : List.of(false, true)) {
                if (left != right && left != DataType.FLOAT32 && right != DataType.FLOAT32) {
                    assertThrows(IllegalArgumentException.class,
                            () -> MetalScanLowering.plan(mixedComparisonQuery(left, right, lengths, reverse)));
                    continue;
                }
                var plan = MetalScanLowering.plan(mixedComparisonQuery(left, right, lengths, reverse));
                var decoded = MetalScanDecoding.decode(MetalScanEncoding.encode(plan));
                assertEquals(left, decoded.values().get(0).descriptor().dataType());
                assertEquals(right, decoded.values().get(1).descriptor().dataType());
                assertEquals(DataType.BOOL, decoded.values().get(2).descriptor().dataType());
                assertEquals(2, decoded.values().get(2).byteSize());
            }
    }

    static OperationCapabilityQuery mixedComparisonQuery(DataType left, DataType right, boolean lengths, boolean reverse) {
        var x = new ValueId(0); var r = new ValueId(1); var y = new ValueId(2);
        var input = descriptor(left, 2, 1);
        var invariant = descriptor(right);
        var output = descriptor(DataType.BOOL, 2, 1);
        var compare = new CompiledNode(new NodeId(0), new Operation(
                io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKind.GREATER_THAN,
                NoOperationAttrs.INSTANCE), List.of(x, r), List.of(y));
        var graph = new CompiledGraphModel(List.of(new GraphValue(x, input), new GraphValue(r, invariant), new GraphValue(y, output)),
                List.of(compare), List.of(x, r), List.of(y), Map.of(compare.id(), GraphPhase.FORWARD));
        var body = new CompiledScanBody(graph, List.of(), List.of(y), List.of(), List.of(0));
        var attrs = lengths ? new ScanAttrs(2, 0, 1, 1, 1, 0, reverse, true, 0, List.of(OptionalInt.of(0)))
                : new ScanAttrs(2, 0, 1, 1, 1, 0, reverse);
        var inputs = new java.util.ArrayList<>(List.of(descriptor(left, 2, 2, 1), invariant));
        if (lengths) inputs.add(descriptor(DataType.INT64, 2));
        return new OperationCapabilityQuery(new Operation(ScanKind.SCAN, attrs), inputs,
                List.of(descriptor(DataType.BOOL, 2, 2, 1)), Optional.of(body), Optional.empty());
    }

    static TensorDescriptor descriptor(DataType type, long... dimensions) {
        Shape shape = Shape.of(dimensions);
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), false);
    }
}
