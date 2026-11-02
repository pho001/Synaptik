package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.scan.ScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.ScanKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;
import org.junit.jupiter.api.Test;

/** Binding/resource behavior; actual arithmetic and Compiler reverse execute in the native test. */
class MetalScanPlanTest {
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
        for (boolean lengths : List.of(false, true)) for (boolean reverse : List.of(false, true)) {
            for (long time : new long[] {0, 3}) for (long batch : new long[] {0, 1}) {
                var query = additiveQuery(time, batch, lengths, reverse);
                assertTrue(MetalScanLowering.supports(query));
                assertEquals(reverse, MetalScanLowering.plan(query).header().reverse());
            }
        }
        var valid = additiveQuery(0, 0, true, false);
        var wrong = new OperationCapabilityQuery(valid.operation(), valid.inputs(),
                List.of(descriptor(DataType.FLOAT32, 1, 1), valid.outputs().get(1), valid.outputs().get(2)), valid.scanBody(), Optional.empty());
        assertFalse(MetalScanLowering.supports(wrong), "T=B=0 does not hide a malformed carry result");
    }

    @Test void independentLowAndSymbolicDomainsAreNotGrantedByTheBinder() {
        var valid = additiveQuery(3, 2, false, false);
        var inputs = new java.util.ArrayList<>(valid.inputs());
        inputs.set(0, descriptor(DataType.FLOAT16, 2, 1));
        assertFalse(MetalScanLowering.supports(new OperationCapabilityQuery(valid.operation(), inputs, valid.outputs(), valid.scanBody(), Optional.empty())));
        assertFalse(MetalScanLowering.supports(new OperationCapabilityQuery(valid.operation(), valid.inputs(), valid.outputs())));
        assertFalse(MetalScanLowering.supports(symbolicQuery()), "CPU named-B qualification is not Metal admission");
    }

    @Test void rawConstructorRejectsStaleGrammarOverlappingWritersAndRepeatedDemand() {
        var plan = MetalScanLowering.plan(additiveQuery(3, 2, false, false));
        long[] valid = MetalScanEncoding.encode(plan);
        assertDoesNotThrow(() -> MetalScanEncoding.validate(valid));
        long[] stale = valid.clone(); stale[1] = 0;
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
        var step = descriptor(DataType.FLOAT32, batch, 1);
        var h = new ValueId(0); var x = new ValueId(1); var next = new ValueId(2);
        var add = new CompiledNode(new NodeId(0), new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE), List.of(h, x), List.of(next));
        var graph = new CompiledGraphModel(List.of(new GraphValue(h, step), new GraphValue(x, step), new GraphValue(next, step)),
                List.of(add), List.of(h, x), List.of(next, h), Map.of(add.id(), GraphPhase.FORWARD));
        var body = new CompiledScanBody(graph, List.of(next), List.of(next), List.of(h), List.of(0, 1, 2));
        var attrs = lengths ? new ScanAttrs(time, 1, 1, 0, 1, 1, reverse, true, 1, List.of(OptionalInt.of(0)))
                : new ScanAttrs(time, 1, 1, 0, 1, 1, reverse);
        var inputs = new java.util.ArrayList<>(List.of(step, descriptor(DataType.FLOAT32, time, batch, 1)));
        if (lengths) inputs.add(descriptor(DataType.INT64, batch));
        return new OperationCapabilityQuery(new Operation(ScanKind.SCAN, attrs), inputs,
                List.of(step, descriptor(DataType.FLOAT32, time, batch, 1), descriptor(DataType.FLOAT32, time, batch, 1)),
                Optional.of(body), Optional.empty());
    }

    static TensorDescriptor descriptor(DataType type, long... dimensions) {
        Shape shape = Shape.of(dimensions);
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), false);
    }
}
