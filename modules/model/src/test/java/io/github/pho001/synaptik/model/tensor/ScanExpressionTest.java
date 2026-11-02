package io.github.pho001.synaptik.model.tensor;

import static io.github.pho001.synaptik.model.tensor.ScanBodyTest.*;
import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.scan.ScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.ScanKind;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.List;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

class ScanExpressionTest {
    @Test
    void typedLengthPortsAndCorrespondingBindingsRejectAtDirectSeamBeforeOutputSuppliers() throws Exception {
        for (var batch : List.of(new io.github.pho001.synaptik.model.shape.StaticDimension(0),
                new io.github.pho001.synaptik.model.shape.DynamicDimension("Customers"))) {
            Shape row = ScanSymbolicLengthSourceTest.shape(batch, ScanSymbolicLengthSourceTest.fixed(2));
            Tensor h = ScanSymbolicLengthSourceTest.source(DataType.FLOAT16, false, row);
            Tensor x = ScanSymbolicLengthSourceTest.source(DataType.FLOAT32, true, row);
            Tensor unused = ScanSymbolicLengthSourceTest.source(DataType.INT32, false, row);
            Tensor inv = leaf(DataType.FLOAT64, false, 2);
            var emissions = java.util.Arrays.stream(DataType.values()).map(h::cast).toList();
            var body = new ScanBody(List.of(h), List.of(x, unused), List.of(inv), List.of(h), emissions);
            var result = ScanSymbolicLengthSourceTest.apply(body, 0, 1,
                    List.of(OptionalInt.of(0), OptionalInt.of(0)), false);
            var p = result.finalCarry().getFirst().provenance().orElseThrow().producer();
            var count = new AtomicInteger();
            for (int port = 0; port < p.outputCount(); port++) {
                var d = p.outputDescriptors().get(port);
                DataType wrongType = d.dataType() == DataType.FLOAT32 ? DataType.FLOAT64 : DataType.FLOAT32;
                var bads = new java.util.ArrayList<TensorDescriptor>();
                bads.add(new TensorDescriptor(wrongType, d.shape(), d.layout(), false));
                if (d.dataType().isDifferentiable()) bads.add(new TensorDescriptor(d.dataType(), d.shape(), d.layout(), false));
                bads.add(new TensorDescriptor(d.dataType(), Shape.of(0, 7), Optional.empty(), false));
                if (d.shape().isFullyStatic()) bads.add(new TensorDescriptor(d.dataType(), d.shape(), Optional.empty(), d.requiresGrad()));
                for (var bad : bads) {
                    var outputs = new java.util.ArrayList<>(p.outputDescriptors()); outputs.set(port, bad);
                    long before = nextId();
                    assertThrows(IllegalArgumentException.class, () -> new TensorProducer(p.operation(), p.inputs(), outputs,
                            Optional.empty(), () -> new TensorId(count.getAndIncrement()), Optional.of(body)));
                    assertEquals(before, nextId());
                }
            }
            for (int position = 0; position < p.inputs().size() - 1; position++) {
                var d = p.inputs().get(position).descriptor();
                var bads = new java.util.ArrayList<Tensor>();
                bads.add(ScanSymbolicLengthSourceTest.source(d.dataType() == DataType.FLOAT64 ? DataType.FLOAT32 : DataType.FLOAT64,
                        false, d.shape()));
                bads.add(ScanSymbolicLengthSourceTest.source(d.dataType(), d.requiresGrad(), Shape.of(0, 7)));
                if (d.dataType().isDifferentiable()) bads.add(ScanSymbolicLengthSourceTest.source(d.dataType(), !d.requiresGrad(), d.shape()));
                for (Tensor bad : bads) {
                    var inputs = new java.util.ArrayList<>(p.inputs()); inputs.set(position, bad);
                    long before = nextId();
                    assertThrows(IllegalArgumentException.class, () -> new TensorProducer(p.operation(), inputs, p.outputDescriptors(),
                            Optional.empty(), () -> new TensorId(count.getAndIncrement()), Optional.of(body)));
                    assertThrows(IllegalArgumentException.class, () -> Scan.apply(inputs.subList(0, 1), inputs.subList(1, 3),
                            inputs.subList(3, 4), inputs.getLast(), 1, List.of(OptionalInt.of(0), OptionalInt.of(0)), body, false));
                    assertEquals(before, nextId());
                }
            }
            assertEquals(0, count.get());
        }
    }

    @Test
    void lengthMetadataAndDirectAttachmentRejectBeforeAnyOutputSupplier() throws Exception {
        Tensor h = leaf(false, 2, 2), x = leaf(false, 2, 2);
        var body = body(h, x, List.of(), h.add(x), h);
        Tensor initial = leaf(false, 2, 2), seq = leaf(false, 0, 2, 2), lengths = leaf(DataType.INT64, false, 2);
        var axes = new java.util.ArrayList<>(List.of(OptionalInt.of(0)));
        var valid = Scan.apply(List.of(initial), List.of(seq), List.of(), lengths, 1, axes, body, true);
        axes.clear();
        var p = valid.finalCarry().getFirst().provenance().orElseThrow().producer();
        assertEquals(List.of(initial, seq, lengths), p.inputs());
        assertEquals(List.of(OptionalInt.of(0)), ((ScanAttrs) p.operation().attrs()).sequenceRowAxes());
        var invalidLengths = List.of(leaf(DataType.INT32, false, 2), leaf(DataType.INT64, false, 3),
                leaf(DataType.INT64, false, 1, 2), leaf(true, 2));
        long before = nextId();
        for (Tensor bad : invalidLengths) {
            assertThrows(IllegalArgumentException.class, () -> Scan.apply(List.of(initial), List.of(seq), List.of(), bad, 1, List.of(OptionalInt.of(0)), body, false));
            var inputs = new java.util.ArrayList<>(p.inputs()); inputs.set(2, bad);
            var count = new AtomicInteger();
            assertThrows(IllegalArgumentException.class, () -> new TensorProducer(p.operation(), inputs, p.outputDescriptors(), Optional.empty(),
                    () -> new TensorId(count.getAndIncrement()), Optional.of(body)));
            assertEquals(0, count.get());
        }
        assertThrows(NullPointerException.class, () -> Scan.apply(List.of(initial), List.of(seq), List.of(), null, 1, List.of(OptionalInt.of(0)), body, false));
        assertThrows(IllegalArgumentException.class, () -> Scan.apply(List.of(initial), List.of(seq), List.of(), lengths, 0, List.of(), body, false));
        assertThrows(NullPointerException.class, () -> Scan.apply(List.of(initial), List.of(seq), List.of(), lengths, 1, null, body, false));
        var outputs = new java.util.ArrayList<>(p.outputDescriptors()); outputs.set(1, initial.descriptor());
        var count = new AtomicInteger();
        Operation wrongRoles = new Operation(ScanKind.SCAN, new ScanAttrs(0, 1, 1, 0, 1, body.retainedValues().size(), true, true, 1, List.of(OptionalInt.empty())));
        assertThrows(IllegalArgumentException.class, () -> new TensorProducer(wrongRoles, p.inputs(), p.outputDescriptors(), Optional.empty(),
                () -> new TensorId(count.getAndIncrement()), Optional.of(body)));
        assertThrows(IllegalArgumentException.class, () -> new TensorProducer(p.operation(), p.inputs(), outputs, Optional.empty(),
                () -> new TensorId(count.getAndIncrement()), Optional.of(body)));
        assertEquals(0, count.get());
        assertEquals(before, nextId());
        // Storage on a body value remains mutable, so direct and factory seams must recheck it.
        h.replaceHostStorage(TensorFactory.zeros(Shape.of(2, 2), DataType.FLOAT32, Optional.empty(), false).hostStorage().orElseThrow());
        assertThrows(IllegalArgumentException.class, () -> new TensorProducer(p.operation(), p.inputs(), p.outputDescriptors(), Optional.empty(),
                () -> new TensorId(count.getAndIncrement()), Optional.of(body)));
        assertEquals(0, count.get());
    }

    @Test
    void tiedActualsRequireCompatibleExplicitOuterCoordinatesAndTemplateReuseKeepsOccurrences() throws Exception {
        Tensor h = leaf(false, 2, 2), x = leaf(false, 2, 2), y = leaf(false, 2, 2);
        var body = new ScanBody(List.of(h), List.of(x, y), List.of(), List.of(h.add(x).add(y)), List.of(h));
        Tensor initial = leaf(false, 2, 2), seq = leaf(false, 2, 2, 2), lengths = leaf(DataType.INT64, false, 2);
        var axes = List.of(OptionalInt.of(0), OptionalInt.of(0));
        var a = Scan.apply(List.of(initial), List.of(seq, seq), List.of(), lengths, 1, axes, body, false);
        var b = Scan.apply(List.of(initial), List.of(seq, seq), List.of(), lengths, 1, axes, body, true);
        assertNotSame(a.finalCarry().getFirst().provenance().orElseThrow().producer(), b.finalCarry().getFirst().provenance().orElseThrow().producer());
        // Unused declarations still cannot reinterpret a tied actual as SHARED or another row axis.
        var unused = new ScanBody(List.of(h), List.of(x, y), List.of(), List.of(h), List.of());
        long before = nextId();
        for (OptionalInt role : List.of(OptionalInt.empty(), OptionalInt.of(1))) {
            assertThrows(IllegalArgumentException.class, () -> Scan.apply(List.of(initial), List.of(seq, seq), List.of(), lengths,
                    1, List.of(OptionalInt.of(0), role), unused, false));
        }
        assertEquals(before, nextId());
    }
    @Test
    void reservesCanonicalGroupsWithDistinctWrappersForAliasedRoles() throws Exception {
        Tensor h = leaf(false, 2), x = leaf(true, 2);
        ScanBody b = body(h, x, List.of(), h.add(x), h, h);
        Tensor initial = leaf(false, 2), sequence = leaf(true, 3, 2);
        long before = nextId();
        ScanResult r = Scan.apply(List.of(initial), List.of(sequence), List.of(), b, true);
        TensorProducer p = r.finalCarry().getFirst().provenance().orElseThrow().producer();
        assertSame(b, p.scanBody().orElseThrow());
        assertEquals(new ScanAttrs(3, 1, 1, 0, 2, 2, true), p.operation().attrs());
        assertEquals(List.of(initial, sequence), p.inputs());
        assertEquals(before + p.outputCount(), nextId());
        List<Tensor> all = java.util.stream.Stream.of(r.finalCarry(), r.outputs(), r.residuals())
                .flatMap(List::stream).toList();
        for (int i = 0; i < all.size(); i++) {
            Tensor value = all.get(i);
            assertSame(value, p.output(i));
            assertSame(p.outputDescriptors().get(i), value.descriptor());
            assertEquals(i, value.provenance().orElseThrow().outputIndex());
            assertTrue(value.hostStorage().isEmpty());
            assertTrue(value.descriptor().requiresGrad());
            assertTrue(value.descriptor().layout().orElseThrow().isContiguous());
        }
        assertNotSame(r.outputs().get(0), r.outputs().get(1));
        assertEquals(Shape.of(3, 2), r.outputs().getFirst().descriptor().shape());
    }

    @Test
    void reusesOneTemplateAcrossIndependentOccurrencesWithoutUnrolling() {
        Tensor h = leaf(false, 2), x = leaf(false, 2);
        ScanBody b = body(h, x, List.of(), h.add(x));
        Tensor initial = leaf(false, 2);
        ScanResult a = Scan.apply(List.of(initial), List.of(leaf(false, 0, 2)), List.of(), b, false);
        ScanResult c = Scan.apply(List.of(initial), List.of(leaf(false, 1000000, 2)), List.of(), b, false);
        TensorProducer p = a.finalCarry().getFirst().provenance().orElseThrow().producer();
        TensorProducer q = c.finalCarry().getFirst().provenance().orElseThrow().producer();
        assertNotSame(p, q);
        assertSame(b, p.scanBody().orElseThrow());
        assertSame(b, q.scanBody().orElseThrow());
        assertEquals(p.outputCount(), q.outputCount());
        assertNotSame(initial, a.finalCarry().getFirst());
        assertEquals(Shape.of(0, 2), a.residuals().getFirst().descriptor().shape());
    }

    @Test
    void retainsRepeatedActualPositionsAndMultipleCarriesSequencesAndInvariants() {
        Tensor h = leaf(false, 2), h2 = leaf(false, 2), x = leaf(false, 2), x2 = leaf(false, 2);
        Tensor a = leaf(false, 2), b = leaf(false, 2);
        ScanBody body = new ScanBody(List.of(h, h2), List.of(x, x2), List.of(a, b),
                List.of(h.add(x), h.add(x)), List.of(a));
        Tensor actual = leaf(false, 2), sequence = leaf(false, 4, 2);
        ScanResult r = Scan.apply(List.of(actual, actual), List.of(sequence, sequence),
                List.of(actual, actual), body, false);
        TensorProducer p = r.finalCarry().getFirst().provenance().orElseThrow().producer();
        assertEquals(List.of(actual, actual, sequence, sequence, actual, actual), p.inputs());
        assertNotSame(r.finalCarry().get(0), r.finalCarry().get(1));
        assertTrue(body.retainedValues().contains(h2));
    }

    @Test
    void rejectsBindingFailuresBeforeAnyOutputId() throws Exception {
        Tensor h = leaf(false, 2), x = leaf(false, 2);
        ScanBody b = body(h, x, List.of(), h);
        Tensor initial = leaf(false, 2), wrong = leaf(false, 3), seq = leaf(false, 2, 2);
        long before = nextId();
        assertThrows(IllegalArgumentException.class, () -> Scan.apply(List.of(h), List.of(seq), List.of(), b, false));
        assertThrows(IllegalArgumentException.class, () -> Scan.apply(List.of(wrong), List.of(seq), List.of(), b, false));
        assertThrows(IllegalArgumentException.class, () -> Scan.apply(List.of(initial), List.of(wrong), List.of(), b, false));
        assertThrows(IllegalArgumentException.class, () -> Scan.apply(List.of(), List.of(seq), List.of(), b, false));
        assertEquals(before, nextId());
    }

    @Test
    void rejectsCommonTimeFlagsTypeAndShapeOverflowBeforeIds() throws Exception {
        Tensor h = leaf(false, 2), x = leaf(false, 2), y = leaf(false, 2);
        ScanBody b = new ScanBody(List.of(h), List.of(x, y), List.of(), List.of(h), List.of());
        Tensor initial = leaf(false, 2), s = leaf(false, 3, 2), t = leaf(false, 4, 2);
        Tensor grad = leaf(true, 3, 2), wrongType = leaf(DataType.FLOAT64, false, 3, 2);
        Tensor huge = TensorFactory.create(new TensorDescriptor(DataType.FLOAT32,
                Shape.of(Long.MAX_VALUE, 2), Optional.empty(), false));
        long before = nextId();
        assertThrows(IllegalArgumentException.class, () -> Scan.apply(List.of(initial), List.of(s, t), List.of(), b, false));
        assertThrows(IllegalArgumentException.class, () -> Scan.apply(List.of(initial), List.of(s, grad), List.of(), b, false));
        assertThrows(IllegalArgumentException.class, () -> Scan.apply(List.of(initial), List.of(s, wrongType), List.of(), b, false));
        assertThrows(ArithmeticException.class, () -> Scan.apply(List.of(initial), List.of(huge, huge), List.of(), b, false));
        assertEquals(before, nextId());
    }

    @Test
    void rejectsStorageAttachedAfterBodyCreationAtApplyAndDirectAttachmentBeforeIds() throws Exception {
        for (int role = 0; role < 5; role++) {
            Tensor h = leaf(false, 2), x = leaf(false, 2), unused = leaf(false, 2);
            Tensor next = h.add(x);
            TopKResult top = next.topK(2, 0, true, true);
            ScanBody b = body(h, x, List.of(unused), top.values());
            Tensor initial = leaf(false, 2), sequence = leaf(false, 3, 2), invariant = leaf(false, 2);
            ScanResult valid = Scan.apply(List.of(initial), List.of(sequence), List.of(invariant), b, false);
            TensorProducer p = valid.finalCarry().getFirst().provenance().orElseThrow().producer();
            Tensor target = List.of(h, x, unused, next, top.indices()).get(role);
            target.replaceHostStorage(TensorFactory.zeros(target.descriptor().shape(),
                    target.descriptor().dataType(), Optional.empty(), false).hostStorage().orElseThrow());
            long before = nextId();
            AtomicInteger supplied = new AtomicInteger();
            assertThrows(IllegalArgumentException.class, () -> Scan.apply(List.of(initial), List.of(sequence), List.of(invariant), b, false));
            assertThrows(IllegalArgumentException.class, () -> new TensorProducer(p.operation(), p.inputs(),
                    p.outputDescriptors(), Optional.empty(), () -> new TensorId(supplied.getAndIncrement()), Optional.of(b)));
            assertThrows(IllegalArgumentException.class, () -> TensorFactory.createDerivedOutputs(
                    p.operation(), p.inputs(), p.outputDescriptors(), Optional.of(b)));
            assertEquals(0, supplied.get());
            assertEquals(before, nextId());
        }
    }

    @Test
    void rejectsDirectMissingWrongKindHeadersAndDescriptorsBeforeSupplierCalls() {
        Tensor h = leaf(false, 2), x = leaf(false, 2);
        ScanBody b = body(h, x, List.of(), h.add(x));
        ScanResult valid = Scan.apply(List.of(leaf(false, 2)), List.of(leaf(false, 3, 2)), List.of(), b, false);
        TensorProducer p = valid.finalCarry().getFirst().provenance().orElseThrow().producer();
        AtomicInteger supplied = new AtomicInteger();
        java.util.function.Supplier<TensorId> ids = () -> new TensorId(supplied.getAndIncrement());
        assertThrows(IllegalArgumentException.class, () -> new TensorProducer(p.operation(), p.inputs(), p.outputDescriptors()));
        assertThrows(IllegalArgumentException.class, () -> new TensorProducer(h.tanh().provenance().orElseThrow().producer().operation(),
                List.of(h), List.of(h.descriptor()), Optional.empty(), ids, Optional.of(b)));
        Operation wrongTime = new Operation(ScanKind.SCAN, new ScanAttrs(2, 1, 1, 0, 0, 2, false));
        assertThrows(IllegalArgumentException.class, () -> new TensorProducer(wrongTime, p.inputs(), p.outputDescriptors(), Optional.empty(), ids, Optional.of(b)));
        var wrongDescriptors = new java.util.ArrayList<>(p.outputDescriptors());
        wrongDescriptors.set(1, h.descriptor());
        assertThrows(IllegalArgumentException.class, () -> new TensorProducer(p.operation(), p.inputs(), wrongDescriptors, Optional.empty(), ids, Optional.of(b)));
        assertEquals(0, supplied.get());
    }

    @Test
    void laterIdentifierFailureConsumesEarlierIdsButReturnsNoPartialOccurrence() {
        Tensor h = leaf(false, 2), x = leaf(false, 2);
        ScanBody b = body(h, x, List.of(), h.add(x));
        ScanResult valid = Scan.apply(List.of(leaf(false, 2)), List.of(leaf(false, 1, 2)), List.of(), b, false);
        TensorProducer p = valid.finalCarry().getFirst().provenance().orElseThrow().producer();
        AtomicInteger supplied = new AtomicInteger();
        assertThrows(IllegalStateException.class, () -> new TensorProducer(p.operation(), p.inputs(),
                p.outputDescriptors(), Optional.empty(), () -> {
                    if (supplied.get() == 1) throw new IllegalStateException("exhausted");
                    return new TensorId(supplied.getAndIncrement());
                }, Optional.of(b)));
        assertEquals(1, supplied.get());
    }

    @Test
    void rejectsNestedAndRandomBodies() {
        Tensor h = leaf(false, 2), x = leaf(false, 2);
        // An independently declared pass-through body can be bound to outer source formals.
        Tensor ih = leaf(false, 2), ix = leaf(false, 2);
        ScanBody inner = body(ih, ix, List.of(), ih);
        Tensor nested = Scan.apply(List.of(h), List.of(x.expandDims(0)), List.of(), inner, false).finalCarry().getFirst();
        assertThrows(IllegalArgumentException.class, () -> body(h, x, List.of(), nested));
        DropoutResult dropout = h.dropout(0.5, GraphRngState.initial(1, 0));
        assertThrows(IllegalArgumentException.class, () -> body(h, x, List.of(), dropout.output()));
    }
}
