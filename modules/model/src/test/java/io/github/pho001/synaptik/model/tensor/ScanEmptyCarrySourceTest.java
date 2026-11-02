package io.github.pho001.synaptik.model.tensor;

import static io.github.pho001.synaptik.model.tensor.ScanBodyTest.nextId;
import static io.github.pho001.synaptik.model.tensor.ScanSymbolicLengthSourceTest.*;
import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.scan.*;
import io.github.pho001.synaptik.model.shape.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

/** Source construction and failure ordering only; newly admitted topologies are not executable. */
class ScanEmptyCarrySourceTest {
    @Test void emissionOnlyKeepsActualAliasesAndCanonicalPortsWithoutFakeState() throws Exception {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (Dimension batch : List.of(fixed(0), fixed(1), new DynamicDimension("Customers")))
                for (long time : new long[] {0, 1}) for (boolean reverse : List.of(false, true))
                    for (boolean length : List.of(false, true)) {
                        Tensor x = source(type, true, shape(batch, fixed(2)));
                        Tensor unused = source(type, false, Shape.of(2));
                        Tensor emitted = x.tanh();
                        var body = new ScanBody(List.of(), List.of(x), List.of(unused), List.of(), List.of(emitted, emitted));
                        Tensor sequence = fresh(x, stacked(time, x.descriptor().shape()));
                        Tensor parameter = fresh(unused, unused.descriptor().shape());
                        Tensor lengths = source(DataType.INT64, false, shape(batch));
                        long before = nextId();
                        var result = length
                                ? Scan.apply(List.of(), List.of(sequence), List.of(parameter), lengths, 0,
                                        List.of(OptionalInt.of(0)), body, reverse)
                                : Scan.apply(List.of(), List.of(sequence), List.of(parameter), body, reverse);
                        assertTrue(result.finalCarry().isEmpty());
                        var p = result.outputs().getFirst().provenance().orElseThrow().producer();
                        var a = (ScanAttrs) p.operation().attrs();
                        assertEquals(length, a.lengthAware()); assertEquals(0, a.carryCount());
                        assertEquals(0, a.rowCarryCount()); assertEquals(reverse, a.reverse());
                        assertEquals(length ? List.of(sequence, parameter, lengths) : List.of(sequence, parameter), p.inputs());
                        assertEquals(a.inputCount(), p.inputs().size());
                        assertEquals(a.outputCount(), p.outputCount());
                        assertEquals(before + p.outputCount(), nextId());
                        assertNotSame(result.outputs().get(0), result.outputs().get(1));
                        for (int slot = 0; slot < p.outputCount(); slot++) {
                            Tensor port = p.output(slot);
                            assertSame(p, port.provenance().orElseThrow().producer());
                            assertSame(port, p.output(slot)); assertTrue(port.hostStorage().isEmpty());
                            assertEquals(stacked(time, x.descriptor().shape()), port.descriptor().shape());
                            assertEquals(type, port.descriptor().dataType()); assertTrue(port.descriptor().requiresGrad());
                            assertEquals(port.descriptor().shape().isFullyStatic()
                                    ? Optional.of(LayoutDescriptor.contiguous(port.descriptor().shape())) : Optional.empty(), port.descriptor().layout());
                        }
                    }
    }

    @Test void zeroRowSharedAccumulatorDerivesBatchFromLengthsNotItsFirstCarry() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (Dimension batch : List.of(fixed(0), fixed(1), new DynamicDimension("Rows")))
                for (long time : new long[] {0, 1}) for (boolean reverse : List.of(false, true))
                    for (boolean emit : List.of(false, true)) {
                    Tensor x = source(type, true, shape(batch, fixed(2)));
                    Tensor old = source(type, false, Shape.of(2));
                    Tensor next = old.add(x.sum(0, false));
                    var body = new ScanBody(List.of(old), List.of(x), List.of(), List.of(next), emit ? List.of(x) : List.of());
                    Tensor initial = fresh(old, old.descriptor().shape());
                    Tensor sequence = fresh(x, stacked(time, x.descriptor().shape()));
                    Tensor lengths = source(DataType.INT64, false, shape(batch));
                    var result = Scan.apply(List.of(initial), List.of(sequence), List.of(), lengths, 0,
                            List.of(OptionalInt.of(0)), body, reverse);
                    assertEquals(Shape.of(2), result.finalCarry().getFirst().descriptor().shape());
                    var p = result.finalCarry().getFirst().provenance().orElseThrow().producer();
                    assertSame(lengths, p.inputs().getLast());
                    assertTrue(((ScanAttrs) p.operation().attrs()).lengthAware());
                    assertEquals(body.formals().size() + 1, p.inputs().size());
                    if (emit) assertEquals(stacked(time, x.descriptor().shape()), result.outputs().getFirst().descriptor().shape());
                    else assertTrue(result.outputs().isEmpty());
                }
    }

    @Test void lengthsMetadataMayDeclareBatchWithoutAnyRowFormal() {
        for (Dimension batch : List.of(fixed(0), fixed(1), new DynamicDimension("Rows")))
            for (long time : new long[] {0, 1}) for (boolean reverse : List.of(false, true)) {
                Tensor old = source(DataType.FLOAT64, false, Shape.of(2));
                Tensor step = fresh(old, old.descriptor().shape());
                var body = new ScanBody(List.of(old), List.of(step), List.of(), List.of(old.add(step)), List.of());
                Tensor initial = fresh(old, old.descriptor().shape());
                Tensor sequence = fresh(step, stacked(time, step.descriptor().shape()));
                Tensor lengths = source(DataType.INT64, false, shape(batch));
                var result = Scan.apply(List.of(initial), List.of(sequence), List.of(), lengths, 0,
                        List.of(OptionalInt.empty()), body, reverse);
                assertTrue(result.outputs().isEmpty());
                var producer = result.finalCarry().getFirst().provenance().orElseThrow().producer();
                assertSame(lengths, producer.inputs().getLast());
                assertEquals(Shape.of(2), result.finalCarry().getFirst().descriptor().shape());
            }
    }

    @Test void emptyCarryTiedSequencesRetainEveryPositionAndRejectContradictoryRolesBeforeIds() throws Exception {
        Tensor x = source(DataType.FLOAT32, true, Shape.of(1, 1));
        Tensor y = fresh(x, x.descriptor().shape());
        var body = new ScanBody(List.of(), List.of(x, y), List.of(), List.of(), List.of(x.add(y)));
        Tensor sequence = fresh(x, stacked(0, x.descriptor().shape()));
        Tensor lengths = source(DataType.INT64, false, Shape.of(1));
        var result = Scan.apply(List.of(), List.of(sequence, sequence), List.of(), lengths, 0,
                List.of(OptionalInt.of(0), OptionalInt.of(0)), body, true);
        var p = result.outputs().getFirst().provenance().orElseThrow().producer();
        assertSame(p.inputs().get(0), p.inputs().get(1));
        long before = nextId();
        assertThrows(IllegalArgumentException.class, () -> Scan.apply(List.of(), List.of(sequence, sequence), List.of(), lengths, 0,
                List.of(OptionalInt.of(0), OptionalInt.of(1)), body, true));
        assertEquals(before, nextId());
        var ids = new AtomicInteger();
        var wrong = new Operation(ScanKind.SCAN, new ScanAttrs(0, 0, 2, 0, 1,
                body.retainedValues().size(), true, true, 0, List.of(OptionalInt.of(0), OptionalInt.of(0))));
        assertThrows(IllegalArgumentException.class, () -> new TensorProducer(wrong,
                List.of(lengths, sequence, sequence), p.outputDescriptors(), Optional.empty(),
                () -> new TensorId(ids.getAndIncrement()), Optional.of(body)));
        assertEquals(0, ids.get());
    }

    @Test void malformedRootsLengthsAndExplicitRowsFailBeforeOutputIdentities() throws Exception {
        Tensor x = source(DataType.FLOAT32, false, Shape.of(0, 2));
        var body = new ScanBody(List.of(), List.of(x), List.of(), List.of(), List.of(x));
        Tensor seq = fresh(x, stacked(0, x.descriptor().shape()));
        Tensor lengths = source(DataType.INT64, false, Shape.of(1));
        long before = nextId();
        assertThrows(IllegalArgumentException.class, () -> new ScanBody(List.of(), List.of(x), List.of(), List.of(), List.of()));
        assertThrows(IllegalArgumentException.class, () -> Scan.apply(List.of(), List.of(seq), List.of(), lengths,
                0, List.of(OptionalInt.of(0)), body, false));
        assertThrows(IllegalArgumentException.class, () -> Scan.apply(List.of(), List.of(seq), List.of(), lengths,
                0, List.of(OptionalInt.empty()), body, false));
        assertThrows(IllegalArgumentException.class, () -> Scan.apply(List.of(), List.of(seq), List.of(), x,
                0, List.of(OptionalInt.of(0)), body, false));
        assertEquals(before, nextId());
    }
}
