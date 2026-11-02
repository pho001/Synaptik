package io.github.pho001.synaptik.model.tensor;

import static io.github.pho001.synaptik.model.tensor.ScanBodyTest.*;
import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.scan.ScanAttrs;
import io.github.pho001.synaptik.model.shape.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

/** Source metadata and pre-identity proofs only; no batch binding or length execution. */
class ScanSymbolicLengthSourceTest {
    private static final DynamicDimension B = new DynamicDimension("Customers");

    static Shape shape(Dimension... axes) { return Shape.ofDimensions(axes); }
    static StaticDimension fixed(long size) { return new StaticDimension(size); }
    static Shape stacked(long time, Shape local) {
        var axes = new ArrayList<Dimension>(); axes.add(fixed(time)); axes.addAll(local.dimensions());
        return shape(axes.toArray(Dimension[]::new));
    }
    static Tensor source(DataType type, boolean grad, Shape shape) {
        return TensorFactory.create(new TensorDescriptor(type, shape, shape.isFullyStatic()
                ? Optional.of(LayoutDescriptor.contiguous(shape)) : Optional.empty(), grad));
    }
    static Tensor fresh(Tensor value, Shape shape) {
        return source(value.descriptor().dataType(), value.descriptor().requiresGrad(), shape);
    }
    static ScanResult apply(ScanBody body, long time, int rows, List<OptionalInt> axes, boolean reverse) {
        var carry = body.carry().stream().map(t -> fresh(t, t.descriptor().shape())).toList();
        var sequences = body.step().stream().map(t -> fresh(t, stacked(time, t.descriptor().shape()))).toList();
        var invariants = body.invariants().stream().map(t -> fresh(t, t.descriptor().shape())).toList();
        Dimension batch = body.carry().getFirst().descriptor().shape().dimension(0);
        if (batch instanceof DynamicDimension named) batch = new DynamicDimension("  " + named.symbol() + "  ");
        return Scan.apply(carry, sequences, invariants, source(DataType.INT64, false, shape(batch)), rows, axes, body, reverse);
    }

    @Test void allValidRemainsWholeTensorAlgebraWithoutInventingBatchLocality() {
        for (long time : new long[] {0, 1, 3}) for (boolean reverse : List.of(false, true)) {
            Tensor h = source(DataType.FLOAT64, false, Shape.of(3));
            Tensor step = source(DataType.FLOAT32, true, Shape.of(2, 3));
            Tensor mixedRows = step.mean(0, false).cast(DataType.FLOAT64);
            var body = new ScanBody(List.of(h), List.of(step), List.of(),
                    List.of(h.add(mixedRows)), List.of(mixedRows));
            Tensor initial = fresh(h, h.descriptor().shape());
            Tensor sequence = fresh(step, stacked(time, step.descriptor().shape()));
            var result = Scan.apply(List.of(initial), List.of(sequence), List.of(), body, reverse);
            assertEquals(DataType.FLOAT64, result.finalCarry().getFirst().descriptor().dataType());
            assertEquals(Shape.of(3), result.finalCarry().getFirst().descriptor().shape());
            assertEquals(Shape.of(time, 3), result.outputs().getFirst().descriptor().shape());
            var producer = result.finalCarry().getFirst().provenance().orElseThrow().producer();
            var attrs = (ScanAttrs) producer.operation().attrs();
            assertFalse(attrs.lengthAware()); assertTrue(attrs.sequenceRowAxes().isEmpty());
            assertEquals(List.of(initial, sequence), producer.inputs());
            assertTrue(result.outputs().getFirst().descriptor().requiresGrad());
        }
    }

    @Test void carryFreeDeclaredHistoriesPreserveEveryTypedRowCoordinateAtAnyRank() {
        for (DataType type : DataType.values())
            for (Dimension batch : List.of(fixed(0), fixed(1), fixed(2), B))
                for (int rank = 1; rank <= 4; rank++) for (int row = 0; row < rank; row++) {
                    Dimension[] dimensions = new Dimension[rank];
                    Arrays.fill(dimensions, fixed(2)); dimensions[row] = batch;
                    Tensor step = source(type, type.isDifferentiable(), shape(dimensions));
                    int[] permutation = new int[rank]; permutation[0] = row;
                    for (int axis = 0, slot = 1; axis < rank; axis++)
                        if (axis != row) permutation[slot++] = axis;
                    Tensor emission = step.permute(permutation);
                    var body = new ScanBody(List.of(), List.of(step), List.of(), List.of(), List.of(emission));
                    var roles = List.of(OptionalInt.of(row));
                    for (long time : new long[] {0, 1, 3}) for (boolean reverse : List.of(false, true)) {
                        Tensor sequence = fresh(step, stacked(time, step.descriptor().shape()));
                        Tensor lengths = source(DataType.INT64, false, shape(batch));
                        var result = Scan.apply(List.of(), List.of(sequence), List.of(), lengths,
                                0, roles, body, reverse);
                        assertTrue(result.finalCarry().isEmpty());
                        Tensor output = result.outputs().getFirst();
                        assertEquals(type, output.descriptor().dataType());
                        assertEquals(stacked(time, emission.descriptor().shape()), output.descriptor().shape());
                        assertEquals(type.isDifferentiable(), output.descriptor().requiresGrad());
                        var producer = output.provenance().orElseThrow().producer();
                        var attrs = (ScanAttrs) producer.operation().attrs();
                        assertTrue(attrs.lengthAware()); assertEquals(roles, attrs.sequenceRowAxes());
                        assertSame(body, producer.scanBody().orElseThrow());
                        assertEquals(OptionalInt.of(0), Scan.ordinarySemantics(
                                emission.provenance().orElseThrow().producer().operation(),
                                List.of(step.descriptor()),
                                List.of(Scan.declaredSemantics(step.descriptor(), OptionalInt.of(row), batch)),
                                List.of(emission.descriptor()), batch).getFirst().rowAxis());
                    }
                }
    }

    @Test void unusedHistoryDeclarationsStillProveExactBatchRankAndTiedCoordinatesBeforeIds() throws Exception {
        Tensor h = source(DataType.FLOAT32, false, shape(B, fixed(2)));
        Tensor unused = source(DataType.BOOL, false, shape(fixed(2), fixed(2), B));
        var body = new ScanBody(List.of(h), List.of(unused), List.of(), List.of(h), List.of());
        Tensor initial = fresh(h, h.descriptor().shape());
        Tensor sequence = fresh(unused, stacked(0, unused.descriptor().shape()));
        Tensor lengths = source(DataType.INT64, false, shape(B));
        assertDoesNotThrow(() -> Scan.apply(List.of(initial), List.of(sequence), List.of(), lengths,
                1, List.of(OptionalInt.of(2)), body, false));
        long before = nextId();
        for (int axis : new int[] {0, 1, 3, Integer.MAX_VALUE})
            assertThrows(IllegalArgumentException.class, () -> Scan.apply(List.of(initial),
                    List.of(sequence), List.of(), lengths, 1, List.of(OptionalInt.of(axis)), body, false));
        assertEquals(before, nextId());

        Tensor row = source(DataType.FLOAT32, false, Shape.of(2, 2, 2));
        Tensor other = fresh(row, row.descriptor().shape());
        var tiedBody = new ScanBody(List.of(), List.of(row, other), List.of(), List.of(), List.of(row));
        Tensor tied = fresh(row, stacked(0, row.descriptor().shape()));
        Tensor staticLengths = source(DataType.INT64, false, Shape.of(2));
        before = nextId();
        assertThrows(IllegalArgumentException.class, () -> Scan.apply(List.of(), List.of(tied, tied),
                List.of(), staticLengths, 0, List.of(OptionalInt.of(0), OptionalInt.of(2)), tiedBody, true));
        assertEquals(before, nextId());
    }

    @Test void allSevenPerValueTypesKeepOwnPortsAndConservativeFlagsAcrossStaticNamedAndEmptyGeometry() {
        for (Dimension batch : List.of(fixed(0), fixed(1), fixed(3), B))
            for (long features : new long[] {0, 3}) for (long time : new long[] {0, 3})
                for (boolean grad : List.of(false, true)) for (int gradientRole = 0; gradientRole < 3; gradientRole++) {
                    var carry = new ArrayList<Tensor>(); var step = new ArrayList<Tensor>();
                    var invariants = new ArrayList<Tensor>(); var next = new ArrayList<Tensor>();
                    var emissions = new ArrayList<Tensor>();
                    for (DataType type : DataType.values()) {
                        Tensor h = source(type, grad && gradientRole == 0 && type.isDifferentiable(), shape(batch, fixed(features)));
                        Tensor x = source(type, grad && gradientRole == 1 && type.isDifferentiable(), h.descriptor().shape());
                        // Unused invariant flags must still reach every differentiable outer port.
                        Tensor unused = source(type, grad && gradientRole == 2 && type.isDifferentiable(), Shape.of(features));
                        carry.add(h); step.add(x); invariants.add(unused);
                        Tensor root = type == DataType.BOOL || type == DataType.INT64 ? h : h.add(x);
                        next.add(root); emissions.add(root);
                    }
                    var body = new ScanBody(carry, step, invariants, next, emissions);
                    for (boolean reverse : List.of(false, true)) {
                        var result = apply(body, time, carry.size(), Collections.nCopies(step.size(), OptionalInt.of(0)), reverse);
                        var p = result.finalCarry().getFirst().provenance().orElseThrow().producer();
                        var roles = new ArrayList<>(next); roles.addAll(emissions); roles.addAll(body.retainedValues());
                        for (int port = 0; port < p.outputCount(); port++) {
                            var local = roles.get(port).descriptor(); var actual = p.output(port).descriptor();
                            assertEquals(local.dataType(), actual.dataType());
                            assertEquals(port < carry.size() ? local.shape() : stacked(time, local.shape()), actual.shape());
                            assertEquals(grad && local.dataType().isDifferentiable(), actual.requiresGrad());
                            assertEquals(actual.shape().isFullyStatic() ? Optional.of(LayoutDescriptor.contiguous(actual.shape())) : Optional.empty(), actual.layout());
                        }
                        for (Tensor formal : body.formals()) {
                            boolean expected = grad && formal.descriptor().dataType().isDifferentiable()
                                    && (gradientRole == 0 ? carry.contains(formal) : gradientRole == 1 ? step.contains(formal) : invariants.contains(formal));
                            assertEquals(expected, formal.descriptor().requiresGrad());
                        }
                        assertEquals(body.formals().size() + 1, p.inputs().size());
                        assertEquals(DataType.INT64, p.inputs().getLast().descriptor().dataType());
                        assertFalse(p.inputs().getLast().descriptor().requiresGrad());
                    }
                }
    }

    @Test void explicitLowToF32WorkAndCastBackRetainEveryTypedSiteAndMixedRowRole() {
        for (DataType low : List.of(DataType.BFLOAT16, DataType.FLOAT16))
            for (Dimension batch : List.of(fixed(0), fixed(1), fixed(3), B))
                for (long features : new long[] {0, 3}) for (long time : new long[] {0, 3}) {
                    Tensor h = source(low, false, shape(batch, fixed(features)));
                    Tensor x = source(low, true, h.descriptor().shape());
                    Tensor y = source(low, false, shape(fixed(features), batch));
                    Tensor shared = source(DataType.FLOAT32, false, Shape.of(features));
                    Tensor old = source(DataType.FLOAT64, false, Shape.of(features));
                    Tensor work = h.cast(DataType.FLOAT32).add(x.cast(DataType.FLOAT32))
                            .mul(y.cast(DataType.FLOAT32).transpose()).add(shared.cast(DataType.FLOAT64).cast(DataType.FLOAT32));
                    Tensor narrowed = work.cast(low);
                    Tensor sum = work.sum(0, false).cast(DataType.FLOAT64);
                    var body = new ScanBody(List.of(h, old), List.of(x, y), List.of(shared),
                            List.of(narrowed, old.add(sum)), List.of(work, narrowed));
                    for (boolean reverse : List.of(false, true)) {
                        var result = apply(body, time, 1, List.of(OptionalInt.of(0), OptionalInt.of(1)), reverse);
                        assertEquals(low, result.finalCarry().getFirst().descriptor().dataType());
                        assertEquals(DataType.FLOAT32, result.outputs().getFirst().descriptor().dataType());
                        assertEquals(DataType.FLOAT32, result.residuals().get(body.retainedValues().indexOf(work)).descriptor().dataType());
                        assertEquals(low, result.residuals().get(body.retainedValues().indexOf(narrowed)).descriptor().dataType());
                        assertEquals(Shape.of(time, features), result.residuals().get(body.retainedValues().indexOf(sum)).descriptor().shape());
                        assertFalse(h.descriptor().requiresGrad()); assertTrue(work.descriptor().requiresGrad());
                    }
                }
    }

    @Test void completeMixedRoleInventoryKeepsExactDimensionsCanonicalPortsAndSharedHistories() throws Exception {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (Dimension batch : List.of(fixed(0), fixed(1), fixed(3), B))
                for (long time : new long[] {0, 3}) for (boolean reverse : List.of(false, true)) {
                    Tensor h = source(type, false, shape(batch, fixed(3))), old = source(type, false, Shape.of(3, 3));
                    Tensor x = fresh(h, h.descriptor().shape()), y = fresh(h, shape(fixed(3), batch));
                    x = source(type, true, x.descriptor().shape());
                    Tensor sharedStep = source(type, false, Shape.of(3, 3)), w = source(type, false, Shape.of(3, 3));
                    Tensor transpose = w.transpose();
                    Tensor row = h.matmul(w).add(x).add(w.matmul(y).transpose()).tanh();
                    Tensor contribution = x.transpose().matmul(row).add(sharedStep).add(transpose);
                    var body = new ScanBody(List.of(h, old), List.of(x, y, sharedStep), List.of(w),
                            List.of(row, old.add(contribution)), List.of(h, row, row));
                    var result = apply(body, time, 1, List.of(OptionalInt.of(0), OptionalInt.of(1), OptionalInt.empty()), reverse);
                    var p = result.finalCarry().getFirst().provenance().orElseThrow().producer();
                    assertEquals(7, p.inputs().size()); assertEquals(6, body.formals().size());
                    assertEquals(reverse, ((ScanAttrs) p.operation().attrs()).reverse());
                    assertSame(body, p.scanBody().orElseThrow());
                    assertEquals(Shape.of(time, 3, 3), result.residuals().get(body.retainedValues().indexOf(transpose)).descriptor().shape());
                    var roles = new ArrayList<>(body.nextCarry()); roles.addAll(body.emissions()); roles.addAll(body.retainedValues());
                    for (int port = 0; port < p.outputCount(); port++) {
                        Tensor value = p.output(port); var d = value.descriptor();
                        Shape local = roles.get(port).descriptor().shape();
                        assertEquals(port < 2 ? local : stacked(time, local), d.shape());
                        assertEquals(d.shape().isFullyStatic() ? Optional.of(LayoutDescriptor.contiguous(d.shape())) : Optional.empty(), d.layout());
                        assertTrue(d.requiresGrad()); assertTrue(value.hostStorage().isEmpty());
                        assertSame(p, value.provenance().orElseThrow().producer());
                        assertEquals(port, value.provenance().orElseThrow().outputIndex());
                    }
                    assertNotSame(result.outputs().get(1), result.outputs().get(2));
                    assertFalse(h.descriptor().requiresGrad());
                    long before = nextId();
                    new TensorProducer(p.operation(), p.inputs(), p.outputDescriptors(), Optional.empty(), TensorFactory::nextTensorId, Optional.of(body));
                    assertEquals(before + p.outputCount(), nextId());
                }
    }

    @Test void provedFeatureEditsAndReductionsPreserveNamedRowButExplicitRowReductionIsTerminal() {
        Tensor h = source(DataType.FLOAT64, false, shape(B, fixed(6))), x = fresh(h, h.descriptor().shape());
        Tensor edited = h.reshape(shape(B, fixed(2), fixed(3))).expandDims(-1).squeeze(-1)
                .sum(new int[] {-1}, true).expand(shape(B, fixed(2), fixed(3)))
                .reshape(h.descriptor().shape()).sumToShape(h.descriptor().shape()).contiguous().cast(DataType.FLOAT64);
        var body = body(h, x, List.of(), edited.add(x).neg().sigmoid(), edited);
        assertDoesNotThrow(() -> apply(body, 0, 1, List.of(OptionalInt.of(0)), true));
        Tensor transposed = h.transpose().sumToShape(shape(fixed(6), B)).sum(0, true).transpose();
        var rowOne = body(h, x, List.of(), h, transposed);
        assertDoesNotThrow(() -> apply(rowOne, 3, 1, List.of(OptionalInt.of(0)), false));
        for (Tensor bad : List.of(h.sum(0, true).expand(h.descriptor().shape()).add(h),
                h.sumToShape(Shape.of(1, 6)).expand(h.descriptor().shape()).add(h))) {
            var invalid = body(h, x, List.of(), bad);
            assertThrows(IllegalArgumentException.class, () -> apply(invalid, 0, 1, List.of(OptionalInt.of(0)), false));
        }
        for (Tensor valid : List.of(h.transpose().expandDims(0).squeeze(0).transpose(), h.relu()))
            assertDoesNotThrow(() -> apply(body(h, x, List.of(), valid), 0, 1, List.of(OptionalInt.of(0)), false));
    }

    @Test void everyDeadPortAndMutableStorageAreRecheckedBeforeSupplierCalls() throws Exception {
        Tensor h = source(DataType.FLOAT32, false, shape(B, fixed(2))), x = fresh(h, h.descriptor().shape());
        Tensor shared = leaf(false, 2), unused = leaf(false, 2), next = h.add(x).add(shared);
        var body = body(h, x, List.of(shared, unused), next, h, next);
        var result = apply(body, 0, 1, List.of(OptionalInt.of(0)), false);
        var p = result.finalCarry().getFirst().provenance().orElseThrow().producer();
        var supplied = new AtomicInteger(); long before = nextId();
        for (int port = 0; port < p.outputCount(); port++) {
            var outputs = new ArrayList<>(p.outputDescriptors()); var d = outputs.get(port);
            outputs.set(port, new TensorDescriptor(d.dataType(), d.shape(), d.layout(), !d.requiresGrad()));
            assertThrows(IllegalArgumentException.class, () -> new TensorProducer(p.operation(), p.inputs(), outputs,
                    Optional.empty(), () -> new TensorId(supplied.getAndIncrement()), Optional.of(body)));
        }
        assertEquals(before, nextId()); assertEquals(0, supplied.get());
        for (Tensor value : List.of(h, x, unused, next)) {
            value.replaceHostStorage(new io.github.pho001.synaptik.model.storage.MemorySegmentStorage(
                    DataType.FLOAT32, 2, java.lang.foreign.MemorySegment.ofArray(new float[2])));
            assertThrows(IllegalArgumentException.class, () -> Scan.apply(p.inputs().subList(0, 1), p.inputs().subList(1, 2),
                    p.inputs().subList(2, 4), p.inputs().getLast(), 1, List.of(OptionalInt.of(0)), body, false));
            assertThrows(IllegalArgumentException.class, () -> new TensorProducer(p.operation(), p.inputs(), p.outputDescriptors(),
                    Optional.empty(), () -> new TensorId(supplied.getAndIncrement()), Optional.of(body)));
            value.clearHostStorage();
        }
        assertEquals(before, nextId()); assertEquals(0, supplied.get());
    }

    @Test void aliasesKeepPositionsAndInvalidLengthsMetadataConsumesNoIds() throws Exception {
        Tensor h = source(DataType.FLOAT32, false, shape(B, fixed(2))), x = fresh(h, h.descriptor().shape()), y = fresh(h, h.descriptor().shape());
        var body = new ScanBody(List.of(h), List.of(x, y), List.of(), List.of(h), List.of(h));
        Tensor initial = fresh(h, h.descriptor().shape()), sequence = fresh(x, stacked(0, x.descriptor().shape()));
        Tensor lengths = source(DataType.INT64, false, shape(new DynamicDimension(" Customers ")));
        try (var arena = java.lang.foreign.Arena.ofConfined()) {
            var storage = arena.allocate(java.lang.foreign.ValueLayout.JAVA_LONG);
            storage.set(java.lang.foreign.ValueLayout.JAVA_LONG, 0, Long.MIN_VALUE);
            lengths.replaceHostStorage(new io.github.pho001.synaptik.model.storage.MemorySegmentStorage(DataType.INT64, 1, storage));
        } // Deliberately dead and invalid lengths: source construction must never inspect them.
        var axes = List.of(OptionalInt.of(0), OptionalInt.of(0));
        var result = Scan.apply(List.of(initial), List.of(sequence, sequence), List.of(), lengths, 1, axes, body, false);
        assertEquals(List.of(initial, sequence, sequence, lengths), result.finalCarry().getFirst().provenance().orElseThrow().inputs());
        var badLengths = List.of(leaf(DataType.INT64, false, 2), source(DataType.INT64, false, shape(new DynamicDimension("Other"))),
                source(DataType.INT32, false, shape(B)), source(DataType.INT64, false, shape(B, fixed(1))));
        long before = nextId();
        for (Tensor bad : badLengths) assertThrows(IllegalArgumentException.class, () -> Scan.apply(List.of(initial),
                List.of(sequence, sequence), List.of(), bad, 1, axes, body, false));
        assertThrows(IllegalArgumentException.class, () -> Scan.apply(List.of(initial), List.of(sequence, sequence), List.of(),
                lengths, 1, List.of(OptionalInt.of(0), OptionalInt.empty()), body, false));
        assertEquals(before, nextId());
    }

    @Test void terminalSharedCarriesAndUnusedSharedSymbolsCannotLaunderRowOrStateDependencies() throws Exception {
        Tensor h = source(DataType.FLOAT32, false, shape(B, fixed(2))), x = fresh(h, h.descriptor().shape());
        Tensor old = leaf(false, 2), other = leaf(false, 2), shared = leaf(false, 1, 2);
        Tensor sum = h.sum(0, false), updated = old.add(sum);
        var invalid = List.of(
                new ScanBody(List.of(h, old), List.of(x), List.of(), List.of(h.add(updated), updated), List.of()),
                new ScanBody(List.of(h, old), List.of(x), List.of(), List.of(h, sum.add(old)), List.of()),
                new ScanBody(List.of(h, old, other), List.of(x), List.of(), List.of(h, updated, other.add(updated.neg())), List.of()),
                body(h, x, List.of(shared), shared.expand(h.descriptor().shape())),
                body(h, x, List.of(fresh(h, h.descriptor().shape())), h));
        for (ScanBody body : invalid) {
            var carry = body.carry().stream().map(t -> fresh(t, t.descriptor().shape())).toList();
            var sequences = body.step().stream().map(t -> fresh(t, stacked(0, t.descriptor().shape()))).toList();
            var invariants = body.invariants().stream().map(t -> fresh(t, t.descriptor().shape())).toList();
            Tensor lengths = source(DataType.INT64, false, shape(B));
            long before = nextId();
            assertThrows(IllegalArgumentException.class, () -> Scan.apply(carry, sequences, invariants, lengths,
                    1, List.of(OptionalInt.of(0)), body, false));
            assertEquals(before, nextId());
        }
    }
}
