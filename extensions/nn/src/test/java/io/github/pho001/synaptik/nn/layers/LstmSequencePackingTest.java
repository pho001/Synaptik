package io.github.pho001.synaptik.nn.layers;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutKind;
import io.github.pho001.synaptik.model.operation.index.AxisGatherKind;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.operation.index.SelectAttrs;
import io.github.pho001.synaptik.model.operation.index.SelectKind;
import io.github.pho001.synaptik.model.operation.layout.TensorCompositionKind;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.Field;
import java.util.ArrayDeque;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;

@Execution(ExecutionMode.SAME_THREAD)
class LstmSequencePackingTest {
    @Test
    void packsFiveThreeOneIntoNineHiddenRowsAndRestoresBothExitStates() {
        LstmSequence sequence = sequence();
        Tensor input = tensor(DataType.FLOAT32, Shape.of(5, 3, 3), false);
        Tensor hidden = tensor(DataType.FLOAT32, Shape.of(3, 4), false);
        Tensor cell = tensor(DataType.FLOAT32, Shape.of(3, 4), false);
        LstmSequenceForwardResult result = sequence.forward(
                input, hidden, cell, new long[] {5, 3, 1});

        assertAll(
                () -> assertEquals(
                        List.of(
                                Shape.of(3, 4), Shape.of(2, 4), Shape.of(2, 4),
                                Shape.of(1, 4), Shape.of(1, 4)),
                        result.packedOutputs().stream()
                                .map(value -> value.descriptor().shape()).toList()),
                () -> assertEquals(9L, result.packedOutputs().stream()
                        .mapToLong(value -> value.descriptor().shape()
                                .knownElementCount().orElseThrow() / 4)
                        .sum()),
                () -> assertEquals(Shape.of(3, 4), result.finalHidden().descriptor().shape()),
                () -> assertEquals(Shape.of(3, 4), result.finalCell().descriptor().shape()),
                () -> assertSame(TensorCompositionKind.STACK,
                        result.finalHidden().provenance().orElseThrow().operation().kind()),
                () -> assertSame(TensorCompositionKind.STACK,
                        result.finalCell().provenance().orElseThrow().operation().kind()));

        List<Tensor> hiddenRows = result.finalHidden().provenance().orElseThrow().inputs();
        assertRowSelection(hiddenRows.get(0), result.packedOutputs().get(4), 0);
        assertRowSelection(hiddenRows.get(1), result.packedOutputs().get(2), 1);
        assertRowSelection(hiddenRows.get(2), result.packedOutputs().get(0), 2);
        List<Tensor> cellRows = result.finalCell().provenance().orElseThrow().inputs();
        assertRowSelection(cellRows.get(0), nextCellOf(result.packedOutputs().get(4)), 0);
        assertRowSelection(cellRows.get(1), nextCellOf(result.packedOutputs().get(2)), 1);
        assertRowSelection(cellRows.get(2), nextCellOf(result.packedOutputs().get(0)), 2);
    }

    @Test
    void unsortedLengthsUseStableIndicesSharedAcrossBothStateGathers() {
        LstmSequence sequence = sequence();
        Tensor input = tensor(DataType.FLOAT32, Shape.of(3, 3, 3), false);
        Tensor hidden = tensor(DataType.FLOAT32, Shape.of(3, 4), false);
        Tensor cell = tensor(DataType.FLOAT32, Shape.of(3, 4), false);
        LstmSequenceForwardResult result = sequence.forward(
                input, hidden, cell, new long[] {1, 3, 2});

        assertAll(
                () -> assertArrayEquals(new long[] {0, 1, 2}, indexValues(compactInput(result, 0))),
                () -> assertArrayEquals(new long[] {1, 2}, indexValues(compactInput(result, 1))),
                () -> assertArrayEquals(new long[] {1}, indexValues(compactInput(result, 2))),
                () -> assertArrayEquals(new long[] {1, 2}, indexValues(compactHidden(result, 1))),
                () -> assertArrayEquals(new long[] {1, 2}, indexValues(compactCell(result, 1))),
                () -> assertArrayEquals(new long[] {0}, indexValues(compactHidden(result, 2))),
                () -> assertArrayEquals(new long[] {0}, indexValues(compactCell(result, 2))),
                () -> assertSame(
                        gatherIndex(compactInput(result, 0)), gatherIndex(compactHidden(result, 0))),
                () -> assertSame(
                        gatherIndex(compactInput(result, 0)), gatherIndex(compactCell(result, 0))),
                () -> assertSame(
                        gatherIndex(compactHidden(result, 1)), gatherIndex(compactCell(result, 1))),
                () -> assertSame(
                        gatherIndex(compactHidden(result, 2)), gatherIndex(compactCell(result, 2))));

        List<Tensor> hiddenRows = result.finalHidden().provenance().orElseThrow().inputs();
        assertRowSelection(hiddenRows.get(0), result.packedOutputs().get(0), 0);
        assertRowSelection(hiddenRows.get(1), result.packedOutputs().get(2), 0);
        assertRowSelection(hiddenRows.get(2), result.packedOutputs().get(1), 1);
        List<Tensor> cellRows = result.finalCell().provenance().orElseThrow().inputs();
        assertRowSelection(cellRows.get(0), nextCellOf(result.packedOutputs().get(0)), 0);
        assertRowSelection(cellRows.get(1), nextCellOf(result.packedOutputs().get(2)), 0);
        assertRowSelection(cellRows.get(2), nextCellOf(result.packedOutputs().get(1)), 1);
    }

    @Test
    void zeroLengthRowsNeverEnterOperandsAndUseMatchingInitialRowsForBothStates() {
        LstmSequence sequence = sequence();
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3, 3), false);
        Tensor hidden = tensor(DataType.FLOAT32, Shape.of(3, 4), false);
        Tensor cell = tensor(DataType.FLOAT32, Shape.of(3, 4), false);
        LstmSequenceForwardResult result = sequence.forward(
                input, hidden, cell, new long[] {0, 2, 0});

        assertAll(
                () -> assertEquals(
                        List.of(Shape.of(1, 4), Shape.of(1, 4)),
                        result.packedOutputs().stream()
                                .map(value -> value.descriptor().shape()).toList()),
                () -> assertArrayEquals(new long[] {1}, indexValues(compactInput(result, 0))),
                () -> assertArrayEquals(new long[] {1}, indexValues(compactHidden(result, 0))),
                () -> assertArrayEquals(new long[] {1}, indexValues(compactCell(result, 0))));

        List<Tensor> hiddenRows = result.finalHidden().provenance().orElseThrow().inputs();
        assertRowSelection(hiddenRows.get(0), hidden, 0);
        assertRowSelection(hiddenRows.get(1), result.packedOutputs().get(1), 0);
        assertRowSelection(hiddenRows.get(2), hidden, 2);
        List<Tensor> cellRows = result.finalCell().provenance().orElseThrow().inputs();
        assertRowSelection(cellRows.get(0), cell, 0);
        assertRowSelection(cellRows.get(1), nextCellOf(result.packedOutputs().get(1)), 0);
        assertRowSelection(cellRows.get(2), cell, 2);
    }

    @Test
    void storageFreeZeroValuedDataRemainsActiveAndNoCumulativeScanAppears() {
        LstmSequence sequence = sequence();
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 2, 3), false);
        Tensor hidden = tensor(DataType.FLOAT32, Shape.of(2, 4), false);
        Tensor cell = tensor(DataType.FLOAT32, Shape.of(2, 4), false);
        assertTrue(input.hostStorage().isEmpty());
        LstmSequenceForwardResult result = sequence.forward(
                input, hidden, cell, new long[] {2, 1});

        assertAll(
                () -> assertArrayEquals(new long[] {0, 1}, indexValues(compactInput(result, 0))),
                () -> assertArrayEquals(new long[] {0}, indexValues(compactInput(result, 1))),
                () -> assertTrue(allOperationKinds(result).stream()
                        .noneMatch(CumulativeScanKind.class::isInstance)));
    }

    @Test
    void snapshotsLengthsAndCreatesExactDenseUnlabeledInt64Indices() {
        LstmSequence sequence = sequence();
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 2, 3), false);
        Tensor hidden = tensor(DataType.FLOAT32, Shape.of(2, 4), false);
        Tensor cell = tensor(DataType.FLOAT32, Shape.of(2, 4), false);
        long[] lengths = {2, 1};
        LstmSequenceForwardResult result = sequence.forward(input, hidden, cell, lengths);
        lengths[0] = 0;
        lengths[1] = 0;

        Tensor activeIndex = gatherIndex(compactInput(result, 0));
        Tensor survivorIndex = gatherIndex(compactHidden(result, 1));
        assertAll(
                () -> assertArrayEquals(new long[] {0, 1}, indexValues(compactInput(result, 0))),
                () -> assertArrayEquals(new long[] {0}, indexValues(compactInput(result, 1))),
                () -> assertSame(DataType.INT64, activeIndex.descriptor().dataType()),
                () -> assertEquals(Shape.of(2), activeIndex.descriptor().shape()),
                () -> assertFalse(activeIndex.descriptor().requiresGrad()),
                () -> assertSame(LayoutKind.DENSE_CONTIGUOUS,
                        activeIndex.descriptor().layout().orElseThrow().kind()),
                () -> assertTrue(activeIndex.label().isEmpty()),
                () -> assertTrue(activeIndex.provenance().isEmpty()),
                () -> assertTrue(activeIndex.hostStorage().isPresent()),
                () -> assertSame(DataType.INT64, survivorIndex.descriptor().dataType()),
                () -> assertSame(
                        survivorIndex, gatherIndex(compactCell(result, 1))));
    }

    @Test
    void eachStepSelectsTimeAndPublishesExactHiddenWhileCarryingExactCell() {
        LstmSequence sequence = sequence();
        Tensor input = tensor(DataType.FLOAT64, Shape.of(3, 2, 3), false);
        Tensor hidden = tensor(DataType.FLOAT32, Shape.of(2, 4), false);
        Tensor cell = tensor(DataType.FLOAT32, Shape.of(2, 4), false);
        LstmSequenceForwardResult result = sequence.forward(
                input, hidden, cell, new long[] {3, 2});

        for (int step = 0; step < result.packedOutputs().size(); step++) {
            Tensor compactInput = compactInput(result, step);
            Tensor timeSlice = compactInput.provenance().orElseThrow().inputs().getFirst();
            assertAll(
                    () -> assertSame(AxisGatherKind.GATHER,
                            compactInput.provenance().orElseThrow().operation().kind()),
                    () -> assertEquals(new IndexAxisAttrs(0),
                            compactInput.provenance().orElseThrow().operation().attrs()),
                    () -> assertSame(SelectKind.SELECT,
                            timeSlice.provenance().orElseThrow().operation().kind()),
                    () -> assertSame(input,
                            timeSlice.provenance().orElseThrow().inputs().getFirst()));
            assertEquals(
                    new SelectAttrs(0, step), timeSlice.provenance().orElseThrow().operation().attrs());
        }
        assertAll(
                () -> assertSame(DataType.FLOAT64,
                        result.packedOutputs().getFirst().descriptor().dataType()),
                () -> assertSame(DataType.FLOAT64, result.finalHidden().descriptor().dataType()),
                () -> assertSame(DataType.FLOAT64, result.finalCell().descriptor().dataType()),
                () -> assertSame(
                        nextCellOf(result.packedOutputs().get(0)),
                        compactCell(result, 1).provenance().orElseThrow().inputs().getFirst()),
                () -> assertSame(
                        result.packedOutputs().get(0),
                        compactHidden(result, 1).provenance().orElseThrow().inputs().getFirst()));
    }

    @Test
    void oneStepUsesTheFixedThirtyFourIdentifierOrder() throws Exception {
        LstmSequence sequence = sequence();
        Tensor input = tensor(DataType.FLOAT32, Shape.of(1, 1, 3), false);
        Tensor hidden = tensor(DataType.FLOAT32, Shape.of(1, 4), false);
        Tensor cell = tensor(DataType.FLOAT32, Shape.of(1, 4), false);
        AtomicLong ids = nextTensorIdState();
        long start = ids.get();

        LstmSequenceForwardResult result = sequence.forward(
                input, hidden, cell, new long[] {1});
        Tensor nextHidden = result.packedOutputs().getFirst();
        Tensor nextCell = nextCellOf(nextHidden);
        Tensor compactInput = compactInput(result, 0);
        Tensor compactHidden = compactHidden(result, 0);
        Tensor compactCell = compactCell(result, 0);
        Tensor timeSlice = compactInput.provenance().orElseThrow().inputs().getFirst();
        Tensor index = gatherIndex(compactInput);
        Tensor outputGate = nextHidden.provenance().orElseThrow().inputs().getFirst();
        Tensor outputAdd = outputGate.provenance().orElseThrow().inputs().getFirst();
        Tensor inputOutputSlice = outputAdd.provenance().orElseThrow().inputs().getFirst();
        Tensor hiddenOutputSlice = outputAdd.provenance().orElseThrow().inputs().get(1);
        Tensor inputProjection = inputOutputSlice.provenance().orElseThrow().inputs().getFirst();
        Tensor hiddenProjection = hiddenOutputSlice.provenance().orElseThrow().inputs().getFirst();
        Tensor inputTranspose = inputProjection.provenance().orElseThrow().inputs().get(1);
        Tensor hiddenTranspose = hiddenProjection.provenance().orElseThrow().inputs().get(1);
        Tensor nextCellTanh = nextHidden.provenance().orElseThrow().inputs().get(1);
        Tensor finalHiddenRow = result.finalHidden().provenance().orElseThrow().inputs().getFirst();
        Tensor finalCellRow = result.finalCell().provenance().orElseThrow().inputs().getFirst();

        assertAll(
                () -> assertEquals(start, timeSlice.id().value()),
                () -> assertEquals(start + 1, index.id().value()),
                () -> assertEquals(start + 2, compactInput.id().value()),
                () -> assertEquals(start + 3, compactHidden.id().value()),
                () -> assertEquals(start + 4, compactCell.id().value()),
                () -> assertEquals(start + 5, inputTranspose.id().value()),
                () -> assertEquals(start + 6, inputProjection.id().value()),
                () -> assertEquals(start + 7, hiddenTranspose.id().value()),
                () -> assertEquals(start + 8, hiddenProjection.id().value()),
                () -> assertEquals(start + 12, inputOutputSlice.id().value()),
                () -> assertEquals(start + 16, hiddenOutputSlice.id().value()),
                () -> assertEquals(start + 24, outputGate.id().value()),
                () -> assertEquals(start + 27, nextCell.id().value()),
                () -> assertEquals(start + 28, nextCellTanh.id().value()),
                () -> assertEquals(start + 29, nextHidden.id().value()),
                () -> assertEquals(start + 30, finalHiddenRow.id().value()),
                () -> assertEquals(start + 31, finalCellRow.id().value()),
                () -> assertEquals(start + 32, result.finalHidden().id().value()),
                () -> assertEquals(start + 33, result.finalCell().id().value()),
                () -> assertEquals(start + 34, ids.get()));
    }

    @Test
    void biasedOneStepCreatesExactlyThirtyFiveIdentifiers() throws Exception {
        LstmCell cell = new LstmCell(
                tensor(DataType.FLOAT32, Shape.of(16, 3), true),
                tensor(DataType.FLOAT32, Shape.of(16, 4), true),
                tensor(DataType.FLOAT32, Shape.of(16), true));
        LstmSequence sequence = new LstmSequence(cell);
        Tensor input = tensor(DataType.FLOAT32, Shape.of(1, 1, 3), false);
        Tensor hidden = tensor(DataType.FLOAT32, Shape.of(1, 4), false);
        Tensor currentCell = tensor(DataType.FLOAT32, Shape.of(1, 4), false);
        AtomicLong ids = nextTensorIdState();
        long start = ids.get();

        LstmSequenceForwardResult result = sequence.forward(
                input, hidden, currentCell, new long[] {1});

        assertAll(
                () -> assertEquals(1, result.packedOutputs().size()),
                () -> assertEquals(start + 35, ids.get()),
                () -> assertEquals(start + 34, result.finalCell().id().value()));
    }

    private static Tensor compactInput(LstmSequenceForwardResult result, int step) {
        Tensor outputGate = result.packedOutputs().get(step)
                .provenance().orElseThrow().inputs().getFirst();
        Tensor outputAdd = outputGate.provenance().orElseThrow().inputs().getFirst();
        Tensor inputOutputSlice = outputAdd.provenance().orElseThrow().inputs().getFirst();
        Tensor inputProjection = inputOutputSlice.provenance().orElseThrow().inputs().getFirst();
        return inputProjection.provenance().orElseThrow().inputs().getFirst();
    }

    private static Tensor compactHidden(LstmSequenceForwardResult result, int step) {
        Tensor outputGate = result.packedOutputs().get(step)
                .provenance().orElseThrow().inputs().getFirst();
        Tensor outputAdd = outputGate.provenance().orElseThrow().inputs().getFirst();
        Tensor hiddenOutputSlice = outputAdd.provenance().orElseThrow().inputs().get(1);
        Tensor hiddenProjection = hiddenOutputSlice.provenance().orElseThrow().inputs().getFirst();
        return hiddenProjection.provenance().orElseThrow().inputs().getFirst();
    }

    private static Tensor compactCell(LstmSequenceForwardResult result, int step) {
        Tensor nextCell = nextCellOf(result.packedOutputs().get(step));
        Tensor forgetProduct = nextCell.provenance().orElseThrow().inputs().getFirst();
        return forgetProduct.provenance().orElseThrow().inputs().get(1);
    }

    private static Tensor nextCellOf(Tensor nextHidden) {
        Tensor activatedCell = nextHidden.provenance().orElseThrow().inputs().get(1);
        return activatedCell.provenance().orElseThrow().inputs().getFirst();
    }

    private static Tensor gatherIndex(Tensor gather) {
        assertSame(AxisGatherKind.GATHER, gather.provenance().orElseThrow().operation().kind());
        return gather.provenance().orElseThrow().inputs().get(1);
    }

    private static long[] indexValues(Tensor gather) {
        Tensor indices = gatherIndex(gather);
        int count = Math.toIntExact(indices.descriptor().shape().knownElementCount().orElseThrow());
        long[] values = new long[count];
        for (int index = 0; index < count; index++) {
            values[index] = indices.hostStorage().orElseThrow().segment()
                    .getAtIndex(ValueLayout.JAVA_LONG, index);
        }
        return values;
    }

    private static void assertRowSelection(Tensor row, Tensor source, long index) {
        assertAll(
                () -> assertSame(SelectKind.SELECT,
                        row.provenance().orElseThrow().operation().kind()),
                () -> assertEquals(new SelectAttrs(0, index),
                        row.provenance().orElseThrow().operation().attrs()),
                () -> assertSame(source, row.provenance().orElseThrow().inputs().getFirst()));
    }

    private static Set<Object> allOperationKinds(LstmSequenceForwardResult result) {
        Set<Tensor> visited = Collections.newSetFromMap(new IdentityHashMap<>());
        Set<Object> kinds = Collections.newSetFromMap(new IdentityHashMap<>());
        ArrayDeque<Tensor> pending = new ArrayDeque<>();
        pending.addAll(result.packedOutputs());
        pending.add(result.finalHidden());
        pending.add(result.finalCell());
        while (!pending.isEmpty()) {
            Tensor tensor = pending.removeFirst();
            if (!visited.add(tensor) || tensor.provenance().isEmpty()) {
                continue;
            }
            var provenance = tensor.provenance().orElseThrow();
            kinds.add(provenance.operation().kind());
            pending.addAll(provenance.inputs());
        }
        return kinds;
    }

    private static LstmSequence sequence() {
        return new LstmSequence(new LstmCell(
                tensor(DataType.FLOAT32, Shape.of(16, 3), true),
                tensor(DataType.FLOAT32, Shape.of(16, 4), true)));
    }

    private static Tensor tensor(DataType dataType, Shape shape, boolean requiresGrad) {
        return TensorFactory.create(new TensorDescriptor(
                dataType, shape, Optional.empty(), requiresGrad));
    }

    private static AtomicLong nextTensorIdState() throws Exception {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }
}
