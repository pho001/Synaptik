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
import java.util.ArrayList;
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
class GruSequencePackingTest {
    @Test
    void packsFiveThreeOneIntoNineActiveRowsAndRestoresExitRows() {
        GruSequence sequence = sequence(false);
        Tensor input = tensor(DataType.FLOAT32, Shape.of(5, 3, 3), false);
        Tensor hidden = tensor(DataType.FLOAT32, Shape.of(3, 4), false);
        GruSequenceForwardResult result = sequence.forward(input, hidden, new long[] {5, 3, 1});

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
                () -> assertSame(TensorCompositionKind.STACK,
                        result.finalHidden().provenance().orElseThrow().operation().kind()));

        List<Tensor> finalRows = result.finalHidden().provenance().orElseThrow().inputs();
        assertRowSelection(finalRows.get(0), result.packedOutputs().get(4), 0);
        assertRowSelection(finalRows.get(1), result.packedOutputs().get(2), 1);
        assertRowSelection(finalRows.get(2), result.packedOutputs().get(0), 2);
    }

    @Test
    void unsortedLengthsUseStableOriginalAndRelativeSurvivorIndicesWithoutSorting() {
        GruSequence sequence = sequence(false);
        Tensor input = tensor(DataType.FLOAT32, Shape.of(3, 3, 3), false);
        Tensor hidden = tensor(DataType.FLOAT32, Shape.of(3, 4), false);
        GruSequenceForwardResult result = sequence.forward(input, hidden, new long[] {1, 3, 2});

        assertAll(
                () -> assertArrayEquals(new long[] {0, 1, 2}, indexValues(compactInput(result, 0))),
                () -> assertArrayEquals(new long[] {1, 2}, indexValues(compactInput(result, 1))),
                () -> assertArrayEquals(new long[] {1}, indexValues(compactInput(result, 2))),
                () -> assertArrayEquals(new long[] {1, 2}, indexValues(compactHidden(result, 1))),
                () -> assertArrayEquals(new long[] {0}, indexValues(compactHidden(result, 2))));

        Tensor stepZeroInputIndex = gatherIndex(compactInput(result, 0));
        Tensor stepZeroHiddenIndex = gatherIndex(compactHidden(result, 0));
        assertSame(stepZeroInputIndex, stepZeroHiddenIndex);

        List<Tensor> finalRows = result.finalHidden().provenance().orElseThrow().inputs();
        assertRowSelection(finalRows.get(0), result.packedOutputs().get(0), 0);
        assertRowSelection(finalRows.get(1), result.packedOutputs().get(2), 0);
        assertRowSelection(finalRows.get(2), result.packedOutputs().get(1), 1);
    }

    @Test
    void zeroLengthRowsNeverEnterCellOperandsAndUseInitialHiddenForFinalState() {
        GruSequence sequence = sequence(false);
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3, 3), false);
        Tensor hidden = tensor(DataType.FLOAT32, Shape.of(3, 4), false);
        GruSequenceForwardResult result = sequence.forward(input, hidden, new long[] {0, 2, 0});

        assertAll(
                () -> assertEquals(List.of(Shape.of(1, 4), Shape.of(1, 4)),
                        result.packedOutputs().stream()
                                .map(value -> value.descriptor().shape()).toList()),
                () -> assertArrayEquals(new long[] {1}, indexValues(compactInput(result, 0))),
                () -> assertArrayEquals(new long[] {1}, indexValues(compactInput(result, 1))));

        List<Tensor> finalRows = result.finalHidden().provenance().orElseThrow().inputs();
        assertRowSelection(finalRows.get(0), hidden, 0);
        assertRowSelection(finalRows.get(1), result.packedOutputs().get(1), 0);
        assertRowSelection(finalRows.get(2), hidden, 2);
    }

    @Test
    void storageFreeZeroValuedInputRemainsActiveAndNoCumulativeScanAppears() {
        GruSequence sequence = sequence(false);
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 2, 3), false);
        Tensor hidden = tensor(DataType.FLOAT32, Shape.of(2, 4), false);
        assertTrue(input.hostStorage().isEmpty());
        GruSequenceForwardResult result = sequence.forward(input, hidden, new long[] {2, 1});

        assertAll(
                () -> assertArrayEquals(new long[] {0, 1}, indexValues(compactInput(result, 0))),
                () -> assertArrayEquals(new long[] {0}, indexValues(compactInput(result, 1))),
                () -> assertTrue(allOperationKinds(result).stream()
                        .noneMatch(CumulativeScanKind.class::isInstance)));
    }

    @Test
    void snapshotsLengthsAndCreatesExactUnlabeledDenseInt64IndexLeaves() {
        GruSequence sequence = sequence(false);
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 2, 3), false);
        Tensor hidden = tensor(DataType.FLOAT32, Shape.of(2, 4), false);
        long[] lengths = {2, 1};
        GruSequenceForwardResult result = sequence.forward(input, hidden, lengths);
        lengths[0] = 0;
        lengths[1] = 0;

        Tensor index = gatherIndex(compactInput(result, 0));
        assertAll(
                () -> assertArrayEquals(new long[] {0, 1}, indexValues(compactInput(result, 0))),
                () -> assertArrayEquals(new long[] {0}, indexValues(compactInput(result, 1))),
                () -> assertSame(DataType.INT64, index.descriptor().dataType()),
                () -> assertEquals(Shape.of(2), index.descriptor().shape()),
                () -> assertFalse(index.descriptor().requiresGrad()),
                () -> assertSame(LayoutKind.DENSE_CONTIGUOUS,
                        index.descriptor().layout().orElseThrow().kind()),
                () -> assertTrue(index.label().isEmpty()),
                () -> assertTrue(index.provenance().isEmpty()),
                () -> assertTrue(index.hostStorage().isPresent()));
    }

    @Test
    void everyStepStartsWithTimeSelectAndEveryGatherUsesAxisZero() {
        GruSequence sequence = sequence(false);
        Tensor input = tensor(DataType.FLOAT64, Shape.of(3, 2, 3), false);
        Tensor hidden = tensor(DataType.FLOAT32, Shape.of(2, 4), false);
        GruSequenceForwardResult result = sequence.forward(input, hidden, new long[] {3, 2});

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
            assertEquals(new SelectAttrs(0, step),
                    timeSlice.provenance().orElseThrow().operation().attrs());
        }
        assertAll(
                () -> assertSame(DataType.FLOAT64,
                        result.packedOutputs().getFirst().descriptor().dataType()),
                () -> assertSame(DataType.FLOAT64, result.finalHidden().descriptor().dataType()));
    }

    @Test
    void oneStepUsesFixedTwentySixAndTwentySevenIdentityOrders() throws Exception {
        assertOneStepIdentityOrder(false, 26);
        assertOneStepIdentityOrder(true, 27);
    }

    private static void assertOneStepIdentityOrder(boolean biased, int expectedCount)
            throws Exception {
        GruSequence sequence = sequence(biased);
        Tensor input = tensor(DataType.FLOAT32, Shape.of(1, 1, 3), false);
        Tensor hidden = tensor(DataType.FLOAT32, Shape.of(1, 4), false);
        AtomicLong ids = nextTensorIdState();
        long start = ids.get();

        GruSequenceForwardResult result = sequence.forward(input, hidden, new long[] {1});
        Tensor compactInput = compactInput(result, 0);
        Tensor compactHidden = compactHidden(result, 0);
        Tensor timeSlice = compactInput.provenance().orElseThrow().inputs().getFirst();
        Tensor index = gatherIndex(compactInput);
        List<Tensor> ordered = new ArrayList<>();
        ordered.add(timeSlice);
        ordered.add(index);
        ordered.add(compactInput);
        ordered.add(compactHidden);
        ordered.addAll(orderedGruTensors(result.packedOutputs().getFirst(), biased));
        ordered.add(result.finalHidden().provenance().orElseThrow().inputs().getFirst());
        ordered.add(result.finalHidden());

        assertEquals(expectedCount, ordered.size());
        for (int indexPosition = 0; indexPosition < ordered.size(); indexPosition++) {
            assertEquals(start + indexPosition, ordered.get(indexPosition).id().value(),
                    "ID at " + indexPosition);
        }
        assertEquals(start + expectedCount, ids.get());
    }

    private static List<Tensor> orderedGruTensors(Tensor result, boolean biased) {
        Tensor candidate = result.provenance().orElseThrow().inputs().getFirst();
        Tensor weighted = result.provenance().orElseThrow().inputs().get(1);
        Tensor update = weighted.provenance().orElseThrow().inputs().getFirst();
        Tensor difference = weighted.provenance().orElseThrow().inputs().get(1);
        Tensor candidateAdd = candidate.provenance().orElseThrow().inputs().getFirst();
        Tensor resetProduct = candidateAdd.provenance().orElseThrow().inputs().get(1);
        Tensor reset = resetProduct.provenance().orElseThrow().inputs().getFirst();
        Tensor resetAdd = reset.provenance().orElseThrow().inputs().getFirst();
        Tensor updateAdd = update.provenance().orElseThrow().inputs().getFirst();
        Tensor inputReset = resetAdd.provenance().orElseThrow().inputs().getFirst();
        Tensor hiddenReset = resetAdd.provenance().orElseThrow().inputs().get(1);
        Tensor inputUpdate = updateAdd.provenance().orElseThrow().inputs().getFirst();
        Tensor hiddenUpdate = updateAdd.provenance().orElseThrow().inputs().get(1);
        Tensor inputCandidate = candidateAdd.provenance().orElseThrow().inputs().getFirst();
        Tensor hiddenCandidate = resetProduct.provenance().orElseThrow().inputs().get(1);
        Tensor inputProjection = inputReset.provenance().orElseThrow().inputs().getFirst();
        Tensor hiddenProjection = hiddenReset.provenance().orElseThrow().inputs().getFirst();
        Tensor inputProduct = biased
                ? inputProjection.provenance().orElseThrow().inputs().getFirst()
                : inputProjection;
        Tensor inputTranspose = inputProduct.provenance().orElseThrow().inputs().get(1);
        Tensor hiddenTranspose = hiddenProjection.provenance().orElseThrow().inputs().get(1);
        List<Tensor> tensors = new ArrayList<>();
        tensors.add(inputTranspose);
        tensors.add(inputProduct);
        if (biased) {
            tensors.add(inputProjection);
        }
        tensors.add(hiddenTranspose);
        tensors.add(hiddenProjection);
        tensors.addAll(List.of(
                inputReset, inputUpdate, inputCandidate,
                hiddenReset, hiddenUpdate, hiddenCandidate,
                resetAdd, reset, updateAdd, update, resetProduct,
                candidateAdd, candidate, difference, weighted, result));
        return tensors;
    }

    private static Tensor compactInput(GruSequenceForwardResult result, int step) {
        Tensor candidate = result.packedOutputs().get(step)
                .provenance().orElseThrow().inputs().getFirst();
        Tensor candidateAdd = candidate.provenance().orElseThrow().inputs().getFirst();
        Tensor inputCandidate = candidateAdd.provenance().orElseThrow().inputs().getFirst();
        Tensor inputProjection = inputCandidate.provenance().orElseThrow().inputs().getFirst();
        Tensor product = inputProjection;
        if (inputProjection.provenance().orElseThrow().operation().kind().toString().equals("ADD")) {
            product = inputProjection.provenance().orElseThrow().inputs().getFirst();
        }
        return product.provenance().orElseThrow().inputs().getFirst();
    }

    private static Tensor compactHidden(GruSequenceForwardResult result, int step) {
        Tensor weighted = result.packedOutputs().get(step)
                .provenance().orElseThrow().inputs().get(1);
        Tensor difference = weighted.provenance().orElseThrow().inputs().get(1);
        return difference.provenance().orElseThrow().inputs().getFirst();
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

    private static Set<Object> allOperationKinds(GruSequenceForwardResult result) {
        Set<Tensor> visited = Collections.newSetFromMap(new IdentityHashMap<>());
        Set<Object> kinds = Collections.newSetFromMap(new IdentityHashMap<>());
        ArrayDeque<Tensor> pending = new ArrayDeque<>();
        pending.addAll(result.packedOutputs());
        pending.add(result.finalHidden());
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

    private static GruSequence sequence(boolean bias) {
        Tensor inputWeight = tensor(DataType.FLOAT32, Shape.of(12, 3), true);
        Tensor hiddenWeight = tensor(DataType.FLOAT32, Shape.of(12, 4), true);
        GruCell cell = bias
                ? new GruCell(inputWeight, hiddenWeight,
                        tensor(DataType.FLOAT32, Shape.of(12), true))
                : new GruCell(inputWeight, hiddenWeight);
        return new GruSequence(cell);
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
