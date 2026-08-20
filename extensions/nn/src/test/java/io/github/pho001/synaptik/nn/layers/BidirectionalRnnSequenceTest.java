package io.github.pho001.synaptik.nn.layers;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.index.AxisGatherKind;
import io.github.pho001.synaptik.model.operation.index.GatherNdKind;
import io.github.pho001.synaptik.model.operation.layout.TensorCompositionKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.initialization.ParameterInitialization;
import io.github.pho001.synaptik.nn.module.ForwardMode;
import java.lang.reflect.Field;
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

class BidirectionalRnnSequenceTest {
    @Test
    void ownsIndependentNamedCellsAndAutomaticParameterTrees() {
        BidirectionalRnnSequence sequence = new BidirectionalRnnSequence(
                3, false, DataType.FLOAT32, ParameterInitialization.zeros(), 11L, 17L);

        assertAll(
                () -> assertNotSame(sequence.forwardCell(), sequence.backwardCell()),
                () -> assertEquals(List.of("forward", "backward"),
                        List.copyOf(sequence.children().keySet())),
                () -> assertThrows(IllegalStateException.class, sequence::parametersRecursively));

        sequence.forward(tensor(Shape.of(1, 1, 2)), new long[] {1});
        assertEquals(
                List.of(
                        "forward.inputWeight", "forward.hiddenWeight",
                        "backward.inputWeight", "backward.hiddenWeight"),
                List.copyOf(sequence.parametersRecursively().keySet()));
        assertNotSame(
                sequence.forwardCell().inputWeight().value(),
                sequence.backwardCell().inputWeight().value());

        sequence.eval();
        assertSame(ForwardMode.EVALUATION, sequence.forwardCell().mode());
        assertSame(ForwardMode.EVALUATION, sequence.backwardCell().mode());
    }

    @Test
    void rejectsSharedOrMismatchedCellsBeforeEitherOwnershipLinkChanges() {
        RnnCell first = cell(3, DataType.FLOAT32);
        RnnCell mismatch = cell(4, DataType.FLOAT32);

        assertThrows(IllegalArgumentException.class, () -> new BidirectionalRnnSequence(first, first));
        assertThrows(IllegalArgumentException.class,
                () -> new BidirectionalRnnSequence(first, mismatch));
        assertTrue(first.children().isEmpty());

        BidirectionalRnnSequence owner = new BidirectionalRnnSequence(
                cell(3, DataType.FLOAT32), cell(3, DataType.FLOAT32));
        RnnCell available = cell(3, DataType.FLOAT32);
        assertThrows(IllegalStateException.class,
                () -> new BidirectionalRnnSequence(available, owner.backwardCell()));
        new BidirectionalRnnSequence(available, cell(3, DataType.FLOAT32));
    }

    @Test
    void reversesOnlyValidPrefixesRealignsOriginalTimeAndMergesForwardFirst() {
        BidirectionalRnnSequence sequence = new BidirectionalRnnSequence(
                cell(3, DataType.FLOAT32), cell(3, DataType.FLOAT32));
        Tensor input = tensor(Shape.of(3, 3, 2));
        Tensor forwardInitial = tensor(Shape.of(3, 3));
        Tensor backwardInitial = tensor(Shape.of(3, 3));

        BidirectionalRnnSequenceForwardResult result = sequence.forward(
                input, forwardInitial, backwardInitial, new long[] {3, 1, 2});

        assertEquals(
                List.of(Shape.of(3, 6), Shape.of(2, 6), Shape.of(1, 6)),
                result.packedOutputs().stream().map(value -> value.descriptor().shape()).toList());
        for (Tensor output : result.packedOutputs()) {
            assertSame(TensorCompositionKind.CONCAT,
                    output.provenance().orElseThrow().operation().kind());
        }

        Tensor flattenedReverse = alignedBackward(result.packedOutputs().getFirst())
                .provenance().orElseThrow().inputs().getFirst();
        List<Tensor> reverseSteps = flattenedReverse.provenance().orElseThrow().inputs();
        assertSame(TensorCompositionKind.CONCAT,
                flattenedReverse.provenance().orElseThrow().operation().kind());
        assertArrayEquals(new long[] {2, 0, 0, 1, 1, 2},
                values(reverseCoordinates(reverseSteps.get(0))));
        assertArrayEquals(new long[] {1, 0, 0, 2},
                values(reverseCoordinates(reverseSteps.get(1))));
        assertArrayEquals(new long[] {0, 0},
                values(reverseCoordinates(reverseSteps.get(2))));

        assertArrayEquals(new long[] {5, 1, 4}, alignmentValues(result.packedOutputs().get(0)));
        assertArrayEquals(new long[] {3, 2}, alignmentValues(result.packedOutputs().get(1)));
        assertArrayEquals(new long[] {0}, alignmentValues(result.packedOutputs().get(2)));
        assertSame(result.packedOutputs().get(0).provenance().orElseThrow().inputs().getFirst(),
                result.packedOutputs().get(0).provenance().orElseThrow().inputs().getFirst());

        Tensor forwardFinalIndex = result.forwardFinalHidden().provenance().orElseThrow().inputs().get(1);
        Tensor backwardFinalIndex = result.backwardFinalHidden().provenance().orElseThrow().inputs().get(1);
        assertSame(forwardFinalIndex, backwardFinalIndex);
        assertArrayEquals(new long[] {8, 4, 7}, values(forwardFinalIndex));
    }

    @Test
    void allZeroLengthsPreserveExactStatesAndDoNotBindEitherAutomaticCell() {
        BidirectionalRnnSequence sequence = new BidirectionalRnnSequence(
                3, true, DataType.FLOAT32, ParameterInitialization.ones(), 1L, 2L);
        Tensor input = tensor(Shape.of(4, 2, 2));
        Tensor forwardInitial = tensor(Shape.of(2, 3));
        Tensor backwardInitial = tensor(Shape.of(2, 3));

        BidirectionalRnnSequenceForwardResult explicit = sequence.forward(
                input, forwardInitial, backwardInitial, new long[] {0, 0});
        assertAll(
                () -> assertTrue(explicit.packedOutputs().isEmpty()),
                () -> assertSame(forwardInitial, explicit.forwardFinalHidden()),
                () -> assertSame(backwardInitial, explicit.backwardFinalHidden()),
                () -> assertThrows(IllegalStateException.class, sequence.forwardCell()::parameters),
                () -> assertThrows(IllegalStateException.class, sequence.backwardCell()::parameters));

        BidirectionalRnnSequenceForwardResult defaults = sequence.forward(input, new long[] {0, 0});
        assertNotSame(defaults.forwardFinalHidden(), defaults.backwardFinalHidden());
        assertTrue(defaults.forwardFinalHidden().provenance().isEmpty());
        assertTrue(defaults.backwardFinalHidden().provenance().isEmpty());
    }

    @Test
    void defaultTypeFailurePrecedesStateAndAutomaticParameterTensorEffects() throws Exception {
        BidirectionalRnnSequence sequence = new BidirectionalRnnSequence(
                3, false, DataType.FLOAT32, ParameterInitialization.zeros(), 1L, 2L);
        Tensor input = tensor(DataType.FLOAT64, Shape.of(1, 2, 2), false);
        AtomicLong ids = nextTensorIdState();
        long before = ids.get();

        assertAll(
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(input, new long[] {0, 1})),
                () -> assertEquals(before, ids.get()),
                () -> assertThrows(IllegalStateException.class, sequence.forwardCell()::parameters),
                () -> assertThrows(IllegalStateException.class, sequence.backwardCell()::parameters));
    }

    @Test
    void strictLoadBindsBothAutomaticCellsWithoutRunningForwardInitialization() {
        Tensor input = tensor(Shape.of(1, 1, 2));
        BidirectionalRnnSequence source = new BidirectionalRnnSequence(
                3, true, DataType.FLOAT32, ParameterInitialization.ones(), 3L, 4L);
        source.forward(input, new long[] {1});
        BidirectionalRnnSequence target = new BidirectionalRnnSequence(
                3, true, DataType.FLOAT32, ParameterInitialization.zeros(), 30L, 40L);

        target.loadStateDictionary(source.stateDictionary());

        assertEquals(
                source.stateDictionary().entries().stream().map(value -> value.path()).toList(),
                target.stateDictionary().entries().stream().map(value -> value.path()).toList());
        assertSame(source.forwardCell().inputWeight().value(),
                target.forwardCell().inputWeight().value());
        assertSame(source.backwardCell().hiddenWeight().value(),
                target.backwardCell().hiddenWeight().value());
    }

    private static Tensor alignedBackward(Tensor merged) {
        Tensor aligned = merged.provenance().orElseThrow().inputs().get(1);
        assertSame(AxisGatherKind.GATHER, aligned.provenance().orElseThrow().operation().kind());
        return aligned;
    }

    private static long[] alignmentValues(Tensor merged) {
        return values(alignedBackward(merged).provenance().orElseThrow().inputs().get(1));
    }

    private static Tensor reverseCoordinates(Tensor reverseOutput) {
        Tensor preactivation = reverseOutput.provenance().orElseThrow().inputs().getFirst();
        Tensor inputProjection = preactivation.provenance().orElseThrow().inputs().getFirst();
        Tensor compactInput = inputProjection.provenance().orElseThrow().inputs().getFirst();
        assertSame(GatherNdKind.GATHER_ND,
                compactInput.provenance().orElseThrow().operation().kind());
        return compactInput.provenance().orElseThrow().inputs().get(1);
    }

    private static long[] values(Tensor indices) {
        int count = Math.toIntExact(indices.descriptor().shape().knownElementCount().orElseThrow());
        long[] values = new long[count];
        for (int index = 0; index < count; index++) {
            values[index] = indices.hostStorage().orElseThrow().segment()
                    .getAtIndex(ValueLayout.JAVA_LONG, index);
        }
        return values;
    }

    private static RnnCell cell(long hiddenSize, DataType type) {
        return new RnnCell(
                parameter(type, Shape.of(hiddenSize, 2)),
                parameter(type, Shape.of(hiddenSize, hiddenSize)));
    }

    private static Tensor parameter(DataType type, Shape shape) {
        return TensorFactory.create(new TensorDescriptor(type, shape, Optional.empty(), true));
    }

    private static Tensor tensor(Shape shape) {
        return tensor(DataType.FLOAT32, shape, false);
    }

    private static Tensor tensor(DataType type, Shape shape, boolean requiresGrad) {
        return TensorFactory.create(new TensorDescriptor(
                type, shape, Optional.empty(), requiresGrad));
    }

    private static AtomicLong nextTensorIdState() throws Exception {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }
}
