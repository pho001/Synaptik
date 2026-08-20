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
import java.lang.foreign.ValueLayout;
import java.util.ArrayDeque;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import org.junit.jupiter.api.Test;

class BidirectionalGruSequenceTest {
    @Test
    void ownsIndependentNamedCellsAndAutomaticParameterTrees() {
        BidirectionalGruSequence sequence = new BidirectionalGruSequence(
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
        GruCell first = cell(3, DataType.FLOAT32);
        GruCell mismatch = cell(4, DataType.FLOAT32);

        assertThrows(IllegalArgumentException.class, () -> new BidirectionalGruSequence(first, first));
        assertThrows(IllegalArgumentException.class,
                () -> new BidirectionalGruSequence(first, mismatch));
        assertTrue(first.children().isEmpty());

        BidirectionalGruSequence owner = new BidirectionalGruSequence(
                cell(3, DataType.FLOAT32), cell(3, DataType.FLOAT32));
        GruCell available = cell(3, DataType.FLOAT32);
        assertThrows(IllegalStateException.class,
                () -> new BidirectionalGruSequence(available, owner.backwardCell()));
        new BidirectionalGruSequence(available, cell(3, DataType.FLOAT32));
    }

    @Test
    void reversesOnlyValidPrefixesRealignsOriginalTimeAndMergesForwardFirst() {
        BidirectionalGruSequence sequence = new BidirectionalGruSequence(
                cell(3, DataType.FLOAT32), cell(3, DataType.FLOAT32));
        Tensor input = tensor(Shape.of(3, 3, 2));
        Tensor forwardInitial = tensor(Shape.of(3, 3));
        Tensor backwardInitial = tensor(Shape.of(3, 3));

        BidirectionalGruSequenceForwardResult result = sequence.forward(
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
        BidirectionalGruSequence sequence = new BidirectionalGruSequence(
                3, true, DataType.FLOAT32, ParameterInitialization.ones(), 1L, 2L);
        Tensor input = tensor(Shape.of(4, 2, 2));
        Tensor forwardInitial = tensor(Shape.of(2, 3));
        Tensor backwardInitial = tensor(Shape.of(2, 3));

        BidirectionalGruSequenceForwardResult explicit = sequence.forward(
                input, forwardInitial, backwardInitial, new long[] {0, 0});
        assertAll(
                () -> assertTrue(explicit.packedOutputs().isEmpty()),
                () -> assertSame(forwardInitial, explicit.forwardFinalHidden()),
                () -> assertSame(backwardInitial, explicit.backwardFinalHidden()),
                () -> assertThrows(IllegalStateException.class, sequence.forwardCell()::parameters),
                () -> assertThrows(IllegalStateException.class, sequence.backwardCell()::parameters));

        BidirectionalGruSequenceForwardResult defaults = sequence.forward(input, new long[] {0, 0});
        assertNotSame(defaults.forwardFinalHidden(), defaults.backwardFinalHidden());
        assertTrue(defaults.forwardFinalHidden().provenance().isEmpty());
        assertTrue(defaults.backwardFinalHidden().provenance().isEmpty());
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
        Set<Tensor> visited = Collections.newSetFromMap(new IdentityHashMap<>());
        ArrayDeque<Tensor> pending = new ArrayDeque<>();
        pending.add(reverseOutput);
        while (!pending.isEmpty()) {
            Tensor candidate = pending.removeFirst();
            if (!visited.add(candidate) || candidate.provenance().isEmpty()) {
                continue;
            }
            if (candidate.provenance().orElseThrow().operation().kind() == GatherNdKind.GATHER_ND) {
                return candidate.provenance().orElseThrow().inputs().get(1);
            }
            pending.addAll(candidate.provenance().orElseThrow().inputs());
        }
        throw new AssertionError("reverse output has no Gather-ND input");
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

    private static GruCell cell(long hiddenSize, DataType type) {
        return new GruCell(
                parameter(type, Shape.of(Math.multiplyExact(hiddenSize, 3L), 2)),
                parameter(type, Shape.of(Math.multiplyExact(hiddenSize, 3L), hiddenSize)));
    }

    private static Tensor parameter(DataType type, Shape shape) {
        return TensorFactory.create(new TensorDescriptor(type, shape, Optional.empty(), true));
    }

    private static Tensor tensor(Shape shape) {
        return TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.empty(), false));
    }
}
