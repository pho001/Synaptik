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
import java.lang.foreign.ValueLayout;
import java.util.ArrayDeque;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import org.junit.jupiter.api.Test;

class BidirectionalLstmSequenceTest {
    @Test
    void ownsTwoIndependentCellTreesInDirectionalOrder() {
        BidirectionalLstmSequence sequence = new BidirectionalLstmSequence(
                3, true, DataType.FLOAT32, ParameterInitialization.zeros(), 5L, 7L);
        assertAll(
                () -> assertNotSame(sequence.forwardCell(), sequence.backwardCell()),
                () -> assertEquals(List.of("forward", "backward"),
                        List.copyOf(sequence.children().keySet())),
                () -> assertThrows(IllegalStateException.class, sequence::parametersRecursively));

        sequence.forward(tensor(Shape.of(1, 1, 2)), new long[] {1});
        assertEquals(
                List.of(
                        "forward.inputWeight", "forward.hiddenWeight", "forward.bias",
                        "backward.inputWeight", "backward.hiddenWeight", "backward.bias"),
                List.copyOf(sequence.parametersRecursively().keySet()));
        assertNotSame(sequence.forwardCell().hiddenWeight().value(),
                sequence.backwardCell().hiddenWeight().value());
    }

    @Test
    void reversePrefixAlignmentMergeAndAllDirectionalFinalStatesUsePlannedSources() {
        BidirectionalLstmSequence sequence = new BidirectionalLstmSequence(
                cell(3), cell(3));
        Tensor input = tensor(Shape.of(3, 3, 2));
        Tensor forwardHidden = tensor(Shape.of(3, 3));
        Tensor forwardCell = tensor(Shape.of(3, 3));
        Tensor backwardHidden = tensor(Shape.of(3, 3));
        Tensor backwardCell = tensor(Shape.of(3, 3));

        BidirectionalLstmSequenceForwardResult result = sequence.forward(
                input, forwardHidden, forwardCell, backwardHidden, backwardCell,
                new long[] {3, 1, 2});

        assertEquals(
                List.of(Shape.of(3, 6), Shape.of(2, 6), Shape.of(1, 6)),
                result.packedOutputs().stream().map(value -> value.descriptor().shape()).toList());
        Tensor flattenedReverse = alignedBackward(result.packedOutputs().getFirst())
                .provenance().orElseThrow().inputs().getFirst();
        List<Tensor> reverseSteps = flattenedReverse.provenance().orElseThrow().inputs();
        assertArrayEquals(new long[] {2, 0, 0, 1, 1, 2},
                values(reverseCoordinates(reverseSteps.get(0))));
        assertArrayEquals(new long[] {1, 0, 0, 2},
                values(reverseCoordinates(reverseSteps.get(1))));
        assertArrayEquals(new long[] {0, 0}, values(reverseCoordinates(reverseSteps.get(2))));
        assertArrayEquals(new long[] {5, 1, 4}, alignmentValues(result.packedOutputs().get(0)));
        assertArrayEquals(new long[] {3, 2}, alignmentValues(result.packedOutputs().get(1)));
        assertArrayEquals(new long[] {0}, alignmentValues(result.packedOutputs().get(2)));

        Tensor finalIndex = result.forwardFinalHidden().provenance().orElseThrow().inputs().get(1);
        assertSame(finalIndex,
                result.forwardFinalCell().provenance().orElseThrow().inputs().get(1));
        assertSame(finalIndex,
                result.backwardFinalHidden().provenance().orElseThrow().inputs().get(1));
        assertSame(finalIndex,
                result.backwardFinalCell().provenance().orElseThrow().inputs().get(1));
        assertArrayEquals(new long[] {8, 4, 7}, values(finalIndex));
        assertSame(TensorCompositionKind.CONCAT,
                result.packedOutputs().getFirst().provenance().orElseThrow().operation().kind());
    }

    @Test
    void allZeroRequestPreservesFourExactStatesAndDefaultStatesAreDistinct() {
        BidirectionalLstmSequence sequence = new BidirectionalLstmSequence(
                3, false, DataType.FLOAT32, ParameterInitialization.ones(), 1L, 2L);
        Tensor input = tensor(Shape.of(2, 2, 2));
        Tensor forwardHidden = tensor(Shape.of(2, 3));
        Tensor forwardCell = tensor(Shape.of(2, 3));
        Tensor backwardHidden = tensor(Shape.of(2, 3));
        Tensor backwardCell = tensor(Shape.of(2, 3));

        BidirectionalLstmSequenceForwardResult explicit = sequence.forward(
                input, forwardHidden, forwardCell, backwardHidden, backwardCell,
                new long[] {0, 0});
        assertAll(
                () -> assertTrue(explicit.packedOutputs().isEmpty()),
                () -> assertSame(forwardHidden, explicit.forwardFinalHidden()),
                () -> assertSame(forwardCell, explicit.forwardFinalCell()),
                () -> assertSame(backwardHidden, explicit.backwardFinalHidden()),
                () -> assertSame(backwardCell, explicit.backwardFinalCell()),
                () -> assertThrows(IllegalStateException.class, sequence.forwardCell()::parameters),
                () -> assertThrows(IllegalStateException.class, sequence.backwardCell()::parameters));

        BidirectionalLstmSequenceForwardResult defaults = sequence.forward(input, new long[] {0, 0});
        assertAll(
                () -> assertNotSame(defaults.forwardFinalHidden(), defaults.forwardFinalCell()),
                () -> assertNotSame(defaults.forwardFinalHidden(), defaults.backwardFinalHidden()),
                () -> assertNotSame(defaults.forwardFinalHidden(), defaults.backwardFinalCell()),
                () -> assertNotSame(defaults.forwardFinalCell(), defaults.backwardFinalHidden()),
                () -> assertNotSame(defaults.forwardFinalCell(), defaults.backwardFinalCell()),
                () -> assertNotSame(defaults.backwardFinalHidden(), defaults.backwardFinalCell()));
    }

    @Test
    void rejectsRepeatedOrSchemaMismatchedCells() {
        LstmCell first = cell(3);
        assertThrows(IllegalArgumentException.class,
                () -> new BidirectionalLstmSequence(first, first));
        assertThrows(IllegalArgumentException.class,
                () -> new BidirectionalLstmSequence(first, cell(4)));
        new BidirectionalLstmSequence(first, cell(3));
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

    private static LstmCell cell(long hiddenSize) {
        long packed = Math.multiplyExact(hiddenSize, 4L);
        return new LstmCell(
                parameter(Shape.of(packed, 2)),
                parameter(Shape.of(packed, hiddenSize)));
    }

    private static Tensor parameter(Shape shape) {
        return TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.empty(), true));
    }

    private static Tensor tensor(Shape shape) {
        return TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.empty(), false));
    }
}
