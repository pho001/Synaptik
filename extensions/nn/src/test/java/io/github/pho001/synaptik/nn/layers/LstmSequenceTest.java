package io.github.pho001.synaptik.nn.layers;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.initialization.ParameterInitialization;
import io.github.pho001.synaptik.nn.module.ForwardMode;
import io.github.pho001.synaptik.nn.module.Module;
import io.github.pho001.synaptik.nn.module.Sequential;
import io.github.pho001.synaptik.nn.module.UnaryTensorModule;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.lang.reflect.RecordComponent;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;

@Execution(ExecutionMode.SAME_THREAD)
class LstmSequenceTest {
    @Test
    void defaultStatesPreflightRejectsMixedFinalRowsBeforeStateOrCellEffects() throws Exception {
        LstmSequence sequence = new LstmSequence(
                4, false, DataType.FLOAT32, ParameterInitialization.zeros(), 1L);
        Tensor input = tensor(DataType.FLOAT64, Shape.of(1, 2, 3), false);
        AtomicLong ids = nextTensorIdState();
        long before = ids.get();

        assertAll(
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(input, new long[] {0, 1})),
                () -> assertEquals(before, ids.get()),
                () -> assertThrows(IllegalStateException.class, sequence.cell()::parameters));
    }

    @Test
    void standardSequenceDerivesDistinctZeroStatesAndBindsOnlyForRepresentedSteps() {
        LstmSequence sequence = new LstmSequence(
                4, false, DataType.FLOAT32, ParameterInitialization.zeros(), 19L);
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 2, 3), false);

        LstmSequenceForwardResult skipped = sequence.forward(input, new long[] {0, 0});

        assertAll(
                () -> assertTrue(skipped.packedOutputs().isEmpty()),
                () -> assertEquals(Shape.of(2, 4), skipped.finalHidden().descriptor().shape()),
                () -> assertEquals(Shape.of(2, 4), skipped.finalCell().descriptor().shape()),
                () -> assertNotSame(skipped.finalHidden(), skipped.finalCell()),
                () -> assertNotEquals(skipped.finalHidden().id(), skipped.finalCell().id()),
                () -> assertFalse(skipped.finalHidden().descriptor().requiresGrad()),
                () -> assertFalse(skipped.finalCell().descriptor().requiresGrad()),
                () -> assertTrue(skipped.finalHidden().hostStorage().isPresent()),
                () -> assertTrue(skipped.finalCell().hostStorage().isPresent()),
                () -> assertThrows(IllegalStateException.class, sequence.cell()::parameters));

        LstmSequenceForwardResult complete = sequence.forward(input);

        assertAll(
                () -> assertEquals(2, complete.packedOutputs().size()),
                () -> assertNotSame(complete.packedOutputs().get(0),
                        complete.packedOutputs().get(1)),
                () -> assertNotEquals(complete.packedOutputs().get(0).id(),
                        complete.packedOutputs().get(1).id()),
                () -> assertSame(sequence.cell().inputWeight().value(),
                        inputWeightOf(complete.packedOutputs().get(0))),
                () -> assertSame(sequence.cell().inputWeight().value(),
                        inputWeightOf(complete.packedOutputs().get(1))),
                () -> assertEquals(List.of("inputWeight", "hiddenWeight"),
                        sequence.cell().parameters().stream().map(value -> value.name()).toList()),
                () -> assertNotSame(skipped.finalHidden(), complete.finalHidden()),
                () -> assertNotSame(skipped.finalCell(), complete.finalCell()));
    }

    @Test
    void exposesExactlyThePlannedDirectModuleAndThreeComponentRecordSurface() throws Exception {
        Set<List<Class<?>>> constructors = Arrays.stream(LstmSequence.class.getDeclaredConstructors())
                .filter(value -> Modifier.isPublic(value.getModifiers()))
                .map(value -> List.of(value.getParameterTypes()))
                .collect(Collectors.toSet());
        Set<String> methods = Arrays.stream(LstmSequence.class.getDeclaredMethods())
                .filter(value -> Modifier.isPublic(value.getModifiers()))
                .map(value -> value.getName())
                .collect(Collectors.toSet());
        RecordComponent[] components = LstmSequenceForwardResult.class.getRecordComponents();

        assertAll(
                () -> assertTrue(Modifier.isFinal(LstmSequence.class.getModifiers())),
                () -> assertSame(Module.class, LstmSequence.class.getSuperclass()),
                () -> assertFalse(UnaryTensorModule.class.isAssignableFrom(LstmSequence.class)),
                () -> assertFalse(Sequential.class.isAssignableFrom(LstmSequence.class)),
                () -> assertEquals(Set.of(
                        List.of(LstmCell.class),
                        List.of(long.class, boolean.class, DataType.class,
                                ParameterInitialization.class, long.class)), constructors),
                () -> assertEquals(Set.of("cell", "forward"), methods),
                () -> assertSame(
                        LstmCell.class, LstmSequence.class.getDeclaredMethod("cell").getReturnType()),
                () -> assertSame(
                        LstmSequenceForwardResult.class,
                        LstmSequence.class.getDeclaredMethod(
                                        "forward",
                                        Tensor.class,
                                        Tensor.class,
                                        Tensor.class,
                                        long[].class)
                                .getReturnType()),
                () -> assertSame(LstmSequenceForwardResult.class,
                        LstmSequence.class.getDeclaredMethod(
                                        "forward", Tensor.class, Tensor.class, Tensor.class)
                                .getReturnType()),
                () -> assertSame(LstmSequenceForwardResult.class,
                        LstmSequence.class.getDeclaredMethod("forward", Tensor.class, long[].class)
                                .getReturnType()),
                () -> assertSame(LstmSequenceForwardResult.class,
                        LstmSequence.class.getDeclaredMethod("forward", Tensor.class)
                                .getReturnType()),
                () -> assertEquals(1, LstmSequence.class.getDeclaredFields().length),
                () -> assertEquals("cell", LstmSequence.class.getDeclaredFields()[0].getName()),
                () -> assertTrue(Arrays.stream(LstmSequence.class.getDeclaredClasses())
                        .noneMatch(type -> Modifier.isPublic(type.getModifiers())
                                || Modifier.isProtected(type.getModifiers()))),
                () -> assertTrue(LstmSequenceForwardResult.class.isRecord()),
                () -> assertEquals(
                        List.of("packedOutputs", "finalHidden", "finalCell"),
                        Arrays.stream(components).map(RecordComponent::getName).toList()),
                () -> assertEquals(
                        List.of(List.class, Tensor.class, Tensor.class),
                        Arrays.stream(components).map(RecordComponent::getType).toList()),
                () -> assertEquals(0, LstmSequenceForwardResult.class.getDeclaredClasses().length));
    }

    @Test
    void resultSnapshotsStructureRetainsExactReferencesAndChecksNullsInOrder() {
        Tensor first = tensor(DataType.FLOAT32, Shape.of(2, 4), false);
        Tensor second = tensor(DataType.FLOAT32, Shape.of(1, 4), false);
        Tensor hidden = tensor(DataType.FLOAT32, Shape.of(3, 4), false);
        Tensor cell = tensor(DataType.FLOAT32, Shape.of(3, 4), false);
        ArrayList<Tensor> source = new ArrayList<>(List.of(first, second));
        LstmSequenceForwardResult result = new LstmSequenceForwardResult(source, hidden, cell);
        source.clear();

        assertAll(
                () -> assertEquals(2, result.packedOutputs().size()),
                () -> assertSame(first, result.packedOutputs().get(0)),
                () -> assertSame(second, result.packedOutputs().get(1)),
                () -> assertSame(hidden, result.finalHidden()),
                () -> assertSame(cell, result.finalCell()),
                () -> assertThrows(
                        UnsupportedOperationException.class,
                        () -> result.packedOutputs().add(first)),
                () -> assertEquals(
                        new LstmSequenceForwardResult(List.of(first, second), hidden, cell), result),
                () -> assertEquals("packedOutputs", assertThrows(
                        NullPointerException.class,
                        () -> new LstmSequenceForwardResult(null, null, null)).getMessage()),
                () -> assertEquals("packedOutputs[1]", assertThrows(
                        NullPointerException.class,
                        () -> new LstmSequenceForwardResult(
                                Arrays.asList(first, null), null, null)).getMessage()),
                () -> assertEquals("finalHidden", assertThrows(
                        NullPointerException.class,
                        () -> new LstmSequenceForwardResult(List.of(first), null, null)).getMessage()),
                () -> assertEquals("finalCell", assertThrows(
                        NullPointerException.class,
                        () -> new LstmSequenceForwardResult(List.of(first), hidden, null)).getMessage()));
    }

    @Test
    void ownsExactlyOneCellAndPreservesRecursiveStateAndModeBehavior() {
        LstmCell cell = cell(DataType.FLOAT32, true);
        LstmSequence sequence = new LstmSequence(cell);

        assertAll(
                () -> assertSame(cell, sequence.cell()),
                () -> assertTrue(sequence.parameters().isEmpty()),
                () -> assertTrue(sequence.buffers().isEmpty()),
                () -> assertEquals(List.of("cell"), List.copyOf(sequence.children().keySet())),
                () -> assertSame(cell, sequence.children().get("cell")),
                () -> assertEquals(
                        List.of("cell.inputWeight", "cell.hiddenWeight", "cell.bias"),
                        List.copyOf(sequence.parametersRecursively().keySet())),
                () -> assertEquals(
                        List.of("cell.inputWeight", "cell.hiddenWeight", "cell.bias"),
                        sequence.stateDictionary().entries().stream()
                                .map(value -> value.path()).toList()),
                () -> assertEquals("cell", assertThrows(
                        NullPointerException.class, () -> new LstmSequence(null)).getMessage()),
                () -> assertThrows(IllegalStateException.class, () -> new LstmSequence(cell)));

        sequence.eval();
        assertEquals(ForwardMode.EVALUATION, cell.mode());
        LstmSequenceForwardResult evaluation = sequence.forward(
                tensor(DataType.FLOAT32, Shape.of(1, 1, 3), false),
                tensor(DataType.FLOAT32, Shape.of(1, 4), false),
                tensor(DataType.FLOAT32, Shape.of(1, 4), false),
                new long[] {1});
        sequence.train();
        LstmSequenceForwardResult training = sequence.forward(
                tensor(DataType.FLOAT32, Shape.of(1, 1, 3), false),
                tensor(DataType.FLOAT32, Shape.of(1, 4), false),
                tensor(DataType.FLOAT32, Shape.of(1, 4), false),
                new long[] {1});
        assertAll(
                () -> assertEquals(ForwardMode.TRAINING, sequence.mode()),
                () -> assertEquals(ForwardMode.TRAINING, cell.mode()),
                () -> assertEquals(
                        evaluation.finalHidden().descriptor(), training.finalHidden().descriptor()),
                () -> assertEquals(
                        evaluation.finalCell().descriptor(), training.finalCell().descriptor()),
                () -> assertNotSame(evaluation.finalHidden(), training.finalHidden()),
                () -> assertNotSame(evaluation.finalCell(), training.finalCell()));
    }

    @Test
    void allZeroZeroTimeAndEmptyBatchReturnBothExactStatesWithoutCreatingIds() throws Exception {
        LstmSequence sequence = new LstmSequence(cell(DataType.FLOAT32, false));
        AtomicLong ids = nextTensorIdState();

        Tensor input = tensor(DataType.FLOAT32, Shape.of(3, 2, 3), false);
        Tensor hidden = tensor(DataType.FLOAT32, Shape.of(2, 4), false);
        Tensor cell = tensor(DataType.FLOAT32, Shape.of(2, 4), false);
        long beforeAllZero = ids.get();
        LstmSequenceForwardResult allZero = sequence.forward(
                input, hidden, cell, new long[] {0, 0});
        assertAll(
                () -> assertTrue(allZero.packedOutputs().isEmpty()),
                () -> assertSame(hidden, allZero.finalHidden()),
                () -> assertSame(cell, allZero.finalCell()),
                () -> assertEquals(beforeAllZero, ids.get()));

        Tensor zeroTimeInput = tensor(DataType.FLOAT32, Shape.of(0, 2, 3), false);
        Tensor zeroTimeHidden = tensor(DataType.FLOAT32, Shape.of(2, 4), false);
        Tensor zeroTimeCell = tensor(DataType.FLOAT32, Shape.of(2, 4), false);
        long beforeZeroTime = ids.get();
        LstmSequenceForwardResult zeroTime = sequence.forward(
                zeroTimeInput, zeroTimeHidden, zeroTimeCell, new long[] {0, 0});
        assertAll(
                () -> assertTrue(zeroTime.packedOutputs().isEmpty()),
                () -> assertSame(zeroTimeHidden, zeroTime.finalHidden()),
                () -> assertSame(zeroTimeCell, zeroTime.finalCell()),
                () -> assertEquals(beforeZeroTime, ids.get()));

        Tensor emptyInput = tensor(DataType.FLOAT32, Shape.of(4, 0, 3), false);
        Tensor emptyHidden = tensor(DataType.FLOAT32, Shape.of(0, 4), false);
        Tensor emptyCell = tensor(DataType.FLOAT32, Shape.of(0, 4), false);
        long beforeEmpty = ids.get();
        LstmSequenceForwardResult empty = sequence.forward(
                emptyInput, emptyHidden, emptyCell, new long[0]);
        assertAll(
                () -> assertTrue(empty.packedOutputs().isEmpty()),
                () -> assertSame(emptyHidden, empty.finalHidden()),
                () -> assertSame(emptyCell, empty.finalCell()),
                () -> assertEquals(beforeEmpty, ids.get()));
    }

    @Test
    void rejectsCallerControlledDescriptorAndLengthFailuresBeforeAnyExpressionId() throws Exception {
        LstmSequence sequence = new LstmSequence(cell(DataType.FLOAT32, true));
        Tensor input = tensor(DataType.FLOAT32, Shape.of(3, 2, 3), false);
        Tensor hidden = tensor(DataType.FLOAT32, Shape.of(2, 4), false);
        Tensor cell = tensor(DataType.FLOAT32, Shape.of(2, 4), false);
        Tensor integralInput = tensor(DataType.INT32, Shape.of(3, 2, 3), false);
        Tensor integralHidden = tensor(DataType.INT64, Shape.of(2, 4), false);
        Tensor integralCell = tensor(DataType.INT32, Shape.of(2, 4), false);
        Tensor wrongInputRank = tensor(DataType.FLOAT32, Shape.of(2, 3), false);
        Tensor wrongHiddenRank = tensor(DataType.FLOAT32, Shape.of(2, 1, 4), false);
        Tensor wrongCellRank = tensor(DataType.FLOAT32, Shape.of(2, 1, 4), false);
        Tensor dynamicInput = tensor(DataType.FLOAT32, Shape.ofDimensions(
                new DynamicDimension("T"), new StaticDimension(2), new StaticDimension(3)), false);
        Tensor dynamicInputBatch = tensor(DataType.FLOAT32, Shape.ofDimensions(
                new StaticDimension(3), new DynamicDimension("B"), new StaticDimension(3)), false);
        Tensor dynamicInputFeature = tensor(DataType.FLOAT32, Shape.ofDimensions(
                new StaticDimension(3), new StaticDimension(2), new DynamicDimension("I")), false);
        Tensor dynamicHidden = tensor(DataType.FLOAT32, Shape.ofDimensions(
                new DynamicDimension("B"), new StaticDimension(4)), false);
        Tensor dynamicHiddenFeature = tensor(DataType.FLOAT32, Shape.ofDimensions(
                new StaticDimension(2), new DynamicDimension("H")), false);
        Tensor dynamicCellBatch = tensor(DataType.FLOAT32, Shape.ofDimensions(
                new DynamicDimension("B"), new StaticDimension(4)), false);
        Tensor dynamicCell = tensor(DataType.FLOAT32, Shape.ofDimensions(
                new StaticDimension(2), new DynamicDimension("H")), false);
        Tensor wrongInputFeature = tensor(DataType.FLOAT32, Shape.of(3, 2, 5), false);
        Tensor wrongHiddenFeature = tensor(DataType.FLOAT32, Shape.of(2, 5), false);
        Tensor wrongCellFeature = tensor(DataType.FLOAT32, Shape.of(2, 5), false);
        Tensor wrongInputBatch = tensor(DataType.FLOAT32, Shape.of(3, 1, 3), false);
        Tensor wrongHiddenBatch = tensor(DataType.FLOAT32, Shape.of(1, 4), false);
        Tensor wrongCellBatch = tensor(DataType.FLOAT32, Shape.of(1, 4), false);
        AtomicLong ids = nextTensorIdState();
        long before = ids.get();

        assertAll(
                () -> assertEquals("input", assertThrows(NullPointerException.class,
                        () -> sequence.forward(null, null, null, null)).getMessage()),
                () -> assertEquals("initialHidden", assertThrows(NullPointerException.class,
                        () -> sequence.forward(input, null, null, null)).getMessage()),
                () -> assertEquals("initialCell", assertThrows(NullPointerException.class,
                        () -> sequence.forward(input, hidden, null, null)).getMessage()),
                () -> assertEquals("lengths", assertThrows(NullPointerException.class,
                        () -> sequence.forward(input, hidden, cell, null)).getMessage()),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(integralInput, hidden, cell, new long[] {1, 1})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(input, integralHidden, cell, new long[] {1, 1})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(input, hidden, integralCell, new long[] {1, 1})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(wrongInputRank, hidden, cell, new long[] {1, 1})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(input, wrongHiddenRank, cell, new long[] {1, 1})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(input, hidden, wrongCellRank, new long[] {1, 1})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(dynamicInput, hidden, cell, new long[] {1, 1})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(dynamicInputBatch, hidden, cell, new long[] {1, 1})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(dynamicInputFeature, hidden, cell, new long[] {1, 1})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(input, dynamicHidden, cell, new long[] {1, 1})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(input, dynamicHiddenFeature, cell, new long[] {1, 1})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(input, hidden, dynamicCellBatch, new long[] {1, 1})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(input, hidden, dynamicCell, new long[] {1, 1})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(wrongInputFeature, hidden, cell, new long[] {1, 1})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(input, wrongHiddenFeature, cell, new long[] {1, 1})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(input, hidden, wrongCellFeature, new long[] {1, 1})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(wrongInputBatch, hidden, cell, new long[] {1})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(input, wrongHiddenBatch, cell, new long[] {1, 1})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(input, hidden, wrongCellBatch, new long[] {1, 1})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(input, hidden, cell, new long[] {1})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(input, hidden, cell, new long[] {-1, 1})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(input, hidden, cell, new long[] {4, 1})),
                () -> assertEquals(before, ids.get()));
    }

    @Test
    void prevalidatesMaximumLengthAndBothIndependentFinalStackTypes() throws Exception {
        LstmSequence floatSequence = new LstmSequence(cell(DataType.FLOAT32, false));
        Tensor hugeInput = tensor(
                DataType.FLOAT32, Shape.of((long) Integer.MAX_VALUE + 1, 1, 3), false);
        Tensor oneHidden = tensor(DataType.FLOAT32, Shape.of(1, 4), false);
        Tensor oneCell = tensor(DataType.FLOAT32, Shape.of(1, 4), false);
        Tensor doubleInput = tensor(DataType.FLOAT64, Shape.of(1, 2, 3), false);
        Tensor floatHidden = tensor(DataType.FLOAT32, Shape.of(2, 4), false);
        Tensor doubleHidden = tensor(DataType.FLOAT64, Shape.of(2, 4), false);
        Tensor floatCell = tensor(DataType.FLOAT32, Shape.of(2, 4), false);
        Tensor doubleCell = tensor(DataType.FLOAT64, Shape.of(2, 4), false);
        AtomicLong ids = nextTensorIdState();
        long before = ids.get();

        assertAll(
                () -> assertThrows(IllegalArgumentException.class,
                        () -> floatSequence.forward(
                                hugeInput,
                                oneHidden,
                                oneCell,
                                new long[] {(long) Integer.MAX_VALUE + 1})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> floatSequence.forward(
                                doubleInput, floatHidden, doubleCell, new long[] {1, 0})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> floatSequence.forward(
                                doubleInput, doubleHidden, floatCell, new long[] {1, 0})),
                () -> assertEquals(before, ids.get()));
    }

    @Test
    void compatibleReplacementAffectsLaterConstructionOnly() {
        LstmCell cell = cell(DataType.FLOAT32, false);
        LstmSequence sequence = new LstmSequence(cell);
        Tensor input = tensor(DataType.FLOAT32, Shape.of(1, 1, 3), false);
        Tensor hidden = tensor(DataType.FLOAT32, Shape.of(1, 4), false);
        Tensor currentCell = tensor(DataType.FLOAT32, Shape.of(1, 4), false);
        Tensor oldWeight = cell.inputWeight().value();
        LstmSequenceForwardResult before = sequence.forward(
                input, hidden, currentCell, new long[] {1});
        Tensor newWeight = tensor(DataType.FLOAT32, Shape.of(16, 3), true);
        cell.inputWeight().replace(newWeight);
        LstmSequenceForwardResult after = sequence.forward(
                input, hidden, currentCell, new long[] {1});

        assertAll(
                () -> assertSame(oldWeight, inputWeightOf(before.packedOutputs().getFirst())),
                () -> assertSame(newWeight, inputWeightOf(after.packedOutputs().getFirst())),
                () -> assertSame(newWeight, cell.inputWeight().value()));
    }

    private static Tensor inputWeightOf(Tensor nextHidden) {
        Tensor outputGate = nextHidden.provenance().orElseThrow().inputs().getFirst();
        Tensor outputPreactivation = outputGate.provenance().orElseThrow().inputs().getFirst();
        Tensor inputOutputSlice = outputPreactivation.provenance().orElseThrow().inputs().getFirst();
        Tensor inputProjection = inputOutputSlice.provenance().orElseThrow().inputs().getFirst();
        Tensor transpose = inputProjection.provenance().orElseThrow().inputs().get(1);
        return transpose.provenance().orElseThrow().inputs().getFirst();
    }

    private static LstmCell cell(DataType dataType, boolean bias) {
        Tensor inputWeight = tensor(dataType, Shape.of(16, 3), true);
        Tensor hiddenWeight = tensor(dataType, Shape.of(16, 4), true);
        return bias
                ? new LstmCell(inputWeight, hiddenWeight, tensor(dataType, Shape.of(16), true))
                : new LstmCell(inputWeight, hiddenWeight);
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
