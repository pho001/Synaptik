package io.github.pho001.synaptik.nn.layers;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
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
class GruSequenceTest {
    @Test
    void exposesExactlyThePlannedDirectModuleAndRecordSurface() throws Exception {
        Set<List<Class<?>>> constructors = Arrays.stream(GruSequence.class.getDeclaredConstructors())
                .filter(value -> Modifier.isPublic(value.getModifiers()))
                .map(value -> List.of(value.getParameterTypes()))
                .collect(Collectors.toSet());
        Set<String> methods = Arrays.stream(GruSequence.class.getDeclaredMethods())
                .filter(value -> Modifier.isPublic(value.getModifiers()))
                .map(value -> value.getName())
                .collect(Collectors.toSet());
        RecordComponent[] components = GruSequenceForwardResult.class.getRecordComponents();

        assertAll(
                () -> assertTrue(Modifier.isFinal(GruSequence.class.getModifiers())),
                () -> assertSame(Module.class, GruSequence.class.getSuperclass()),
                () -> assertFalse(UnaryTensorModule.class.isAssignableFrom(GruSequence.class)),
                () -> assertFalse(Sequential.class.isAssignableFrom(GruSequence.class)),
                () -> assertEquals(Set.of(List.of(GruCell.class)), constructors),
                () -> assertEquals(Set.of("cell", "forward"), methods),
                () -> assertSame(GruCell.class, GruSequence.class.getDeclaredMethod("cell").getReturnType()),
                () -> assertSame(
                        GruSequenceForwardResult.class,
                        GruSequence.class
                                .getDeclaredMethod("forward", Tensor.class, Tensor.class, long[].class)
                                .getReturnType()),
                () -> assertEquals(1, GruSequence.class.getDeclaredFields().length),
                () -> assertEquals("cell", GruSequence.class.getDeclaredFields()[0].getName()),
                () -> assertTrue(GruSequenceForwardResult.class.isRecord()),
                () -> assertEquals(List.of("packedOutputs", "finalHidden"),
                        Arrays.stream(components).map(RecordComponent::getName).toList()),
                () -> assertEquals(List.of(List.class, Tensor.class),
                        Arrays.stream(components).map(RecordComponent::getType).toList()),
                () -> assertEquals(0, GruSequence.class.getDeclaredClasses().length),
                () -> assertEquals(0, GruSequenceForwardResult.class.getDeclaredClasses().length));
    }

    @Test
    void resultSnapshotsListRetainsExactTensorsAndChecksNullsInOrder() {
        Tensor first = tensor(DataType.FLOAT32, Shape.of(2, 4), false);
        Tensor second = tensor(DataType.FLOAT32, Shape.of(1, 4), false);
        Tensor hidden = tensor(DataType.FLOAT32, Shape.of(3, 4), false);
        ArrayList<Tensor> source = new ArrayList<>(List.of(first, second));
        GruSequenceForwardResult result = new GruSequenceForwardResult(source, hidden);
        source.clear();

        assertAll(
                () -> assertEquals(2, result.packedOutputs().size()),
                () -> assertSame(first, result.packedOutputs().get(0)),
                () -> assertSame(second, result.packedOutputs().get(1)),
                () -> assertSame(hidden, result.finalHidden()),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> result.packedOutputs().add(first)),
                () -> assertEquals("packedOutputs", assertThrows(
                        NullPointerException.class,
                        () -> new GruSequenceForwardResult(null, null)).getMessage()),
                () -> assertEquals("packedOutputs[1]", assertThrows(
                        NullPointerException.class,
                        () -> new GruSequenceForwardResult(Arrays.asList(first, null), null))
                                .getMessage()),
                () -> assertEquals("finalHidden", assertThrows(
                        NullPointerException.class,
                        () -> new GruSequenceForwardResult(List.of(first), null)).getMessage()));
    }

    @Test
    void ownsOneCellAndPreservesRecursiveStateAndModeBehavior() {
        GruCell cell = cell(DataType.FLOAT32, true);
        GruSequence sequence = new GruSequence(cell);

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
                        sequence.stateDictionary().entries().stream().map(value -> value.path()).toList()),
                () -> assertEquals("cell", assertThrows(
                        NullPointerException.class, () -> new GruSequence(null)).getMessage()),
                () -> assertThrows(IllegalStateException.class, () -> new GruSequence(cell)));

        sequence.eval();
        assertEquals(ForwardMode.EVALUATION, cell.mode());
        GruSequenceForwardResult evaluation = sequence.forward(
                tensor(DataType.FLOAT32, Shape.of(1, 1, 3), false),
                tensor(DataType.FLOAT32, Shape.of(1, 4), false),
                new long[] {1});
        sequence.train();
        GruSequenceForwardResult training = sequence.forward(
                tensor(DataType.FLOAT32, Shape.of(1, 1, 3), false),
                tensor(DataType.FLOAT32, Shape.of(1, 4), false),
                new long[] {1});
        assertAll(
                () -> assertEquals(ForwardMode.TRAINING, sequence.mode()),
                () -> assertEquals(ForwardMode.TRAINING, cell.mode()),
                () -> assertEquals(evaluation.finalHidden().descriptor(), training.finalHidden().descriptor()),
                () -> assertNotSame(evaluation.finalHidden(), training.finalHidden()));
    }

    @Test
    void allZeroZeroTimeAndEmptyBatchReturnExactHiddenWithoutCreatingTensorIds() throws Exception {
        GruSequence sequence = new GruSequence(cell(DataType.FLOAT32, false));
        AtomicLong ids = nextTensorIdState();

        Tensor input = tensor(DataType.FLOAT32, Shape.of(3, 2, 3), false);
        Tensor hidden = tensor(DataType.FLOAT32, Shape.of(2, 4), false);
        long beforeAllZero = ids.get();
        GruSequenceForwardResult allZero = sequence.forward(input, hidden, new long[] {0, 0});

        Tensor zeroTimeInput = tensor(DataType.FLOAT32, Shape.of(0, 2, 3), false);
        Tensor zeroTimeHidden = tensor(DataType.FLOAT32, Shape.of(2, 4), false);
        long beforeZeroTime = ids.get();
        GruSequenceForwardResult zeroTime = sequence.forward(
                zeroTimeInput, zeroTimeHidden, new long[] {0, 0});

        Tensor emptyInput = tensor(DataType.FLOAT32, Shape.of(4, 0, 3), false);
        Tensor emptyHidden = tensor(DataType.FLOAT32, Shape.of(0, 4), false);
        long beforeEmpty = ids.get();
        GruSequenceForwardResult empty = sequence.forward(emptyInput, emptyHidden, new long[0]);

        assertAll(
                () -> assertTrue(allZero.packedOutputs().isEmpty()),
                () -> assertSame(hidden, allZero.finalHidden()),
                () -> assertEquals(beforeAllZero, zeroTimeInput.id().value()),
                () -> assertTrue(zeroTime.packedOutputs().isEmpty()),
                () -> assertSame(zeroTimeHidden, zeroTime.finalHidden()),
                () -> assertEquals(beforeZeroTime, emptyInput.id().value()),
                () -> assertTrue(empty.packedOutputs().isEmpty()),
                () -> assertSame(emptyHidden, empty.finalHidden()),
                () -> assertEquals(beforeEmpty, ids.get()));
    }

    @Test
    void rejectsCallerControlledFailuresBeforeAnyExpressionIdentity() throws Exception {
        GruSequence sequence = new GruSequence(cell(DataType.FLOAT32, true));
        Tensor validInput = tensor(DataType.FLOAT32, Shape.of(3, 2, 3), false);
        Tensor validHidden = tensor(DataType.FLOAT32, Shape.of(2, 4), false);
        Tensor integralInput = tensor(DataType.INT32, Shape.of(3, 2, 3), false);
        Tensor integralHidden = tensor(DataType.INT64, Shape.of(2, 4), false);
        Tensor wrongInputRank = tensor(DataType.FLOAT32, Shape.of(2, 3), false);
        Tensor wrongHiddenRank = tensor(DataType.FLOAT32, Shape.of(2, 1, 4), false);
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
        Tensor wrongInputFeature = tensor(DataType.FLOAT32, Shape.of(3, 2, 5), false);
        Tensor wrongHiddenFeature = tensor(DataType.FLOAT32, Shape.of(2, 5), false);
        Tensor wrongBatch = tensor(DataType.FLOAT32, Shape.of(3, 1, 3), false);
        AtomicLong ids = nextTensorIdState();
        long before = ids.get();

        assertAll(
                () -> assertEquals("input", assertThrows(NullPointerException.class,
                        () -> sequence.forward(null, null, null)).getMessage()),
                () -> assertEquals("initialHidden", assertThrows(NullPointerException.class,
                        () -> sequence.forward(validInput, null, null)).getMessage()),
                () -> assertEquals("lengths", assertThrows(NullPointerException.class,
                        () -> sequence.forward(validInput, validHidden, null)).getMessage()),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(integralInput, validHidden, new long[] {1, 1})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(validInput, integralHidden, new long[] {1, 1})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(wrongInputRank, validHidden, new long[] {1, 1})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(validInput, wrongHiddenRank, new long[] {1, 1})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(dynamicInput, validHidden, new long[] {1, 1})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(dynamicInputBatch, validHidden, new long[] {1, 1})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(dynamicInputFeature, validHidden, new long[] {1, 1})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(validInput, dynamicHidden, new long[] {1, 1})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(validInput, dynamicHiddenFeature, new long[] {1, 1})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(wrongInputFeature, validHidden, new long[] {1, 1})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(validInput, wrongHiddenFeature, new long[] {1, 1})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(wrongBatch, validHidden, new long[] {1})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(validInput, validHidden, new long[] {1})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(validInput, validHidden, new long[] {-1, 1})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(validInput, validHidden, new long[] {4, 1})),
                () -> assertEquals(before, ids.get()));
    }

    @Test
    void prevalidatesMaximumLengthMixedFinalTypeAndCurrentCellSchema() throws Exception {
        GruSequence sequence = new GruSequence(cell(DataType.FLOAT32, false));
        Tensor hugeTime = tensor(DataType.FLOAT32,
                Shape.of((long) Integer.MAX_VALUE + 1, 1, 3), false);
        Tensor hidden = tensor(DataType.FLOAT32, Shape.of(1, 4), false);
        Tensor mixedInput = tensor(DataType.FLOAT64, Shape.of(1, 2, 3), false);
        Tensor mixedHidden = tensor(DataType.FLOAT32, Shape.of(2, 4), false);

        GruSequence corruptedSequence = new GruSequence(cell(DataType.FLOAT32, false));
        Field value = corruptedSequence.cell().inputWeight().getClass().getDeclaredField("value");
        value.setAccessible(true);
        value.set(corruptedSequence.cell().inputWeight(),
                tensor(DataType.FLOAT32, Shape.of(13, 3), true));
        Tensor corruptedInput = tensor(DataType.FLOAT32, Shape.of(1, 1, 3), false);
        Tensor corruptedHidden = tensor(DataType.FLOAT32, Shape.of(1, 4), false);

        AtomicLong ids = nextTensorIdState();
        long before = ids.get();
        assertAll(
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(
                                hugeTime, hidden, new long[] {(long) Integer.MAX_VALUE + 1})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sequence.forward(mixedInput, mixedHidden, new long[] {1, 0})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> corruptedSequence.forward(
                                corruptedInput,
                                corruptedHidden,
                                new long[] {1})),
                () -> assertEquals(before, ids.get()));
    }

    @Test
    void compatibleReplacementAffectsLaterConstructionOnly() {
        GruCell cell = cell(DataType.FLOAT32, false);
        GruSequence sequence = new GruSequence(cell);
        Tensor input = tensor(DataType.FLOAT32, Shape.of(1, 1, 3), false);
        Tensor hidden = tensor(DataType.FLOAT32, Shape.of(1, 4), false);
        Tensor oldWeight = cell.inputWeight().value();
        GruSequenceForwardResult before = sequence.forward(input, hidden, new long[] {1});
        Tensor newWeight = tensor(DataType.FLOAT32, Shape.of(12, 3), true);
        cell.inputWeight().replace(newWeight);
        GruSequenceForwardResult after = sequence.forward(input, hidden, new long[] {1});

        assertAll(
                () -> assertSame(oldWeight, inputWeightOf(before.packedOutputs().getFirst())),
                () -> assertSame(newWeight, inputWeightOf(after.packedOutputs().getFirst())),
                () -> assertSame(newWeight, cell.inputWeight().value()));
    }

    private static Tensor inputWeightOf(Tensor output) {
        Tensor candidate = output.provenance().orElseThrow().inputs().getFirst();
        Tensor candidateAdd = candidate.provenance().orElseThrow().inputs().getFirst();
        Tensor inputCandidate = candidateAdd.provenance().orElseThrow().inputs().getFirst();
        Tensor inputProjection = inputCandidate.provenance().orElseThrow().inputs().getFirst();
        Tensor inputProduct = inputProjection;
        if (inputProjection.provenance().orElseThrow().inputs().size() == 2
                && inputProjection.provenance().orElseThrow().inputs().get(1).provenance().isEmpty()) {
            inputProduct = inputProjection.provenance().orElseThrow().inputs().getFirst();
        }
        Tensor transpose = inputProduct.provenance().orElseThrow().inputs().get(1);
        return transpose.provenance().orElseThrow().inputs().getFirst();
    }

    private static GruCell cell(DataType dataType, boolean bias) {
        Tensor inputWeight = tensor(dataType, Shape.of(12, 3), true);
        Tensor hiddenWeight = tensor(dataType, Shape.of(12, 4), true);
        return bias
                ? new GruCell(inputWeight, hiddenWeight, tensor(dataType, Shape.of(12), true))
                : new GruCell(inputWeight, hiddenWeight);
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
