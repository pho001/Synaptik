package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.random.DropoutAttrs;
import io.github.pho001.synaptik.model.operation.random.DropoutKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

public final class TensorDropoutExpressionTest {
    @Test
    void helperTensorMethodAndResultExposeOnlyTheExactRequiredSurface()
            throws ReflectiveOperationException {
        var helperFields = Arrays.stream(TensorDropoutExpressions.class.getDeclaredFields())
                .filter(field -> !field.isSynthetic())
                .toList();
        var helperConstructors = TensorDropoutExpressions.class.getDeclaredConstructors();
        var helperMethods = Arrays.stream(TensorDropoutExpressions.class.getDeclaredMethods())
                .filter(method -> !method.isSynthetic())
                .toList();
        var apply = TensorDropoutExpressions.class.getDeclaredMethod(
                "apply", Tensor.class, double.class, GraphRngState.class);
        var dropout = Tensor.class.getDeclaredMethod(
                "dropout", double.class, GraphRngState.class);
        var components = DropoutResult.class.getRecordComponents();

        assertAll(
                () -> assertTrue(Modifier.isFinal(TensorDropoutExpressions.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(TensorDropoutExpressions.class.getModifiers())),
                () -> assertFalse(TensorDropoutExpressions.class.isRecord()),
                () -> assertEquals(Set.of(), Set.of(TensorDropoutExpressions.class.getInterfaces())),
                () -> assertTrue(helperFields.isEmpty()),
                () -> assertEquals(1, helperConstructors.length),
                () -> assertTrue(Modifier.isPrivate(helperConstructors[0].getModifiers())),
                () -> assertEquals(0, helperConstructors[0].getParameterCount()),
                () -> assertEquals(List.of(apply), helperMethods),
                () -> assertTrue(Modifier.isStatic(apply.getModifiers())),
                () -> assertFalse(Modifier.isPublic(apply.getModifiers())),
                () -> assertEquals(DropoutResult.class, apply.getReturnType()),
                () -> assertTrue(Modifier.isPublic(dropout.getModifiers())),
                () -> assertFalse(Modifier.isStatic(dropout.getModifiers())),
                () -> assertEquals(DropoutResult.class, dropout.getReturnType()),
                () -> assertTrue(DropoutResult.class.isRecord()),
                () -> assertEquals(List.of("output", "nextState"),
                        Arrays.stream(components).map(component -> component.getName()).toList()),
                () -> assertEquals(List.of(Tensor.class, GraphRngState.class),
                        Arrays.stream(components).map(component -> component.getType()).toList()),
                () -> assertEquals(0, DropoutResult.class.getDeclaredClasses().length));
    }

    @Test
    void createsExactlyThreeIndexedOutputsUnderOneExactProducer()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        Shape shape = Shape.of(2, 3);
        Tensor input = tensor(DataType.FLOAT32, shape, true);
        GraphRngState state = GraphRngState.initial(0x1234L, 7L);
        Tensor stateTensor = state.tensor();
        long before = next.get();

        DropoutResult result = input.dropout(0.25d, state);
        Tensor output = result.output();
        Tensor nextStateTensor = result.nextState().tensor();
        TensorProvenance outputProvenance = output.provenance().orElseThrow();
        TensorProvenance stateProvenance = nextStateTensor.provenance().orElseThrow();
        TensorProducer producer = outputProvenance.producer();
        List<TensorDescriptor> descriptors = producer.outputDescriptors();

        assertAll(
                () -> assertEquals(before, output.id().value()),
                () -> assertEquals(before + 2, nextStateTensor.id().value()),
                () -> assertEquals(before + 3, next.get()),
                () -> assertEquals(3, producer.outputCount()),
                () -> assertSame(producer, stateProvenance.producer()),
                () -> assertSame(DropoutKind.DROPOUT, producer.operation().kind()),
                () -> assertEquals(new DropoutAttrs(0.25d), producer.operation().attrs()),
                () -> assertEquals(2, producer.inputs().size()),
                () -> assertSame(input, producer.inputs().get(0)),
                () -> assertSame(stateTensor, producer.inputs().get(1)),
                () -> assertEquals(0, outputProvenance.outputIndex()),
                () -> assertEquals(2, stateProvenance.outputIndex()),
                () -> assertSame(output.descriptor(), descriptors.get(0)),
                () -> assertSame(nextStateTensor.descriptor(), descriptors.get(2)),
                () -> assertEquals(DataType.BOOL, descriptors.get(1).dataType()),
                () -> assertSame(shape, descriptors.get(1).shape()),
                () -> assertTrue(descriptors.get(1).layout().isEmpty()),
                () -> assertFalse(descriptors.get(1).requiresGrad()),
                () -> assertEquals(List.of("output", "nextState"),
                        Arrays.stream(DropoutResult.class.getRecordComponents())
                                .map(component -> component.getName()).toList()));
    }

    @Test
    void derivesExactStorageFreeDescriptorsForEveryFloatingTypeAndShapeForm() {
        List<Shape> shapes = List.of(
                Shape.scalar(),
                Shape.of(0, 4),
                Shape.of(2, 3),
                Shape.ofDimensions(new DynamicDimension("batch"), new DynamicDimension("width")));

        for (DataType dataType : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16)) {
            for (Shape shape : shapes) {
                Tensor input = tensor(dataType, shape, true);
                DropoutResult result = input.dropout(-0.0d, GraphRngState.initial(1L, 2L));
                Tensor output = result.output();
                Tensor nextState = result.nextState().tensor();

                assertAll(
                        () -> assertSame(dataType, output.descriptor().dataType()),
                        () -> assertSame(shape, output.descriptor().shape()),
                        () -> assertTrue(output.descriptor().layout().isEmpty()),
                        () -> assertTrue(output.descriptor().requiresGrad()),
                        () -> assertTrue(output.label().isEmpty()),
                        () -> assertTrue(output.hostStorage().isEmpty()),
                        () -> assertSame(DataType.INT64, nextState.descriptor().dataType()),
                        () -> assertEquals(Shape.of(2), nextState.descriptor().shape()),
                        () -> assertTrue(nextState.descriptor().layout().isEmpty()),
                        () -> assertFalse(nextState.descriptor().requiresGrad()),
                        () -> assertTrue(nextState.label().isEmpty()),
                        () -> assertTrue(nextState.hostStorage().isEmpty()));
            }
        }
    }

    @Test
    void deliberatelyDropsInputLayoutLabelAndStorageMetadata() {
        Shape shape = Shape.of(2);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT64,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                false);
        Tensor input = TensorFactory.allocate(descriptor, Optional.of("input"));

        Tensor output = input.dropout(0.5d, GraphRngState.initial(1L, 2L)).output();

        assertAll(
                () -> assertTrue(input.descriptor().layout().isPresent()),
                () -> assertEquals(Optional.of("input"), input.label()),
                () -> assertTrue(input.hostStorage().isPresent()),
                () -> assertSame(shape, output.descriptor().shape()),
                () -> assertTrue(output.descriptor().layout().isEmpty()),
                () -> assertTrue(output.label().isEmpty()),
                () -> assertTrue(output.hostStorage().isEmpty()),
                () -> assertFalse(output.descriptor().requiresGrad()));
    }

    @Test
    void distinctBranchReplayAndThreadedCallsCreateDistinctOccurrencesWithoutMutation() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(4), false);
        GraphRngState state = GraphRngState.initial(-1L, Long.MAX_VALUE);
        Tensor stateTensor = state.tensor();

        DropoutResult branchOne = input.dropout(0.0d, state);
        DropoutResult branchTwo = input.dropout(-0.0d, state);
        DropoutResult threaded = branchOne.output().dropout(0.0d, branchOne.nextState());

        assertAll(
                () -> assertNotSame(branchOne.output(), branchTwo.output()),
                () -> assertNotSame(branchOne.nextState(), branchTwo.nextState()),
                () -> assertNotEquals(branchOne.output().id(), branchTwo.output().id()),
                () -> assertSame(stateTensor,
                        branchOne.output().provenance().orElseThrow().inputs().get(1)),
                () -> assertSame(stateTensor,
                        branchTwo.output().provenance().orElseThrow().inputs().get(1)),
                () -> assertSame(branchOne.nextState().tensor(),
                        threaded.output().provenance().orElseThrow().inputs().get(1)),
                () -> assertSame(stateTensor, state.tensor()),
                () -> assertTrue(stateTensor.hostStorage().isEmpty()));
    }

    @Test
    void validatesInTheExactOrderAndConsumesNoIdsForLocalFailures()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        Tensor integral = tensor(DataType.INT32, Shape.scalar(), false);
        Tensor int64 = tensor(DataType.INT64, Shape.scalar(), false);
        Tensor bool = tensor(DataType.BOOL, Shape.scalar(), false);
        Tensor floating = tensor(DataType.FLOAT32, Shape.scalar(), false);
        GraphRngState state = GraphRngState.initial(1L, 2L);
        long before = next.get();

        NullPointerException nullInput = assertThrows(
                NullPointerException.class,
                () -> TensorDropoutExpressions.apply(null, Double.NaN, null));
        IllegalArgumentException intFailure = assertThrows(
                IllegalArgumentException.class,
                () -> integral.dropout(Double.NaN, null));
        IllegalArgumentException int64Failure = assertThrows(
                IllegalArgumentException.class,
                () -> int64.dropout(0.25d, state));
        IllegalArgumentException boolFailure = assertThrows(
                IllegalArgumentException.class,
                () -> bool.dropout(0.25d, null));
        IllegalArgumentException probabilityFailure = assertThrows(
                IllegalArgumentException.class,
                () -> floating.dropout(1.0d, null));
        NullPointerException stateFailure = assertThrows(
                NullPointerException.class,
                () -> floating.dropout(0.25d, null));

        assertAll(
                () -> assertEquals("input", nullInput.getMessage()),
                () -> assertEquals("dropout input data type must be floating: INT32",
                        intFailure.getMessage()),
                () -> assertEquals("dropout input data type must be floating: INT64",
                        int64Failure.getMessage()),
                () -> assertEquals("dropout input data type must be floating: BOOL",
                        boolFailure.getMessage()),
                () -> assertEquals("probability must be finite and in [0.0, 1.0): 1.0",
                        probabilityFailure.getMessage()),
                () -> assertEquals("state", stateFailure.getMessage()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void resultRejectsNullComponentsInOrderAndRetainsExactReferences() {
        Tensor output = tensor(DataType.FLOAT32, Shape.scalar(), false);
        GraphRngState state = GraphRngState.initial(1L, 2L);

        NullPointerException outputFailure = assertThrows(
                NullPointerException.class, () -> new DropoutResult(null, null));
        NullPointerException stateFailure = assertThrows(
                NullPointerException.class, () -> new DropoutResult(output, null));
        DropoutResult result = new DropoutResult(output, state);

        assertAll(
                () -> assertEquals("output", outputFailure.getMessage()),
                () -> assertEquals("nextState", stateFailure.getMessage()),
                () -> assertSame(output, result.output()),
                () -> assertSame(state, result.nextState()),
                () -> assertEquals(result, new DropoutResult(output, state)));
    }

    private static Tensor tensor(DataType dataType, Shape shape, boolean requiresGrad) {
        return TensorFactory.create(new TensorDescriptor(
                dataType, shape, Optional.empty(), requiresGrad));
    }

    private static AtomicLong nextTensorIdState() throws ReflectiveOperationException {
        var field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }
}
