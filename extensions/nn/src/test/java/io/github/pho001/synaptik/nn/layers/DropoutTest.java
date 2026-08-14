package io.github.pho001.synaptik.nn.layers;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.random.DropoutAttrs;
import io.github.pho001.synaptik.model.operation.random.DropoutKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.GraphRngState;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProducer;
import io.github.pho001.synaptik.nn.module.ForwardContext;
import io.github.pho001.synaptik.nn.module.ForwardMode;
import io.github.pho001.synaptik.nn.module.Module;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

public final class DropoutTest {
    @Test
    void exposesOnlyTheExactFinalLayerAndRecordSurfaces() throws ReflectiveOperationException {
        var constructors = Dropout.class.getDeclaredConstructors();
        var visibleMethods = Arrays.stream(Dropout.class.getDeclaredMethods())
                .filter(method -> !method.isSynthetic())
                .filter(method -> Modifier.isPublic(method.getModifiers())
                        || Modifier.isProtected(method.getModifiers()))
                .toList();
        var fields = Arrays.stream(Dropout.class.getDeclaredFields())
                .filter(field -> !field.isSynthetic())
                .toList();
        var forward = Dropout.class.getDeclaredMethod(
                "forward", Tensor.class, GraphRngState.class, ForwardContext.class);
        var resultConstructors = DropoutForwardResult.class.getDeclaredConstructors();
        var components = DropoutForwardResult.class.getRecordComponents();
        var resultFields = Arrays.stream(DropoutForwardResult.class.getDeclaredFields())
                .filter(field -> !field.isSynthetic())
                .toList();
        Set<String> resultMethodNames = Arrays.stream(DropoutForwardResult.class.getDeclaredMethods())
                .filter(method -> !method.isSynthetic())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .map(method -> method.getName())
                .collect(java.util.stream.Collectors.toSet());

        assertAll(
                () -> assertTrue(Modifier.isPublic(Dropout.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(Dropout.class.getModifiers())),
                () -> assertSame(Module.class, Dropout.class.getSuperclass()),
                () -> assertEquals(0, Dropout.class.getDeclaredClasses().length),
                () -> assertEquals(Set.of(), Set.of(Dropout.class.getInterfaces())),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPublic(constructors[0].getModifiers())),
                () -> assertEquals(List.of(double.class),
                        List.of(constructors[0].getParameterTypes())),
                () -> assertEquals(List.of(forward), visibleMethods),
                () -> assertEquals(DropoutForwardResult.class, forward.getReturnType()),
                () -> assertEquals(1, fields.size()),
                () -> assertEquals(double.class, fields.getFirst().getType()),
                () -> assertTrue(Modifier.isPrivate(fields.getFirst().getModifiers())),
                () -> assertTrue(Modifier.isFinal(fields.getFirst().getModifiers())),
                () -> assertTrue(DropoutForwardResult.class.isRecord()),
                () -> assertTrue(Modifier.isPublic(DropoutForwardResult.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(DropoutForwardResult.class.getModifiers())),
                () -> assertEquals(0, DropoutForwardResult.class.getDeclaredClasses().length),
                () -> assertEquals(1, resultConstructors.length),
                () -> assertTrue(Modifier.isPublic(resultConstructors[0].getModifiers())),
                () -> assertEquals(List.of(Tensor.class, GraphRngState.class),
                        List.of(resultConstructors[0].getParameterTypes())),
                () -> assertEquals(List.of("output", "nextState"),
                        Arrays.stream(components).map(component -> component.getName()).toList()),
                () -> assertEquals(List.of(Tensor.class, GraphRngState.class),
                        Arrays.stream(components).map(component -> component.getType()).toList()),
                () -> assertEquals(List.of("output", "nextState"),
                        resultFields.stream().map(Field::getName).toList()),
                () -> assertEquals(List.of(Tensor.class, GraphRngState.class),
                        resultFields.stream().map(Field::getType).toList()),
                () -> assertTrue(resultFields.stream().allMatch(field ->
                        Modifier.isPrivate(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers()))),
                () -> assertEquals(
                        Set.of("output", "nextState", "equals", "hashCode", "toString"),
                        resultMethodNames));
    }

    @Test
    void constructionValidatesTheCompleteDomainPreservesSignedZeroAndCreatesNoState()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();
        Field probabilityField = Dropout.class.getDeclaredField("probability");
        probabilityField.setAccessible(true);

        for (double probability : new double[] {
                0.0d, -0.0d, Double.MIN_VALUE, 0.25d, Math.nextDown(1.0d)
        }) {
            Dropout layer = new Dropout(probability);
            assertEquals(
                    Double.doubleToRawLongBits(probability),
                    Double.doubleToRawLongBits(probabilityField.getDouble(layer)));
            assertAll(
                    () -> assertTrue(layer.parameters().isEmpty()),
                    () -> assertTrue(layer.buffers().isEmpty()),
                    () -> assertTrue(layer.children().isEmpty()),
                    () -> assertTrue(layer.parametersRecursively().isEmpty()),
                    () -> assertTrue(layer.buffersRecursively().isEmpty()),
                    () -> assertEquals(ForwardMode.TRAINING, layer.mode()));
        }

        for (double probability : new double[] {
                Double.NaN,
                Double.NEGATIVE_INFINITY,
                Double.POSITIVE_INFINITY,
                -Double.MIN_VALUE,
                -1.0d,
                1.0d,
                Math.nextUp(1.0d),
                Double.MAX_VALUE
        }) {
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class, () -> new Dropout(probability));
            assertEquals(
                    "probability must be finite and in [0.0, 1.0): " + probability,
                    failure.getMessage());
        }
        assertEquals(before, next.get());
    }

    @Test
    void resultRejectsNullComponentsInOrderAndRetainsExactReferences() {
        Tensor output = tensor(DataType.FLOAT32, Shape.of(2), false);
        GraphRngState state = GraphRngState.initial(1L, 2L);

        NullPointerException outputFailure = assertThrows(
                NullPointerException.class, () -> new DropoutForwardResult(null, null));
        NullPointerException stateFailure = assertThrows(
                NullPointerException.class, () -> new DropoutForwardResult(output, null));
        DropoutForwardResult first = new DropoutForwardResult(output, state);
        DropoutForwardResult equal = new DropoutForwardResult(output, state);

        assertAll(
                () -> assertEquals("output", outputFailure.getMessage()),
                () -> assertEquals("nextState", stateFailure.getMessage()),
                () -> assertSame(output, first.output()),
                () -> assertSame(state, first.nextState()),
                () -> assertNotSame(first, equal),
                () -> assertEquals(first, equal),
                () -> assertEquals(first.hashCode(), equal.hashCode()));
    }

    @Test
    void evaluationBypassesModelAndPreservesExactReferencesForEveryDataType()
            throws ReflectiveOperationException {
        Dropout layer = new Dropout(0.25d);
        layer.eval();
        ForwardContext evaluation = layer.forwardContext();
        GraphRngState state = GraphRngState.initial(7L, 11L);

        for (DataType dataType : DataType.values()) {
            Tensor input = tensor(dataType, Shape.of(2, 3), dataType.isDifferentiable());
            AtomicLong next = nextTensorIdState();
            long before = next.get();

            DropoutForwardResult first = layer.forward(input, state, evaluation);
            DropoutForwardResult second = layer.forward(input, state, evaluation);

            assertAll(
                    () -> assertNotSame(first, second),
                    () -> assertSame(input, first.output()),
                    () -> assertSame(input, second.output()),
                    () -> assertSame(state, first.nextState()),
                    () -> assertSame(state, second.nextState()),
                    () -> assertEquals(before, next.get()),
                    () -> assertEquals(ForwardMode.EVALUATION, layer.mode()),
                    () -> assertEquals(ForwardMode.EVALUATION, evaluation.mode()));
        }
    }

    @Test
    void trainingCreatesOneExactModelProducerAndWrapsItsExactPublicOutputs()
            throws ReflectiveOperationException {
        double probability = -0.0d;
        Dropout layer = new Dropout(probability);
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3), true);
        GraphRngState state = GraphRngState.initial(0x1234L, 7L);
        Tensor stateTensor = stateTensor(state);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        DropoutForwardResult result = layer.forward(input, state, layer.forwardContext());
        Tensor output = result.output();
        Tensor nextStateTensor = stateTensor(result.nextState());
        TensorProducer producer = output.provenance().orElseThrow().producer();
        Tensor mask = producer.output(1);
        DropoutAttrs attrs = (DropoutAttrs) producer.operation().attrs();

        assertAll(
                () -> assertEquals(before, output.id().value()),
                () -> assertEquals(before + 2, nextStateTensor.id().value()),
                () -> assertEquals(before + 3, next.get()),
                () -> assertSame(DropoutKind.DROPOUT, producer.operation().kind()),
                () -> assertEquals(
                        Double.doubleToRawLongBits(probability),
                        Double.doubleToRawLongBits(attrs.probability())),
                () -> assertEquals(List.of(input, stateTensor), producer.inputs()),
                () -> assertEquals(3, producer.outputCount()),
                () -> assertSame(output, producer.output(0)),
                () -> assertSame(mask, producer.output(1)),
                () -> assertSame(nextStateTensor, producer.output(2)),
                () -> assertEquals(0, output.provenance().orElseThrow().outputIndex()),
                () -> assertEquals(1, mask.provenance().orElseThrow().outputIndex()),
                () -> assertEquals(2, nextStateTensor.provenance().orElseThrow().outputIndex()),
                () -> assertSame(producer, mask.provenance().orElseThrow().producer()),
                () -> assertSame(producer, nextStateTensor.provenance().orElseThrow().producer()),
                () -> assertSame(input.descriptor().shape(), output.descriptor().shape()),
                () -> assertSame(DataType.FLOAT32, output.descriptor().dataType()),
                () -> assertTrue(output.descriptor().requiresGrad()),
                () -> assertTrue(output.descriptor().layout().isEmpty()),
                () -> assertTrue(output.label().isEmpty()),
                () -> assertTrue(output.hostStorage().isEmpty()),
                () -> assertSame(DataType.BOOL, mask.descriptor().dataType()),
                () -> assertSame(input.descriptor().shape(), mask.descriptor().shape()),
                () -> assertFalse(mask.descriptor().requiresGrad()),
                () -> assertSame(DataType.INT64, nextStateTensor.descriptor().dataType()),
                () -> assertEquals(Shape.of(2), nextStateTensor.descriptor().shape()),
                () -> assertFalse(nextStateTensor.descriptor().requiresGrad()),
                () -> assertNotSame(input, result.output()),
                () -> assertNotSame(state, result.nextState()),
                () -> assertSame(result.output(), producer.output(0)),
                () -> assertSame(stateTensor(result.nextState()), producer.output(2)));
    }

    @Test
    void suppliedContextAloneSelectsTheBranchAcrossLaterAndForeignModeChanges()
            throws ReflectiveOperationException {
        Dropout layer = new Dropout(0.5d);
        Tensor input = tensor(DataType.FLOAT32, Shape.of(4), false);
        GraphRngState state = GraphRngState.initial(3L, 5L);

        layer.eval();
        ForwardContext evaluationSnapshot = layer.forwardContext();
        layer.train();
        AtomicLong next = nextTensorIdState();
        long beforeEvaluation = next.get();
        DropoutForwardResult evaluation = layer.forward(input, state, evaluationSnapshot);

        Dropout foreign = new Dropout(0.1d);
        ForwardContext foreignTraining = foreign.forwardContext();
        layer.eval();
        long beforeTraining = next.get();
        DropoutForwardResult training = layer.forward(input, state, foreignTraining);

        assertAll(
                () -> assertSame(input, evaluation.output()),
                () -> assertSame(state, evaluation.nextState()),
                () -> assertEquals(beforeEvaluation, beforeTraining),
                () -> assertEquals(beforeTraining + 3, next.get()),
                () -> assertSame(DropoutKind.DROPOUT,
                        training.output().provenance().orElseThrow().operation().kind()),
                () -> assertEquals(ForwardMode.EVALUATION, layer.mode()),
                () -> assertEquals(ForwardMode.TRAINING, foreign.mode()),
                () -> assertEquals(ForwardMode.EVALUATION, evaluationSnapshot.mode()),
                () -> assertEquals(ForwardMode.TRAINING, foreignTraining.mode()));
    }

    @Test
    void probabilityZeroStillCreatesFreshBranchesAndExplicitThreadingUsesReturnedState()
            throws ReflectiveOperationException {
        Dropout layer = new Dropout(0.0d);
        Tensor input = tensor(DataType.FLOAT64, Shape.of(4), false);
        GraphRngState initial = GraphRngState.initial(9L, 13L);
        Tensor initialTensor = stateTensor(initial);
        ForwardContext training = new ForwardContext(ForwardMode.TRAINING);

        DropoutForwardResult branchOne = layer.forward(input, initial, training);
        DropoutForwardResult branchTwo = layer.forward(input, initial, training);
        DropoutForwardResult threaded = layer.forward(input, branchOne.nextState(), training);
        TensorProducer firstProducer = branchOne.output().provenance().orElseThrow().producer();
        TensorProducer secondProducer = branchTwo.output().provenance().orElseThrow().producer();
        TensorProducer threadedProducer = threaded.output().provenance().orElseThrow().producer();

        assertAll(
                () -> assertNotSame(branchOne, branchTwo),
                () -> assertNotSame(branchOne.output(), branchTwo.output()),
                () -> assertNotSame(branchOne.nextState(), branchTwo.nextState()),
                () -> assertNotSame(firstProducer, secondProducer),
                () -> assertSame(initialTensor, firstProducer.inputs().get(1)),
                () -> assertSame(initialTensor, secondProducer.inputs().get(1)),
                () -> assertSame(stateTensor(branchOne.nextState()),
                        threadedProducer.inputs().get(1)),
                () -> assertEquals(new DropoutAttrs(0.0d), firstProducer.operation().attrs()),
                () -> assertTrue(layer.parameters().isEmpty()),
                () -> assertTrue(layer.buffers().isEmpty()),
                () -> assertTrue(layer.children().isEmpty()),
                () -> assertSame(initialTensor, stateTensor(initial)));
    }

    @Test
    void forwardValidatesArgumentsInOrderAndOnlyTrainingRequiresFloatingInput()
            throws ReflectiveOperationException {
        Dropout layer = new Dropout(0.25d);
        Tensor floating = tensor(DataType.FLOAT32, Shape.scalar(), false);
        Tensor integral = tensor(DataType.INT32, Shape.scalar(), false);
        Tensor bool = tensor(DataType.BOOL, Shape.scalar(), false);
        GraphRngState state = GraphRngState.initial(1L, 2L);
        ForwardContext training = new ForwardContext(ForwardMode.TRAINING);
        ForwardContext evaluation = new ForwardContext(ForwardMode.EVALUATION);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        NullPointerException inputFailure = assertThrows(
                NullPointerException.class, () -> layer.forward(null, null, null));
        NullPointerException stateFailure = assertThrows(
                NullPointerException.class, () -> layer.forward(floating, null, null));
        NullPointerException contextFailure = assertThrows(
                NullPointerException.class, () -> layer.forward(floating, state, null));
        IllegalArgumentException integralFailure = assertThrows(
                IllegalArgumentException.class,
                () -> layer.forward(integral, state, training));
        IllegalArgumentException boolFailure = assertThrows(
                IllegalArgumentException.class,
                () -> layer.forward(bool, state, training));
        DropoutForwardResult integralEvaluation = layer.forward(integral, state, evaluation);
        DropoutForwardResult boolEvaluation = layer.forward(bool, state, evaluation);

        assertAll(
                () -> assertEquals("input", inputFailure.getMessage()),
                () -> assertEquals("state", stateFailure.getMessage()),
                () -> assertEquals("context", contextFailure.getMessage()),
                () -> assertEquals("dropout input data type must be floating: INT32",
                        integralFailure.getMessage()),
                () -> assertEquals("dropout input data type must be floating: BOOL",
                        boolFailure.getMessage()),
                () -> assertSame(integral, integralEvaluation.output()),
                () -> assertSame(bool, boolEvaluation.output()),
                () -> assertSame(state, integralEvaluation.nextState()),
                () -> assertSame(state, boolEvaluation.nextState()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void partialModelIdentifierExhaustionConsumesThePrefixAndRetainsNoLayerState()
            throws ReflectiveOperationException {
        Dropout layer = new Dropout(0.25d);
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2), false);
        GraphRngState state = GraphRngState.initial(1L, 2L);
        Tensor exactStateTensor = stateTensor(state);
        AtomicLong next = nextTensorIdState();
        AtomicBoolean maximumClaimed = maximumTensorIdClaimedState();
        long savedNext = next.get();
        boolean savedClaimed = maximumClaimed.get();
        try {
            next.set(Long.MAX_VALUE - 1);
            maximumClaimed.set(false);

            IllegalStateException failure = assertThrows(
                    IllegalStateException.class,
                    () -> layer.forward(
                            input, state, new ForwardContext(ForwardMode.TRAINING)));

            assertAll(
                    () -> assertEquals("tensor identifier space exhausted", failure.getMessage()),
                    () -> assertEquals(Long.MAX_VALUE, next.get()),
                    () -> assertTrue(maximumClaimed.get()),
                    () -> assertTrue(layer.parameters().isEmpty()),
                    () -> assertTrue(layer.buffers().isEmpty()),
                    () -> assertTrue(layer.children().isEmpty()),
                    () -> assertSame(exactStateTensor, stateTensor(state)));
        } finally {
            next.set(savedNext);
            maximumClaimed.set(savedClaimed);
        }
    }

    private static Tensor tensor(DataType dataType, Shape shape, boolean requiresGrad) {
        return TensorFactory.create(new TensorDescriptor(
                dataType, shape, Optional.empty(), requiresGrad));
    }

    private static Tensor stateTensor(GraphRngState state) throws ReflectiveOperationException {
        Field field = GraphRngState.class.getDeclaredField("tensor");
        field.setAccessible(true);
        return (Tensor) field.get(state);
    }

    private static AtomicLong nextTensorIdState() throws ReflectiveOperationException {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }

    private static AtomicBoolean maximumTensorIdClaimedState()
            throws ReflectiveOperationException {
        Field field = TensorFactory.class.getDeclaredField("MAXIMUM_TENSOR_ID_CLAIMED");
        field.setAccessible(true);
        return (AtomicBoolean) field.get(null);
    }
}
