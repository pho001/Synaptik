package io.github.pho001.synaptik.nn.module;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.initialization.ParameterInitialization;
import io.github.pho001.synaptik.nn.layers.Embedding;
import io.github.pho001.synaptik.nn.layers.GruCell;
import io.github.pho001.synaptik.nn.layers.GruSequence;
import io.github.pho001.synaptik.nn.layers.Linear;
import io.github.pho001.synaptik.nn.layers.LstmCell;
import io.github.pho001.synaptik.nn.layers.LstmSequence;
import io.github.pho001.synaptik.nn.layers.RnnCell;
import io.github.pho001.synaptik.nn.layers.RnnSequence;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;

@Execution(ExecutionMode.SAME_THREAD)
class ModuleFactoryTest {
    private static final RandomGeneratorFactory<RandomGenerator> STANDARD_RANDOM_FACTORY =
            RandomGeneratorFactory.of("L64X128MixRandom");

    @Test
    void exposesExactlyThePlannedFinalStatelessSurface() throws ReflectiveOperationException {
        Constructor<ModuleFactory> constructor = ModuleFactory.class.getDeclaredConstructor();
        Field standardField = ModuleFactory.class.getDeclaredField("STANDARD");
        Set<MethodShape> methods = Arrays.stream(ModuleFactory.class.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .map(MethodShape::from)
                .collect(Collectors.toSet());

        assertAll(
                () -> assertTrue(Modifier.isPublic(ModuleFactory.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(ModuleFactory.class.getModifiers())),
                () -> assertSame(Object.class, ModuleFactory.class.getSuperclass()),
                () -> assertEquals(0, ModuleFactory.class.getInterfaces().length),
                () -> assertFalse(Modifier.isPublic(constructor.getModifiers())),
                () -> assertFalse(Modifier.isProtected(constructor.getModifiers())),
                () -> assertTrue(Modifier.isPrivate(constructor.getModifiers())),
                () -> assertEquals(1, ModuleFactory.class.getDeclaredConstructors().length),
                () -> assertEquals(1, ModuleFactory.class.getDeclaredFields().length),
                () -> assertTrue(Modifier.isPrivate(standardField.getModifiers())),
                () -> assertTrue(Modifier.isStatic(standardField.getModifiers())),
                () -> assertTrue(Modifier.isFinal(standardField.getModifiers())),
                () -> assertSame(ModuleFactory.class, standardField.getType()),
                () -> assertEquals(0, ModuleFactory.class.getDeclaredClasses().length),
                () -> assertEquals(0, ModuleFactory.class.getFields().length),
                () -> assertEquals(Set.of(
                                new MethodShape("standard", ModuleFactory.class, List.of(), true),
                                new MethodShape(
                                        "embedding",
                                        Embedding.class,
                                        recipeParameters(true),
                                        false),
                                new MethodShape(
                                        "linear", Linear.class, recipeParameters(false), false),
                                new MethodShape(
                                        "rnn", RnnSequence.class, recipeParameters(false), false),
                                new MethodShape(
                                        "gru", GruSequence.class, recipeParameters(false), false),
                                new MethodShape(
                                        "lstm", LstmSequence.class, recipeParameters(false), false)),
                        methods),
                () -> assertFalse(Arrays.stream(ModuleFactory.class.getDeclaredMethods())
                        .anyMatch(method -> Modifier.isProtected(method.getModifiers()))));
    }

    @Test
    void standardIsOneIdentityAndConstructionRetainsNoPerCallState()
            throws ReflectiveOperationException {
        AtomicLong nextTensorId = nextTensorIdState();
        long before = nextTensorId.get();

        ModuleFactory first = ModuleFactory.standard();
        ModuleFactory second = ModuleFactory.standard();
        Linear linear = first.linear(
                4, true, DataType.FLOAT32, ParameterInitialization.glorotUniform(), 11L);
        RnnSequence rnn = first.rnn(
                4, true, DataType.FLOAT32, ParameterInitialization.glorotUniform(), 12L);
        GruSequence gru = first.gru(
                4, true, DataType.FLOAT32, ParameterInitialization.glorotUniform(), 13L);
        LstmSequence lstm = first.lstm(
                4, true, DataType.FLOAT32, ParameterInitialization.glorotUniform(), 14L);

        assertAll(
                () -> assertSame(first, second),
                () -> assertEquals(before, nextTensorId.get()),
                () -> assertThrows(IllegalStateException.class, linear::weight),
                () -> assertThrows(IllegalStateException.class, rnn.cell()::inputWeight),
                () -> assertThrows(IllegalStateException.class, gru.cell()::inputWeight),
                () -> assertThrows(IllegalStateException.class, lstm.cell()::inputWeight));
    }

    @Test
    void everyRecipeReturnsTheExactFreshConcreteTypeAndOwnedCell() {
        ModuleFactory modules = ModuleFactory.standard();
        ParameterInitialization policy = ParameterInitialization.zeros();

        Embedding firstEmbedding =
                modules.embedding(5, 3, DataType.FLOAT32, policy, 21L);
        Embedding secondEmbedding =
                modules.embedding(5, 3, DataType.FLOAT32, policy, 21L);
        Linear firstLinear = modules.linear(4, false, DataType.FLOAT32, policy, 22L);
        Linear secondLinear = modules.linear(4, false, DataType.FLOAT32, policy, 22L);
        RnnSequence firstRnn = modules.rnn(4, false, DataType.FLOAT32, policy, 23L);
        RnnSequence secondRnn = modules.rnn(4, false, DataType.FLOAT32, policy, 23L);
        GruSequence firstGru = modules.gru(4, false, DataType.FLOAT32, policy, 24L);
        GruSequence secondGru = modules.gru(4, false, DataType.FLOAT32, policy, 24L);
        LstmSequence firstLstm = modules.lstm(4, false, DataType.FLOAT32, policy, 25L);
        LstmSequence secondLstm = modules.lstm(4, false, DataType.FLOAT32, policy, 25L);

        assertAll(
                () -> assertNotSame(firstEmbedding, secondEmbedding),
                () -> assertNotSame(firstEmbedding.weight(), secondEmbedding.weight()),
                () -> assertNotSame(
                        firstEmbedding.weight().value(), secondEmbedding.weight().value()),
                () -> assertNotEquals(
                        firstEmbedding.weight().value().id(),
                        secondEmbedding.weight().value().id()),
                () -> assertNotSame(firstLinear, secondLinear),
                () -> assertNotSame(firstRnn, secondRnn),
                () -> assertNotSame(firstRnn.cell(), secondRnn.cell()),
                () -> assertSame(firstRnn.cell(), firstRnn.children().get("cell")),
                () -> assertNotSame(firstGru, secondGru),
                () -> assertNotSame(firstGru.cell(), secondGru.cell()),
                () -> assertSame(firstGru.cell(), firstGru.children().get("cell")),
                () -> assertNotSame(firstLstm, secondLstm),
                () -> assertNotSame(firstLstm.cell(), secondLstm.cell()),
                () -> assertSame(firstLstm.cell(), firstLstm.children().get("cell")));
    }

    @Test
    void embeddingDelegatesEagerInitializationStateAndTensorEffectsExactly()
            throws ReflectiveOperationException {
        ModuleFactory modules = ModuleFactory.standard();
        ParameterInitialization policy = ParameterInitialization.uniform(-0.75d, 0.5d);
        AtomicLong nextTensorId = nextTensorIdState();
        long before = nextTensorId.get();

        Embedding actual = modules.embedding(4, 3, DataType.FLOAT32, policy, 31L);
        long afterFactory = nextTensorId.get();
        Embedding expected = new Embedding(4, 3, DataType.FLOAT32, policy, 31L);

        assertAll(
                () -> assertEquals(before + 1, afterFactory),
                () -> assertEquals(afterFactory + 1, nextTensorId.get()),
                () -> assertArrayEquals(
                        storage(expected.weight().value()), storage(actual.weight().value())),
                () -> assertEquals(Shape.of(4, 3), actual.weight().value().descriptor().shape()),
                () -> assertSame(DataType.FLOAT32, actual.weight().value().descriptor().dataType()),
                () -> assertTrue(actual.weight().value().descriptor().requiresGrad()),
                () -> assertEquals(List.of("weight"),
                        List.copyOf(actual.parametersRecursively().keySet())),
                () -> assertTrue(actual.buffersRecursively().isEmpty()),
                () -> assertTrue(actual.children().isEmpty()));
    }

    @Test
    void linearUsesTheExactStandardFactoryAndPreservesAutomaticBindingState() {
        ModuleFactory modules = ModuleFactory.standard();
        ParameterInitialization policy = ParameterInitialization.glorotUniform();
        Linear actual = modules.linear(4, true, DataType.FLOAT32, policy, 41L);
        Linear expected = new Linear(
                4, true, DataType.FLOAT32, policy, STANDARD_RANDOM_FACTORY, 41L);
        Tensor input = tensor(Shape.of(2, 3), DataType.FLOAT32, false);

        Tensor actualResult = actual.forward(input);
        Tensor expectedResult = expected.forward(input);

        assertAll(
                () -> assertEquals(Shape.of(2, 4), actualResult.descriptor().shape()),
                () -> assertEquals(actualResult.descriptor(), expectedResult.descriptor()),
                () -> assertArrayEquals(
                        storage(expected.weight().value()), storage(actual.weight().value())),
                () -> assertArrayEquals(
                        storage(expected.bias().orElseThrow().value()),
                        storage(actual.bias().orElseThrow().value())),
                () -> assertNotSame(expected.weight(), actual.weight()),
                () -> assertNotEquals(expected.weight().value().id(), actual.weight().value().id()),
                () -> assertEquals(List.of("weight", "bias"),
                        List.copyOf(actual.parametersRecursively().keySet())));
    }

    @Test
    void recurrentRecipesDelegateAutomaticCellInitializationAndStatePaths() {
        ModuleFactory modules = ModuleFactory.standard();
        ParameterInitialization policy = ParameterInitialization.glorotUniform();
        Tensor input = tensor(Shape.of(2, 3), DataType.FLOAT32, false);
        Tensor hidden = tensor(Shape.of(2, 4), DataType.FLOAT32, false);
        Tensor cellState = tensor(Shape.of(2, 4), DataType.FLOAT32, false);

        RnnSequence actualRnn = modules.rnn(4, true, DataType.FLOAT32, policy, 51L);
        RnnSequence expectedRnn =
                new RnnSequence(4, true, DataType.FLOAT32, policy, 51L);
        actualRnn.cell().forward(input, hidden);
        expectedRnn.cell().forward(input, hidden);

        GruSequence actualGru = modules.gru(4, true, DataType.FLOAT32, policy, 52L);
        GruSequence expectedGru =
                new GruSequence(4, true, DataType.FLOAT32, policy, 52L);
        actualGru.cell().forward(input, hidden);
        expectedGru.cell().forward(input, hidden);

        LstmSequence actualLstm = modules.lstm(4, true, DataType.FLOAT32, policy, 53L);
        LstmSequence expectedLstm =
                new LstmSequence(4, true, DataType.FLOAT32, policy, 53L);
        actualLstm.cell().forward(input, hidden, cellState);
        expectedLstm.cell().forward(input, hidden, cellState);

        assertAll(
                () -> assertRecurrentStateParity(actualRnn, expectedRnn),
                () -> assertRecurrentStateParity(actualGru, expectedGru),
                () -> assertRecurrentStateParity(actualLstm, expectedLstm));
    }

    @Test
    void delegatesConstructorFailuresWithoutTensorEffectsOrTranslation()
            throws ReflectiveOperationException {
        ModuleFactory modules = ModuleFactory.standard();
        AtomicLong nextTensorId = nextTensorIdState();
        long before = nextTensorId.get();

        assertSameFailure(
                () -> new Embedding(
                        0, 2, DataType.FLOAT32, ParameterInitialization.ones(), 61L),
                () -> modules.embedding(
                        0, 2, DataType.FLOAT32, ParameterInitialization.ones(), 61L));
        assertSameFailure(
                () -> new Linear(
                        0,
                        true,
                        DataType.FLOAT32,
                        ParameterInitialization.ones(),
                        STANDARD_RANDOM_FACTORY,
                        62L),
                () -> modules.linear(
                        0, true, DataType.FLOAT32, ParameterInitialization.ones(), 62L));
        assertSameFailure(
                () -> new RnnSequence(
                        0, true, DataType.FLOAT32, ParameterInitialization.ones(), 63L),
                () -> modules.rnn(
                        0, true, DataType.FLOAT32, ParameterInitialization.ones(), 63L));
        assertSameFailure(
                () -> new GruSequence(
                        0, true, DataType.FLOAT32, ParameterInitialization.ones(), 64L),
                () -> modules.gru(
                        0, true, DataType.FLOAT32, ParameterInitialization.ones(), 64L));
        assertSameFailure(
                () -> new LstmSequence(
                        1, true, null, ParameterInitialization.ones(), 65L),
                () -> modules.lstm(
                        1, true, null, ParameterInitialization.ones(), 65L));

        assertEquals(before, nextTensorId.get());
    }

    @Test
    void factoryResultsComposeThroughTopologyWithoutAddingAFactoryPathSegment() {
        ModuleFactory modules = ModuleFactory.standard();
        Embedding embedding = modules.embedding(
                7, 4, DataType.FLOAT32, ParameterInitialization.ones(), 71L);
        Linear projection = modules.linear(
                3, true, DataType.FLOAT32, ParameterInitialization.zeros(), 72L);
        Model<Tensor, Tensor> model = Model.<Tensor, Tensor>define(topology -> {
            assertSame(embedding, topology.addModule("embedding", embedding));
            assertSame(projection, topology.addModule("projection", projection));
            return indices -> projection.forward(embedding.forward(indices));
        });

        Tensor result = model.forward(tensor(Shape.of(2), DataType.INT64, false));

        assertAll(
                () -> assertEquals(Shape.of(2, 3), result.descriptor().shape()),
                () -> assertSame(embedding, model.children().get("embedding")),
                () -> assertSame(projection, model.children().get("projection")),
                () -> assertEquals(
                        List.of("embedding.weight", "projection.weight", "projection.bias"),
                        List.copyOf(model.parametersRecursively().keySet())),
                () -> assertFalse(model.parametersRecursively().keySet().stream()
                        .anyMatch(path -> path.contains("factory") || path.contains("moduleFactory"))),
                () -> assertThrows(
                        IllegalStateException.class,
                        () -> Model.define(topology -> {
                            topology.addModule("reused", embedding);
                            return (Object value) -> value;
                        })));
    }

    @Test
    void representativeAdvancedDirectConstructorsRemainAvailable()
            throws ReflectiveOperationException {
        assertAll(
                () -> assertSame(Tensor.class,
                        Embedding.class.getDeclaredConstructor(Tensor.class).getParameterTypes()[0]),
                () -> Linear.class.getDeclaredConstructor(
                        long.class,
                        long.class,
                        boolean.class,
                        DataType.class,
                        RandomGenerator.class),
                () -> Linear.class.getDeclaredConstructor(
                        long.class,
                        boolean.class,
                        DataType.class,
                        ParameterInitialization.class,
                        RandomGeneratorFactory.class,
                        long.class),
                () -> RnnCell.class.getDeclaredConstructor(
                        long.class,
                        long.class,
                        boolean.class,
                        DataType.class,
                        RandomGenerator.class),
                () -> GruCell.class.getDeclaredConstructor(
                        long.class,
                        long.class,
                        boolean.class,
                        DataType.class,
                        RandomGenerator.class),
                () -> LstmCell.class.getDeclaredConstructor(
                        long.class,
                        long.class,
                        boolean.class,
                        DataType.class,
                        RandomGenerator.class),
                () -> RnnSequence.class.getDeclaredConstructor(RnnCell.class),
                () -> GruSequence.class.getDeclaredConstructor(GruCell.class),
                () -> LstmSequence.class.getDeclaredConstructor(LstmCell.class));
    }

    private static List<Class<?>> recipeParameters(boolean embedding) {
        return embedding
                ? List.of(
                        long.class,
                        long.class,
                        DataType.class,
                        ParameterInitialization.class,
                        long.class)
                : List.of(
                        long.class,
                        boolean.class,
                        DataType.class,
                        ParameterInitialization.class,
                        long.class);
    }

    private static Tensor tensor(Shape shape, DataType dataType, boolean requiresGrad) {
        return TensorFactory.zeros(shape, dataType, Optional.empty(), requiresGrad);
    }

    private static float[] storage(Tensor tensor) {
        return (float[]) tensor.hostStorage().orElseThrow().segment().heapBase().orElseThrow();
    }

    private static void assertRecurrentStateParity(Module actual, Module expected) {
        List<String> expectedPaths = List.of(
                "cell.inputWeight", "cell.hiddenWeight", "cell.bias");
        assertEquals(expectedPaths, List.copyOf(actual.parametersRecursively().keySet()));
        assertEquals(expectedPaths, List.copyOf(expected.parametersRecursively().keySet()));
        for (String path : expectedPaths) {
            Tensor actualValue = actual.parametersRecursively().get(path).value();
            Tensor expectedValue = expected.parametersRecursively().get(path).value();
            assertArrayEquals(storage(expectedValue), storage(actualValue), path);
            assertNotSame(expectedValue, actualValue, path);
            assertNotEquals(expectedValue.id(), actualValue.id(), path);
        }
    }

    private static void assertSameFailure(ThrowingCall direct, ThrowingCall factory) {
        RuntimeException directFailure = assertThrows(RuntimeException.class, direct::run);
        RuntimeException factoryFailure = assertThrows(RuntimeException.class, factory::run);
        assertAll(
                () -> assertSame(directFailure.getClass(), factoryFailure.getClass()),
                () -> assertEquals(directFailure.getMessage(), factoryFailure.getMessage()));
    }

    private static AtomicLong nextTensorIdState() throws ReflectiveOperationException {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }

    private record MethodShape(
            String name, Class<?> returnType, List<Class<?>> parameterTypes, boolean staticMethod) {
        private static MethodShape from(Method method) {
            return new MethodShape(
                    method.getName(),
                    method.getReturnType(),
                    List.of(method.getParameterTypes()),
                    Modifier.isStatic(method.getModifiers()));
        }
    }

    @FunctionalInterface
    private interface ThrowingCall {
        void run();
    }
}
