package io.github.pho001.synaptik.nn.layers;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutKind;
import io.github.pho001.synaptik.model.operation.index.AxisGatherKind;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProvenance;
import io.github.pho001.synaptik.nn.initialization.ParameterInitialization;
import io.github.pho001.synaptik.nn.initialization.ParameterInitializers;
import io.github.pho001.synaptik.nn.module.ForwardMode;
import io.github.pho001.synaptik.nn.module.Parameter;
import io.github.pho001.synaptik.nn.module.StateDictionary;
import io.github.pho001.synaptik.nn.module.StateEntry;
import io.github.pho001.synaptik.nn.module.StateKind;
import io.github.pho001.synaptik.nn.module.UnaryTensorModule;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;

@Execution(ExecutionMode.SAME_THREAD)
class EmbeddingTest {
    @Test
    void exposesExactlyThePlannedFinalPublicSurface() throws ReflectiveOperationException {
        Set<List<Class<?>>> constructors = Arrays.stream(Embedding.class.getDeclaredConstructors())
                .filter(constructor -> Modifier.isPublic(constructor.getModifiers()))
                .map(Constructor::getParameterTypes)
                .map(List::of)
                .collect(Collectors.toSet());
        Set<String> publicMethods = Arrays.stream(Embedding.class.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .map(Method::getName)
                .collect(Collectors.toSet());
        Method weight = Embedding.class.getDeclaredMethod("weight");
        Method forward = Embedding.class.getDeclaredMethod("forward", Tensor.class);

        assertAll(
                () -> assertTrue(Modifier.isPublic(Embedding.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(Embedding.class.getModifiers())),
                () -> assertSame(UnaryTensorModule.class, Embedding.class.getSuperclass()),
                () -> assertEquals(
                        Set.of(
                                List.of(Tensor.class),
                                List.of(
                                        long.class,
                                        long.class,
                                        DataType.class,
                                        ParameterInitialization.class,
                                        long.class)),
                        constructors),
                () -> assertEquals(Set.of("weight", "forward"), publicMethods),
                () -> assertSame(Parameter.class, weight.getReturnType()),
                () -> assertSame(Tensor.class, forward.getReturnType()),
                () -> assertEquals(2, Arrays.stream(Embedding.class.getDeclaredMethods())
                        .filter(method -> Modifier.isPublic(method.getModifiers()))
                        .count()),
                () -> assertFalse(Arrays.stream(Embedding.class.getDeclaredMethods())
                        .anyMatch(method -> Modifier.isProtected(method.getModifiers()))),
                () -> assertEquals(1, Embedding.class.getDeclaredFields().length),
                () -> assertSame(Parameter.class, Embedding.class.getDeclaredField("weight").getType()),
                () -> assertEquals(0, Embedding.class.getDeclaredClasses().length),
                () -> assertFalse(Arrays.stream(UnaryTensorModule.class.getSuperclass().getDeclaredMethods())
                        .anyMatch(method -> method.getName().equals("forward"))));
    }

    @Test
    void retainsOneExactWeightParameterAndRecursivePath() {
        Tensor weight = tensor(DataType.FLOAT32, Shape.of(10, 4), true);

        Embedding layer = new Embedding(weight);

        assertAll(
                () -> assertSame(weight, layer.weight().value()),
                () -> assertEquals("weight", layer.weight().name()),
                () -> assertEquals(List.of(layer.weight()), layer.parameters()),
                () -> assertEquals(List.of("weight"),
                        List.copyOf(layer.parametersRecursively().keySet())),
                () -> assertSame(layer.weight(), layer.parametersRecursively().get("weight")),
                () -> assertTrue(layer.buffers().isEmpty()),
                () -> assertTrue(layer.buffersRecursively().isEmpty()),
                () -> assertTrue(layer.children().isEmpty()));
    }

    @Test
    void initializedConstructorCoversEveryPolicyAndFloatingTypeExactly()
            throws ReflectiveOperationException {
        Shape expectedShape = Shape.of(3, 2);
        long seed = 0x531E_77A2_19D4L;
        for (ParameterInitialization initialization : initializationPolicies()) {
            for (DataType dataType : floatingTypes()) {
                AtomicLong next = nextTensorIdState();
                long before = next.get();

                Embedding layer = new Embedding(3, 2, dataType, initialization, seed);
                Tensor actual = layer.weight().value();

                assertAll(
                        () -> assertEquals(before + 1, next.get()),
                        () -> assertInitializedLeaf(actual, expectedShape, dataType),
                        () -> assertSame(actual, layer.weight().value()),
                        () -> assertEquals("weight", layer.weight().name()),
                        () -> assertEquals(List.of(layer.weight()), layer.parameters()),
                        () -> assertEquals(List.of("weight"),
                                List.copyOf(layer.parametersRecursively().keySet())),
                        () -> assertTrue(layer.buffers().isEmpty()),
                        () -> assertTrue(layer.buffersRecursively().isEmpty()),
                        () -> assertTrue(layer.children().isEmpty()),
                        () -> assertEquals(
                                List.of(new StateEntry("weight", StateKind.PARAMETER, actual)),
                                layer.stateDictionary().entries()));

                Tensor expected = directInitialization(
                        expectedShape, dataType, initialization, seed);
                assertRepresentedValuesEqual(expected, actual, dataType);
            }
        }
    }

    @Test
    void fanPoliciesUseVocabularyAsFanOutAndEmbeddingWidthAsFanIn() {
        Shape wholeTable = Shape.of(7, 2);
        long seed = 991_734L;
        for (ParameterInitialization initialization : fanPolicies()) {
            Embedding layer = new Embedding(
                    7, 2, DataType.FLOAT64, initialization, seed);
            Tensor expected = ParameterInitializers.initialize(
                    wholeTable,
                    DataType.FLOAT64,
                    initialization,
                    standardGenerator(seed));

            assertRepresentedValuesEqual(
                    expected, layer.weight().value(), DataType.FLOAT64);
        }
    }

    @Test
    void equalConfigurationsRestartTheStandardStreamButOwnDistinctStateAndStorage() {
        ParameterInitialization policy = ParameterInitialization.uniform(-0.75d, 0.5d);
        Embedding first = new Embedding(4, 3, DataType.FLOAT32, policy, 7729L);
        Embedding second = new Embedding(4, 3, DataType.FLOAT32, policy, 7729L);
        Embedding differentSeed = new Embedding(4, 3, DataType.FLOAT32, policy, 7730L);

        assertRepresentedValuesEqual(
                first.weight().value(), second.weight().value(), DataType.FLOAT32);
        assertAll(
                () -> assertNotSame(first.weight(), second.weight()),
                () -> assertNotSame(first.weight().value(), second.weight().value()),
                () -> assertNotEquals(first.weight().value().id(), second.weight().value().id()),
                () -> assertNotSame(
                        heapBase(first.weight().value()), heapBase(second.weight().value())),
                () -> assertFalse(Arrays.equals(
                        heapArray(first.weight().value(), float[].class),
                        heapArray(differentSeed.weight().value(), float[].class))));
    }

    @Test
    void zeroAndOneIgnoreSeedAndEveryRowIncludingRowZeroIsOrdinaryTrainableState() {
        for (ParameterInitialization policy : List.of(
                ParameterInitialization.zeros(), ParameterInitialization.ones())) {
            for (DataType dataType : floatingTypes()) {
                Embedding first = new Embedding(3, 2, dataType, policy, Long.MIN_VALUE);
                Embedding second = new Embedding(3, 2, dataType, policy, Long.MAX_VALUE);

                assertRepresentedValuesEqual(
                        first.weight().value(), second.weight().value(), dataType);
                assertConstantTable(
                        first.weight().value(), dataType,
                        policy.equals(ParameterInitialization.ones()));
                assertAll(
                        () -> assertTrue(first.weight().value().descriptor().requiresGrad()),
                        () -> assertEquals(Shape.of(3, 2),
                                first.weight().value().descriptor().shape()),
                        () -> assertEquals(1, first.parameters().size()),
                        () -> assertTrue(first.buffers().isEmpty()));
            }
        }
    }

    @Test
    void initializedValidationPrecedesEveryEagerTensorEffectInExactOrder()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        assertAll(
                () -> assertEquals(
                        "vocabularySize must be positive: 0",
                        initializedFailure(0, 0, null, null)),
                () -> assertEquals(
                        "embeddingSize must be positive: 0",
                        initializedFailure(1, 0, null, null)),
                () -> assertEquals(
                        "dataType",
                        assertThrows(
                                NullPointerException.class,
                                () -> new Embedding(1, 1, null, null, 0L)).getMessage()),
                () -> assertEquals(
                        "weightInitialization",
                        assertThrows(
                                NullPointerException.class,
                                () -> new Embedding(1, 1, DataType.FLOAT32, null, 0L))
                                .getMessage()),
                () -> assertEquals(
                        "embedding data type must be floating: INT32",
                        initializedFailure(
                                Long.MAX_VALUE,
                                2,
                                DataType.INT32,
                                ParameterInitialization.ones())),
                () -> assertThrows(
                        ArithmeticException.class,
                        () -> new Embedding(
                                Long.MAX_VALUE,
                                2,
                                DataType.FLOAT32,
                                ParameterInitialization.zeros(),
                                0L)),
                () -> assertEquals(
                        "embedding weight element count exceeds Java array limit: count="
                                + (2L * Integer.MAX_VALUE) + ", maximum=" + Integer.MAX_VALUE,
                        initializedFailure(
                                Integer.MAX_VALUE,
                                2,
                                DataType.FLOAT32,
                                ParameterInitialization.ones())),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void identifierExhaustionFromRandomAndConstantInitializationPublishesNoLayer()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        AtomicBoolean maximumClaimed = maximumTensorIdClaimedState();
        long saved = next.get();
        boolean savedMaximumClaimed = maximumClaimed.get();
        try {
            next.set(Long.MAX_VALUE);
            maximumClaimed.set(true);

            IllegalStateException randomFailure = assertThrows(
                    IllegalStateException.class,
                    () -> new Embedding(
                            2,
                            2,
                            DataType.FLOAT32,
                            ParameterInitialization.normal(0.0d, 1.0d),
                            7L));
            IllegalStateException constantFailure = assertThrows(
                    IllegalStateException.class,
                    () -> new Embedding(
                            2,
                            2,
                            DataType.FLOAT32,
                            ParameterInitialization.zeros(),
                            9L));

            assertAll(
                    () -> assertEquals(
                            "tensor identifier space exhausted", randomFailure.getMessage()),
                    () -> assertEquals(
                            "tensor identifier space exhausted", constantFailure.getMessage()),
                    () -> assertEquals(Long.MAX_VALUE, next.get()),
                    () -> assertTrue(maximumClaimed.get()));
        } finally {
            next.set(saved);
            maximumClaimed.set(savedMaximumClaimed);
        }
    }

    @Test
    void initializedStateExportsLoadsAndReplacesWithTheExistingExactSchema() {
        Embedding layer = new Embedding(
                5, 3, DataType.FLOAT32, ParameterInitialization.ones(), 17L);
        Parameter wrapper = layer.weight();
        Tensor original = wrapper.value();
        Tensor indices = tensor(DataType.INT64, Shape.of(2), false);
        Tensor before = layer.forward(indices);
        Tensor loaded = tensor(DataType.FLOAT32, Shape.of(5, 3), true);
        layer.eval();

        layer.loadStateDictionary(new StateDictionary(List.of(
                new StateEntry("weight", StateKind.PARAMETER, loaded))));
        Tensor after = layer.forward(indices);

        assertGather(before, original, indices);
        assertGather(after, loaded, indices);
        assertAll(
                () -> assertSame(wrapper, layer.weight()),
                () -> assertSame(loaded, wrapper.value()),
                () -> assertEquals(ForwardMode.EVALUATION, layer.mode()),
                () -> assertEquals(
                        List.of(new StateEntry("weight", StateKind.PARAMETER, loaded)),
                        layer.stateDictionary().entries()),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> layer.loadStateDictionary(new StateDictionary(List.of(
                                new StateEntry("weight", StateKind.BUFFER, loaded))))),
                () -> assertSame(loaded, wrapper.value()),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> layer.loadStateDictionary(new StateDictionary(List.of(
                                new StateEntry(
                                        "weight",
                                        StateKind.PARAMETER,
                                        tensor(DataType.FLOAT64, Shape.of(5, 3), true)))))),
                () -> assertSame(loaded, wrapper.value()),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> layer.loadStateDictionary(new StateDictionary(List.of(
                                new StateEntry(
                                        "weight",
                                        StateKind.PARAMETER,
                                        tensor(DataType.FLOAT32, Shape.of(6, 3), true)))))),
                () -> assertSame(loaded, wrapper.value()),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> layer.loadStateDictionary(new StateDictionary(List.of(
                                new StateEntry(
                                        "weight",
                                        StateKind.PARAMETER,
                                        tensor(DataType.FLOAT32, Shape.of(5, 3), false)))))),
                () -> assertSame(loaded, wrapper.value()));
    }

    @Test
    void validatesSuppliedWeightInOrderWithoutTensorSideEffects()
            throws ReflectiveOperationException {
        Tensor integral = tensor(DataType.INT32, Shape.scalar(), false);
        Tensor noGradient = tensor(DataType.FLOAT32, Shape.scalar(), false);
        Tensor scalar = tensor(DataType.FLOAT32, Shape.scalar(), true);
        Tensor dynamic = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("vocabulary"), new StaticDimension(4)),
                true);
        Tensor zeroVocabulary = tensor(DataType.FLOAT32, Shape.of(0, 0), true);
        Tensor zeroEmbedding = tensor(DataType.FLOAT32, Shape.of(10, 0), true);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        assertAll(
                () -> assertEquals(
                        "weight",
                        assertThrows(NullPointerException.class, () -> new Embedding(null))
                                .getMessage()),
                () -> assertEquals(
                        "embedding weight must have a floating data type: INT32",
                        failure(integral)),
                () -> assertEquals(
                        "embedding weight must have requiresGrad == true",
                        failure(noGradient)),
                () -> assertEquals(
                        "embedding weight must have rank two: 0",
                        failure(scalar)),
                () -> assertEquals(
                        "embedding weight must have a fully static shape: Shape[vocabulary, 4]",
                        failure(dynamic)),
                () -> assertEquals(
                        "embedding weight must have positive vocabularySize: 0",
                        failure(zeroVocabulary)),
                () -> assertEquals(
                        "embedding weight must have positive embeddingSize: 0",
                        failure(zeroEmbedding)),
                () -> assertEquals(before, next.get()),
                () -> assertTrue(integral.provenance().isEmpty()),
                () -> assertTrue(noGradient.provenance().isEmpty()),
                () -> assertTrue(scalar.provenance().isEmpty()),
                () -> assertTrue(dynamic.provenance().isEmpty()),
                () -> assertTrue(zeroVocabulary.provenance().isEmpty()),
                () -> assertTrue(zeroEmbedding.provenance().isEmpty()));
    }

    @Test
    void delegatesEveryFloatingTableAndExactIndexTypeToOneOrdinaryGather() {
        for (DataType weightType : List.of(
                DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64)) {
            for (DataType indexType : List.of(DataType.INT32, DataType.INT64)) {
                Tensor weight = tensor(weightType, Shape.of(10, 4), true);
                Tensor indices = tensor(indexType, Shape.of(2, 3), false);
                Embedding layer = new Embedding(weight);

                Tensor result = layer.forward(indices);

                assertGather(result, weight, indices);
                assertAll(
                        () -> assertSame(weightType, result.descriptor().dataType()),
                        () -> assertEquals(Shape.of(2, 3, 4), result.descriptor().shape()),
                        () -> assertTrue(result.descriptor().requiresGrad()),
                        () -> assertTrue(result.descriptor().layout().isEmpty()),
                        () -> assertTrue(result.label().isEmpty()),
                        () -> assertTrue(result.hostStorage().isEmpty()));
            }
        }
    }

    @Test
    void retainsScalarAndMultiAxisIndexDimensionsAndCreatesFreshResults() {
        StaticDimension vocabularySize = new StaticDimension(10);
        StaticDimension embeddingSize = new StaticDimension(4);
        StaticDimension batch = new StaticDimension(2);
        StaticDimension sequence = new StaticDimension(3);
        Tensor weight = tensor(
                DataType.FLOAT64,
                Shape.ofDimensions(vocabularySize, embeddingSize),
                true);
        Tensor scalarIndices = tensor(DataType.INT32, Shape.scalar(), false);
        Tensor shapedIndices = tensor(
                DataType.INT64, Shape.ofDimensions(batch, sequence), false);
        Embedding layer = new Embedding(weight);

        Tensor scalar = layer.forward(scalarIndices);
        Tensor first = layer.forward(shapedIndices);
        Tensor second = layer.forward(shapedIndices);

        assertAll(
                () -> assertEquals(1, scalar.descriptor().shape().rank()),
                () -> assertSame(embeddingSize, scalar.descriptor().shape().dimension(0)),
                () -> assertSame(batch, first.descriptor().shape().dimension(0)),
                () -> assertSame(sequence, first.descriptor().shape().dimension(1)),
                () -> assertSame(embeddingSize, first.descriptor().shape().dimension(2)),
                () -> assertNotSame(first, second),
                () -> assertNotEquals(first.id(), second.id()),
                () -> assertNotSame(
                        first.provenance().orElseThrow().producer(),
                        second.provenance().orElseThrow().producer()));
        assertGather(scalar, weight, scalarIndices);
        assertGather(first, weight, shapedIndices);
        assertGather(second, weight, shapedIndices);
    }

    @Test
    void forwardIsModeInsensitiveAndRejectsNullOrInvalidIndicesThroughModel() {
        Tensor weight = tensor(DataType.FLOAT32, Shape.of(10, 4), true);
        Tensor indices = tensor(DataType.INT64, Shape.of(2), false);
        Tensor invalidIndices = tensor(DataType.BOOL, Shape.of(2), false);
        Embedding layer = new Embedding(weight);

        layer.eval();
        Tensor evaluation = layer.forward(indices);
        layer.train();
        Tensor training = layer.forward(indices);
        NullPointerException nullIndices = assertThrows(
                NullPointerException.class, () -> layer.forward(null));
        IllegalArgumentException invalidType = assertThrows(
                IllegalArgumentException.class, () -> layer.forward(invalidIndices));

        assertGather(evaluation, weight, indices);
        assertGather(training, weight, indices);
        assertAll(
                () -> assertEquals(ForwardMode.TRAINING, layer.mode()),
                () -> assertEquals(evaluation.descriptor(), training.descriptor()),
                () -> assertEquals("indices", nullIndices.getMessage()),
                () -> assertEquals(
                        "embedding indices data type must be INT32 or INT64: BOOL",
                        invalidType.getMessage()));
    }

    @Test
    void compatibleReplacementChangesOnlyLaterForwardSnapshots() {
        Tensor oldWeight = tensor(DataType.FLOAT32, Shape.of(10, 4), true);
        Tensor indices = tensor(DataType.INT64, Shape.of(2, 3), false);
        Embedding layer = new Embedding(oldWeight);
        Parameter handle = layer.weight();

        Tensor before = layer.forward(indices);
        Tensor newWeight = tensor(DataType.FLOAT32, Shape.of(10, 4), true);
        handle.replace(newWeight);
        Tensor after = layer.forward(indices);

        assertGather(before, oldWeight, indices);
        assertGather(after, newWeight, indices);
        assertAll(
                () -> assertSame(handle, layer.weight()),
                () -> assertSame(newWeight, handle.value()),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> handle.replace(tensor(DataType.FLOAT64, Shape.of(10, 4), true))),
                () -> assertSame(newWeight, handle.value()),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> handle.replace(tensor(DataType.FLOAT32, Shape.of(11, 4), true))),
                () -> assertSame(newWeight, handle.value()),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> handle.replace(tensor(DataType.FLOAT32, Shape.of(10, 4), false))),
                () -> assertSame(newWeight, handle.value()));
    }

    @Test
    void inheritsTensorIdentifierExhaustionWithoutChangingTheBinding()
            throws ReflectiveOperationException {
        Tensor weight = tensor(DataType.FLOAT32, Shape.of(10, 4), true);
        Tensor indices = tensor(DataType.INT32, Shape.of(1), false);
        Embedding layer = new Embedding(weight);
        AtomicLong next = nextTensorIdState();
        AtomicBoolean maximumClaimed = maximumTensorIdClaimedState();
        long saved = next.get();
        boolean savedMaximumClaimed = maximumClaimed.get();
        try {
            next.set(Long.MAX_VALUE);
            maximumClaimed.set(true);

            IllegalStateException exhausted = assertThrows(
                    IllegalStateException.class, () -> layer.forward(indices));

            assertAll(
                    () -> assertEquals(
                            "tensor identifier space exhausted", exhausted.getMessage()),
                    () -> assertEquals(Long.MAX_VALUE, next.get()),
                    () -> assertSame(weight, layer.weight().value()));
        } finally {
            next.set(saved);
            maximumClaimed.set(savedMaximumClaimed);
        }
    }

    private static List<DataType> floatingTypes() {
        return List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16);
    }

    private static List<ParameterInitialization> initializationPolicies() {
        return List.of(
                ParameterInitialization.glorotNormal(),
                ParameterInitialization.glorotUniform(),
                ParameterInitialization.kaimingReluNormal(),
                ParameterInitialization.kaimingReluUniform(),
                ParameterInitialization.normal(-0.25d, 0.75d),
                ParameterInitialization.uniform(-0.5d, 0.625d),
                ParameterInitialization.zeros(),
                ParameterInitialization.ones());
    }

    private static List<ParameterInitialization> fanPolicies() {
        return List.of(
                ParameterInitialization.glorotNormal(),
                ParameterInitialization.glorotUniform(),
                ParameterInitialization.kaimingReluNormal(),
                ParameterInitialization.kaimingReluUniform());
    }

    private static Tensor directInitialization(
            Shape shape,
            DataType dataType,
            ParameterInitialization initialization,
            long seed) {
        if (initialization.requiresRandomGenerator()) {
            return ParameterInitializers.initialize(
                    shape, dataType, initialization, standardGenerator(seed));
        }
        return ParameterInitializers.initialize(shape, dataType, initialization);
    }

    private static RandomGenerator standardGenerator(long seed) {
        return RandomGeneratorFactory.<RandomGenerator>of("L64X128MixRandom").create(seed);
    }

    private static void assertInitializedLeaf(
            Tensor tensor, Shape shape, DataType dataType) {
        assertAll(
                () -> assertEquals(shape, tensor.descriptor().shape()),
                () -> assertSame(dataType, tensor.descriptor().dataType()),
                () -> assertTrue(tensor.descriptor().requiresGrad()),
                () -> assertEquals(
                        LayoutKind.DENSE_CONTIGUOUS,
                        tensor.descriptor().layout().orElseThrow().kind()),
                () -> assertFalse(tensor.descriptor().layout().orElseThrow().isView()),
                () -> assertTrue(tensor.label().isEmpty()),
                () -> assertTrue(tensor.provenance().isEmpty()),
                () -> assertEquals(
                        shape.knownElementCount().orElseThrow(),
                        tensor.hostStorage().orElseThrow().elementCapacity()));
    }

    private static void assertRepresentedValuesEqual(
            Tensor expected, Tensor actual, DataType dataType) {
        switch (dataType) {
            case FLOAT64 -> assertArrayEquals(
                    heapArray(expected, double[].class), heapArray(actual, double[].class));
            case FLOAT32 -> assertArrayEquals(
                    heapArray(expected, float[].class), heapArray(actual, float[].class));
            case BFLOAT16 -> assertArrayEquals(
                    heapArray(expected, short[].class), heapArray(actual, short[].class));
            case INT32, INT64, BOOL -> throw new AssertionError("unexpected non-floating type");
        }
    }

    private static void assertConstantTable(Tensor tensor, DataType dataType, boolean one) {
        int length = Math.toIntExact(
                tensor.descriptor().shape().knownElementCount().orElseThrow());
        switch (dataType) {
            case FLOAT64 -> {
                double[] expected = new double[length];
                Arrays.fill(expected, one ? 1.0d : 0.0d);
                assertArrayEquals(expected, heapArray(tensor, double[].class));
            }
            case FLOAT32 -> {
                float[] expected = new float[length];
                Arrays.fill(expected, one ? 1.0f : 0.0f);
                assertArrayEquals(expected, heapArray(tensor, float[].class));
            }
            case BFLOAT16 -> {
                short[] expected = new short[length];
                Arrays.fill(expected, BFloat16Bits.fromFloat(one ? 1.0f : 0.0f));
                assertArrayEquals(expected, heapArray(tensor, short[].class));
            }
            case INT32, INT64, BOOL -> throw new AssertionError("unexpected non-floating type");
        }
    }

    private static Object heapBase(Tensor tensor) {
        return tensor.hostStorage().orElseThrow().segment().heapBase().orElseThrow();
    }

    private static <T> T heapArray(Tensor tensor, Class<T> carrier) {
        return carrier.cast(heapBase(tensor));
    }

    private static String initializedFailure(
            long vocabularySize,
            long embeddingSize,
            DataType dataType,
            ParameterInitialization initialization) {
        return assertThrows(
                IllegalArgumentException.class,
                () -> new Embedding(
                        vocabularySize,
                        embeddingSize,
                        dataType,
                        initialization,
                        0L)).getMessage();
    }

    private static String failure(Tensor weight) {
        return assertThrows(IllegalArgumentException.class, () -> new Embedding(weight))
                .getMessage();
    }

    private static void assertGather(Tensor result, Tensor weight, Tensor indices) {
        TensorProvenance provenance = result.provenance().orElseThrow();
        assertAll(
                () -> assertSame(AxisGatherKind.GATHER, provenance.operation().kind()),
                () -> assertEquals(new IndexAxisAttrs(0), provenance.operation().attrs()),
                () -> assertEquals(List.of(weight, indices), provenance.inputs()),
                () -> assertEquals(0, provenance.outputIndex()),
                () -> assertEquals(1, provenance.producer().outputCount()),
                () -> assertSame(result.descriptor(), provenance.outputDescriptor()));
    }

    private static Tensor tensor(DataType dataType, Shape shape, boolean requiresGrad) {
        return TensorFactory.create(new TensorDescriptor(
                dataType, shape, Optional.empty(), requiresGrad));
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
