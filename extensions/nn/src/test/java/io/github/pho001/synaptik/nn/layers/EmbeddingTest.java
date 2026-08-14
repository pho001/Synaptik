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
import io.github.pho001.synaptik.model.operation.index.AxisGatherKind;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProvenance;
import io.github.pho001.synaptik.nn.module.ForwardMode;
import io.github.pho001.synaptik.nn.module.Module;
import io.github.pho001.synaptik.nn.module.Parameter;
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
                () -> assertSame(Module.class, Embedding.class.getSuperclass()),
                () -> assertEquals(Set.of(List.of(Tensor.class)), constructors),
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
                () -> assertFalse(Arrays.stream(Module.class.getDeclaredMethods())
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
