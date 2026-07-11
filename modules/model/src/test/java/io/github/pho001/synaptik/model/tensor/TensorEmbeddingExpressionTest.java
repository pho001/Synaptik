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
import io.github.pho001.synaptik.model.operation.index.AxisGatherKind;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class TensorEmbeddingExpressionTest {
    private static final AtomicLong INPUT_IDS = new AtomicLong(190_000);

    @Test
    void exposesExactlyOnePublicConvenienceAndOnePackagePrivateHelper() throws Exception {
        Method publicMethod = Tensor.class.getDeclaredMethod("embedding", Tensor.class);
        Method helper = TensorAxisGatherExpressions.class.getDeclaredMethod(
                "embedding", Tensor.class, Tensor.class);

        assertAll(
                () -> assertTrue(Modifier.isPublic(publicMethod.getModifiers())),
                () -> assertFalse(Modifier.isStatic(publicMethod.getModifiers())),
                () -> assertEquals(Tensor.class, publicMethod.getReturnType()),
                () -> assertTrue(Modifier.isStatic(helper.getModifiers())),
                () -> assertFalse(Modifier.isPublic(helper.getModifiers())),
                () -> assertFalse(Modifier.isProtected(helper.getModifiers())),
                () -> assertFalse(Modifier.isPrivate(helper.getModifiers())),
                () -> assertEquals(Tensor.class, helper.getReturnType()));

        List<Method> embeddingMethods = Arrays.stream(Tensor.class.getDeclaredMethods())
                .filter(method -> method.getName().equals("embedding"))
                .toList();
        assertEquals(List.of(publicMethod), embeddingMethods);
        Set<String> operationKinds = Arrays.stream(AxisGatherKind.values())
                .map(Enum::name)
                .collect(Collectors.toSet());
        assertEquals(Set.of("GATHER", "GATHER_ELEMENTS"), operationKinds);
    }

    @Test
    void acceptsEveryFloatingWeightAndIntegralIndexTypeWithWeightOnlyEligibility() {
        for (DataType weightType : List.of(
                DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64)) {
            for (DataType indexType : List.of(DataType.INT32, DataType.INT64)) {
                Tensor weights = tensor(weightType, Shape.of(10, 4), true);
                Tensor indices = tensor(indexType, Shape.of(2, 3), false);

                Tensor result = weights.embedding(indices);

                assertAll(
                        () -> assertSame(weightType, result.descriptor().dataType()),
                        () -> assertTrue(result.descriptor().requiresGrad()),
                        () -> assertEquals(Shape.of(2, 3, 4), result.descriptor().shape()),
                        () -> assertTrue(result.descriptor().layout().isEmpty()),
                        () -> assertTrue(result.label().isEmpty()),
                        () -> assertTrue(result.hostStorage().isEmpty()));
            }
        }

        Tensor weights = tensor(DataType.FLOAT32, Shape.of(3, 2), false);
        Tensor result = weights.embedding(tensor(DataType.INT64, Shape.of(1), false));
        assertFalse(result.descriptor().requiresGrad());
    }

    @Test
    void retainsScalarZeroStaticNamedAndExpressionDimensionsExactly() {
        DynamicDimension query = new DynamicDimension("query");
        Dimension expression = DimensionExpressions.addConstant(
                new DynamicDimension("batch"), 1);
        StaticDimension embeddingSize = new StaticDimension(0);
        Tensor weights = tensor(
                DataType.FLOAT64,
                Shape.ofDimensions(new DynamicDimension("vocabulary"), embeddingSize),
                true);

        Tensor scalar = weights.embedding(tensor(DataType.INT32, Shape.scalar(), false));
        StaticDimension zero = new StaticDimension(0);
        StaticDimension staticExtent = new StaticDimension(3);
        Tensor shaped = weights.embedding(tensor(
                DataType.INT64,
                Shape.ofDimensions(zero, staticExtent, query, expression),
                false));

        assertAll(
                () -> assertEquals(1, scalar.descriptor().shape().rank()),
                () -> assertSame(
                        embeddingSize, scalar.descriptor().shape().dimensions().getFirst()),
                () -> assertEquals(5, shaped.descriptor().shape().rank()),
                () -> assertSame(zero, shaped.descriptor().shape().dimensions().get(0)),
                () -> assertSame(staticExtent, shaped.descriptor().shape().dimensions().get(1)),
                () -> assertSame(query, shaped.descriptor().shape().dimensions().get(2)),
                () -> assertSame(expression, shaped.descriptor().shape().dimensions().get(3)),
                () -> assertSame(embeddingSize, shaped.descriptor().shape().dimensions().get(4)));

        Tensor zeroVocabulary = tensor(DataType.BFLOAT16, Shape.of(0, 5), false);
        assertEquals(
                Shape.of(2, 5),
                zeroVocabulary.embedding(tensor(DataType.INT32, Shape.of(2), false))
                        .descriptor().shape());
    }

    @Test
    void createsOneOrdinaryGatherProducerAndOneIdWithoutMutatingInputs() throws Exception {
        Tensor weights = tensor(DataType.FLOAT32, Shape.of(10, 4), true);
        Tensor indices = tensor(DataType.INT64, Shape.of(2, 3), false);
        TensorDescriptor weightsDescriptor = weights.descriptor();
        TensorDescriptor indicesDescriptor = indices.descriptor();
        TensorId weightsId = weights.id();
        TensorId indicesId = indices.id();
        Optional<String> weightsLabel = weights.label();
        Optional<String> indicesLabel = indices.label();
        Optional<TensorProvenance> weightsProvenance = weights.provenance();
        Optional<TensorProvenance> indicesProvenance = indices.provenance();
        long before = nextTensorIdState().get();

        Tensor first = weights.embedding(indices);
        Tensor second = weights.embedding(indices);

        TensorProvenance provenance = first.provenance().orElseThrow();
        TensorProducer producer = provenance.producer();
        assertAll(
                () -> assertEquals(before, first.id().value()),
                () -> assertEquals(before + 1, second.id().value()),
                () -> assertEquals(before + 2, nextTensorIdState().get()),
                () -> assertNotSame(first, second),
                () -> assertNotEquals(first.id(), second.id()),
                () -> assertNotSame(
                        producer, second.provenance().orElseThrow().producer()),
                () -> assertSame(AxisGatherKind.GATHER, producer.operation().kind()),
                () -> assertEquals(new IndexAxisAttrs(0), producer.operation().attrs()),
                () -> assertEquals(2, producer.inputs().size()),
                () -> assertSame(weights, producer.inputs().get(0)),
                () -> assertSame(indices, producer.inputs().get(1)),
                () -> assertEquals(1, producer.outputCount()),
                () -> assertSame(first.descriptor(), producer.outputDescriptors().getFirst()),
                () -> assertEquals(0, provenance.outputIndex()),
                () -> assertSame(weightsDescriptor, weights.descriptor()),
                () -> assertSame(indicesDescriptor, indices.descriptor()),
                () -> assertSame(weightsId, weights.id()),
                () -> assertSame(indicesId, indices.id()),
                () -> assertEquals(weightsLabel, weights.label()),
                () -> assertEquals(indicesLabel, indices.label()),
                () -> assertEquals(weightsProvenance, weights.provenance()),
                () -> assertEquals(indicesProvenance, indices.provenance()),
                () -> assertTrue(weights.hostStorage().isEmpty()),
                () -> assertTrue(indices.hostStorage().isEmpty()));
    }

    @Test
    void validatesNullRankWeightTypeAndIndexTypeInOrderWithoutAllocatingIds() throws Exception {
        Tensor rankOneIntegral = tensor(DataType.INT64, Shape.of(4), false);
        Tensor rankTwoIntegral = tensor(DataType.INT64, Shape.of(10, 4), false);
        Tensor validWeights = tensor(DataType.FLOAT32, Shape.of(10, 4), true);
        Tensor invalidIndices = tensor(DataType.BOOL, Shape.of(2, 3), false);
        long before = nextTensorIdState().get();

        NullPointerException weightsNull = assertThrows(
                NullPointerException.class,
                () -> TensorAxisGatherExpressions.embedding(null, null));
        NullPointerException indicesNull = assertThrows(
                NullPointerException.class,
                () -> TensorAxisGatherExpressions.embedding(validWeights, null));
        IllegalArgumentException rankBeforeTypes = assertThrows(
                IllegalArgumentException.class,
                () -> rankOneIntegral.embedding(invalidIndices));
        IllegalArgumentException weightTypeBeforeIndexType = assertThrows(
                IllegalArgumentException.class,
                () -> rankTwoIntegral.embedding(invalidIndices));
        IllegalArgumentException indexType = assertThrows(
                IllegalArgumentException.class,
                () -> validWeights.embedding(invalidIndices));

        assertAll(
                () -> assertEquals("weights", weightsNull.getMessage()),
                () -> assertEquals("indices", indicesNull.getMessage()),
                () -> assertEquals(
                        "embedding weights rank must be 2: actual=1",
                        rankBeforeTypes.getMessage()),
                () -> assertEquals(
                        "embedding weights data type must be BFLOAT16, FLOAT32, or FLOAT64: INT64",
                        weightTypeBeforeIndexType.getMessage()),
                () -> assertEquals(
                        "embedding indices data type must be INT32 or INT64: BOOL",
                        indexType.getMessage()),
                () -> assertEquals(before, nextTensorIdState().get()));

        for (int rank : List.of(0, 1, 3)) {
            Shape shape = switch (rank) {
                case 0 -> Shape.scalar();
                case 1 -> Shape.of(4);
                default -> Shape.of(2, 3, 4);
            };
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class,
                    () -> tensor(DataType.FLOAT32, shape, true).embedding(
                            tensor(DataType.INT32, Shape.scalar(), false)));
            assertEquals(
                    "embedding weights rank must be 2: actual=" + rank,
                    failure.getMessage());
        }
        assertEquals(before, nextTensorIdState().get());
    }

    @Test
    void rejectsEveryNonFloatingWeightAndNonIntegralIndexWithoutValueInspection() throws Exception {
        long before = nextTensorIdState().get();
        for (DataType weightType : List.of(DataType.INT32, DataType.INT64, DataType.BOOL)) {
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class,
                    () -> tensor(weightType, Shape.of(0, 0), false).embedding(
                            tensor(DataType.INT32, Shape.of(0), false)));
            assertEquals(
                    "embedding weights data type must be BFLOAT16, FLOAT32, or FLOAT64: "
                            + weightType,
                    failure.getMessage());
        }
        for (DataType indexType : DataType.values()) {
            if (indexType == DataType.INT32 || indexType == DataType.INT64) {
                continue;
            }
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class,
                    () -> tensor(DataType.FLOAT32, Shape.of(0, 0), false).embedding(
                            tensor(indexType, Shape.of(0), false)));
            assertEquals(
                    "embedding indices data type must be INT32 or INT64: " + indexType,
                    failure.getMessage());
        }
        assertEquals(before, nextTensorIdState().get());
    }

    @Test
    void doesNotReadStoredNegativeOrOutOfRangeIndexValues() {
        Shape indicesShape = Shape.of(2);
        Tensor indices = TensorFactory.fromFlatArray(
                new TensorDescriptor(
                        DataType.INT32,
                        indicesShape,
                        Optional.of(LayoutDescriptor.contiguous(indicesShape)),
                        false),
                Optional.of("unchecked indices"),
                new int[] {-1, 10});
        Tensor weights = tensor(DataType.FLOAT32, Shape.of(10, 4), true);

        Tensor result = weights.embedding(indices);

        assertAll(
                () -> assertEquals(Shape.of(2, 4), result.descriptor().shape()),
                () -> assertSame(indices, result.provenance().orElseThrow().inputs().get(1)),
                () -> assertTrue(indices.hostStorage().isPresent()),
                () -> assertEquals(Optional.of("unchecked indices"), indices.label()),
                () -> assertTrue(result.hostStorage().isEmpty()),
                () -> assertTrue(result.descriptor().layout().isEmpty()));
    }

    private static Tensor tensor(DataType type, Shape shape, boolean requiresGrad) {
        return new Tensor(
                new TensorId(INPUT_IDS.getAndIncrement()),
                new TensorDescriptor(type, shape, Optional.empty(), requiresGrad),
                Optional.empty(),
                Optional.empty(),
                Optional.empty());
    }

    private static AtomicLong nextTensorIdState() throws ReflectiveOperationException {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }
}
