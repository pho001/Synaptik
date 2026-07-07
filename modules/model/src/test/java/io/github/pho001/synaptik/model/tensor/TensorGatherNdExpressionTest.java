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
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.index.GatherNdAttrs;
import io.github.pho001.synaptik.model.operation.index.GatherNdKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

class TensorGatherNdExpressionTest {
    private static final AtomicLong IDS = new AtomicLong(90_000);

    @Test
    void exposesExactlyTwoPublicMethodsAndEightMethodFieldFreeHelper() throws Exception {
        Method defaultGatherNd = Tensor.class.getDeclaredMethod("gatherNd", Tensor.class);
        Method explicitGatherNd =
                Tensor.class.getDeclaredMethod("gatherNd", Tensor.class, int.class);
        assertPublicInstance(defaultGatherNd, List.of(Tensor.class));
        assertPublicInstance(explicitGatherNd, List.of(Tensor.class, int.class));

        Method helperDefault = TensorGatherNdExpressions.class.getDeclaredMethod(
                "gatherNd", Tensor.class, Tensor.class);
        Method helperExplicit = TensorGatherNdExpressions.class.getDeclaredMethod(
                "gatherNd", Tensor.class, Tensor.class, int.class);
        Method validateIndexType = TensorGatherNdExpressions.class.getDeclaredMethod(
                "validateIndexType", TensorDescriptor.class);
        Method validateBatchDimensions = TensorGatherNdExpressions.class.getDeclaredMethod(
                "validateBatchDimensions", Shape.class, Shape.class, int.class);
        Method validateBatchPrefix = TensorGatherNdExpressions.class.getDeclaredMethod(
                "validateBatchPrefix", Shape.class, Shape.class, int.class);
        Method tupleDepth = TensorGatherNdExpressions.class.getDeclaredMethod(
                "tupleDepth", Shape.class, Shape.class, int.class);
        Method resultShape = TensorGatherNdExpressions.class.getDeclaredMethod(
                "resultShape", Shape.class, Shape.class, int.class, int.class);
        Method create = TensorGatherNdExpressions.class.getDeclaredMethod(
                "create",
                Tensor.class,
                Tensor.class,
                TensorDescriptor.class,
                Shape.class,
                GatherNdAttrs.class);
        var constructor = TensorGatherNdExpressions.class.getDeclaredConstructor();

        assertAll(
                () -> assertTrue(Modifier.isFinal(
                        TensorGatherNdExpressions.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(
                        TensorGatherNdExpressions.class.getModifiers())),
                () -> assertEquals(0,
                        TensorGatherNdExpressions.class.getDeclaredFields().length),
                () -> assertEquals(0,
                        TensorGatherNdExpressions.class.getDeclaredClasses().length),
                () -> assertEquals(1,
                        TensorGatherNdExpressions.class.getDeclaredConstructors().length),
                () -> assertTrue(Modifier.isPrivate(constructor.getModifiers())),
                () -> assertEquals(
                        List.of(
                                "create",
                                "gatherNd",
                                "gatherNd",
                                "resultShape",
                                "tupleDepth",
                                "validateBatchDimensions",
                                "validateBatchPrefix",
                                "validateIndexType"),
                        Arrays.stream(TensorGatherNdExpressions.class.getDeclaredMethods())
                                .map(Method::getName)
                                .sorted()
                                .toList()),
                () -> assertPackagePrivateStatic(helperDefault, Tensor.class),
                () -> assertPackagePrivateStatic(helperExplicit, Tensor.class),
                () -> assertPrivateStatic(validateIndexType, void.class),
                () -> assertPrivateStatic(validateBatchDimensions, void.class),
                () -> assertPrivateStatic(validateBatchPrefix, void.class),
                () -> assertPrivateStatic(tupleDepth, int.class),
                () -> assertPrivateStatic(resultShape, Shape.class),
                () -> assertPrivateStatic(create, Tensor.class));
    }

    @Test
    void defaultDelegatesToZeroBatchAndBothFormsRecordExactOrderedSemantics() {
        Tensor data = tensor(DataType.FLOAT32, Shape.of(2, 3, 4), true);
        Tensor indices = tensor(DataType.INT32, Shape.of(5, 2), false);
        Tensor defaultResult = data.gatherNd(indices);
        Tensor explicitResult = data.gatherNd(indices, 0);

        assertOperation(defaultResult, 0, data, indices);
        assertOperation(explicitResult, 0, data, indices);
        assertAll(
                () -> assertEquals(defaultResult.descriptor(), explicitResult.descriptor()),
                () -> assertNotSame(defaultResult, explicitResult),
                () -> assertNotEquals(defaultResult.id(), explicitResult.id()));
    }

    @Test
    void acceptsEveryDataTypeAndBothExactIndexTypesWhileRetainingOnlyDataMetadata() {
        for (DataType dataType : DataType.values()) {
            for (boolean requiresGrad : dataType.isDifferentiable()
                    ? List.of(false, true)
                    : List.of(false)) {
                for (DataType indexType : List.of(DataType.INT32, DataType.INT64)) {
                    Tensor data = tensor(dataType, Shape.of(2, 3), requiresGrad);
                    Tensor indices = tensor(indexType, Shape.of(4, 1), false);
                    Tensor result = data.gatherNd(indices);
                    TensorProvenance provenance = result.provenance().orElseThrow();

                    assertAll(
                            () -> assertSame(dataType, result.descriptor().dataType()),
                            () -> assertEquals(
                                    requiresGrad, result.descriptor().requiresGrad()),
                            () -> assertEquals(Shape.of(4, 3), result.descriptor().shape()),
                            () -> assertTrue(result.descriptor().layout().isEmpty()),
                            () -> assertTrue(result.label().isEmpty()),
                            () -> assertTrue(result.hostStorage().isEmpty()),
                            () -> assertEquals(List.of(data, indices), provenance.inputs()));
                }
            }
        }
    }

    @Test
    void derivesPrefixPlusSuffixExamplesAndCanonicalScalarWithExactDimensions() {
        Tensor zeroBatchData = tensor(DataType.FLOAT64, Shape.of(2, 3, 4), true);
        Tensor zeroBatchIndices = tensor(DataType.INT32, Shape.of(5, 2), false);
        assertEquals(
                Shape.of(5, 4), zeroBatchData.gatherNd(zeroBatchIndices).descriptor().shape());

        DynamicDimension batch = new DynamicDimension("N");
        DynamicDimension queries = new DynamicDimension("M");
        StaticDimension suffix = new StaticDimension(4);
        Shape dataShape = Shape.ofDimensions(batch, new StaticDimension(3), suffix);
        Shape indicesShape =
                Shape.ofDimensions(batch, queries, new StaticDimension(1));
        Shape resultShape = tensor(DataType.BFLOAT16, dataShape, true)
                .gatherNd(tensor(DataType.INT64, indicesShape, false), 1)
                .descriptor()
                .shape();

        Shape scalarShape = tensor(DataType.BOOL, Shape.of(2, 3), false)
                .gatherNd(tensor(DataType.INT32, Shape.of(2), false))
                .descriptor()
                .shape();

        assertAll(
                () -> assertEquals(Shape.ofDimensions(batch, queries, suffix), resultShape),
                () -> assertSame(batch, resultShape.dimensions().get(0)),
                () -> assertSame(queries, resultShape.dimensions().get(1)),
                () -> assertSame(suffix, resultShape.dimensions().get(2)),
                () -> assertSame(Shape.scalar(), scalarShape));
    }

    @Test
    void validatesNullTypeRankAndNegativeBatchInExactOrderWithoutConsumingIdentity()
            throws Exception {
        AtomicLong next = nextTensorIdState();
        Tensor data = tensor(DataType.FLOAT32, Shape.of(2, 3), true);
        Tensor floatingScalar = tensor(DataType.FLOAT64, Shape.scalar(), false);
        Tensor integralScalar = tensor(DataType.INT32, Shape.scalar(), false);
        Tensor validIndices = tensor(DataType.INT64, Shape.of(1), false);
        long before = next.get();

        NullPointerException nullData = assertThrows(
                NullPointerException.class,
                () -> TensorGatherNdExpressions.gatherNd(null, null, -1));
        NullPointerException nullIndices = assertThrows(
                NullPointerException.class,
                () -> TensorGatherNdExpressions.gatherNd(data, null, -1));
        IllegalArgumentException type = assertThrows(
                IllegalArgumentException.class, () -> data.gatherNd(floatingScalar, -1));
        IllegalArgumentException rank = assertThrows(
                IllegalArgumentException.class, () -> data.gatherNd(integralScalar, -1));
        IllegalArgumentException negative = assertThrows(
                IllegalArgumentException.class, () -> data.gatherNd(validIndices, -1));

        assertAll(
                () -> assertEquals("data", nullData.getMessage()),
                () -> assertEquals("indices", nullIndices.getMessage()),
                () -> assertEquals(
                        "gatherNd indices data type must be INT32 or INT64: FLOAT64",
                        type.getMessage()),
                () -> assertEquals(
                        "gatherNd indices rank must be at least 1", rank.getMessage()),
                () -> assertEquals(
                        "batchDimensions must be non-negative: -1", negative.getMessage()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void validatesBatchRankThenPrefixInExactOrderAndSupportsEqualDynamicSymbols() {
        Tensor rankTwoData = tensor(DataType.FLOAT32, Shape.of(2, 3), true);
        IllegalArgumentException indicesRank = assertThrows(
                IllegalArgumentException.class,
                () -> rankTwoData.gatherNd(
                        tensor(DataType.INT32, Shape.of(2, 1), false), 2));
        IllegalArgumentException dataRank = assertThrows(
                IllegalArgumentException.class,
                () -> tensor(DataType.FLOAT32, Shape.scalar(), false)
                        .gatherNd(tensor(DataType.INT32, Shape.of(1), false), 0));

        DynamicDimension dataBatch = new DynamicDimension("batch");
        DynamicDimension equalBatch = new DynamicDimension("batch");
        DynamicDimension otherBatch = new DynamicDimension("other");
        Tensor dynamicData = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(dataBatch, new StaticDimension(3)),
                true);
        Tensor validIndices = tensor(
                DataType.INT64,
                Shape.ofDimensions(equalBatch, new StaticDimension(1)),
                false);
        IllegalArgumentException mismatch = assertThrows(
                IllegalArgumentException.class,
                () -> dynamicData.gatherNd(
                        tensor(
                                DataType.INT64,
                                Shape.ofDimensions(otherBatch, new StaticDimension(1)),
                                false),
                        1));

        Tensor valid = dynamicData.gatherNd(validIndices, 1);
        assertAll(
                () -> assertEquals(
                        "gatherNd batchDimensions must be less than indices rank: "
                                + "batchDimensions=2, indicesRank=2",
                        indicesRank.getMessage()),
                () -> assertEquals(
                        "gatherNd batchDimensions must be less than data rank: "
                                + "batchDimensions=0, dataRank=0",
                        dataRank.getMessage()),
                () -> assertEquals(
                        "gatherNd batch dimension at axis 0 must match data: expected="
                                + dataBatch + ", actual=" + otherBatch,
                        mismatch.getMessage()),
                () -> assertEquals(Shape.ofDimensions(equalBatch),
                        valid.descriptor().shape()),
                () -> assertSame(equalBatch,
                        valid.descriptor().shape().dimensions().get(0)));
    }

    @Test
    void requiresStaticPositiveTupleDepthWithinUnbatchedDataRank() {
        Tensor data = tensor(DataType.FLOAT32, Shape.of(2, 3, 4), true);
        IllegalArgumentException dynamic = assertThrows(
                IllegalArgumentException.class,
                () -> data.gatherNd(tensor(
                        DataType.INT32,
                        Shape.ofDimensions(
                                new StaticDimension(5), new DynamicDimension("K")),
                        false)));
        IllegalArgumentException zero = assertThrows(
                IllegalArgumentException.class,
                () -> data.gatherNd(tensor(DataType.INT64, Shape.of(5, 0), false)));
        IllegalArgumentException tooLarge = assertThrows(
                IllegalArgumentException.class,
                () -> data.gatherNd(tensor(DataType.INT32, Shape.of(5, 4), false)));

        assertAll(
                () -> assertEquals(
                        "gatherNd tuple depth must be statically known", dynamic.getMessage()),
                () -> assertEquals(
                        "gatherNd tuple depth must be in [1, data rank - batchDimensions]: "
                                + "depth=0, maximum=3",
                        zero.getMessage()),
                () -> assertEquals(
                        "gatherNd tuple depth must be in [1, data rank - batchDimensions]: "
                                + "depth=4, maximum=3",
                        tooLarge.getMessage()));
    }

    @Test
    void rejectsEveryUnsupportedIndexTypeBeforeShapeValidation() {
        Tensor data = tensor(DataType.FLOAT32, Shape.of(2), true);
        for (DataType indexType : List.of(
                DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16, DataType.BOOL)) {
            Tensor indices = tensor(indexType, Shape.scalar(), false);
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class, () -> data.gatherNd(indices, -1));
            assertEquals(
                    "gatherNd indices data type must be INT32 or INT64: " + indexType,
                    failure.getMessage());
        }
    }

    @Test
    void discardsResolvedLayoutsAndDoesNotReadOrMutateInputStorageOrMetadata() {
        float[] dataValues = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        int[] indexValues = {1, 0};
        Shape dataShape = Shape.of(2, 3);
        Shape indicesShape = Shape.of(2);
        LayoutDescriptor dataLayout = LayoutDescriptor.contiguous(dataShape);
        LayoutDescriptor indicesLayout = LayoutDescriptor.contiguous(indicesShape);
        HostTensorStorage dataStorage = new MemorySegmentStorage(
                DataType.FLOAT32, dataValues.length, MemorySegment.ofArray(dataValues));
        HostTensorStorage indicesStorage = new MemorySegmentStorage(
                DataType.INT32, indexValues.length, MemorySegment.ofArray(indexValues));
        Tensor data = new Tensor(
                new TensorId(IDS.getAndIncrement()),
                new TensorDescriptor(
                        DataType.FLOAT32, dataShape, Optional.of(dataLayout), true),
                Optional.of("data"),
                Optional.empty(),
                Optional.of(dataStorage));
        Tensor indices = new Tensor(
                new TensorId(IDS.getAndIncrement()),
                new TensorDescriptor(
                        DataType.INT32, indicesShape, Optional.of(indicesLayout), false),
                Optional.of("indices"),
                Optional.empty(),
                Optional.of(indicesStorage));

        Tensor result = data.gatherNd(indices);

        assertAll(
                () -> assertSame(dataLayout, data.descriptor().layout().orElseThrow()),
                () -> assertSame(indicesLayout, indices.descriptor().layout().orElseThrow()),
                () -> assertSame(dataStorage, data.hostStorage().orElseThrow()),
                () -> assertSame(indicesStorage, indices.hostStorage().orElseThrow()),
                () -> assertEquals(
                        List.of(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f),
                        List.of(
                                dataValues[0],
                                dataValues[1],
                                dataValues[2],
                                dataValues[3],
                                dataValues[4],
                                dataValues[5])),
                () -> assertEquals(List.of(1, 0), List.of(indexValues[0], indexValues[1])),
                () -> assertTrue(result.descriptor().layout().isEmpty()),
                () -> assertTrue(result.label().isEmpty()),
                () -> assertTrue(result.hostStorage().isEmpty()));
    }

    @Test
    void repeatedAndNestedRequestsAreFreshAndIdentifierExhaustionPropagates() throws Exception {
        Tensor data = tensor(DataType.FLOAT64, Shape.of(2, 3), true);
        Tensor indices = tensor(DataType.INT32, Shape.of(4, 1), false);
        Tensor first = data.gatherNd(indices);
        Tensor second = data.gatherNd(indices);
        Tensor nestedIndices = tensor(DataType.INT64, Shape.of(2), false);
        Tensor nested = first.gatherNd(nestedIndices);

        assertAll(
                () -> assertNotSame(first, second),
                () -> assertNotEquals(first.id(), second.id()),
                () -> assertSame(first,
                        nested.provenance().orElseThrow().inputs().get(0)),
                () -> assertSame(nestedIndices,
                        nested.provenance().orElseThrow().inputs().get(1)));

        AtomicLong next = nextTensorIdState();
        AtomicBoolean claimed = maximumClaimedState();
        long originalNext = next.get();
        boolean originalClaimed = claimed.get();
        try {
            next.set(Long.MAX_VALUE);
            claimed.set(true);
            IllegalStateException failure = assertThrows(
                    IllegalStateException.class, () -> data.gatherNd(indices));
            assertEquals("tensor identifier space exhausted", failure.getMessage());
        } finally {
            next.set(originalNext);
            claimed.set(originalClaimed);
        }
    }

    private static void assertPublicInstance(Method method, List<Class<?>> parameters) {
        assertAll(
                () -> assertSame(Tensor.class, method.getReturnType()),
                () -> assertEquals(parameters, Arrays.asList(method.getParameterTypes())),
                () -> assertTrue(Modifier.isPublic(method.getModifiers())),
                () -> assertFalse(Modifier.isStatic(method.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(method.getModifiers())),
                () -> assertFalse(method.isVarArgs()));
    }

    private static void assertPackagePrivateStatic(Method method, Class<?> returnType) {
        assertAll(
                () -> assertSame(returnType, method.getReturnType()),
                () -> assertTrue(Modifier.isStatic(method.getModifiers())),
                () -> assertFalse(Modifier.isPublic(method.getModifiers())),
                () -> assertFalse(Modifier.isProtected(method.getModifiers())),
                () -> assertFalse(Modifier.isPrivate(method.getModifiers())));
    }

    private static void assertPrivateStatic(Method method, Class<?> returnType) {
        assertAll(
                () -> assertSame(returnType, method.getReturnType()),
                () -> assertTrue(Modifier.isStatic(method.getModifiers())),
                () -> assertTrue(Modifier.isPrivate(method.getModifiers())));
    }

    private static void assertOperation(
            Tensor result, int batchDimensions, Tensor data, Tensor indices) {
        TensorProvenance provenance = result.provenance().orElseThrow();
        Operation operation = provenance.operation();
        assertAll(
                () -> assertSame(GatherNdKind.GATHER_ND, operation.kind()),
                () -> assertEquals(new GatherNdAttrs(batchDimensions), operation.attrs()),
                () -> assertEquals(List.of(data, indices), provenance.inputs()),
                () -> assertSame(data, provenance.inputs().get(0)),
                () -> assertSame(indices, provenance.inputs().get(1)));
    }

    private static AtomicLong nextTensorIdState() throws Exception {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }

    private static AtomicBoolean maximumClaimedState() throws Exception {
        Field field = TensorFactory.class.getDeclaredField("MAXIMUM_TENSOR_ID_CLAIMED");
        field.setAccessible(true);
        return (AtomicBoolean) field.get(null);
    }

    private static Tensor tensor(DataType dataType, Shape shape, boolean requiresGrad) {
        return new Tensor(
                new TensorId(IDS.getAndIncrement()),
                new TensorDescriptor(dataType, shape, Optional.empty(), requiresGrad),
                Optional.empty(),
                Optional.empty(),
                Optional.empty());
    }
}
