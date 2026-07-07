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
import io.github.pho001.synaptik.model.operation.index.ScatterNdAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterNdKind;
import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
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

class TensorScatterNdExpressionTest {
    private static final AtomicLong IDS = new AtomicLong(110_000);

    @Test
    void exposesExactlyThreePublicMethodsAndElevenMethodFieldFreeHelper() throws Exception {
        Method shortScatterNd = Tensor.class.getDeclaredMethod(
                "scatterNd", Tensor.class, Tensor.class);
        Method reductionScatterNd = Tensor.class.getDeclaredMethod(
                "scatterNd", Tensor.class, Tensor.class, ScatterReduction.class);
        Method explicitScatterNd = Tensor.class.getDeclaredMethod(
                "scatterNd",
                Tensor.class,
                Tensor.class,
                ScatterReduction.class,
                int.class);
        assertPublicInstance(shortScatterNd, List.of(Tensor.class, Tensor.class));
        assertPublicInstance(
                reductionScatterNd,
                List.of(Tensor.class, Tensor.class, ScatterReduction.class));
        assertPublicInstance(
                explicitScatterNd,
                List.of(Tensor.class, Tensor.class, ScatterReduction.class, int.class));

        Method helperShort = TensorScatterNdExpressions.class.getDeclaredMethod(
                "scatterNd", Tensor.class, Tensor.class, Tensor.class);
        Method helperReduction = TensorScatterNdExpressions.class.getDeclaredMethod(
                "scatterNd",
                Tensor.class,
                Tensor.class,
                Tensor.class,
                ScatterReduction.class);
        Method helperExplicit = TensorScatterNdExpressions.class.getDeclaredMethod(
                "scatterNd",
                Tensor.class,
                Tensor.class,
                Tensor.class,
                ScatterReduction.class,
                int.class);
        Method validateIndexType = TensorScatterNdExpressions.class.getDeclaredMethod(
                "validateIndexType", TensorDescriptor.class);
        Method validateMatchingDataType = TensorScatterNdExpressions.class.getDeclaredMethod(
                "validateMatchingDataType", TensorDescriptor.class, TensorDescriptor.class);
        Method validateReductionDataType = TensorScatterNdExpressions.class.getDeclaredMethod(
                "validateReductionDataType", TensorDescriptor.class, ScatterReduction.class);
        Method validateBatchDimensions = TensorScatterNdExpressions.class.getDeclaredMethod(
                "validateBatchDimensions", Shape.class, Shape.class, int.class);
        Method validateBatchPrefix = TensorScatterNdExpressions.class.getDeclaredMethod(
                "validateBatchPrefix", Shape.class, Shape.class, int.class);
        Method tupleDepth = TensorScatterNdExpressions.class.getDeclaredMethod(
                "tupleDepth", Shape.class, Shape.class, int.class);
        Method expectedUpdatesShape = TensorScatterNdExpressions.class.getDeclaredMethod(
                "expectedUpdatesShape", Shape.class, Shape.class, int.class, int.class);
        Method create = TensorScatterNdExpressions.class.getDeclaredMethod(
                "create",
                Tensor.class,
                Tensor.class,
                Tensor.class,
                TensorDescriptor.class,
                TensorDescriptor.class,
                ScatterNdAttrs.class);
        var constructor = TensorScatterNdExpressions.class.getDeclaredConstructor();

        assertAll(
                () -> assertTrue(Modifier.isFinal(
                        TensorScatterNdExpressions.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(
                        TensorScatterNdExpressions.class.getModifiers())),
                () -> assertEquals(0,
                        TensorScatterNdExpressions.class.getDeclaredFields().length),
                () -> assertEquals(0,
                        TensorScatterNdExpressions.class.getDeclaredClasses().length),
                () -> assertEquals(1,
                        TensorScatterNdExpressions.class.getDeclaredConstructors().length),
                () -> assertTrue(Modifier.isPrivate(constructor.getModifiers())),
                () -> assertEquals(
                        List.of(
                                "create",
                                "expectedUpdatesShape",
                                "scatterNd",
                                "scatterNd",
                                "scatterNd",
                                "tupleDepth",
                                "validateBatchDimensions",
                                "validateBatchPrefix",
                                "validateIndexType",
                                "validateMatchingDataType",
                                "validateReductionDataType"),
                        Arrays.stream(TensorScatterNdExpressions.class.getDeclaredMethods())
                                .map(Method::getName)
                                .sorted()
                                .toList()),
                () -> assertPackagePrivateStatic(helperShort, Tensor.class),
                () -> assertPackagePrivateStatic(helperReduction, Tensor.class),
                () -> assertPackagePrivateStatic(helperExplicit, Tensor.class),
                () -> assertPrivateStatic(validateIndexType, void.class),
                () -> assertPrivateStatic(validateMatchingDataType, void.class),
                () -> assertPrivateStatic(validateReductionDataType, void.class),
                () -> assertPrivateStatic(validateBatchDimensions, void.class),
                () -> assertPrivateStatic(validateBatchPrefix, void.class),
                () -> assertPrivateStatic(tupleDepth, int.class),
                () -> assertPrivateStatic(expectedUpdatesShape, Shape.class),
                () -> assertPrivateStatic(create, Tensor.class));
    }

    @Test
    void defaultsDelegateToNoneAndZeroBatchAndRecordExactOrderedSemantics() {
        Tensor data = tensor(DataType.FLOAT32, Shape.of(2, 3, 4), true);
        Tensor indices = tensor(DataType.INT32, Shape.of(5, 2), false);
        Tensor updates = tensor(DataType.FLOAT32, Shape.of(5, 4), false);

        Tensor replacement = data.scatterNd(indices, updates);
        Tensor addition = data.scatterNd(indices, updates, ScatterReduction.ADD);
        Tensor explicit = data.scatterNd(indices, updates, ScatterReduction.MUL, 0);

        assertOperation(replacement, 0, ScatterReduction.NONE, data, indices, updates);
        assertOperation(addition, 0, ScatterReduction.ADD, data, indices, updates);
        assertOperation(explicit, 0, ScatterReduction.MUL, data, indices, updates);
        assertAll(
                () -> assertEquals(replacement.descriptor(), addition.descriptor()),
                () -> assertNotSame(replacement, addition),
                () -> assertNotEquals(replacement.id(), addition.id()));
    }

    @Test
    void acceptsExactTypeDomainsAndCombinesOnlyDataAndUpdateEligibility() {
        Shape dataShape = Shape.of(2, 3);
        Shape indicesShape = Shape.of(4, 1);
        Shape updatesShape = Shape.of(4, 3);
        for (DataType indexType : List.of(DataType.INT32, DataType.INT64)) {
            for (DataType dataType : DataType.values()) {
                Tensor data = tensor(dataType, dataShape, false);
                Tensor indices = tensor(indexType, indicesShape, false);
                Tensor updates = tensor(
                        dataType, updatesShape, dataType.isDifferentiable());
                Tensor replacement = data.scatterNd(indices, updates);
                assertResult(replacement, data, dataType.isDifferentiable());

                if (!dataType.isBoolean()) {
                    for (ScatterReduction reduction : List.of(
                            ScatterReduction.ADD,
                            ScatterReduction.MUL,
                            ScatterReduction.MAX,
                            ScatterReduction.MIN)) {
                        Tensor arithmetic = data.scatterNd(indices, updates, reduction);
                        assertResult(arithmetic, data, dataType.isDifferentiable());
                    }
                }
            }
        }

        Tensor gradData = tensor(DataType.FLOAT64, dataShape, true);
        Tensor noGradUpdates = tensor(DataType.FLOAT64, updatesShape, false);
        assertTrue(gradData.scatterNd(
                        tensor(DataType.INT32, indicesShape, false), noGradUpdates)
                .descriptor().requiresGrad());
    }

    @Test
    void validatesNullTypeAndReductionInExactOrderWithoutConsumingIdentity() throws Exception {
        AtomicLong next = nextTensorIdState();
        Tensor data = tensor(DataType.FLOAT32, Shape.of(2, 3), true);
        Tensor floatingIndices = tensor(DataType.FLOAT64, Shape.scalar(), false);
        Tensor wrongUpdates = tensor(DataType.FLOAT64, Shape.scalar(), true);
        long before = next.get();

        NullPointerException nullData = assertThrows(
                NullPointerException.class,
                () -> TensorScatterNdExpressions.scatterNd(null, null, null, null, -1));
        NullPointerException nullIndices = assertThrows(
                NullPointerException.class,
                () -> TensorScatterNdExpressions.scatterNd(data, null, null, null, -1));
        NullPointerException nullUpdates = assertThrows(
                NullPointerException.class,
                () -> TensorScatterNdExpressions.scatterNd(
                        data, floatingIndices, null, null, -1));
        NullPointerException nullReduction = assertThrows(
                NullPointerException.class,
                () -> TensorScatterNdExpressions.scatterNd(
                        data, floatingIndices, wrongUpdates, null, -1));
        IllegalArgumentException indexType = assertThrows(
                IllegalArgumentException.class,
                () -> data.scatterNd(
                        floatingIndices, wrongUpdates, ScatterReduction.ADD, -1));
        Tensor validIndices = tensor(DataType.INT32, Shape.of(1), false);
        IllegalArgumentException updateType = assertThrows(
                IllegalArgumentException.class,
                () -> data.scatterNd(
                        validIndices, wrongUpdates, ScatterReduction.ADD, -1));

        assertAll(
                () -> assertEquals("data", nullData.getMessage()),
                () -> assertEquals("indices", nullIndices.getMessage()),
                () -> assertEquals("updates", nullUpdates.getMessage()),
                () -> assertEquals("reduction", nullReduction.getMessage()),
                () -> assertEquals(
                        "scatterNd indices data type must be INT32 or INT64: FLOAT64",
                        indexType.getMessage()),
                () -> assertEquals(
                        "scatterNd updates data type must match data: "
                                + "expected=FLOAT32, actual=FLOAT64",
                        updateType.getMessage()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void rejectsUnsupportedIndexTypesAndBoolArithmeticBeforeShapeValidation() {
        Tensor data = tensor(DataType.FLOAT32, Shape.of(2), true);
        Tensor updates = tensor(DataType.FLOAT32, Shape.scalar(), false);
        for (DataType indexType : List.of(
                DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16, DataType.BOOL)) {
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class,
                    () -> data.scatterNd(
                            tensor(indexType, Shape.scalar(), false),
                            tensor(DataType.FLOAT32, Shape.of(9), false),
                            ScatterReduction.ADD,
                            -1));
            assertEquals(
                    "scatterNd indices data type must be INT32 or INT64: " + indexType,
                    failure.getMessage());
        }

        Tensor boolData = tensor(DataType.BOOL, Shape.of(2, 3), false);
        Tensor boolIndices = tensor(DataType.INT64, Shape.of(4, 1), false);
        Tensor boolUpdates = tensor(DataType.BOOL, Shape.of(4, 3), false);
        assertResult(boolData.scatterNd(boolIndices, boolUpdates), boolData, false);
        for (ScatterReduction reduction : List.of(
                ScatterReduction.ADD,
                ScatterReduction.MUL,
                ScatterReduction.MAX,
                ScatterReduction.MIN)) {
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class,
                    () -> boolData.scatterNd(
                            tensor(DataType.INT32, Shape.scalar(), false),
                            tensor(DataType.BOOL, Shape.scalar(), false),
                            reduction,
                            -1));
            assertEquals(
                    "scatterNd BOOL data supports only NONE reduction: " + reduction,
                    failure.getMessage());
        }
    }

    @Test
    void validatesRankNegativeBatchBatchFitAndPrefixInExactOrder() {
        Tensor data = tensor(DataType.FLOAT32, Shape.of(2, 3), true);
        IllegalArgumentException rank = assertThrows(
                IllegalArgumentException.class,
                () -> data.scatterNd(
                        tensor(DataType.INT32, Shape.scalar(), false),
                        tensor(DataType.FLOAT32, Shape.scalar(), false),
                        ScatterReduction.NONE,
                        -1));
        IllegalArgumentException negative = assertThrows(
                IllegalArgumentException.class,
                () -> data.scatterNd(
                        tensor(DataType.INT32, Shape.of(1), false),
                        tensor(DataType.FLOAT32, Shape.of(3), false),
                        ScatterReduction.NONE,
                        -1));
        IllegalArgumentException indicesRank = assertThrows(
                IllegalArgumentException.class,
                () -> data.scatterNd(
                        tensor(DataType.INT32, Shape.of(2, 1), false),
                        tensor(DataType.FLOAT32, Shape.of(2, 3), false),
                        ScatterReduction.NONE,
                        2));
        IllegalArgumentException dataRank = assertThrows(
                IllegalArgumentException.class,
                () -> tensor(DataType.FLOAT32, Shape.scalar(), false).scatterNd(
                        tensor(DataType.INT64, Shape.of(1), false),
                        tensor(DataType.FLOAT32, Shape.scalar(), false),
                        ScatterReduction.NONE,
                        0));

        DynamicDimension dataBatch = new DynamicDimension("batch");
        DynamicDimension equalBatch = new DynamicDimension("batch");
        DynamicDimension otherBatch = new DynamicDimension("other");
        Tensor dynamicData = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(dataBatch, new StaticDimension(3)),
                true);
        Tensor valid = dynamicData.scatterNd(
                tensor(
                        DataType.INT64,
                        Shape.ofDimensions(equalBatch, new StaticDimension(1)),
                        false),
                tensor(DataType.FLOAT32, Shape.ofDimensions(equalBatch), false),
                ScatterReduction.ADD,
                1);
        IllegalArgumentException mismatch = assertThrows(
                IllegalArgumentException.class,
                () -> dynamicData.scatterNd(
                        tensor(
                                DataType.INT64,
                                Shape.ofDimensions(otherBatch, new StaticDimension(1)),
                                false),
                        tensor(DataType.FLOAT32, Shape.ofDimensions(otherBatch), false),
                        ScatterReduction.ADD,
                        1));

        assertAll(
                () -> assertEquals("scatterNd indices rank must be at least 1", rank.getMessage()),
                () -> assertEquals(
                        "batchDimensions must be non-negative: -1", negative.getMessage()),
                () -> assertEquals(
                        "scatterNd batchDimensions must be less than indices rank: "
                                + "batchDimensions=2, indicesRank=2",
                        indicesRank.getMessage()),
                () -> assertEquals(
                        "scatterNd batchDimensions must be less than data rank: "
                                + "batchDimensions=0, dataRank=0",
                        dataRank.getMessage()),
                () -> assertEquals(
                        "scatterNd batch dimension at axis 0 must match data: expected="
                                + dataBatch + ", actual=" + otherBatch,
                        mismatch.getMessage()),
                () -> assertSame(dynamicData.descriptor().shape(), valid.descriptor().shape()));
    }

    @Test
    void requiresStaticPositiveTupleDepthAndExactUpdatesShape() {
        Tensor data = tensor(DataType.FLOAT32, Shape.of(2, 3, 4), true);
        IllegalArgumentException dynamic = assertThrows(
                IllegalArgumentException.class,
                () -> data.scatterNd(
                        tensor(
                                DataType.INT32,
                                Shape.ofDimensions(
                                        new StaticDimension(5), new DynamicDimension("K")),
                                false),
                        tensor(DataType.FLOAT32, Shape.of(9), false)));
        IllegalArgumentException zero = assertThrows(
                IllegalArgumentException.class,
                () -> data.scatterNd(
                        tensor(DataType.INT64, Shape.of(5, 0), false),
                        tensor(DataType.FLOAT32, Shape.of(9), false)));
        IllegalArgumentException tooLarge = assertThrows(
                IllegalArgumentException.class,
                () -> data.scatterNd(
                        tensor(DataType.INT32, Shape.of(5, 4), false),
                        tensor(DataType.FLOAT32, Shape.of(9), false)));
        IllegalArgumentException updatesShape = assertThrows(
                IllegalArgumentException.class,
                () -> data.scatterNd(
                        tensor(DataType.INT32, Shape.of(5, 2), false),
                        tensor(DataType.FLOAT32, Shape.of(5, 3), false)));

        assertAll(
                () -> assertEquals(
                        "scatterNd tuple depth must be statically known", dynamic.getMessage()),
                () -> assertEquals(
                        "scatterNd tuple depth must be in [1, data rank - batchDimensions]: "
                                + "depth=0, maximum=3",
                        zero.getMessage()),
                () -> assertEquals(
                        "scatterNd tuple depth must be in [1, data rank - batchDimensions]: "
                                + "depth=4, maximum=3",
                        tooLarge.getMessage()),
                () -> assertEquals(
                        "scatterNd updates shape must equal indices prefix plus data suffix: "
                                + "expected=Shape[5, 4], actual=Shape[5, 3]",
                        updatesShape.getMessage()));
    }

    @Test
    void derivesAllShapeExamplesAndCanonicalScalarWithExactDimensions() {
        Tensor zeroBatchData = tensor(DataType.FLOAT64, Shape.of(2, 3, 4), true);
        Tensor zeroBatchIndices = tensor(DataType.INT32, Shape.of(5, 2), false);
        Tensor zeroBatchUpdates = tensor(DataType.FLOAT64, Shape.of(5, 4), false);
        assertSame(
                zeroBatchData.descriptor().shape(),
                zeroBatchData.scatterNd(zeroBatchIndices, zeroBatchUpdates)
                        .descriptor().shape());

        DynamicDimension batch = new DynamicDimension("N");
        DynamicDimension queries = new DynamicDimension("M");
        StaticDimension suffix = new StaticDimension(4);
        Shape dataShape = Shape.ofDimensions(batch, new StaticDimension(3), suffix);
        Shape indicesShape = Shape.ofDimensions(batch, queries, new StaticDimension(1));
        Shape updatesShape = Shape.ofDimensions(batch, queries, suffix);
        Tensor data = tensor(DataType.BFLOAT16, dataShape, true);
        Tensor result = data.scatterNd(
                tensor(DataType.INT64, indicesShape, false),
                tensor(DataType.BFLOAT16, updatesShape, true),
                ScatterReduction.MAX,
                1);

        Tensor scalarUpdates = tensor(DataType.BOOL, Shape.scalar(), false);
        Tensor scalarResult = tensor(DataType.BOOL, Shape.of(2, 3), false).scatterNd(
                tensor(DataType.INT32, Shape.of(2), false), scalarUpdates);

        assertAll(
                () -> assertSame(dataShape, result.descriptor().shape()),
                () -> assertSame(data, result.provenance().orElseThrow().inputs().get(0)),
                () -> assertSame(Shape.scalar(), scalarUpdates.descriptor().shape()),
                () -> assertEquals(Shape.of(2, 3), scalarResult.descriptor().shape()),
                () -> assertSame(suffix, updatesShape.dimensions().get(2)));
    }

    @Test
    void discardsLayoutsAndDoesNotReadOrMutateInputStorageOrMetadata() {
        float[] dataValues = {1, 2, 3, 4, 5, 6};
        int[] indexValues = {99, 99};
        float[] updateValues = {7};
        Tensor data = storedTensor(
                DataType.FLOAT32, Shape.of(2, 3), dataValues, Optional.of("data"), true);
        Tensor indices = storedTensor(
                DataType.INT32, Shape.of(2), indexValues, Optional.of("indices"), false);
        Tensor updates = storedTensor(
                DataType.FLOAT32, Shape.scalar(), updateValues, Optional.of("updates"), true);

        LayoutDescriptor dataLayout = data.descriptor().layout().orElseThrow();
        LayoutDescriptor indicesLayout = indices.descriptor().layout().orElseThrow();
        LayoutDescriptor updatesLayout = updates.descriptor().layout().orElseThrow();
        HostTensorStorage dataStorage = data.hostStorage().orElseThrow();
        HostTensorStorage indicesStorage = indices.hostStorage().orElseThrow();
        HostTensorStorage updatesStorage = updates.hostStorage().orElseThrow();
        Tensor result = data.scatterNd(indices, updates, ScatterReduction.NONE);

        assertAll(
                () -> assertSame(dataLayout, data.descriptor().layout().orElseThrow()),
                () -> assertSame(indicesLayout, indices.descriptor().layout().orElseThrow()),
                () -> assertSame(updatesLayout, updates.descriptor().layout().orElseThrow()),
                () -> assertSame(dataStorage, data.hostStorage().orElseThrow()),
                () -> assertSame(indicesStorage, indices.hostStorage().orElseThrow()),
                () -> assertSame(updatesStorage, updates.hostStorage().orElseThrow()),
                () -> assertEquals(List.of(1f, 2f, 3f, 4f, 5f, 6f),
                        List.of(
                                dataValues[0], dataValues[1], dataValues[2],
                                dataValues[3], dataValues[4], dataValues[5])),
                () -> assertEquals(List.of(99, 99), List.of(indexValues[0], indexValues[1])),
                () -> assertEquals(List.of(7f), List.of(updateValues[0])),
                () -> assertTrue(result.descriptor().layout().isEmpty()),
                () -> assertTrue(result.label().isEmpty()),
                () -> assertTrue(result.hostStorage().isEmpty()));
    }

    @Test
    void repeatedAndNestedRequestsAreFreshAndIdentifierExhaustionPropagates() throws Exception {
        Tensor data = tensor(DataType.FLOAT64, Shape.of(2, 3), true);
        Tensor indices = tensor(DataType.INT32, Shape.of(4, 1), false);
        Tensor updates = tensor(DataType.FLOAT64, Shape.of(4, 3), true);
        Tensor first = data.scatterNd(indices, updates, ScatterReduction.ADD);
        Tensor second = data.scatterNd(indices, updates, ScatterReduction.ADD);
        Tensor nestedIndices = tensor(DataType.INT64, Shape.of(5, 1), false);
        Tensor nestedUpdates = tensor(DataType.FLOAT64, Shape.of(5, 3), false);
        Tensor nested = first.scatterNd(nestedIndices, nestedUpdates);

        assertAll(
                () -> assertNotSame(first, second),
                () -> assertNotEquals(first.id(), second.id()),
                () -> assertSame(first, nested.provenance().orElseThrow().inputs().get(0)),
                () -> assertSame(
                        nestedIndices, nested.provenance().orElseThrow().inputs().get(1)),
                () -> assertSame(
                        nestedUpdates, nested.provenance().orElseThrow().inputs().get(2)));

        AtomicLong next = nextTensorIdState();
        AtomicBoolean claimed = maximumClaimedState();
        long originalNext = next.get();
        boolean originalClaimed = claimed.get();
        try {
            next.set(Long.MAX_VALUE);
            claimed.set(true);
            IllegalStateException failure = assertThrows(
                    IllegalStateException.class, () -> data.scatterNd(indices, updates));
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
            Tensor result,
            int batchDimensions,
            ScatterReduction reduction,
            Tensor data,
            Tensor indices,
            Tensor updates) {
        TensorProvenance provenance = result.provenance().orElseThrow();
        Operation operation = provenance.operation();
        assertAll(
                () -> assertSame(ScatterNdKind.SCATTER_ND, operation.kind()),
                () -> assertEquals(
                        new ScatterNdAttrs(batchDimensions, reduction), operation.attrs()),
                () -> assertEquals(List.of(data, indices, updates), provenance.inputs()),
                () -> assertSame(data, provenance.inputs().get(0)),
                () -> assertSame(indices, provenance.inputs().get(1)),
                () -> assertSame(updates, provenance.inputs().get(2)));
    }

    private static void assertResult(Tensor result, Tensor data, boolean requiresGrad) {
        assertAll(
                () -> assertSame(data.descriptor().shape(), result.descriptor().shape()),
                () -> assertSame(data.descriptor().dataType(), result.descriptor().dataType()),
                () -> assertEquals(requiresGrad, result.descriptor().requiresGrad()),
                () -> assertTrue(result.descriptor().layout().isEmpty()),
                () -> assertTrue(result.label().isEmpty()),
                () -> assertTrue(result.hostStorage().isEmpty()));
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

    private static Tensor storedTensor(
            DataType dataType,
            Shape shape,
            Object values,
            Optional<String> label,
            boolean requiresGrad) {
        LayoutDescriptor layout = LayoutDescriptor.contiguous(shape);
        MemorySegment segment = switch (values) {
            case float[] array -> MemorySegment.ofArray(array);
            case int[] array -> MemorySegment.ofArray(array);
            default -> throw new AssertionError("unsupported test carrier");
        };
        HostTensorStorage storage = new MemorySegmentStorage(
                dataType, shape.knownElementCount().orElseThrow(), segment);
        return new Tensor(
                new TensorId(IDS.getAndIncrement()),
                new TensorDescriptor(dataType, shape, Optional.of(layout), requiresGrad),
                label,
                Optional.empty(),
                Optional.of(storage));
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
