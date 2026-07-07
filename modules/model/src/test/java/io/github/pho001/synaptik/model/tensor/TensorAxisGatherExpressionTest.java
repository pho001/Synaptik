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
import io.github.pho001.synaptik.model.operation.index.AxisGatherKind;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
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

class TensorAxisGatherExpressionTest {
    private static final AtomicLong IDS = new AtomicLong(80_000);

    @Test
    void exposesExactlyFourPublicMethodsAndNineMethodFieldFreeHelper() throws Exception {
        for (String methodName : List.of("gather", "gatherAxis", "takeAlongAxis")) {
            Method method = Tensor.class.getDeclaredMethod(
                    methodName, Tensor.class, int.class);
            assertPublicInstance(method, List.of(Tensor.class, int.class));
        }
        Method take = Tensor.class.getDeclaredMethod("take", int.class, Tensor.class);
        assertPublicInstance(take, List.of(int.class, Tensor.class));

        Method gather = TensorAxisGatherExpressions.class.getDeclaredMethod(
                "gather", Tensor.class, Tensor.class, int.class);
        Method gatherAxis = TensorAxisGatherExpressions.class.getDeclaredMethod(
                "gatherAxis", Tensor.class, Tensor.class, int.class);
        Method helperTake = TensorAxisGatherExpressions.class.getDeclaredMethod(
                "take", Tensor.class, int.class, Tensor.class);
        Method takeAlongAxis = TensorAxisGatherExpressions.class.getDeclaredMethod(
                "takeAlongAxis", Tensor.class, Tensor.class, int.class);
        Method validateIndexType = TensorAxisGatherExpressions.class.getDeclaredMethod(
                "validateIndexType", String.class, TensorDescriptor.class);
        Method removeAxis = TensorAxisGatherExpressions.class.getDeclaredMethod(
                "removeAxis", Shape.class, int.class);
        Method gatherAxisShape = TensorAxisGatherExpressions.class.getDeclaredMethod(
                "gatherAxisShape", Shape.class, Shape.class, int.class);
        Method validateTakeAlongAxis = TensorAxisGatherExpressions.class.getDeclaredMethod(
                "validateTakeAlongAxis", Shape.class, Shape.class, int.class);
        Method create = TensorAxisGatherExpressions.class.getDeclaredMethod(
                "create",
                Tensor.class,
                Tensor.class,
                TensorDescriptor.class,
                Shape.class,
                AxisGatherKind.class,
                IndexAxisAttrs.class);
        var constructor = TensorAxisGatherExpressions.class.getDeclaredConstructor();

        assertAll(
                () -> assertTrue(Modifier.isFinal(
                        TensorAxisGatherExpressions.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(
                        TensorAxisGatherExpressions.class.getModifiers())),
                () -> assertEquals(0,
                        TensorAxisGatherExpressions.class.getDeclaredFields().length),
                () -> assertEquals(0,
                        TensorAxisGatherExpressions.class.getDeclaredClasses().length),
                () -> assertEquals(1,
                        TensorAxisGatherExpressions.class.getDeclaredConstructors().length),
                () -> assertTrue(Modifier.isPrivate(constructor.getModifiers())),
                () -> assertEquals(
                        List.of(
                                "create",
                                "gather",
                                "gatherAxis",
                                "gatherAxisShape",
                                "removeAxis",
                                "take",
                                "takeAlongAxis",
                                "validateIndexType",
                                "validateTakeAlongAxis"),
                        Arrays.stream(TensorAxisGatherExpressions.class.getDeclaredMethods())
                                .map(Method::getName)
                                .sorted()
                                .toList()),
                () -> assertPackagePrivateStatic(gather, Tensor.class),
                () -> assertPackagePrivateStatic(gatherAxis, Tensor.class),
                () -> assertPackagePrivateStatic(helperTake, Tensor.class),
                () -> assertPackagePrivateStatic(takeAlongAxis, Tensor.class),
                () -> assertPrivateStatic(validateIndexType, void.class),
                () -> assertPrivateStatic(removeAxis, Shape.class),
                () -> assertPrivateStatic(gatherAxisShape, Shape.class),
                () -> assertPrivateStatic(validateTakeAlongAxis, void.class),
                () -> assertPrivateStatic(create, Tensor.class));
    }

    @Test
    void delegatesToExactKindsAndNormalizesAxesWithTakeAsGatherAxisAlias() {
        Tensor data = tensor(DataType.FLOAT32, Shape.of(2, 3, 4), true);
        Tensor gatherIndices = tensor(DataType.INT32, Shape.of(2, 4), false);
        Tensor axisIndices = tensor(DataType.INT64, Shape.of(5), false);
        Tensor alongIndices = tensor(DataType.INT32, Shape.of(2, 7, 4), false);

        Tensor gathered = data.gather(gatherIndices, -2);
        Tensor gatheredAxis = data.gatherAxis(axisIndices, 1);
        Tensor taken = data.take(-2, axisIndices);
        Tensor takenAlong = data.takeAlongAxis(alongIndices, -2);

        assertOperation(gathered, AxisGatherKind.GATHER, 1, data, gatherIndices);
        assertOperation(gatheredAxis, AxisGatherKind.GATHER_AXIS, 1, data, axisIndices);
        assertOperation(taken, AxisGatherKind.GATHER_AXIS, 1, data, axisIndices);
        assertOperation(takenAlong, AxisGatherKind.TAKE_ALONG_AXIS, 1, data, alongIndices);
        assertAll(
                () -> assertEquals(gatheredAxis.descriptor(), taken.descriptor()),
                () -> assertNotSame(gatheredAxis, taken),
                () -> assertNotEquals(gatheredAxis.id(), taken.id()));
    }

    @Test
    void acceptsEveryDataTypeBothIndexTypesAndRetainsOnlyDataMetadata() {
        Shape dataShape = Shape.of(2, 3);
        Shape indicesShape = Shape.of(4);
        for (DataType dataType : DataType.values()) {
            for (boolean requiresGrad : dataType.isDifferentiable()
                    ? List.of(false, true)
                    : List.of(false)) {
                for (DataType indexType : List.of(DataType.INT32, DataType.INT64)) {
                    Tensor data = tensor(dataType, dataShape, requiresGrad);
                    Tensor indices = tensor(indexType, indicesShape, false);
                    Tensor result = data.gatherAxis(indices, 1);
                    TensorProvenance provenance = result.provenance().orElseThrow();

                    assertAll(
                            () -> assertSame(dataType, result.descriptor().dataType()),
                            () -> assertEquals(requiresGrad,
                                    result.descriptor().requiresGrad()),
                            () -> assertEquals(Shape.of(2, 4),
                                    result.descriptor().shape()),
                            () -> assertTrue(result.descriptor().layout().isEmpty()),
                            () -> assertTrue(result.label().isEmpty()),
                            () -> assertTrue(result.hostStorage().isEmpty()),
                            () -> assertEquals(List.of(data, indices), provenance.inputs()));
                }
            }
        }
    }

    @Test
    void gatherRemovesAxisRetainsDataDimensionsAndRequiresExactReducedShape() {
        DynamicDimension batch = new DynamicDimension("batch");
        StaticDimension feature = new StaticDimension(4);
        Shape dataShape = Shape.ofDimensions(batch, new StaticDimension(3), feature);
        Shape indicesShape = Shape.ofDimensions(batch, feature);
        Tensor data = tensor(DataType.FLOAT64, dataShape, true);
        Tensor indices = tensor(DataType.INT32, indicesShape, false);

        Tensor result = data.gather(indices, 1);
        Tensor rankOneResult = tensor(DataType.BOOL, Shape.of(5), false)
                .gather(tensor(DataType.INT64, Shape.scalar(), false), 0);
        IllegalArgumentException mismatch = assertThrows(
                IllegalArgumentException.class,
                () -> data.gather(tensor(DataType.INT32, Shape.of(2, 4), false), 1));

        assertAll(
                () -> assertEquals(indicesShape, result.descriptor().shape()),
                () -> assertNotSame(indicesShape, result.descriptor().shape()),
                () -> assertSame(batch,
                        result.descriptor().shape().dimensions().get(0)),
                () -> assertSame(feature,
                        result.descriptor().shape().dimensions().get(1)),
                () -> assertSame(Shape.scalar(), rankOneResult.descriptor().shape()),
                () -> assertEquals(
                        "gather indices shape must equal data shape without gathered axis: "
                                + "expected=Shape[batch, 4], actual=Shape[2, 4]",
                        mismatch.getMessage()));
    }

    @Test
    void gatherAxisInsertsEveryExactIndicesDimensionIncludingScalarAndDynamicShapes() {
        DynamicDimension tail = new DynamicDimension("tail");
        DynamicDimension firstIndex = new DynamicDimension("query");
        StaticDimension secondIndex = new StaticDimension(6);
        Shape dataShape = Shape.ofDimensions(
                new StaticDimension(2), new StaticDimension(3), tail);
        Shape indicesShape = Shape.ofDimensions(firstIndex, secondIndex);
        Tensor data = tensor(DataType.INT64, dataShape, false);

        Shape result = data.gatherAxis(
                tensor(DataType.INT32, indicesShape, false), 1).descriptor().shape();
        Shape scalarResult = data.gatherAxis(
                tensor(DataType.INT64, Shape.scalar(), false), 1).descriptor().shape();

        assertAll(
                () -> assertEquals(Shape.ofDimensions(
                                dataShape.dimensions().get(0),
                                firstIndex,
                                secondIndex,
                                tail),
                        result),
                () -> assertSame(dataShape.dimensions().get(0),
                        result.dimensions().get(0)),
                () -> assertSame(firstIndex, result.dimensions().get(1)),
                () -> assertSame(secondIndex, result.dimensions().get(2)),
                () -> assertSame(tail, result.dimensions().get(3)),
                () -> assertEquals(Shape.ofDimensions(
                                dataShape.dimensions().get(0), tail),
                        scalarResult));
    }

    @Test
    void takeAlongAxisRetainsExactIndicesShapeAndChecksRankThenNonAxisDimensions() {
        DynamicDimension batch = new DynamicDimension("batch");
        DynamicDimension selected = new DynamicDimension("selected");
        StaticDimension tail = new StaticDimension(4);
        Shape dataShape = Shape.ofDimensions(batch, selected, tail);
        Shape indicesShape = Shape.ofDimensions(batch, new StaticDimension(7), tail);
        Tensor data = tensor(DataType.BFLOAT16, dataShape, true);
        Tensor indices = tensor(DataType.INT64, indicesShape, false);

        Tensor result = data.takeAlongAxis(indices, 1);
        IllegalArgumentException rank = assertThrows(
                IllegalArgumentException.class,
                () -> data.takeAlongAxis(
                        tensor(DataType.INT32, Shape.of(2, 3), false), 1));
        DynamicDimension otherBatch = new DynamicDimension("other");
        IllegalArgumentException dimension = assertThrows(
                IllegalArgumentException.class,
                () -> data.takeAlongAxis(
                        tensor(
                                DataType.INT32,
                                Shape.ofDimensions(otherBatch, new StaticDimension(2), tail),
                                false),
                        1));

        assertAll(
                () -> assertSame(indicesShape, result.descriptor().shape()),
                () -> assertEquals(
                        "takeAlongAxis indices rank must match data rank: expected=3, actual=2",
                        rank.getMessage()),
                () -> assertEquals(
                        "takeAlongAxis indices dimension at axis 0 must match data: expected="
                                + batch + ", actual=" + otherBatch,
                        dimension.getMessage()));
    }

    @Test
    void validatesNullTypeAxisAndShapeInExactOrderWithoutConsumingIdentity() throws Exception {
        AtomicLong next = nextTensorIdState();
        Tensor data = tensor(DataType.FLOAT32, Shape.of(2, 3), true);
        Tensor floatingIndices = tensor(DataType.FLOAT64, Shape.of(9), false);
        long before = next.get();

        NullPointerException nullData = assertThrows(
                NullPointerException.class,
                () -> TensorAxisGatherExpressions.gather(null, null, 9));
        NullPointerException nullIndices = assertThrows(
                NullPointerException.class,
                () -> TensorAxisGatherExpressions.gather(data, null, 9));
        IllegalArgumentException gatherType = assertThrows(
                IllegalArgumentException.class, () -> data.gather(floatingIndices, 9));
        IllegalArgumentException gatherAxisType = assertThrows(
                IllegalArgumentException.class, () -> data.gatherAxis(floatingIndices, 9));
        IllegalArgumentException takeType = assertThrows(
                IllegalArgumentException.class, () -> data.take(9, floatingIndices));
        IllegalArgumentException alongType = assertThrows(
                IllegalArgumentException.class, () -> data.takeAlongAxis(floatingIndices, 9));
        Tensor validIndices = tensor(DataType.INT32, Shape.of(9), false);
        IndexOutOfBoundsException axis = assertThrows(
                IndexOutOfBoundsException.class, () -> data.gatherAxis(validIndices, 2));
        IllegalArgumentException shape = assertThrows(
                IllegalArgumentException.class, () -> data.gather(validIndices, 1));

        assertAll(
                () -> assertEquals("data", nullData.getMessage()),
                () -> assertEquals("indices", nullIndices.getMessage()),
                () -> assertEquals(
                        "gather indices data type must be INT32 or INT64: FLOAT64",
                        gatherType.getMessage()),
                () -> assertEquals(
                        "gatherAxis indices data type must be INT32 or INT64: FLOAT64",
                        gatherAxisType.getMessage()),
                () -> assertEquals(
                        "gatherAxis indices data type must be INT32 or INT64: FLOAT64",
                        takeType.getMessage()),
                () -> assertEquals(
                        "takeAlongAxis indices data type must be INT32 or INT64: FLOAT64",
                        alongType.getMessage()),
                () -> assertEquals("Axis 2 is outside shape rank 2", axis.getMessage()),
                () -> assertEquals(
                        "gather indices shape must equal data shape without gathered axis: "
                                + "expected=Shape[2], actual=Shape[9]",
                        shape.getMessage()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void rejectsEveryUnsupportedIndexTypeAndScalarDataAxis() {
        Tensor data = tensor(DataType.FLOAT32, Shape.of(2), true);
        for (DataType indexType : List.of(
                DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16, DataType.BOOL)) {
            Tensor indices = tensor(indexType, Shape.scalar(), false);
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class, () -> data.gather(indices, 0));
            assertEquals(
                    "gather indices data type must be INT32 or INT64: " + indexType,
                    failure.getMessage());
        }

        IndexOutOfBoundsException scalar = assertThrows(
                IndexOutOfBoundsException.class,
                () -> tensor(DataType.FLOAT32, Shape.scalar(), false)
                        .gatherAxis(tensor(DataType.INT32, Shape.scalar(), false), 0));
        assertEquals("Axis 0 is outside shape rank 0", scalar.getMessage());
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

        Tensor result = data.gather(indices, 1);

        assertAll(
                () -> assertSame(dataLayout, data.descriptor().layout().orElseThrow()),
                () -> assertSame(indicesLayout, indices.descriptor().layout().orElseThrow()),
                () -> assertSame(dataStorage, data.hostStorage().orElseThrow()),
                () -> assertSame(indicesStorage, indices.hostStorage().orElseThrow()),
                () -> assertEquals(List.of(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f),
                        List.of(
                                dataValues[0], dataValues[1], dataValues[2], dataValues[3],
                                dataValues[4], dataValues[5])),
                () -> assertEquals(List.of(1, 0),
                        List.of(indexValues[0], indexValues[1])),
                () -> assertTrue(result.descriptor().layout().isEmpty()),
                () -> assertTrue(result.label().isEmpty()),
                () -> assertTrue(result.hostStorage().isEmpty()));
    }

    @Test
    void repeatedAndNestedRequestsAreFreshAndIdentifierExhaustionPropagates() throws Exception {
        Tensor data = tensor(DataType.FLOAT64, Shape.of(2, 3), true);
        Tensor indices = tensor(DataType.INT32, Shape.of(4), false);
        Tensor first = data.gatherAxis(indices, 1);
        Tensor second = data.gatherAxis(indices, 1);
        Tensor nestedIndices = tensor(DataType.INT64, Shape.of(5), false);
        Tensor nested = first.take(1, nestedIndices);

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
                    IllegalStateException.class, () -> data.gatherAxis(indices, 1));
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
            AxisGatherKind kind,
            int axis,
            Tensor data,
            Tensor indices) {
        TensorProvenance provenance = result.provenance().orElseThrow();
        Operation operation = provenance.operation();
        assertAll(
                () -> assertSame(kind, operation.kind()),
                () -> assertEquals(new IndexAxisAttrs(axis), operation.attrs()),
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
