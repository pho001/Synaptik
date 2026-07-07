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
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.index.AxisScatterKind;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterElementsAttrs;
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

class TensorAxisScatterExpressionTest {
    private static final AtomicLong IDS = new AtomicLong(100_000);

    @Test
    void exposesExactlyFourPublicMethodsAndElevenMethodFieldFreeHelper() throws Exception {
        for (String methodName : List.of("scatterAdd", "scatterAxisAdd")) {
            Method method = Tensor.class.getDeclaredMethod(
                    methodName, Tensor.class, Tensor.class, int.class);
            assertPublicInstance(
                    method, List.of(Tensor.class, Tensor.class, int.class));
        }
        Method shortScatterElements = Tensor.class.getDeclaredMethod(
                "scatterElements", Tensor.class, Tensor.class, int.class);
        Method explicitScatterElements = Tensor.class.getDeclaredMethod(
                "scatterElements",
                Tensor.class,
                Tensor.class,
                int.class,
                ScatterReduction.class);
        assertPublicInstance(
                shortScatterElements, List.of(Tensor.class, Tensor.class, int.class));
        assertPublicInstance(
                explicitScatterElements,
                List.of(Tensor.class, Tensor.class, int.class, ScatterReduction.class));

        Method scatterAdd = TensorAxisScatterExpressions.class.getDeclaredMethod(
                "scatterAdd", Tensor.class, Tensor.class, Tensor.class, int.class);
        Method scatterAxisAdd = TensorAxisScatterExpressions.class.getDeclaredMethod(
                "scatterAxisAdd", Tensor.class, Tensor.class, Tensor.class, int.class);
        Method helperShort = TensorAxisScatterExpressions.class.getDeclaredMethod(
                "scatterElements", Tensor.class, Tensor.class, Tensor.class, int.class);
        Method helperExplicit = TensorAxisScatterExpressions.class.getDeclaredMethod(
                "scatterElements",
                Tensor.class,
                Tensor.class,
                Tensor.class,
                int.class,
                ScatterReduction.class);
        Method validateIndexType = TensorAxisScatterExpressions.class.getDeclaredMethod(
                "validateIndexType", String.class, TensorDescriptor.class);
        Method validateMatchingDataType = TensorAxisScatterExpressions.class.getDeclaredMethod(
                "validateMatchingDataType",
                String.class,
                TensorDescriptor.class,
                TensorDescriptor.class);
        Method validateFloating = TensorAxisScatterExpressions.class.getDeclaredMethod(
                "validateFloating", String.class, TensorDescriptor.class);
        Method removeAxis = TensorAxisScatterExpressions.class.getDeclaredMethod(
                "removeAxis", Shape.class, int.class);
        Method gatherAxisShape = TensorAxisScatterExpressions.class.getDeclaredMethod(
                "gatherAxisShape", Shape.class, Shape.class, int.class);
        Method validateScatterElementsShape = TensorAxisScatterExpressions.class.getDeclaredMethod(
                "validateScatterElementsShape",
                Shape.class,
                Shape.class,
                Shape.class,
                int.class);
        Method create = TensorAxisScatterExpressions.class.getDeclaredMethod(
                "create",
                Tensor.class,
                Tensor.class,
                Tensor.class,
                TensorDescriptor.class,
                TensorDescriptor.class,
                AxisScatterKind.class,
                OperationAttrs.class);
        var constructor = TensorAxisScatterExpressions.class.getDeclaredConstructor();

        assertAll(
                () -> assertTrue(Modifier.isFinal(
                        TensorAxisScatterExpressions.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(
                        TensorAxisScatterExpressions.class.getModifiers())),
                () -> assertEquals(0,
                        TensorAxisScatterExpressions.class.getDeclaredFields().length),
                () -> assertEquals(0,
                        TensorAxisScatterExpressions.class.getDeclaredClasses().length),
                () -> assertEquals(1,
                        TensorAxisScatterExpressions.class.getDeclaredConstructors().length),
                () -> assertTrue(Modifier.isPrivate(constructor.getModifiers())),
                () -> assertEquals(
                        List.of(
                                "create",
                                "gatherAxisShape",
                                "removeAxis",
                                "scatterAdd",
                                "scatterAxisAdd",
                                "scatterElements",
                                "scatterElements",
                                "validateFloating",
                                "validateIndexType",
                                "validateMatchingDataType",
                                "validateScatterElementsShape"),
                        Arrays.stream(TensorAxisScatterExpressions.class.getDeclaredMethods())
                                .map(Method::getName)
                                .sorted()
                                .toList()),
                () -> assertPackagePrivateStatic(scatterAdd, Tensor.class),
                () -> assertPackagePrivateStatic(scatterAxisAdd, Tensor.class),
                () -> assertPackagePrivateStatic(helperShort, Tensor.class),
                () -> assertPackagePrivateStatic(helperExplicit, Tensor.class),
                () -> assertPrivateStatic(validateIndexType, void.class),
                () -> assertPrivateStatic(validateMatchingDataType, void.class),
                () -> assertPrivateStatic(validateFloating, void.class),
                () -> assertPrivateStatic(removeAxis, Shape.class),
                () -> assertPrivateStatic(gatherAxisShape, Shape.class),
                () -> assertPrivateStatic(validateScatterElementsShape, void.class),
                () -> assertPrivateStatic(create, Tensor.class));
    }

    @Test
    void createsExactKindsNormalizesAxesAndDefaultsScatterElementsToNone() {
        Tensor data = tensor(DataType.FLOAT32, Shape.of(2, 3, 4), true);
        Tensor reducedIndices = tensor(DataType.INT32, Shape.of(2, 4), false);
        Tensor reducedUpdates = tensor(DataType.FLOAT32, Shape.of(2, 4), false);
        Tensor axisIndices = tensor(DataType.INT64, Shape.of(5), false);
        Tensor axisUpdates = tensor(DataType.FLOAT32, Shape.of(2, 5, 4), true);
        Tensor elementsIndices = tensor(DataType.INT32, Shape.of(2, 7, 4), false);
        Tensor elementsUpdates = tensor(DataType.FLOAT32, Shape.of(2, 7, 4), false);

        Tensor scatterAdd = data.scatterAdd(reducedIndices, reducedUpdates, -2);
        Tensor scatterAxisAdd = data.scatterAxisAdd(axisIndices, axisUpdates, -2);
        Tensor replacement = data.scatterElements(elementsIndices, elementsUpdates, -2);
        Tensor multiplication = data.scatterElements(
                elementsIndices, elementsUpdates, 1, ScatterReduction.MUL);

        assertOperation(
                scatterAdd,
                AxisScatterKind.SCATTER_ADD,
                new IndexAxisAttrs(1),
                data,
                reducedIndices,
                reducedUpdates);
        assertOperation(
                scatterAxisAdd,
                AxisScatterKind.SCATTER_AXIS_ADD,
                new IndexAxisAttrs(1),
                data,
                axisIndices,
                axisUpdates);
        assertOperation(
                replacement,
                AxisScatterKind.SCATTER_ELEMENTS,
                new ScatterElementsAttrs(1, ScatterReduction.NONE),
                data,
                elementsIndices,
                elementsUpdates);
        assertOperation(
                multiplication,
                AxisScatterKind.SCATTER_ELEMENTS,
                new ScatterElementsAttrs(1, ScatterReduction.MUL),
                data,
                elementsIndices,
                elementsUpdates);
    }

    @Test
    void acceptsExactTypeDomainsAndCombinesOnlyDataAndUpdateEligibility() {
        Shape dataShape = Shape.of(2, 3);
        Shape elementsShape = Shape.of(2, 4);
        for (DataType indexType : List.of(DataType.INT32, DataType.INT64)) {
            for (DataType dataType : DataType.values()) {
                Tensor data = tensor(dataType, dataShape, false);
                Tensor indices = tensor(indexType, elementsShape, false);
                Tensor updates = tensor(
                        dataType, elementsShape, dataType.isDifferentiable());
                Tensor replacement = data.scatterElements(indices, updates, 1);
                assertResult(replacement, data, dataType.isDifferentiable());

                if (!dataType.isBoolean()) {
                    for (ScatterReduction reduction : List.of(
                            ScatterReduction.ADD,
                            ScatterReduction.MUL,
                            ScatterReduction.MAX,
                            ScatterReduction.MIN)) {
                        Tensor arithmetic = data.scatterElements(
                                indices, updates, 1, reduction);
                        assertResult(arithmetic, data, dataType.isDifferentiable());
                    }
                }
            }
        }

        for (DataType floating : List.of(
                DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16)) {
            Tensor data = tensor(floating, dataShape, false);
            Tensor indices = tensor(DataType.INT32, Shape.of(2), false);
            Tensor updates = tensor(floating, Shape.of(2), true);
            assertResult(data.scatterAdd(indices, updates, 1), data, true);
        }
    }

    @Test
    void scatterAddRemovesAxisRetainsDimensionsAndValidatesIndicesBeforeUpdates() {
        DynamicDimension batch = new DynamicDimension("batch");
        StaticDimension feature = new StaticDimension(4);
        Shape dataShape = Shape.ofDimensions(batch, new StaticDimension(3), feature);
        Shape reducedShape = Shape.ofDimensions(batch, feature);
        Tensor data = tensor(DataType.FLOAT64, dataShape, true);
        Tensor indices = tensor(DataType.INT32, reducedShape, false);
        Tensor updates = tensor(DataType.FLOAT64, reducedShape, false);

        Tensor result = data.scatterAdd(indices, updates, 1);
        Tensor scalar = tensor(DataType.FLOAT32, Shape.of(5), false).scatterAdd(
                tensor(DataType.INT64, Shape.scalar(), false),
                tensor(DataType.FLOAT32, Shape.scalar(), false),
                0);
        IllegalArgumentException indicesFailure = assertThrows(
                IllegalArgumentException.class,
                () -> data.scatterAdd(
                        tensor(DataType.INT32, Shape.of(2, 4), false),
                        tensor(DataType.FLOAT64, Shape.of(9), false),
                        1));
        IllegalArgumentException updatesFailure = assertThrows(
                IllegalArgumentException.class,
                () -> data.scatterAdd(
                        indices, tensor(DataType.FLOAT64, Shape.of(2, 4), false), 1));

        assertAll(
                () -> assertSame(dataShape, result.descriptor().shape()),
                () -> assertEquals(Shape.of(5), scalar.descriptor().shape()),
                () -> assertEquals(
                        "scatterAdd indices shape must equal data shape without scattered axis: "
                                + "expected=Shape[batch, 4], actual=Shape[2, 4]",
                        indicesFailure.getMessage()),
                () -> assertEquals(
                        "scatterAdd updates shape must equal data shape without scattered axis: "
                                + "expected=Shape[batch, 4], actual=Shape[2, 4]",
                        updatesFailure.getMessage()));
    }

    @Test
    void scatterAxisAddRequiresGatherAxisShapeIncludingScalarAndDynamicIndices() {
        DynamicDimension tail = new DynamicDimension("tail");
        DynamicDimension query = new DynamicDimension("query");
        StaticDimension six = new StaticDimension(6);
        Shape dataShape = Shape.ofDimensions(
                new StaticDimension(2), new StaticDimension(3), tail);
        Tensor data = tensor(DataType.BFLOAT16, dataShape, true);
        Tensor indices = tensor(
                DataType.INT64, Shape.ofDimensions(query, six), false);
        Shape updatesShape = Shape.ofDimensions(
                dataShape.dimensions().get(0), query, six, tail);

        Tensor result = data.scatterAxisAdd(
                indices, tensor(DataType.BFLOAT16, updatesShape, false), 1);
        Tensor scalarIndicesResult = data.scatterAxisAdd(
                tensor(DataType.INT32, Shape.scalar(), false),
                tensor(
                        DataType.BFLOAT16,
                        Shape.ofDimensions(dataShape.dimensions().get(0), tail),
                        false),
                1);
        IllegalArgumentException failure = assertThrows(
                IllegalArgumentException.class,
                () -> data.scatterAxisAdd(
                        indices,
                        tensor(DataType.BFLOAT16, Shape.of(2, 6, 4), false),
                        1));

        assertAll(
                () -> assertSame(dataShape, result.descriptor().shape()),
                () -> assertSame(dataShape, scalarIndicesResult.descriptor().shape()),
                () -> assertEquals(
                        "scatterAxisAdd updates shape must match gatherAxis result shape: "
                                + "expected=Shape[2, query, 6, tail], actual=Shape[2, 6, 4]",
                        failure.getMessage()));
    }

    @Test
    void scatterElementsValidatesRanksThenUpdatesThenNonAxisDataDimensions() {
        DynamicDimension batch = new DynamicDimension("batch");
        StaticDimension tail = new StaticDimension(4);
        Tensor data = tensor(
                DataType.INT64,
                Shape.ofDimensions(batch, new StaticDimension(3), tail),
                false);
        Shape validShape = Shape.ofDimensions(batch, new StaticDimension(7), tail);
        Tensor validIndices = tensor(DataType.INT32, validShape, false);
        Tensor validUpdates = tensor(DataType.INT64, validShape, false);

        Tensor result = data.scatterElements(
                validIndices, validUpdates, 1, ScatterReduction.MAX);
        IllegalArgumentException indicesRank = assertThrows(
                IllegalArgumentException.class,
                () -> data.scatterElements(
                        tensor(DataType.INT32, Shape.of(2, 3), false),
                        tensor(DataType.INT64, Shape.of(9), false),
                        1));
        IllegalArgumentException updatesRank = assertThrows(
                IllegalArgumentException.class,
                () -> data.scatterElements(
                        validIndices,
                        tensor(DataType.INT64, Shape.of(2, 3), false),
                        1));
        Shape mismatchedUpdates = Shape.ofDimensions(batch, new StaticDimension(8), tail);
        IllegalArgumentException updatesDimension = assertThrows(
                IllegalArgumentException.class,
                () -> data.scatterElements(
                        validIndices,
                        tensor(DataType.INT64, mismatchedUpdates, false),
                        1));
        DynamicDimension otherBatch = new DynamicDimension("other");
        Shape mismatchedData = Shape.ofDimensions(otherBatch, new StaticDimension(7), tail);
        IllegalArgumentException dataDimension = assertThrows(
                IllegalArgumentException.class,
                () -> data.scatterElements(
                        tensor(DataType.INT32, mismatchedData, false),
                        tensor(DataType.INT64, mismatchedData, false),
                        1));

        assertAll(
                () -> assertSame(data.descriptor().shape(), result.descriptor().shape()),
                () -> assertEquals(
                        "scatterElements indices rank must match data rank: expected=3, actual=2",
                        indicesRank.getMessage()),
                () -> assertEquals(
                        "scatterElements updates rank must match indices rank: expected=3, actual=2",
                        updatesRank.getMessage()),
                () -> assertEquals(
                        "scatterElements updates dimension at axis 1 must match indices: "
                                + "expected=StaticDimension[size=7], "
                                + "actual=StaticDimension[size=8]",
                        updatesDimension.getMessage()),
                () -> assertEquals(
                        "scatterElements indices dimension at axis 0 must match data: "
                                + "expected=" + batch + ", actual=" + otherBatch,
                        dataDimension.getMessage()));
    }

    @Test
    void validatesNullTypeReductionAxisAndShapeInExactOrderWithoutConsumingIdentity()
            throws Exception {
        AtomicLong next = nextTensorIdState();
        Tensor data = tensor(DataType.FLOAT32, Shape.of(2, 3), true);
        Tensor floatingIndices = tensor(DataType.FLOAT64, Shape.of(2), false);
        Tensor wrongUpdates = tensor(DataType.FLOAT64, Shape.of(2), true);
        long before = next.get();

        NullPointerException nullData = assertThrows(
                NullPointerException.class,
                () -> TensorAxisScatterExpressions.scatterElements(
                        null, null, null, 9, null));
        NullPointerException nullIndices = assertThrows(
                NullPointerException.class,
                () -> TensorAxisScatterExpressions.scatterElements(
                        data, null, null, 9, null));
        NullPointerException nullUpdates = assertThrows(
                NullPointerException.class,
                () -> TensorAxisScatterExpressions.scatterElements(
                        data, floatingIndices, null, 9, null));
        NullPointerException nullReduction = assertThrows(
                NullPointerException.class,
                () -> TensorAxisScatterExpressions.scatterElements(
                        data, floatingIndices, wrongUpdates, 9, null));
        IllegalArgumentException indexType = assertThrows(
                IllegalArgumentException.class,
                () -> data.scatterElements(
                        floatingIndices, wrongUpdates, 9, ScatterReduction.ADD));
        Tensor validIndices = tensor(DataType.INT32, Shape.of(2), false);
        IllegalArgumentException updateType = assertThrows(
                IllegalArgumentException.class,
                () -> data.scatterElements(
                        validIndices, wrongUpdates, 9, ScatterReduction.ADD));
        Tensor validUpdates = tensor(DataType.FLOAT32, Shape.of(2), false);
        IndexOutOfBoundsException axis = assertThrows(
                IndexOutOfBoundsException.class,
                () -> data.scatterElements(
                        validIndices, validUpdates, 2, ScatterReduction.ADD));

        assertAll(
                () -> assertEquals("data", nullData.getMessage()),
                () -> assertEquals("indices", nullIndices.getMessage()),
                () -> assertEquals("updates", nullUpdates.getMessage()),
                () -> assertEquals("reduction", nullReduction.getMessage()),
                () -> assertEquals(
                        "scatterElements indices data type must be INT32 or INT64: FLOAT64",
                        indexType.getMessage()),
                () -> assertEquals(
                        "scatterElements updates data type must match data: "
                                + "expected=FLOAT32, actual=FLOAT64",
                        updateType.getMessage()),
                () -> assertEquals("Axis 2 is outside shape rank 2", axis.getMessage()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void rejectsUnsupportedIndexUpdateFixedAddAndBoolReductionTypesWithExactMessages() {
        Tensor floatingData = tensor(DataType.FLOAT32, Shape.of(2, 3), true);
        Tensor reducedUpdates = tensor(DataType.FLOAT32, Shape.of(2), false);
        for (DataType indexType : List.of(
                DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16, DataType.BOOL)) {
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class,
                    () -> floatingData.scatterAdd(
                            tensor(indexType, Shape.of(2), false), reducedUpdates, 1));
            assertEquals(
                    "scatterAdd indices data type must be INT32 or INT64: " + indexType,
                    failure.getMessage());
        }

        Tensor indices = tensor(DataType.INT32, Shape.of(2), false);
        IllegalArgumentException updateType = assertThrows(
                IllegalArgumentException.class,
                () -> floatingData.scatterAdd(
                        indices, tensor(DataType.FLOAT64, Shape.of(2), true), 1));
        assertEquals(
                "scatterAdd updates data type must match data: expected=FLOAT32, actual=FLOAT64",
                updateType.getMessage());

        for (DataType rejected : List.of(DataType.INT32, DataType.INT64, DataType.BOOL)) {
            Tensor data = tensor(rejected, Shape.of(2, 3), false);
            Tensor updates = tensor(rejected, Shape.of(2), false);
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class,
                    () -> data.scatterAdd(indices, updates, 1));
            assertEquals(
                    "scatterAdd data and updates must use floating data type: " + rejected,
                    failure.getMessage());
        }

        Tensor boolData = tensor(DataType.BOOL, Shape.of(2, 3), false);
        Tensor boolIndices = tensor(DataType.INT64, Shape.of(2, 4), false);
        Tensor boolUpdates = tensor(DataType.BOOL, Shape.of(2, 4), false);
        assertResult(boolData.scatterElements(boolIndices, boolUpdates, 1), boolData, false);
        for (ScatterReduction reduction : List.of(
                ScatterReduction.ADD,
                ScatterReduction.MUL,
                ScatterReduction.MAX,
                ScatterReduction.MIN)) {
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class,
                    () -> boolData.scatterElements(
                            boolIndices, boolUpdates, 1, reduction));
            assertEquals(
                    "scatterElements BOOL data supports only NONE reduction: " + reduction,
                    failure.getMessage());
        }
    }

    @Test
    void discardsLayoutsAndDoesNotReadOrMutateInputStorageOrMetadata() {
        float[] dataValues = {1, 2, 3, 4, 5, 6};
        int[] indexValues = {1, 0};
        float[] updateValues = {7, 8};
        Tensor data = storedTensor(
                DataType.FLOAT32, Shape.of(2, 3), dataValues, Optional.of("data"), true);
        Tensor indices = storedTensor(
                DataType.INT32, Shape.of(2), indexValues, Optional.of("indices"), false);
        Tensor updates = storedTensor(
                DataType.FLOAT32, Shape.of(2), updateValues, Optional.of("updates"), true);

        LayoutDescriptor dataLayout = data.descriptor().layout().orElseThrow();
        LayoutDescriptor indicesLayout = indices.descriptor().layout().orElseThrow();
        LayoutDescriptor updatesLayout = updates.descriptor().layout().orElseThrow();
        HostTensorStorage dataStorage = data.hostStorage().orElseThrow();
        HostTensorStorage indicesStorage = indices.hostStorage().orElseThrow();
        HostTensorStorage updatesStorage = updates.hostStorage().orElseThrow();
        Tensor result = data.scatterAdd(indices, updates, 1);

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
                () -> assertEquals(List.of(1, 0), List.of(indexValues[0], indexValues[1])),
                () -> assertEquals(List.of(7f, 8f), List.of(updateValues[0], updateValues[1])),
                () -> assertTrue(result.descriptor().layout().isEmpty()),
                () -> assertTrue(result.label().isEmpty()),
                () -> assertTrue(result.hostStorage().isEmpty()));
    }

    @Test
    void repeatedAndNestedRequestsAreFreshAndIdentifierExhaustionPropagates() throws Exception {
        Tensor data = tensor(DataType.FLOAT64, Shape.of(2, 3), true);
        Tensor indices = tensor(DataType.INT32, Shape.of(2, 4), false);
        Tensor updates = tensor(DataType.FLOAT64, Shape.of(2, 4), true);
        Tensor first = data.scatterElements(indices, updates, 1, ScatterReduction.ADD);
        Tensor second = data.scatterElements(indices, updates, 1, ScatterReduction.ADD);
        Tensor nestedIndices = tensor(DataType.INT64, Shape.of(2, 5), false);
        Tensor nestedUpdates = tensor(DataType.FLOAT64, Shape.of(2, 5), false);
        Tensor nested = first.scatterElements(nestedIndices, nestedUpdates, 1);

        assertAll(
                () -> assertNotSame(first, second),
                () -> assertNotEquals(first.id(), second.id()),
                () -> assertSame(first, nested.provenance().orElseThrow().inputs().get(0)),
                () -> assertSame(nestedIndices,
                        nested.provenance().orElseThrow().inputs().get(1)),
                () -> assertSame(nestedUpdates,
                        nested.provenance().orElseThrow().inputs().get(2)));

        AtomicLong next = nextTensorIdState();
        AtomicBoolean claimed = maximumClaimedState();
        long originalNext = next.get();
        boolean originalClaimed = claimed.get();
        try {
            next.set(Long.MAX_VALUE);
            claimed.set(true);
            IllegalStateException failure = assertThrows(
                    IllegalStateException.class,
                    () -> data.scatterElements(indices, updates, 1));
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
            AxisScatterKind kind,
            OperationAttrs attrs,
            Tensor data,
            Tensor indices,
            Tensor updates) {
        TensorProvenance provenance = result.provenance().orElseThrow();
        Operation operation = provenance.operation();
        assertAll(
                () -> assertSame(kind, operation.kind()),
                () -> assertEquals(attrs, operation.attrs()),
                () -> assertEquals(List.of(data, indices, updates), provenance.inputs()),
                () -> assertSame(data, provenance.inputs().get(0)),
                () -> assertSame(indices, provenance.inputs().get(1)),
                () -> assertSame(updates, provenance.inputs().get(2)));
    }

    private static void assertResult(
            Tensor result, Tensor data, boolean requiresGrad) {
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
