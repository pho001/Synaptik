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
import io.github.pho001.synaptik.model.operation.OperationSignature;
import io.github.pho001.synaptik.model.operation.index.AxisScatterKind;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterElementsAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
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

class TensorAxisScatterExpressionTest {
    private static final AtomicLong INPUT_IDS = new AtomicLong(40_000_000);

    @Test
    void exposesExactlyOneScatterAddMethodAndTheNineMethodFieldFreeHelper()
            throws ReflectiveOperationException {
        Method scatterAdd = Tensor.class.getDeclaredMethod(
                "scatterAdd", Tensor.class, Tensor.class, int.class);
        Set<String> scatterNames = Arrays.stream(Tensor.class.getDeclaredMethods())
                .map(Method::getName)
                .filter(name -> name.startsWith("scatter"))
                .collect(Collectors.toSet());
        Set<String> helperMethods = Arrays.stream(
                        TensorAxisScatterExpressions.class.getDeclaredMethods())
                .map(TensorAxisScatterExpressionTest::methodSignature)
                .collect(Collectors.toSet());
        var constructors = TensorAxisScatterExpressions.class.getDeclaredConstructors();

        assertAll(
                () -> assertEquals(
                        Set.of("scatterAdd", "scatterElements", "scatterNd"), scatterNames),
                () -> assertEquals(1, Arrays.stream(Tensor.class.getDeclaredMethods())
                        .filter(method -> method.getName().equals("scatterAdd"))
                        .count()),
                () -> assertEquals(Tensor.class, scatterAdd.getReturnType()),
                () -> assertTrue(Modifier.isPublic(scatterAdd.getModifiers())),
                () -> assertFalse(Modifier.isStatic(scatterAdd.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(scatterAdd.getModifiers())),
                () -> assertFalse(scatterAdd.isVarArgs()),
                () -> assertTrue(Modifier.isFinal(
                        TensorAxisScatterExpressions.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(
                        TensorAxisScatterExpressions.class.getModifiers())),
                () -> assertFalse(TensorAxisScatterExpressions.class.isRecord()),
                () -> assertEquals(0, TensorAxisScatterExpressions.class.getDeclaredFields().length),
                () -> assertEquals(0, TensorAxisScatterExpressions.class.getDeclaredClasses().length),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPrivate(constructors[0].getModifiers())),
                () -> assertEquals(0, constructors[0].getParameterCount()),
                () -> assertEquals(
                        Set.of(
                                "scatterAdd(io.github.pho001.synaptik.model.tensor.Tensor,io.github.pho001.synaptik.model.tensor.Tensor,io.github.pho001.synaptik.model.tensor.Tensor,int):io.github.pho001.synaptik.model.tensor.Tensor",
                                "scatterElements(io.github.pho001.synaptik.model.tensor.Tensor,io.github.pho001.synaptik.model.tensor.Tensor,io.github.pho001.synaptik.model.tensor.Tensor,int):io.github.pho001.synaptik.model.tensor.Tensor",
                                "scatterElements(io.github.pho001.synaptik.model.tensor.Tensor,io.github.pho001.synaptik.model.tensor.Tensor,io.github.pho001.synaptik.model.tensor.Tensor,int,io.github.pho001.synaptik.model.operation.index.ScatterReduction):io.github.pho001.synaptik.model.tensor.Tensor",
                                "validateIndexType(java.lang.String,io.github.pho001.synaptik.model.tensor.TensorDescriptor):void",
                                "validateMatchingDataType(java.lang.String,io.github.pho001.synaptik.model.tensor.TensorDescriptor,io.github.pho001.synaptik.model.tensor.TensorDescriptor):void",
                                "validateAddDataType(io.github.pho001.synaptik.model.tensor.TensorDescriptor):void",
                                "gatherResultShape(io.github.pho001.synaptik.model.shape.Shape,io.github.pho001.synaptik.model.shape.Shape,int):io.github.pho001.synaptik.model.shape.Shape",
                                "validateScatterElementsShape(io.github.pho001.synaptik.model.shape.Shape,io.github.pho001.synaptik.model.shape.Shape,io.github.pho001.synaptik.model.shape.Shape,int):void",
                                "create(io.github.pho001.synaptik.model.tensor.Tensor,io.github.pho001.synaptik.model.tensor.Tensor,io.github.pho001.synaptik.model.tensor.Tensor,io.github.pho001.synaptik.model.tensor.TensorDescriptor,io.github.pho001.synaptik.model.tensor.TensorDescriptor,io.github.pho001.synaptik.model.operation.Operation):io.github.pho001.synaptik.model.tensor.Tensor"),
                        helperMethods));
    }

    @Test
    void buildsStaticGatherCompatibleShapeAndExactMetadataForEveryNumericType() {
        for (DataType dataType : List.of(
                DataType.BFLOAT16,
                DataType.FLOAT32,
                DataType.FLOAT64,
                DataType.INT32,
                DataType.INT64)) {
            for (DataType indexType : List.of(DataType.INT32, DataType.INT64)) {
                Shape dataShape = Shape.of(2, 3, 4);
                Shape indicesShape = Shape.of(5, 6);
                Shape updatesShape = Shape.of(2, 5, 6, 4);
                Tensor data = tensor(dataType, dataShape, dataType.isDifferentiable(), Optional.empty());
                Tensor indices = tensor(indexType, indicesShape, false, Optional.empty());
                Tensor updates = tensor(dataType, updatesShape, false, Optional.empty());

                Tensor result = data.scatterAdd(indices, updates, -2);
                TensorProvenance provenance = result.provenance().orElseThrow();

                assertAll(
                        () -> assertSame(dataType, result.descriptor().dataType()),
                        () -> assertSame(dataShape, result.descriptor().shape()),
                        () -> assertEquals(dataType.isDifferentiable(),
                                result.descriptor().requiresGrad()),
                        () -> assertTrue(result.descriptor().layout().isEmpty()),
                        () -> assertTrue(result.label().isEmpty()),
                        () -> assertTrue(result.hostStorage().isEmpty()),
                        () -> assertSame(AxisScatterKind.SCATTER_ADD,
                                provenance.operation().kind()),
                        () -> assertEquals(new IndexAxisAttrs(1),
                                provenance.operation().attrs()),
                        () -> assertEquals(
                                OperationSignature.fixed(IndexAxisAttrs.class, 3, 1),
                                provenance.operation().signature()),
                        () -> assertEquals(List.of(data, indices, updates), provenance.inputs()),
                        () -> assertEquals(0, provenance.outputIndex()),
                        () -> assertEquals(1, provenance.producer().outputCount()),
                        () -> assertSame(result.descriptor(), provenance.outputDescriptor()));
            }
        }
    }

    @Test
    void preservesDataAndUpdateEligibilityOrAndCreatesFreshProducers()
            throws ReflectiveOperationException {
        Shape dataShape = Shape.of(2, 3);
        Shape indicesShape = Shape.of(4);
        Shape updatesShape = Shape.of(2, 4);
        LayoutDescriptor dataLayout = LayoutDescriptor.contiguous(dataShape);
        Tensor data = tensor(DataType.FLOAT64, dataShape, false, Optional.of(dataLayout));
        Tensor indices = tensor(DataType.INT32, indicesShape, false, Optional.empty());
        Tensor updates = tensor(DataType.FLOAT64, updatesShape, true, Optional.empty());

        AtomicLong next = nextTensorIdState();
        long before = next.get();
        Tensor first = data.scatterAdd(indices, updates, 1);
        Tensor second = data.scatterAdd(indices, updates, 1);

        assertAll(
                () -> assertTrue(first.descriptor().requiresGrad()),
                () -> assertEquals(before, first.id().value()),
                () -> assertEquals(before + 1, second.id().value()),
                () -> assertEquals(before + 2, next.get()),
                () -> assertNotSame(first, second),
                () -> assertNotEquals(first.id(), second.id()),
                () -> assertNotSame(
                        first.provenance().orElseThrow().producer(),
                        second.provenance().orElseThrow().producer()),
                () -> assertSame(dataShape, data.descriptor().shape()),
                () -> assertSame(dataLayout, data.descriptor().layout().orElseThrow()),
                () -> assertFalse(data.descriptor().requiresGrad()),
                () -> assertTrue(data.provenance().isEmpty()),
                () -> assertTrue(indices.provenance().isEmpty()),
                () -> assertTrue(updates.provenance().isEmpty()));
    }

    @Test
    void acceptsScalarZeroNamedExpressionAndUnresolvedIndicesShapesExactly() {
        DynamicDimension batch = new DynamicDimension("batch");
        DynamicDimension vocabulary = new DynamicDimension("vocabulary");
        DynamicDimension width = new DynamicDimension("width");
        DynamicDimension tokens = new DynamicDimension("tokens");
        Dimension expression = DimensionExpressions.addConstant(tokens, 2);
        Dimension unknown = DimensionExpressions.unknown(0, Optional.empty());
        Shape dynamicDataShape = Shape.ofDimensions(batch, vocabulary, width);
        Tensor dynamicData = tensor(
                DataType.FLOAT32, dynamicDataShape, true, Optional.empty());

        Tensor scalar = dynamicData.scatterAdd(
                tensor(DataType.INT64, Shape.scalar(), false, Optional.empty()),
                tensor(
                        DataType.FLOAT32,
                        Shape.ofDimensions(batch, width),
                        false,
                        Optional.empty()),
                1);
        Tensor namedExpression = dynamicData.scatterAdd(
                tensor(
                        DataType.INT32,
                        Shape.ofDimensions(tokens, expression),
                        false,
                        Optional.empty()),
                tensor(
                        DataType.FLOAT32,
                        Shape.ofDimensions(batch, tokens, expression, width),
                        false,
                        Optional.empty()),
                1);
        Tensor unresolved = dynamicData.scatterAdd(
                tensor(DataType.INT64, Shape.ofDimensions(unknown), false, Optional.empty()),
                tensor(
                        DataType.FLOAT32,
                        Shape.ofDimensions(batch, unknown, width),
                        false,
                        Optional.empty()),
                1);
        Tensor zero = tensor(DataType.INT64, Shape.of(2, 3, 4), false, Optional.empty())
                .scatterAdd(
                        tensor(DataType.INT32, Shape.of(0, 6), false, Optional.empty()),
                        tensor(DataType.INT64, Shape.of(2, 0, 6, 4), false, Optional.empty()),
                        1);
        Tensor zeroSelectedExtent = tensor(
                        DataType.FLOAT64, Shape.of(2, 0, 4), true, Optional.empty())
                .scatterAdd(
                        tensor(DataType.INT64, Shape.of(5), false, Optional.empty()),
                        tensor(DataType.FLOAT64, Shape.of(2, 5, 4), true, Optional.empty()),
                        1);

        assertAll(
                () -> assertSame(dynamicDataShape, scalar.descriptor().shape()),
                () -> assertSame(dynamicDataShape, namedExpression.descriptor().shape()),
                () -> assertSame(dynamicDataShape, unresolved.descriptor().shape()),
                () -> assertEquals(Shape.of(2, 3, 4), zero.descriptor().shape()),
                () -> assertEquals(Shape.of(2, 0, 4),
                        zeroSelectedExtent.descriptor().shape()));
    }

    @Test
    void validatesNullAndTypesBeforeAxisThenShapeWithExactMessagesAndNoIds()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        Tensor numeric = tensor(DataType.FLOAT32, Shape.of(2, 3), true, Optional.empty());
        Tensor bool = tensor(DataType.BOOL, Shape.of(2, 3), false, Optional.empty());
        Tensor indices = tensor(DataType.INT32, Shape.of(4), false, Optional.empty());
        Tensor wrongIndices = tensor(DataType.FLOAT32, Shape.of(4), false, Optional.empty());
        Tensor updates = tensor(DataType.FLOAT32, Shape.of(2, 4), true, Optional.empty());
        Tensor wrongType = tensor(DataType.FLOAT64, Shape.of(2, 4), true, Optional.empty());
        Tensor boolUpdates = tensor(DataType.BOOL, Shape.of(2, 4), false, Optional.empty());
        Tensor wrongShape = tensor(DataType.FLOAT32, Shape.of(2, 5), true, Optional.empty());
        long before = next.get();

        NullPointerException nullData = assertThrows(
                NullPointerException.class,
                () -> TensorAxisScatterExpressions.scatterAdd(null, null, null, 9));
        NullPointerException nullIndices = assertThrows(
                NullPointerException.class,
                () -> TensorAxisScatterExpressions.scatterAdd(numeric, null, null, 9));
        NullPointerException nullUpdates = assertThrows(
                NullPointerException.class,
                () -> TensorAxisScatterExpressions.scatterAdd(numeric, indices, null, 9));
        IllegalArgumentException indexType = assertThrows(
                IllegalArgumentException.class,
                () -> numeric.scatterAdd(wrongIndices, wrongType, 9));
        IllegalArgumentException updateType = assertThrows(
                IllegalArgumentException.class,
                () -> numeric.scatterAdd(indices, wrongType, 9));
        IllegalArgumentException boolType = assertThrows(
                IllegalArgumentException.class,
                () -> bool.scatterAdd(indices, boolUpdates, 9));
        IndexOutOfBoundsException axis = assertThrows(
                IndexOutOfBoundsException.class,
                () -> numeric.scatterAdd(indices, wrongShape, 2));
        IllegalArgumentException shape = assertThrows(
                IllegalArgumentException.class,
                () -> numeric.scatterAdd(indices, wrongShape, 1));

        assertAll(
                () -> assertEquals("data", nullData.getMessage()),
                () -> assertEquals("indices", nullIndices.getMessage()),
                () -> assertEquals("updates", nullUpdates.getMessage()),
                () -> assertEquals(
                        "scatterAdd indices data type must be INT32 or INT64: FLOAT32",
                        indexType.getMessage()),
                () -> assertEquals(
                        "scatterAdd updates data type must match data: expected=FLOAT32, actual=FLOAT64",
                        updateType.getMessage()),
                () -> assertEquals(
                        "scatterAdd data type must be numeric: BOOL", boolType.getMessage()),
                () -> assertEquals("Axis 2 is outside shape rank 2", axis.getMessage()),
                () -> assertEquals(
                        "scatterAdd updates shape must match gather result shape: "
                                + "expected=Shape[2, 4], actual=Shape[2, 5]",
                        shape.getMessage()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void rejectsFirstStructuralMismatchWithoutBroadcastingOrBinding() {
        DynamicDimension batch = new DynamicDimension("batch");
        DynamicDimension otherBatch = new DynamicDimension("otherBatch");
        Tensor data = tensor(
                DataType.INT64,
                Shape.ofDimensions(batch, new StaticDimension(3), new StaticDimension(4)),
                false,
                Optional.empty());
        Tensor indices = tensor(DataType.INT64, Shape.of(1, 6), false, Optional.empty());
        Tensor updates = tensor(
                DataType.INT64,
                Shape.ofDimensions(otherBatch, new StaticDimension(1), new StaticDimension(6),
                        new StaticDimension(4)),
                false,
                Optional.empty());

        IllegalArgumentException mismatch = assertThrows(
                IllegalArgumentException.class,
                () -> data.scatterAdd(indices, updates, 1));

        assertEquals(
                "scatterAdd updates shape must match gather result shape: "
                        + "expected=Shape[batch, 1, 6, 4], actual=Shape[otherBatch, 1, 6, 4]",
                mismatch.getMessage());
    }

    @Test
    void retainsExistingScatterElementsMetadataAndValidation() {
        DynamicDimension batch = new DynamicDimension("batch");
        Shape dataShape = Shape.ofDimensions(batch, new StaticDimension(3), new StaticDimension(4));
        Shape updateShape = Shape.ofDimensions(batch, new StaticDimension(7), new StaticDimension(4));
        Tensor data = tensor(DataType.FLOAT32, dataShape, true, Optional.empty());
        Tensor indices = tensor(DataType.INT64, updateShape, false, Optional.empty());
        Tensor updates = tensor(DataType.FLOAT32, updateShape, false, Optional.empty());

        Tensor replacement = data.scatterElements(indices, updates, -2);
        Tensor addition = data.scatterElements(indices, updates, 1, ScatterReduction.ADD);
        TensorProvenance replacementProvenance = replacement.provenance().orElseThrow();
        TensorProvenance additionProvenance = addition.provenance().orElseThrow();

        assertAll(
                () -> assertSame(AxisScatterKind.SCATTER_ELEMENTS,
                        replacementProvenance.operation().kind()),
                () -> assertEquals(
                        new ScatterElementsAttrs(1, ScatterReduction.NONE),
                        replacementProvenance.operation().attrs()),
                () -> assertEquals(
                        new ScatterElementsAttrs(1, ScatterReduction.ADD),
                        additionProvenance.operation().attrs()),
                () -> assertEquals(List.of(data, indices, updates), replacementProvenance.inputs()),
                () -> assertSame(dataShape, replacement.descriptor().shape()),
                () -> assertNotEquals(replacement.id(), addition.id()));
    }

    private static AtomicLong nextTensorIdState() throws ReflectiveOperationException {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }

    private static Tensor tensor(
            DataType dataType,
            Shape shape,
            boolean requiresGrad,
            Optional<LayoutDescriptor> layout) {
        return new Tensor(
                new TensorId(INPUT_IDS.getAndIncrement()),
                new TensorDescriptor(dataType, shape, layout, requiresGrad),
                Optional.empty(),
                Optional.empty(),
                Optional.empty());
    }

    private static String methodSignature(Method method) {
        return method.getName() + "("
                + Arrays.stream(method.getParameterTypes())
                        .map(Class::getName)
                        .collect(Collectors.joining(","))
                + "):" + method.getReturnType().getName();
    }
}
