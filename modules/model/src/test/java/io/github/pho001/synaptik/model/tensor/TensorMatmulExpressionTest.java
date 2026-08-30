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
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
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
import org.junit.jupiter.api.Test;

class TensorMatmulExpressionTest {
    @Test
    void helperAndPublicMethodHaveExactlyTheRequiredSurface() throws ReflectiveOperationException {
        int modifiers = TensorMatmulExpressions.class.getModifiers();
        var constructors = TensorMatmulExpressions.class.getDeclaredConstructors();
        Set<String> methods = Arrays.stream(TensorMatmulExpressions.class.getDeclaredMethods())
                .map(Method::getName)
                .collect(java.util.stream.Collectors.toSet());
        Method apply = TensorMatmulExpressions.class.getDeclaredMethod(
                "apply", Tensor.class, Tensor.class);
        Method matmul = Tensor.class.getDeclaredMethod("matmul", Tensor.class);
        long publicTensorMethods = Arrays.stream(Tensor.class.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .count();
        long publicMatmulMethods = Arrays.stream(Tensor.class.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .filter(method -> method.getName().equals("matmul"))
                .count();

        assertAll(
                () -> assertTrue(Modifier.isFinal(modifiers)),
                () -> assertFalse(Modifier.isPublic(modifiers)),
                () -> assertFalse(Modifier.isProtected(modifiers)),
                () -> assertFalse(TensorMatmulExpressions.class.isRecord()),
                () -> assertEquals(Set.of(), Set.of(TensorMatmulExpressions.class.getInterfaces())),
                () -> assertEquals(0, TensorMatmulExpressions.class.getDeclaredFields().length),
                () -> assertEquals(0, TensorMatmulExpressions.class.getDeclaredClasses().length),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPrivate(constructors[0].getModifiers())),
                () -> assertEquals(0, constructors[0].getParameterCount()),
                () -> assertEquals(Set.of("apply", "broadcastBatchDimension"), methods),
                () -> assertTrue(Modifier.isStatic(apply.getModifiers())),
                () -> assertFalse(Modifier.isPublic(apply.getModifiers())),
                () -> assertFalse(Modifier.isProtected(apply.getModifiers())),
                () -> assertFalse(Modifier.isPrivate(apply.getModifiers())),
                () -> assertEquals(Tensor.class, apply.getReturnType()),
                () -> assertTrue(Modifier.isPublic(matmul.getModifiers())),
                () -> assertFalse(Modifier.isStatic(matmul.getModifiers())),
                () -> assertEquals(Tensor.class, matmul.getReturnType()),
                () -> assertEquals(208, publicTensorMethods),
                () -> assertEquals(1, publicMatmulMethods));
    }

    @Test
    void derivesEveryVectorMatrixAndBatchRankFamily() {
        assertEquals(Shape.scalar(), matmul(Shape.of(3), Shape.of(3)).descriptor().shape());
        assertEquals(Shape.of(2), matmul(Shape.of(2, 3), Shape.of(3)).descriptor().shape());
        assertEquals(Shape.of(4), matmul(Shape.of(3), Shape.of(3, 4)).descriptor().shape());
        assertEquals(Shape.of(2, 4), matmul(Shape.of(2, 3), Shape.of(3, 4)).descriptor().shape());
        assertEquals(Shape.of(5, 2, 4),
                matmul(Shape.of(5, 2, 3), Shape.of(1, 3, 4)).descriptor().shape());
        assertEquals(Shape.of(5, 4),
                matmul(Shape.of(3), Shape.of(5, 3, 4)).descriptor().shape());
        assertEquals(Shape.of(5, 2),
                matmul(Shape.of(5, 2, 3), Shape.of(3)).descriptor().shape());
        assertEquals(Shape.of(7, 5, 2, 4),
                matmul(Shape.of(5, 2, 3), Shape.of(7, 1, 3, 4)).descriptor().shape());
    }

    @Test
    void preservesExactRetainedStaticNamedAndExpressionDimensionReferences() {
        Dimension batch = new DynamicDimension("B");
        Dimension rows = DimensionExpressions.add(new DynamicDimension("M"), new StaticDimension(2));
        Dimension inner = new DynamicDimension("K");
        Dimension columns = DimensionExpressions.multiply(new DynamicDimension("N"), 2);
        Shape leftShape = Shape.ofDimensions(batch, rows, inner);
        Shape rightShape = Shape.ofDimensions(batch, inner, columns);

        Tensor result = tensor(DataType.FLOAT32, leftShape, true)
                .matmul(tensor(DataType.FLOAT32, rightShape, false));
        List<Dimension> resultDimensions = result.descriptor().shape().dimensions();

        assertAll(
                () -> assertSame(batch, resultDimensions.get(0)),
                () -> assertSame(rows, resultDimensions.get(1)),
                () -> assertSame(columns, resultDimensions.get(2)),
                () -> assertTrue(result.descriptor().requiresGrad()));
    }

    @Test
    void appliesExactBatchBroadcastReferenceAndDeferredPolicies() {
        Dimension named = new DynamicDimension("N");
        Dimension expression = DimensionExpressions.add(new DynamicDimension("B"), new StaticDimension(1));
        Dimension staticFour = new StaticDimension(4);
        Tensor namedAgainstSingleton = matmul(
                Shape.ofDimensions(named, new StaticDimension(2), new StaticDimension(3)),
                Shape.of(1, 3, 5));
        Tensor expressionAgainstStatic = matmul(
                Shape.ofDimensions(expression, new StaticDimension(2), new StaticDimension(3)),
                Shape.ofDimensions(staticFour, new StaticDimension(3), new StaticDimension(5)));
        Tensor unpaired = matmul(
                Shape.of(2, 3),
                Shape.ofDimensions(named, new StaticDimension(1), new StaticDimension(3),
                        new StaticDimension(5)));

        assertAll(
                () -> assertSame(named, namedAgainstSingleton.descriptor().shape().dimension(0)),
                () -> assertSame(staticFour, expressionAgainstStatic.descriptor().shape().dimension(0)),
                () -> assertSame(named, unpaired.descriptor().shape().dimension(0)));
    }

    @Test
    void acceptsDeferredInnerObligationsAndEqualUnresolvedBatches() {
        Dimension leftInner = new DynamicDimension("K");
        Dimension rightInner = new DynamicDimension("Q");
        Dimension batchLeft = new DynamicDimension("B");
        Dimension batchRightEqual = new DynamicDimension("B");
        Tensor result = matmul(
                Shape.ofDimensions(batchLeft, new StaticDimension(2), leftInner),
                Shape.ofDimensions(batchRightEqual, rightInner, new StaticDimension(5)));

        assertAll(
                () -> assertEquals(Shape.ofDimensions(batchLeft, new StaticDimension(2),
                        new StaticDimension(5)), result.descriptor().shape()),
                () -> assertSame(batchLeft, result.descriptor().shape().dimension(0)));
    }

    @Test
    void rejectsExactInnerAndBatchFailuresWithTaskOwnedMessages() {
        IllegalArgumentException inner = assertThrows(IllegalArgumentException.class,
                () -> matmul(Shape.of(2, 3), Shape.of(4, 5)));
        IllegalArgumentException staticBatch = assertThrows(IllegalArgumentException.class,
                () -> matmul(Shape.of(2, 3, 4), Shape.of(5, 4, 6)));
        Dimension leftBatch = new DynamicDimension("B");
        Dimension rightBatch = DimensionExpressions.add(
                new DynamicDimension("C"), new StaticDimension(1));
        IllegalArgumentException unresolvedBatch = assertThrows(IllegalArgumentException.class,
                () -> matmul(
                        Shape.ofDimensions(leftBatch, new StaticDimension(3), new StaticDimension(4)),
                        Shape.ofDimensions(rightBatch, new StaticDimension(4), new StaticDimension(6))));

        assertAll(
                () -> assertEquals("matmul inner dimensions must match: left=StaticDimension[size=3], right=StaticDimension[size=4]",
                        inner.getMessage()),
                () -> assertEquals("cannot broadcast matmul batch dimensions at result batch axis 0: left=StaticDimension[size=2], right=StaticDimension[size=5]",
                        staticBatch.getMessage()),
                () -> assertEquals("cannot derive exact matmul batch dimension at result batch axis 0: left="
                                + leftBatch + ", right=" + rightBatch,
                        unresolvedBatch.getMessage()));
    }

    @Test
    void validatesInTheRequiredOrderAndConsumesNoIdOnLocalFailures() throws Exception {
        AtomicLong nextId = nextTensorIdState();
        Tensor scalarFloat = tensor(DataType.FLOAT32, Shape.scalar(), false);
        Tensor vectorFloat = tensor(DataType.FLOAT32, Shape.of(2), false);
        Tensor scalarBool = tensor(DataType.BOOL, Shape.scalar(), false);
        Tensor vectorBool = tensor(DataType.BOOL, Shape.of(2), false);
        long before = nextId.get();

        NullPointerException leftNull = assertThrows(NullPointerException.class,
                () -> TensorMatmulExpressions.apply(null, null));
        NullPointerException rightNull = assertThrows(NullPointerException.class,
                () -> TensorMatmulExpressions.apply(vectorFloat, null));
        IllegalArgumentException promotionBeforeRank = assertThrows(IllegalArgumentException.class,
                () -> TensorMatmulExpressions.apply(scalarBool, vectorBool));
        IllegalArgumentException leftRank = assertThrows(IllegalArgumentException.class,
                () -> TensorMatmulExpressions.apply(scalarFloat, scalarFloat));
        IllegalArgumentException rightRank = assertThrows(IllegalArgumentException.class,
                () -> TensorMatmulExpressions.apply(vectorFloat, scalarFloat));
        IllegalArgumentException innerBeforeBatch = assertThrows(IllegalArgumentException.class,
                () -> matmul(Shape.of(2, 3, 4), Shape.of(5, 6, 7)));

        assertAll(
                () -> assertEquals("left", leftNull.getMessage()),
                () -> assertEquals("right", rightNull.getMessage()),
                () -> assertEquals("left must be a numeric data type, but was BOOL",
                        promotionBeforeRank.getMessage()),
                () -> assertEquals("left rank must be at least 1: 0", leftRank.getMessage()),
                () -> assertEquals("right rank must be at least 1: 0", rightRank.getMessage()),
                () -> assertTrue(innerBeforeBatch.getMessage().startsWith(
                        "matmul inner dimensions must match:")),
                () -> assertEquals(before, nextId.get()));
    }

    @Test
    void acceptsEveryOrderedFloatingAndIntegralWidthPairAndRejectsOtherPairs() {
        DataType[] floating = {DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64};
        for (DataType left : floating) {
            for (DataType right : floating) {
                assertSame(widestFloating(left, right), tensor(left, Shape.of(2), false)
                        .matmul(tensor(right, Shape.of(2), false)).descriptor().dataType());
            }
        }
        DataType[] integral = {DataType.INT32, DataType.INT64};
        for (DataType left : integral) {
            for (DataType right : integral) {
                DataType expected = left == DataType.INT64 || right == DataType.INT64
                        ? DataType.INT64 : DataType.INT32;
                assertSame(expected, tensor(left, Shape.of(2), false)
                        .matmul(tensor(right, Shape.of(2), false)).descriptor().dataType());
            }
        }

        assertThrows(IllegalArgumentException.class,
                () -> tensor(DataType.BOOL, Shape.of(2), false)
                        .matmul(tensor(DataType.BOOL, Shape.of(2), false)));
        assertThrows(IllegalArgumentException.class,
                () -> tensor(DataType.FLOAT32, Shape.of(2), false)
                        .matmul(tensor(DataType.INT32, Shape.of(2), false)));
    }

    @Test
    void supportsZeroContractionsAsMetadataWithoutReadingValues() {
        Tensor floating = matmul(Shape.of(2, 0), Shape.of(0, 4));
        Tensor integral = tensor(DataType.INT32, Shape.of(2, 0), false)
                .matmul(tensor(DataType.INT64, Shape.of(0, 4), false));

        assertAll(
                () -> assertEquals(Shape.of(2, 4), floating.descriptor().shape()),
                () -> assertSame(DataType.FLOAT32, floating.descriptor().dataType()),
                () -> assertTrue(floating.hostStorage().isEmpty()),
                () -> assertEquals(Shape.of(2, 4), integral.descriptor().shape()),
                () -> assertSame(DataType.INT64, integral.descriptor().dataType()),
                () -> assertTrue(integral.hostStorage().isEmpty()));
    }

    @Test
    void createsFreshUnlabeledStorageFreeExactOrderedProvenanceWithoutMutatingInputs() {
        Tensor left = tensor(DataType.FLOAT32, Shape.of(2, 3), true);
        Tensor right = tensor(DataType.FLOAT64, Shape.of(3, 4), false);
        TensorDescriptor leftDescriptor = left.descriptor();
        TensorDescriptor rightDescriptor = right.descriptor();
        Tensor first = left.matmul(right);
        Tensor second = left.matmul(right);
        TensorProvenance provenance = first.provenance().orElseThrow();

        assertAll(
                () -> assertNotSame(first, second),
                () -> assertNotEquals(first.id(), second.id()),
                () -> assertNotSame(provenance.producer(), second.provenance().orElseThrow().producer()),
                () -> assertSame(DataType.FLOAT64, first.descriptor().dataType()),
                () -> assertEquals(Shape.of(2, 4), first.descriptor().shape()),
                () -> assertTrue(first.descriptor().layout().isEmpty()),
                () -> assertTrue(first.descriptor().requiresGrad()),
                () -> assertTrue(first.label().isEmpty()),
                () -> assertTrue(first.hostStorage().isEmpty()),
                () -> assertSame(MatmulKind.MATMUL, provenance.operation().kind()),
                () -> assertSame(NoOperationAttrs.INSTANCE, provenance.operation().attrs()),
                () -> assertEquals(2, provenance.inputs().size()),
                () -> assertSame(left, provenance.inputs().get(0)),
                () -> assertSame(right, provenance.inputs().get(1)),
                () -> assertEquals(0, provenance.outputIndex()),
                () -> assertEquals(1, provenance.producer().outputCount()),
                () -> assertSame(first.descriptor(), provenance.outputDescriptor()),
                () -> assertSame(leftDescriptor, left.descriptor()),
                () -> assertSame(rightDescriptor, right.descriptor()),
                () -> assertTrue(left.provenance().isEmpty()),
                () -> assertTrue(right.provenance().isEmpty()));
    }

    @Test
    void publicMethodNullCheckUsesTheRightParameterName() {
        Tensor left = tensor(DataType.FLOAT32, Shape.of(2), false);
        NullPointerException exception = assertThrows(NullPointerException.class,
                () -> left.matmul(null));
        assertEquals("right", exception.getMessage());
    }

    private static Tensor matmul(Shape left, Shape right) {
        return tensor(DataType.FLOAT32, left, false)
                .matmul(tensor(DataType.FLOAT32, right, false));
    }

    private static Tensor tensor(DataType dataType, Shape shape, boolean requiresGrad) {
        return new Tensor(
                new TensorId(1),
                new TensorDescriptor(dataType, shape, Optional.empty(), requiresGrad),
                Optional.empty(),
                Optional.empty(),
                Optional.empty());
    }

    private static DataType widestFloating(DataType left, DataType right) {
        if (left == DataType.FLOAT64 || right == DataType.FLOAT64) {
            return DataType.FLOAT64;
        }
        if (left == DataType.FLOAT32 || right == DataType.FLOAT32) {
            return DataType.FLOAT32;
        }
        return DataType.BFLOAT16;
    }

    private static AtomicLong nextTensorIdState() throws ReflectiveOperationException {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }
}
