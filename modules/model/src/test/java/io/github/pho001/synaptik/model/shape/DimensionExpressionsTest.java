package io.github.pho001.synaptik.model.shape;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.junit.jupiter.api.Test;

class DimensionExpressionsTest {
    @Test
    void exposesExactlyThePlannedExpressionFormsAndFactorySurface() {
        assertTrue(DimensionExpression.class.isSealed());
        assertEquals(
                Set.of(
                        DimensionExpression.LinearCombination.class,
                        DimensionExpression.Product.class,
                        DimensionExpression.FloorDivision.class,
                        DimensionExpression.CeilingDivision.class,
                        DimensionExpression.Unknown.class),
                Set.of(DimensionExpression.class.getPermittedSubclasses()));
        assertEquals(
                Set.of(
                        "add(Dimension,Dimension)",
                        "addConstant(Dimension,long)",
                        "multiply(Dimension,long)",
                        "multiply(Dimension,Dimension)",
                        "floorDivide(Dimension,long)",
                        "ceilingDivide(Dimension,long)",
                        "unknown(long,Optional)"),
                Set.of(DimensionExpressions.class.getDeclaredMethods()).stream()
                        .filter(method -> Modifier.isPublic(method.getModifiers()))
                        .map(DimensionExpressionsTest::methodSignature)
                        .collect(java.util.stream.Collectors.toSet()));
        assertEquals(0, DimensionExpressions.class.getDeclaredFields().length);
        assertTrue(Modifier.isPrivate(
                DimensionExpressions.class.getDeclaredConstructors()[0].getModifiers()));

        assertPackagePrivateConstructors(
                ExpressionDimension.class,
                DimensionExpression.LinearCombination.class,
                DimensionExpression.Product.class,
                DimensionExpression.FloorDivision.class,
                DimensionExpression.CeilingDivision.class,
                DimensionExpression.Unknown.class);
        assertEquals(
                Set.of("expression"),
                Set.of(ExpressionDimension.class.getDeclaredFields()).stream()
                        .map(Field::getName)
                        .collect(java.util.stream.Collectors.toSet()));
        assertEquals(
                Set.of("factors", "coefficient"),
                Set.of(DimensionExpression.Product.class.getDeclaredFields()).stream()
                        .map(Field::getName)
                        .collect(java.util.stream.Collectors.toSet()));
        assertEquals(
                Set.of("factors", "coefficient", "equals", "hashCode", "toString"),
                Set.of(DimensionExpression.Product.class.getDeclaredMethods()).stream()
                        .filter(method -> Modifier.isPublic(method.getModifiers()))
                        .map(Method::getName)
                        .collect(java.util.stream.Collectors.toSet()));
    }

    @Test
    void foldsStaticArithmeticAndRejectsNegativeStaticResults() {
        assertEquals(
                new StaticDimension(9),
                DimensionExpressions.add(new StaticDimension(4), new StaticDimension(5)));
        assertEquals(
                new StaticDimension(3),
                DimensionExpressions.addConstant(new StaticDimension(5), -2));
        assertEquals(
                new StaticDimension(20),
                DimensionExpressions.multiply(new StaticDimension(5), 4));
        assertEquals(
                new StaticDimension(2),
                DimensionExpressions.floorDivide(new StaticDimension(8), 3));
        assertEquals(
                new StaticDimension(3),
                DimensionExpressions.ceilingDivide(new StaticDimension(8), 3));
        assertEquals(
                new StaticDimension(1),
                DimensionExpressions.ceilingDivide(new StaticDimension(Long.MAX_VALUE),
                        Long.MAX_VALUE));
        assertEquals(
                "dimension expression must not produce a negative static size: -1",
                assertThrows(
                        IllegalArgumentException.class,
                        () -> DimensionExpressions.addConstant(new StaticDimension(2), -3))
                        .getMessage());
    }

    @Test
    void preservesNeutralOperationReferences() {
        Dimension dimension = DimensionExpressions.addConstant(new DynamicDimension("N"), 2);

        assertSame(dimension, DimensionExpressions.add(dimension, new StaticDimension(0)));
        assertSame(dimension, DimensionExpressions.add(new StaticDimension(0), dimension));
        assertSame(dimension, DimensionExpressions.addConstant(dimension, 0));
        assertSame(dimension, DimensionExpressions.multiply(dimension, 1));
        assertSame(dimension, DimensionExpressions.floorDivide(dimension, 1));
        assertSame(dimension, DimensionExpressions.ceilingDivide(dimension, 1));
        assertEquals(new StaticDimension(0), DimensionExpressions.multiply(dimension, 0));
    }

    @Test
    void canonicalizesNestedLinearTermsOffsetsAndOperandOrder() {
        Dimension n = new DynamicDimension("N");
        Dimension m = new DynamicDimension("M");
        Dimension left = DimensionExpressions.add(
                DimensionExpressions.addConstant(n, -3),
                DimensionExpressions.multiply(m, 2));
        Dimension right = DimensionExpressions.add(
                DimensionExpressions.multiply(m, 2),
                DimensionExpressions.addConstant(n, -3));

        assertEquals(left, right);
        assertEquals(left.hashCode(), right.hashCode());
        DimensionExpression.LinearCombination combination = linearCombination(left);
        assertEquals(Map.of(n, 1L, m, 2L), combination.coefficients());
        assertEquals(-3, combination.offset());
        assertThrows(
                UnsupportedOperationException.class,
                () -> combination.coefficients().put(n, 9L));
    }

    @Test
    void combinesRepeatedTermsAndFlattensScaledLinearCombinations() {
        Dimension n = new DynamicDimension("N");
        Dimension doubledByAddition = DimensionExpressions.add(n, n);
        Dimension doubledByMultiplication = DimensionExpressions.multiply(n, 2);
        Dimension scaled = DimensionExpressions.multiply(
                DimensionExpressions.addConstant(n, 3), 4);

        assertEquals(doubledByAddition, doubledByMultiplication);
        assertEquals(Map.of(n, 2L), linearCombination(doubledByAddition).coefficients());
        assertEquals(Map.of(n, 4L), linearCombination(scaled).coefficients());
        assertEquals(12, linearCombination(scaled).offset());
    }

    @Test
    void canonicalizesSymbolicProductsIndependentOfOrderAndNesting() {
        Dimension h = new DynamicDimension("H");
        Dimension w = new DynamicDimension("W");
        Dimension left = DimensionExpressions.multiply(
                DimensionExpressions.multiply(h, w),
                DimensionExpressions.multiply(w, new StaticDimension(3)));
        Dimension right = DimensionExpressions.multiply(
                new StaticDimension(3),
                DimensionExpressions.multiply(w, DimensionExpressions.multiply(w, h)));

        assertEquals(left, right);
        assertEquals(left.hashCode(), right.hashCode());
        DimensionExpression.Product product = product(left);
        assertEquals(Map.of(h, 1L, w, 2L), product.factors());
        assertEquals(3, product.coefficient());
        assertThrows(UnsupportedOperationException.class, () -> product.factors().put(h, 9L));
        assertEquals("3 * H * W^2", product.toString());
    }

    @Test
    void foldsProductStaticValuesAndPreservesZeroAndOneReferences() {
        Dimension h = new DynamicDimension("H");
        Dimension product = DimensionExpressions.multiply(h, new DynamicDimension("W"));

        assertEquals(
                new StaticDimension(42),
                DimensionExpressions.multiply(new StaticDimension(6), new StaticDimension(7)));
        assertEquals(
                new StaticDimension(0),
                DimensionExpressions.multiply(new StaticDimension(0), h));
        assertEquals(
                new StaticDimension(0),
                DimensionExpressions.multiply(h, new StaticDimension(0)));
        assertSame(h, DimensionExpressions.multiply(new StaticDimension(1), h));
        assertSame(h, DimensionExpressions.multiply(h, new StaticDimension(1)));

        Dimension scaled = DimensionExpressions.multiply(product, 5);
        assertEquals(5, product(scaled).coefficient());
        assertEquals(product(product).factors(), product(scaled).factors());
    }

    @Test
    void combinesRepeatedProductFactorsAndChecksExponentAndCoefficientOverflow() {
        Dimension h = new DynamicDimension("H");
        Dimension squared = DimensionExpressions.multiply(h, h);
        Dimension highExponent = new ExpressionDimension(new DimensionExpression.Product(
                Map.of(h, Long.MAX_VALUE), 1));
        Dimension highCoefficient = DimensionExpressions.multiply(
                DimensionExpressions.multiply(h, new DynamicDimension("W")), Long.MAX_VALUE);

        assertEquals(Map.of(h, 2L), product(squared).factors());
        assertThrows(
                ArithmeticException.class,
                () -> DimensionExpressions.multiply(highExponent, h));
        assertThrows(
                ArithmeticException.class,
                () -> DimensionExpressions.multiply(highCoefficient, new StaticDimension(2)));
        assertThrows(
                ArithmeticException.class,
                () -> DimensionExpressions.multiply(
                        new StaticDimension(Long.MAX_VALUE), new StaticDimension(2)));
    }

    @Test
    void productConstructorValidatesAndSnapshotsInExactOrder() {
        Dimension h = new DynamicDimension("H");
        java.util.LinkedHashMap<Dimension, Long> factors = new java.util.LinkedHashMap<>();
        factors.put(h, 2L);
        DimensionExpression.Product product = new DimensionExpression.Product(factors, 3);
        factors.put(new DynamicDimension("W"), 1L);

        assertEquals(Map.of(h, 2L), product.factors());
        assertEquals("factors", assertThrows(
                NullPointerException.class,
                () -> new DimensionExpression.Product(null, 0)).getMessage());
        assertEquals("factors must not be empty", assertThrows(
                IllegalArgumentException.class,
                () -> new DimensionExpression.Product(Map.of(), 0)).getMessage());
        java.util.HashMap<Dimension, Long> nullKey = new java.util.HashMap<>();
        nullKey.put(null, 1L);
        assertEquals("factors key", assertThrows(
                NullPointerException.class,
                () -> new DimensionExpression.Product(nullKey, 0)).getMessage());
        java.util.HashMap<Dimension, Long> nullValue = new java.util.HashMap<>();
        nullValue.put(h, null);
        assertEquals("factors value", assertThrows(
                NullPointerException.class,
                () -> new DimensionExpression.Product(nullValue, 0)).getMessage());
        assertEquals("exponent must be positive: 0", assertThrows(
                IllegalArgumentException.class,
                () -> new DimensionExpression.Product(Map.of(h, 0L), 0)).getMessage());
        assertEquals("coefficient must be positive: 0", assertThrows(
                IllegalArgumentException.class,
                () -> new DimensionExpression.Product(Map.of(h, 1L), 0)).getMessage());
    }

    @Test
    void keepsDivisionNodesStructuralWithoutReassociation() {
        Dimension n = new DynamicDimension("N");
        Dimension floor = DimensionExpressions.floorDivide(
                DimensionExpressions.addConstant(n, 2), 3);
        Dimension equalFloor = DimensionExpressions.floorDivide(
                DimensionExpressions.add(new StaticDimension(2), new DynamicDimension("N")), 3);
        Dimension ceiling = DimensionExpressions.ceilingDivide(n, 3);

        assertEquals(floor, equalFloor);
        assertNotEquals(floor, ceiling);
        DimensionExpression.FloorDivision floorExpression = assertInstanceOf(
                DimensionExpression.FloorDivision.class,
                ((ExpressionDimension) floor).expression());
        assertEquals(3, floorExpression.divisor());
        assertEquals(
                DimensionExpressions.addConstant(n, 2),
                floorExpression.dividend());
        DimensionExpression.CeilingDivision ceilingExpression = assertInstanceOf(
                DimensionExpression.CeilingDivision.class,
                ((ExpressionDimension) ceiling).expression());
        assertEquals(n, ceilingExpression.dividend());
        assertEquals(3, ceilingExpression.divisor());
    }

    @Test
    void createsIdentityBasedUnknownsAndRetainsBounds() {
        Dimension maximum = DimensionExpressions.addConstant(new DynamicDimension("N"), 2);
        Dimension first = DimensionExpressions.unknown(1, Optional.of(maximum));
        Dimension second = DimensionExpressions.unknown(1, Optional.of(maximum));
        Dimension unbounded = DimensionExpressions.unknown(0, Optional.empty());

        assertNotEquals(first, second);
        assertSame(first, first);
        DimensionExpression.Unknown expression = assertInstanceOf(
                DimensionExpression.Unknown.class,
                ((ExpressionDimension) first).expression());
        assertEquals(1, expression.minimum());
        assertSame(maximum, expression.maximum().orElseThrow());
        assertTrue(((DimensionExpression.Unknown)
                ((ExpressionDimension) unbounded).expression()).maximum().isEmpty());
    }

    @Test
    void validatesArgumentsInTheSpecifiedOrderWithSpecifiedMessages() {
        assertThrows(
                NullPointerException.class,
                () -> DimensionExpressions.add(null, null),
                "left is checked before right");
        assertEquals(
                "left",
                assertThrows(
                        NullPointerException.class,
                        () -> DimensionExpressions.add(null, new StaticDimension(1)))
                        .getMessage());
        assertEquals(
                "right",
                assertThrows(
                        NullPointerException.class,
                        () -> DimensionExpressions.add(new StaticDimension(1), null))
                        .getMessage());
        assertEquals(
                "left",
                assertThrows(
                        NullPointerException.class,
                        () -> DimensionExpressions.multiply(null, (Dimension) null))
                        .getMessage());
        assertEquals(
                "right",
                assertThrows(
                        NullPointerException.class,
                        () -> DimensionExpressions.multiply(new StaticDimension(1), null))
                        .getMessage());
        assertEquals(
                "input",
                assertThrows(
                        NullPointerException.class,
                        () -> DimensionExpressions.multiply(null, -1))
                        .getMessage());
        assertEquals(
                "factor must be non-negative: -1",
                assertThrows(
                        IllegalArgumentException.class,
                        () -> DimensionExpressions.multiply(new DynamicDimension("N"), -1))
                        .getMessage());
        assertEquals(
                "divisor must be positive: 0",
                assertThrows(
                        IllegalArgumentException.class,
                        () -> DimensionExpressions.floorDivide(new DynamicDimension("N"), 0))
                        .getMessage());
        assertEquals(
                "divisor must be positive: -2",
                assertThrows(
                        IllegalArgumentException.class,
                        () -> DimensionExpressions.ceilingDivide(new DynamicDimension("N"), -2))
                        .getMessage());
        assertEquals(
                "minimum must be non-negative: -1",
                assertThrows(
                        IllegalArgumentException.class,
                        () -> DimensionExpressions.unknown(-1, null))
                        .getMessage());
        assertEquals(
                "maximum",
                assertThrows(
                        NullPointerException.class,
                        () -> DimensionExpressions.unknown(0, null))
                        .getMessage());
        assertEquals(
                "maximum static size must be greater than or equal to minimum: minimum=3, maximum=2",
                assertThrows(
                        IllegalArgumentException.class,
                        () -> DimensionExpressions.unknown(
                                3, Optional.of(new StaticDimension(2))))
                        .getMessage());
    }

    @Test
    void propagatesCheckedLongOverflow() {
        Dimension n = new DynamicDimension("N");
        Dimension maximumCoefficient = DimensionExpressions.multiply(n, Long.MAX_VALUE);
        Dimension maximumOffset = DimensionExpressions.addConstant(n, Long.MAX_VALUE);

        assertThrows(
                ArithmeticException.class,
                () -> DimensionExpressions.add(
                        new StaticDimension(Long.MAX_VALUE), new StaticDimension(1)));
        assertThrows(
                ArithmeticException.class,
                () -> DimensionExpressions.multiply(new StaticDimension(Long.MAX_VALUE), 2));
        assertThrows(
                ArithmeticException.class,
                () -> DimensionExpressions.add(maximumCoefficient, n));
        assertThrows(
                ArithmeticException.class,
                () -> DimensionExpressions.addConstant(maximumOffset, 1));
        assertThrows(
                ArithmeticException.class,
                () -> DimensionExpressions.multiply(maximumOffset, 2));
    }

    @Test
    void diagnosticsDistinguishExactDivisionAndUnknownExpressions() {
        Dimension n = new DynamicDimension("N");

        assertEquals("N - 3", DimensionExpressions.addConstant(n, -3).toString());
        assertEquals("floorDiv(N, 2)", DimensionExpressions.floorDivide(n, 2).toString());
        assertEquals("ceilDiv(N, 2)", DimensionExpressions.ceilingDivide(n, 2).toString());
        assertEquals(
                "unknown(min=1, max=N)",
                DimensionExpressions.unknown(1, Optional.of(n)).toString());
        assertEquals("unknown(min=0)", DimensionExpressions.unknown(0, Optional.empty()).toString());
    }

    private static DimensionExpression.LinearCombination linearCombination(Dimension dimension) {
        return assertInstanceOf(
                DimensionExpression.LinearCombination.class,
                assertInstanceOf(ExpressionDimension.class, dimension).expression());
    }

    private static DimensionExpression.Product product(Dimension dimension) {
        return assertInstanceOf(
                DimensionExpression.Product.class,
                assertInstanceOf(ExpressionDimension.class, dimension).expression());
    }

    private static void assertPackagePrivateConstructors(Class<?>... types) {
        for (Class<?> type : types) {
            for (Constructor<?> constructor : type.getDeclaredConstructors()) {
                assertFalse(Modifier.isPublic(constructor.getModifiers()));
                assertFalse(Modifier.isProtected(constructor.getModifiers()));
                assertFalse(Modifier.isPrivate(constructor.getModifiers()));
            }
        }
    }

    private static String methodSignature(Method method) {
        String parameters = java.util.Arrays.stream(method.getParameterTypes())
                .map(Class::getSimpleName)
                .collect(java.util.stream.Collectors.joining(","));
        return method.getName() + "(" + parameters + ")";
    }
}
