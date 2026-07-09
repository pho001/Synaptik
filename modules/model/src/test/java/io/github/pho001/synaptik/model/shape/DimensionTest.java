package io.github.pho001.synaptik.model.shape;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Optional;
import java.util.OptionalLong;
import java.util.Set;
import org.junit.jupiter.api.Test;

class DimensionTest {
    @Test
    void permitsExactlyStaticNamedAndExpressionVariants() {
        assertTrue(Dimension.class.isSealed());
        assertEquals(
                Set.of(StaticDimension.class, DynamicDimension.class, ExpressionDimension.class),
                Set.of(Dimension.class.getPermittedSubclasses()));
    }

    @Test
    void staticDimensionAcceptsEveryNonNegativeLongBoundary() {
        assertEquals(0, new StaticDimension(0).size());
        assertEquals(Long.MAX_VALUE, new StaticDimension(Long.MAX_VALUE).size());
        assertThrows(IllegalArgumentException.class, () -> new StaticDimension(-1));
        assertThrows(IllegalArgumentException.class, () -> new StaticDimension(Long.MIN_VALUE));
    }

    @Test
    void staticDimensionExposesOnlyStaticInspectionData() {
        Dimension dimension = new StaticDimension(17);

        assertTrue(dimension.isStatic());
        assertFalse(dimension.isDynamic());
        assertEquals(OptionalLong.of(17), dimension.staticSize());
        assertEquals(Optional.empty(), dimension.dynamicSymbol());
    }

    @Test
    void dynamicDimensionCanonicalizesAndValidatesItsSymbol() {
        DynamicDimension dimension = new DynamicDimension("\u2003batch\u2003");

        assertEquals("batch", dimension.symbol());
        assertThrows(NullPointerException.class, () -> new DynamicDimension(null));
        assertThrows(IllegalArgumentException.class, () -> new DynamicDimension(" \t\n"));
        assertThrows(IllegalArgumentException.class, () -> new DynamicDimension("\u2003"));
    }

    @Test
    void dynamicDimensionExposesOnlySymbolicInspectionData() {
        Dimension dimension = new DynamicDimension("sequence");

        assertFalse(dimension.isStatic());
        assertTrue(dimension.isDynamic());
        assertEquals(OptionalLong.empty(), dimension.staticSize());
        assertEquals(Optional.of("sequence"), dimension.dynamicSymbol());
    }

    @Test
    void expressionDimensionIsDynamicWithoutANameOrStaticSize() {
        Dimension dimension = DimensionExpressions.addConstant(new DynamicDimension("N"), 2);

        assertFalse(dimension.isStatic());
        assertTrue(dimension.isDynamic());
        assertEquals(OptionalLong.empty(), dimension.staticSize());
        assertEquals(Optional.empty(), dimension.dynamicSymbol());
        assertTrue(dimension instanceof ExpressionDimension);
    }

    @Test
    void dimensionEqualityAndHashingUseCanonicalValues() {
        assertEquals(new StaticDimension(5), new StaticDimension(5));
        assertEquals(new StaticDimension(5).hashCode(), new StaticDimension(5).hashCode());
        assertEquals(new DynamicDimension("batch"), new DynamicDimension(" batch "));
        assertEquals(
                new DynamicDimension("batch").hashCode(),
                new DynamicDimension(" batch ").hashCode());
    }
}
