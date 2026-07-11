package io.github.pho001.synaptik.model.operation.ordering;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignatureTest;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class OrderingSemanticsTest {
    @Test
    void declaresExactlySortThenArgsortWithTypedIdentityAndSignatures() {
        assertAll(
                () -> assertArrayEquals(
                        new OrderingKind[] {OrderingKind.SORT, OrderingKind.ARGSORT},
                        OrderingKind.values()),
                () -> assertEquals("SORT", OrderingKind.SORT.name()),
                () -> assertEquals("ARGSORT", OrderingKind.ARGSORT.name()),
                () -> assertSame(OrderingKind.SORT, OrderingKind.valueOf("SORT")),
                () -> assertTrue(OrderingKind.SORT instanceof OperationKind),
                () -> assertNotEquals(OrderingKind.SORT, OrderingKind.ARGSORT));
        OperationSignatureTest.assertSignatureEnumShape(OrderingKind.class);
    }

    @Test
    void exposesOnlyTheExactAttributesRecordSurface() {
        var components = SortAttrs.class.getRecordComponents();
        var constructor = SortAttrs.class.getDeclaredConstructors();
        var fields = SortAttrs.class.getDeclaredFields();

        assertAll(
                () -> assertTrue(Modifier.isPublic(SortAttrs.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(SortAttrs.class.getModifiers())),
                () -> assertTrue(SortAttrs.class.isRecord()),
                () -> assertEquals(List.of(OperationAttrs.class),
                        List.of(SortAttrs.class.getInterfaces())),
                () -> assertEquals(List.of("axis", "descending"),
                        Arrays.stream(components).map(component -> component.getName()).toList()),
                () -> assertEquals(List.of(int.class, boolean.class),
                        Arrays.stream(components).map(component -> component.getType()).toList()),
                () -> assertEquals(1, constructor.length),
                () -> assertEquals(List.of(int.class, boolean.class),
                        List.of(constructor[0].getParameterTypes())),
                () -> assertEquals(List.of("axis", "descending"),
                        Arrays.stream(fields).map(field -> field.getName()).toList()),
                () -> assertEquals(0, SortAttrs.class.getDeclaredClasses().length));
    }

    @Test
    void validatesAxisAndUsesCompleteRecordValueSemantics() {
        SortAttrs attrs = new SortAttrs(2, true);
        assertAll(
                () -> assertEquals(2, attrs.axis()),
                () -> assertTrue(attrs.descending()),
                () -> assertEquals(attrs, new SortAttrs(2, true)),
                () -> assertNotEquals(attrs, new SortAttrs(2, false)),
                () -> assertEquals("SortAttrs[axis=2, descending=true]", attrs.toString()),
                () -> assertEquals(
                        "axis must be non-negative: -1",
                        assertThrows(IllegalArgumentException.class,
                                () -> new SortAttrs(-1, false)).getMessage()));
    }

    @Test
    void composesBothKindsWithTheExactAttributesReference() {
        SortAttrs attrs = new SortAttrs(1, false);
        Operation sort = new Operation(OrderingKind.SORT, attrs);
        Operation argsort = new Operation(OrderingKind.ARGSORT, attrs);

        assertAll(
                () -> assertSame(attrs, sort.attrs()),
                () -> assertSame(attrs, argsort.attrs()),
                () -> assertSame(OrderingKind.SORT, sort.kind()),
                () -> assertSame(OrderingKind.ARGSORT, argsort.kind()));
    }

    @Test
    void locksStableFloatingOrderRepresentationsForBothDirections() {
        double[] values = {3.0, Double.longBitsToDouble(0x7ff8000000000001L), -0.0, +0.0, 3.0,
                Double.longBitsToDouble(0xfff8000000000002L),
                Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY};

        assertArrayEquals(new int[] {6, 2, 3, 0, 4, 7, 1, 5}, stableOrder(values, false));
        assertArrayEquals(new int[] {7, 0, 4, 3, 2, 6, 1, 5}, stableOrder(values, true));
        int[] ascending = stableOrder(values, false);
        assertAll(
                () -> assertEquals(0x8000000000000000L,
                        Double.doubleToRawLongBits(values[ascending[1]])),
                () -> assertEquals(0L, Double.doubleToRawLongBits(values[ascending[2]])),
                () -> assertEquals(0x7ff8000000000001L,
                        Double.doubleToRawLongBits(values[ascending[6]])),
                () -> assertEquals(0xfff8000000000002L,
                        Double.doubleToRawLongBits(values[ascending[7]])));
    }

    private static int[] stableOrder(double[] values, boolean descending) {
        Integer[] indices = java.util.stream.IntStream.range(0, values.length).boxed()
                .toArray(Integer[]::new);
        Arrays.sort(indices, (left, right) -> {
            double a = values[left];
            double b = values[right];
            if (Double.isNaN(a) || Double.isNaN(b)) {
                return Double.isNaN(a) == Double.isNaN(b) ? 0 : Double.isNaN(a) ? 1 : -1;
            }
            int comparison = Double.compare(a, b);
            return descending ? -comparison : comparison;
        });
        return Arrays.stream(indices).mapToInt(Integer::intValue).toArray();
    }
}
