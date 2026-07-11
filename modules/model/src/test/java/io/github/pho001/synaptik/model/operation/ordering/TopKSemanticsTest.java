package io.github.pho001.synaptik.model.operation.ordering;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import io.github.pho001.synaptik.model.operation.OperationSignatureTest;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import org.junit.jupiter.api.Test;

final class TopKSemanticsTest {
    @Test
    void declaresOneFocusedKindWithExactTwoOutputSignature() {
        assertAll(
                () -> assertArrayEquals(new TopKKind[] {TopKKind.TOP_K}, TopKKind.values()),
                () -> assertSame(TopKKind.TOP_K, TopKKind.valueOf("TOP_K")),
                () -> assertTrue(TopKKind.TOP_K instanceof OperationKind),
                () -> assertEquals(
                        List.of(OperationSignature.fixed(TopKAttrs.class, 1, 2)),
                        TopKKind.TOP_K.signatures()));
        OperationSignatureTest.assertSignatureEnumShape(TopKKind.class);
    }

    @Test
    void exposesOnlyTheExactAttributesRecordSurface() {
        var components = TopKAttrs.class.getRecordComponents();
        var constructor = TopKAttrs.class.getDeclaredConstructors();
        var fields = TopKAttrs.class.getDeclaredFields();
        assertAll(
                () -> assertTrue(Modifier.isPublic(TopKAttrs.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(TopKAttrs.class.getModifiers())),
                () -> assertTrue(TopKAttrs.class.isRecord()),
                () -> assertEquals(List.of(OperationAttrs.class),
                        List.of(TopKAttrs.class.getInterfaces())),
                () -> assertEquals(List.of("axis", "k", "largest", "sorted"),
                        Arrays.stream(components).map(component -> component.getName()).toList()),
                () -> assertEquals(List.of(int.class, long.class, boolean.class, boolean.class),
                        Arrays.stream(components).map(component -> component.getType()).toList()),
                () -> assertEquals(1, constructor.length),
                () -> assertEquals(List.of(int.class, long.class, boolean.class, boolean.class),
                        List.of(constructor[0].getParameterTypes())),
                () -> assertEquals(List.of("axis", "k", "largest", "sorted"),
                        Arrays.stream(fields).map(field -> field.getName()).toList()),
                () -> assertEquals(0, TopKAttrs.class.getDeclaredClasses().length));
    }

    @Test
    void validatesAxisThenKAndUsesCompleteRecordValueSemantics() {
        TopKAttrs attrs = new TopKAttrs(2, 3, true, false);
        assertAll(
                () -> assertEquals(2, attrs.axis()),
                () -> assertEquals(3, attrs.k()),
                () -> assertTrue(attrs.largest()),
                () -> assertFalse(attrs.sorted()),
                () -> assertEquals(attrs, new TopKAttrs(2, 3, true, false)),
                () -> assertNotEquals(attrs, new TopKAttrs(2, 3, false, false)),
                () -> assertEquals(
                        "TopKAttrs[axis=2, k=3, largest=true, sorted=false]",
                        attrs.toString()),
                () -> assertEquals("axis must be non-negative: -1",
                        assertThrows(IllegalArgumentException.class,
                                () -> new TopKAttrs(-1, -1, true, true)).getMessage()),
                () -> assertEquals("k must be non-negative: -1",
                        assertThrows(IllegalArgumentException.class,
                                () -> new TopKAttrs(0, -1, true, true)).getMessage()));
    }

    @Test
    void composesTheKindWithTheExactAttributesReference() {
        TopKAttrs attrs = new TopKAttrs(1, 0, false, true);
        Operation operation = new Operation(TopKKind.TOP_K, attrs);
        assertAll(
                () -> assertSame(TopKKind.TOP_K, operation.kind()),
                () -> assertSame(attrs, operation.attrs()));
    }

    @Test
    void locksSelectedSetAndSortedOrLogicalIndexOutputPolicy() {
        double[] input = {3.0, Double.NaN, -0.0, +0.0, 3.0};
        assertAll(
                () -> assertArrayEquals(new int[] {0, 4, 3}, topK(input, 3, true, true)),
                () -> assertArrayEquals(new int[] {0, 3, 4}, topK(input, 3, true, false)),
                () -> assertArrayEquals(new int[] {2, 3, 0}, topK(input, 3, false, true)),
                () -> assertArrayEquals(new int[] {0, 2, 3}, topK(input, 3, false, false)));
    }

    @Test
    void locksNaNsLastInfinitiesSignedZeroAndStableBoundaryTies() {
        double[] input = {
                Double.NaN, 2.0, Double.NEGATIVE_INFINITY, 2.0, -0.0, +0.0,
                Double.POSITIVE_INFINITY, Double.longBitsToDouble(0xfff8000000000002L)};
        assertAll(
                () -> assertArrayEquals(new int[] {6, 1, 3, 5}, topK(input, 4, true, true)),
                () -> assertArrayEquals(new int[] {2, 4, 5, 1}, topK(input, 4, false, true)),
                () -> assertArrayEquals(new int[] {0, 7}, topK(input, 8, true, true)
                        .length == 8 ? Arrays.copyOfRange(topK(input, 8, true, true), 6, 8)
                        : new int[0]));
    }

    private static int[] topK(double[] values, int k, boolean largest, boolean sorted) {
        Integer[] indices = java.util.stream.IntStream.range(0, values.length).boxed()
                .toArray(Integer[]::new);
        Arrays.sort(indices, stableComparator(values, largest));
        int[] selected = Arrays.stream(indices).limit(k).mapToInt(Integer::intValue).toArray();
        if (!sorted) {
            Arrays.sort(selected);
        }
        return selected;
    }

    private static Comparator<Integer> stableComparator(double[] values, boolean largest) {
        return (left, right) -> {
            double a = values[left];
            double b = values[right];
            if (Double.isNaN(a) || Double.isNaN(b)) {
                return Double.isNaN(a) == Double.isNaN(b) ? Integer.compare(left, right)
                        : Double.isNaN(a) ? 1 : -1;
            }
            int comparison = Double.compare(a, b);
            if (largest) {
                comparison = -comparison;
            }
            return comparison != 0 ? comparison : Integer.compare(left, right);
        };
    }
}
