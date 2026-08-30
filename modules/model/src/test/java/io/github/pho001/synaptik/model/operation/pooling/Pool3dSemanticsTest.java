package io.github.pho001.synaptik.model.operation.pooling;

import static io.github.pho001.synaptik.model.operation.OperationSignatureTest.assertSignatureEnumShape;
import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

final class Pool3dSemanticsTest {
    private static final List<String> COMPONENT_NAMES = List.of(
            "kernelDepth", "kernelHeight", "kernelWidth",
            "strideDepth", "strideHeight", "strideWidth",
            "paddingDepth", "paddingHeight", "paddingWidth",
            "dilationDepth", "dilationHeight", "dilationWidth", "ceilMode");

    @Test
    void exposesExactImmutableAttributeAndKindSurface() {
        var maxComponents = MaxPool3dAttrs.class.getRecordComponents();
        var averageComponents = AveragePool3dAttrs.class.getRecordComponents();
        MaxPool3dAttrs maxAttrs = maxValues(true);
        AveragePool3dAttrs averageAttrs = averageValues(true);
        OperationSignature maxSignature = OperationSignature.fixed(MaxPool3dAttrs.class, 1, 1);
        OperationSignature averageSignature = OperationSignature.fixed(
                AveragePool3dAttrs.class, 1, 1);
        Operation maxOperation = new Operation(Pool3dKind.MAX_POOL3D, maxAttrs);
        Operation averageOperation = new Operation(Pool3dKind.AVERAGE_POOL3D, averageAttrs);

        assertAll(
                () -> assertTrue(Modifier.isPublic(MaxPool3dAttrs.class.getModifiers())),
                () -> assertTrue(Modifier.isPublic(AveragePool3dAttrs.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(MaxPool3dAttrs.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(AveragePool3dAttrs.class.getModifiers())),
                () -> assertTrue(MaxPool3dAttrs.class.isRecord()),
                () -> assertTrue(AveragePool3dAttrs.class.isRecord()),
                () -> assertEquals(List.of(OperationAttrs.class),
                        List.of(MaxPool3dAttrs.class.getInterfaces())),
                () -> assertEquals(List.of(OperationAttrs.class),
                        List.of(AveragePool3dAttrs.class.getInterfaces())),
                () -> assertEquals(COMPONENT_NAMES, Arrays.stream(maxComponents)
                        .map(component -> component.getName()).toList()),
                () -> assertEquals(COMPONENT_NAMES, Arrays.stream(averageComponents)
                        .map(component -> component.getName()).toList()),
                () -> assertEquals(
                        List.of(long.class, long.class, long.class, long.class, long.class,
                                long.class, long.class, long.class, long.class, long.class,
                                long.class, long.class, boolean.class),
                        Arrays.stream(maxComponents).map(component -> component.getType()).toList()),
                () -> assertEquals(Arrays.stream(maxComponents)
                                .map(component -> component.getType()).toList(),
                        Arrays.stream(averageComponents)
                                .map(component -> component.getType()).toList()),
                () -> assertEquals(0, MaxPool3dAttrs.class.getDeclaredClasses().length),
                () -> assertEquals(0, AveragePool3dAttrs.class.getDeclaredClasses().length),
                () -> assertEquals(List.of(Pool3dKind.MAX_POOL3D, Pool3dKind.AVERAGE_POOL3D),
                        List.of(Pool3dKind.values())),
                () -> assertEquals(List.of(OperationKind.class),
                        List.of(Pool3dKind.class.getInterfaces())),
                () -> assertEquals(List.of(maxSignature), Pool3dKind.MAX_POOL3D.signatures()),
                () -> assertEquals(List.of(averageSignature),
                        Pool3dKind.AVERAGE_POOL3D.signatures()),
                () -> assertSame(Pool3dKind.MAX_POOL3D.signatures(),
                        Pool3dKind.MAX_POOL3D.signatures()),
                () -> assertSame(Pool3dKind.AVERAGE_POOL3D.signatures(),
                        Pool3dKind.AVERAGE_POOL3D.signatures()),
                () -> assertEquals(maxSignature, maxOperation.signature()),
                () -> assertEquals(averageSignature, averageOperation.signature()),
                () -> assertSame(maxAttrs, maxOperation.attrs()),
                () -> assertSame(averageAttrs, averageOperation.attrs()),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new Operation(Pool3dKind.MAX_POOL3D, averageAttrs)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new Operation(Pool3dKind.AVERAGE_POOL3D, maxAttrs)),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> Pool3dKind.MAX_POOL3D.signatures().clear()),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> Pool3dKind.AVERAGE_POOL3D.signatures().clear()));
        assertSignatureEnumShape(Pool3dKind.class);
    }

    @Test
    void retainsEveryComponentInEqualityAndLiteralCeilMode() {
        MaxPool3dAttrs max = maxValues(true);
        AveragePool3dAttrs average = averageValues(true);

        assertAll(
                () -> assertEquals(maxValues(true), max),
                () -> assertNotEquals(maxValues(false), max),
                () -> assertEquals(max.hashCode(), maxValues(true).hashCode()),
                () -> assertEquals(averageValues(true), average),
                () -> assertNotEquals(averageValues(false), average),
                () -> assertEquals(average.hashCode(), averageValues(true).hashCode()),
                () -> assertTrue(max.ceilMode()),
                () -> assertTrue(average.ceilMode()));
    }

    @Test
    void validatesMaximumComponentsInExactDeclarationOrder() {
        for (int index = 0; index < 12; index++) {
            long[] values = validComponents();
            for (int later = index + 1; later < 12; later++) {
                values[later] = invalidValue(later);
            }
            values[index] = invalidValue(index);
            String name = COMPONENT_NAMES.get(index);
            String qualifier = index >= 6 && index <= 8
                    ? " must be non-negative: -1" : " must be positive: 0";
            assertMaxFailure(values, name + qualifier);
        }
    }

    @Test
    void validatesAverageComponentsInExactDeclarationOrder() {
        for (int index = 0; index < 12; index++) {
            long[] values = validComponents();
            for (int later = index + 1; later < 12; later++) {
                values[later] = invalidValue(later);
            }
            values[index] = invalidValue(index);
            String name = COMPONENT_NAMES.get(index);
            String qualifier = index >= 6 && index <= 8
                    ? " must be non-negative: -1" : " must be positive: 0";
            assertAverageFailure(values, name + qualifier);
        }
    }

    private static long[] validComponents() {
        return new long[] {1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1};
    }

    private static long invalidValue(int index) {
        return index >= 6 && index <= 8 ? -1 : 0;
    }

    private static void assertMaxFailure(long[] values, String expected) {
        IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
                () -> new MaxPool3dAttrs(
                        values[0], values[1], values[2], values[3], values[4], values[5],
                        values[6], values[7], values[8], values[9], values[10], values[11], false));
        assertEquals(expected, failure.getMessage());
    }

    private static void assertAverageFailure(long[] values, String expected) {
        IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
                () -> new AveragePool3dAttrs(
                        values[0], values[1], values[2], values[3], values[4], values[5],
                        values[6], values[7], values[8], values[9], values[10], values[11], false));
        assertEquals(expected, failure.getMessage());
    }

    private static MaxPool3dAttrs maxValues(boolean ceilMode) {
        return new MaxPool3dAttrs(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, ceilMode);
    }

    private static AveragePool3dAttrs averageValues(boolean ceilMode) {
        return new AveragePool3dAttrs(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, ceilMode);
    }
}
