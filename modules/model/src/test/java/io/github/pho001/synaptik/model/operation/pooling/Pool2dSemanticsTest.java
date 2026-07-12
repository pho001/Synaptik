package io.github.pho001.synaptik.model.operation.pooling;

import static io.github.pho001.synaptik.model.operation.OperationSignatureTest.assertSignatureEnumShape;
import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

public final class Pool2dSemanticsTest {
    @Test
    void exposesExactAttributeAndKindSurface() {
        var maxComponents = MaxPool2dAttrs.class.getRecordComponents();
        var averageComponents = AveragePool2dAttrs.class.getRecordComponents();
        MaxPool2dAttrs maxAttrs = new MaxPool2dAttrs(2, 3, 4, 5, 6, 7, 8, 9, true);
        AveragePool2dAttrs averageAttrs = new AveragePool2dAttrs(
                2, 3, 4, 5, 6, 7, 8, 9, true);
        OperationSignature maxSignature = OperationSignature.fixed(MaxPool2dAttrs.class, 1, 1);
        OperationSignature averageSignature = OperationSignature.fixed(
                AveragePool2dAttrs.class, 1, 1);
        Operation maxOperation = new Operation(Pool2dKind.MAX_POOL2D, maxAttrs);
        Operation averageOperation = new Operation(Pool2dKind.AVERAGE_POOL2D, averageAttrs);

        assertAll(
                () -> assertTrue(MaxPool2dAttrs.class.isRecord()),
                () -> assertTrue(AveragePool2dAttrs.class.isRecord()),
                () -> assertEquals(List.of(OperationAttrs.class),
                        List.of(MaxPool2dAttrs.class.getInterfaces())),
                () -> assertEquals(List.of(OperationAttrs.class),
                        List.of(AveragePool2dAttrs.class.getInterfaces())),
                () -> assertEquals(
                        List.of("kernelHeight", "kernelWidth", "strideHeight", "strideWidth",
                                "paddingHeight", "paddingWidth", "dilationHeight",
                                "dilationWidth", "ceilMode"),
                        Arrays.stream(maxComponents).map(component -> component.getName()).toList()),
                () -> assertEquals(
                        Arrays.stream(maxComponents).map(component -> component.getName()).toList(),
                        Arrays.stream(averageComponents)
                                .map(component -> component.getName()).toList()),
                () -> assertEquals(
                        List.of(long.class, long.class, long.class, long.class, long.class,
                                long.class, long.class, long.class, boolean.class),
                        Arrays.stream(maxComponents).map(component -> component.getType()).toList()),
                () -> assertEquals(
                        Arrays.stream(maxComponents).map(component -> component.getType()).toList(),
                        Arrays.stream(averageComponents)
                                .map(component -> component.getType()).toList()),
                () -> assertEquals(0, MaxPool2dAttrs.class.getDeclaredClasses().length),
                () -> assertEquals(0, AveragePool2dAttrs.class.getDeclaredClasses().length),
                () -> assertEquals(List.of(Pool2dKind.MAX_POOL2D, Pool2dKind.AVERAGE_POOL2D),
                        List.of(Pool2dKind.values())),
                () -> assertEquals(List.of(maxSignature), Pool2dKind.MAX_POOL2D.signatures()),
                () -> assertEquals(List.of(averageSignature),
                        Pool2dKind.AVERAGE_POOL2D.signatures()),
                () -> assertEquals(maxSignature, maxOperation.signature()),
                () -> assertEquals(averageSignature, averageOperation.signature()),
                () -> assertSame(maxAttrs, maxOperation.attrs()),
                () -> assertSame(averageAttrs, averageOperation.attrs()),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new Operation(Pool2dKind.MAX_POOL2D, averageAttrs)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new Operation(Pool2dKind.AVERAGE_POOL2D, maxAttrs)),
                () -> assertEquals(List.of(OperationKind.class),
                        List.of(Pool2dKind.class.getInterfaces())),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> Pool2dKind.MAX_POOL2D.signatures().clear()),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> Pool2dKind.AVERAGE_POOL2D.signatures().clear()));
        assertSignatureEnumShape(Pool2dKind.class);
    }

    @Test
    void averageAttrsRetainCeilModeAndValidateComponentsInDeclarationOrder() {
        assertTrue(new AveragePool2dAttrs(1, 1, 1, 1, 0, 0, 1, 1, true).ceilMode());
        assertAverageFailure(new Arguments(0, 0, 0, 0, -1, -1, 0, 0),
                "kernelHeight must be positive: 0");
        assertAverageFailure(new Arguments(1, 0, 0, 0, -1, -1, 0, 0),
                "kernelWidth must be positive: 0");
        assertAverageFailure(new Arguments(1, 1, 0, 0, -1, -1, 0, 0),
                "strideHeight must be positive: 0");
        assertAverageFailure(new Arguments(1, 1, 1, 0, -1, -1, 0, 0),
                "strideWidth must be positive: 0");
        assertAverageFailure(new Arguments(1, 1, 1, 1, -1, -1, 0, 0),
                "paddingHeight must be non-negative: -1");
        assertAverageFailure(new Arguments(1, 1, 1, 1, 0, -1, 0, 0),
                "paddingWidth must be non-negative: -1");
        assertAverageFailure(new Arguments(1, 1, 1, 1, 0, 0, 0, 0),
                "dilationHeight must be positive: 0");
        assertAverageFailure(new Arguments(1, 1, 1, 1, 0, 0, 1, 0),
                "dilationWidth must be positive: 0");
    }

    @Test
    void retainsCeilModeAndValidatesComponentsInDeclarationOrder() {
        assertTrue(new MaxPool2dAttrs(1, 1, 1, 1, 0, 0, 1, 1, true).ceilMode());
        assertFailure(new Arguments(0, 0, 0, 0, -1, -1, 0, 0),
                "kernelHeight must be positive: 0");
        assertFailure(new Arguments(1, 0, 0, 0, -1, -1, 0, 0),
                "kernelWidth must be positive: 0");
        assertFailure(new Arguments(1, 1, 0, 0, -1, -1, 0, 0),
                "strideHeight must be positive: 0");
        assertFailure(new Arguments(1, 1, 1, 0, -1, -1, 0, 0),
                "strideWidth must be positive: 0");
        assertFailure(new Arguments(1, 1, 1, 1, -1, -1, 0, 0),
                "paddingHeight must be non-negative: -1");
        assertFailure(new Arguments(1, 1, 1, 1, 0, -1, 0, 0),
                "paddingWidth must be non-negative: -1");
        assertFailure(new Arguments(1, 1, 1, 1, 0, 0, 0, 0),
                "dilationHeight must be positive: 0");
        assertFailure(new Arguments(1, 1, 1, 1, 0, 0, 1, 0),
                "dilationWidth must be positive: 0");
    }

    private static void assertFailure(Arguments values, String expected) {
        IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
                () -> new MaxPool2dAttrs(
                        values.kernelHeight, values.kernelWidth,
                        values.strideHeight, values.strideWidth,
                        values.paddingHeight, values.paddingWidth,
                        values.dilationHeight, values.dilationWidth, false));
        assertEquals(expected, failure.getMessage());
    }

    private static void assertAverageFailure(Arguments values, String expected) {
        IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
                () -> new AveragePool2dAttrs(
                        values.kernelHeight, values.kernelWidth,
                        values.strideHeight, values.strideWidth,
                        values.paddingHeight, values.paddingWidth,
                        values.dilationHeight, values.dilationWidth, false));
        assertEquals(expected, failure.getMessage());
    }

    private record Arguments(
            long kernelHeight,
            long kernelWidth,
            long strideHeight,
            long strideWidth,
            long paddingHeight,
            long paddingWidth,
            long dilationHeight,
            long dilationWidth) {
    }
}
