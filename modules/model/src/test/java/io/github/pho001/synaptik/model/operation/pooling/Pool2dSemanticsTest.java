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
        var components = MaxPool2dAttrs.class.getRecordComponents();
        MaxPool2dAttrs attrs = new MaxPool2dAttrs(2, 3, 4, 5, 6, 7, 8, 9, true);
        OperationSignature expected = OperationSignature.fixed(MaxPool2dAttrs.class, 1, 1);
        Operation operation = new Operation(Pool2dKind.MAX_POOL2D, attrs);

        assertAll(
                () -> assertTrue(MaxPool2dAttrs.class.isRecord()),
                () -> assertEquals(List.of(OperationAttrs.class),
                        List.of(MaxPool2dAttrs.class.getInterfaces())),
                () -> assertEquals(
                        List.of("kernelHeight", "kernelWidth", "strideHeight", "strideWidth",
                                "paddingHeight", "paddingWidth", "dilationHeight",
                                "dilationWidth", "ceilMode"),
                        Arrays.stream(components).map(component -> component.getName()).toList()),
                () -> assertEquals(
                        List.of(long.class, long.class, long.class, long.class, long.class,
                                long.class, long.class, long.class, boolean.class),
                        Arrays.stream(components).map(component -> component.getType()).toList()),
                () -> assertEquals(0, MaxPool2dAttrs.class.getDeclaredClasses().length),
                () -> assertEquals(List.of(expected), Pool2dKind.MAX_POOL2D.signatures()),
                () -> assertEquals(expected, operation.signature()),
                () -> assertSame(attrs, operation.attrs()),
                () -> assertEquals(List.of(OperationKind.class),
                        List.of(Pool2dKind.class.getInterfaces())),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> Pool2dKind.MAX_POOL2D.signatures().clear()));
        assertSignatureEnumShape(Pool2dKind.class);
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
