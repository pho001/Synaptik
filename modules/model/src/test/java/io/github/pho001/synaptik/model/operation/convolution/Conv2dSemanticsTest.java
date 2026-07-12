package io.github.pho001.synaptik.model.operation.convolution;

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

public final class Conv2dSemanticsTest {
    @Test
    void exposesExactAttributeAndKindSurface() {
        var components = Conv2dAttrs.class.getRecordComponents();
        Conv2dAttrs attrs = new Conv2dAttrs(2, 3, 4, 5, 6, 7, 8);
        OperationSignature expected = OperationSignature.inputRange(
                Conv2dAttrs.class, 2, 3, 1);
        Operation operation = new Operation(Conv2dKind.CONV2D, attrs);

        assertAll(
                () -> assertTrue(Conv2dAttrs.class.isRecord()),
                () -> assertEquals(List.of(OperationAttrs.class),
                        List.of(Conv2dAttrs.class.getInterfaces())),
                () -> assertEquals(
                        List.of("strideHeight", "strideWidth", "paddingHeight", "paddingWidth",
                                "dilationHeight", "dilationWidth", "groups"),
                        Arrays.stream(components).map(component -> component.getName()).toList()),
                () -> assertTrue(Arrays.stream(components)
                        .allMatch(component -> component.getType() == long.class)),
                () -> assertEquals(0, Conv2dAttrs.class.getDeclaredClasses().length),
                () -> assertEquals(List.of(expected), Conv2dKind.CONV2D.signatures()),
                () -> assertEquals(expected, operation.signature()),
                () -> assertSame(attrs, operation.attrs()),
                () -> assertEquals(List.of(OperationKind.class),
                        List.of(Conv2dKind.class.getInterfaces())),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> Conv2dKind.CONV2D.signatures().clear()));
        assertSignatureEnumShape(Conv2dKind.class);
    }

    @Test
    void defaultsSelectUnitGeometryZeroPaddingAndOneGroup() {
        assertEquals(new Conv2dAttrs(1, 1, 0, 0, 1, 1, 1), Conv2dAttrs.defaults());
    }

    @Test
    void validatesComponentsInDeclarationOrderWithExactMessages() {
        assertFailure(new Conv2dAttrsArguments(0, 0, -1, -1, 0, 0, 0),
                "strideHeight must be positive: 0");
        assertFailure(new Conv2dAttrsArguments(1, 0, -1, -1, 0, 0, 0),
                "strideWidth must be positive: 0");
        assertFailure(new Conv2dAttrsArguments(1, 1, -1, -1, 0, 0, 0),
                "paddingHeight must be non-negative: -1");
        assertFailure(new Conv2dAttrsArguments(1, 1, 0, -1, 0, 0, 0),
                "paddingWidth must be non-negative: -1");
        assertFailure(new Conv2dAttrsArguments(1, 1, 0, 0, 0, 0, 0),
                "dilationHeight must be positive: 0");
        assertFailure(new Conv2dAttrsArguments(1, 1, 0, 0, 1, 0, 0),
                "dilationWidth must be positive: 0");
        assertFailure(new Conv2dAttrsArguments(1, 1, 0, 0, 1, 1, 0),
                "groups must be positive: 0");
    }

    private static void assertFailure(Conv2dAttrsArguments values, String expected) {
        IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
                () -> new Conv2dAttrs(
                        values.strideHeight, values.strideWidth,
                        values.paddingHeight, values.paddingWidth,
                        values.dilationHeight, values.dilationWidth, values.groups));
        assertEquals(expected, failure.getMessage());
    }

    private record Conv2dAttrsArguments(
            long strideHeight,
            long strideWidth,
            long paddingHeight,
            long paddingWidth,
            long dilationHeight,
            long dilationWidth,
            long groups) {
    }
}
