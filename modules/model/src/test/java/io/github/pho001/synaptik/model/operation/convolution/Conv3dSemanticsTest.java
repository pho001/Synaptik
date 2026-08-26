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

public final class Conv3dSemanticsTest {
    @Test
    void exposesExactAttributeAndKindSurface() {
        var components = Conv3dAttrs.class.getRecordComponents();
        Conv3dAttrs attrs = new Conv3dAttrs(2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
        OperationSignature expected = OperationSignature.inputRange(
                Conv3dAttrs.class, 2, 3, 1);
        Operation operation = new Operation(Conv3dKind.CONV3D, attrs);

        assertAll(
                () -> assertTrue(Conv3dAttrs.class.isRecord()),
                () -> assertEquals(List.of(OperationAttrs.class),
                        List.of(Conv3dAttrs.class.getInterfaces())),
                () -> assertEquals(
                        List.of("strideDepth", "strideHeight", "strideWidth",
                                "paddingDepth", "paddingHeight", "paddingWidth",
                                "dilationDepth", "dilationHeight", "dilationWidth", "groups"),
                        Arrays.stream(components).map(component -> component.getName()).toList()),
                () -> assertTrue(Arrays.stream(components)
                        .allMatch(component -> component.getType() == long.class)),
                () -> assertEquals(0, Conv3dAttrs.class.getDeclaredClasses().length),
                () -> assertEquals(List.of(Conv3dKind.CONV3D),
                        List.of(Conv3dKind.values())),
                () -> assertEquals(List.of(expected), Conv3dKind.CONV3D.signatures()),
                () -> assertEquals(expected, operation.signature()),
                () -> assertSame(attrs, operation.attrs()),
                () -> assertEquals(List.of(OperationKind.class),
                        List.of(Conv3dKind.class.getInterfaces())),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> Conv3dKind.CONV3D.signatures().clear()));
        assertSignatureEnumShape(Conv3dKind.class);
    }

    @Test
    void defaultsSelectUnitGeometryZeroPaddingAndOneGroup() {
        assertEquals(new Conv3dAttrs(1, 1, 1, 0, 0, 0, 1, 1, 1, 1),
                Conv3dAttrs.defaults());
    }

    @Test
    void validatesComponentsInDeclarationOrderWithExactMessages() {
        assertFailure(new Arguments(0, 0, 0, -1, -1, -1, 0, 0, 0, 0),
                "strideDepth must be positive: 0");
        assertFailure(new Arguments(1, 0, 0, -1, -1, -1, 0, 0, 0, 0),
                "strideHeight must be positive: 0");
        assertFailure(new Arguments(1, 1, 0, -1, -1, -1, 0, 0, 0, 0),
                "strideWidth must be positive: 0");
        assertFailure(new Arguments(1, 1, 1, -1, -1, -1, 0, 0, 0, 0),
                "paddingDepth must be non-negative: -1");
        assertFailure(new Arguments(1, 1, 1, 0, -1, -1, 0, 0, 0, 0),
                "paddingHeight must be non-negative: -1");
        assertFailure(new Arguments(1, 1, 1, 0, 0, -1, 0, 0, 0, 0),
                "paddingWidth must be non-negative: -1");
        assertFailure(new Arguments(1, 1, 1, 0, 0, 0, 0, 0, 0, 0),
                "dilationDepth must be positive: 0");
        assertFailure(new Arguments(1, 1, 1, 0, 0, 0, 1, 0, 0, 0),
                "dilationHeight must be positive: 0");
        assertFailure(new Arguments(1, 1, 1, 0, 0, 0, 1, 1, 0, 0),
                "dilationWidth must be positive: 0");
        assertFailure(new Arguments(1, 1, 1, 0, 0, 0, 1, 1, 1, 0),
                "groups must be positive: 0");
    }

    private static void assertFailure(Arguments values, String expected) {
        IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
                () -> new Conv3dAttrs(
                        values.strideDepth, values.strideHeight, values.strideWidth,
                        values.paddingDepth, values.paddingHeight, values.paddingWidth,
                        values.dilationDepth, values.dilationHeight, values.dilationWidth,
                        values.groups));
        assertEquals(expected, failure.getMessage());
    }

    private record Arguments(
            long strideDepth,
            long strideHeight,
            long strideWidth,
            long paddingDepth,
            long paddingHeight,
            long paddingWidth,
            long dilationDepth,
            long dilationHeight,
            long dilationWidth,
            long groups) {
    }
}
