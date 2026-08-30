package io.github.pho001.synaptik.model.operation.pooling;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

final class Pool1dAttrsTest {
    @Test
    void exposesExactPublicImmutableCompositionParameterSurface() {
        var maxComponents = MaxPool1dAttrs.class.getRecordComponents();
        var averageComponents = AveragePool1dAttrs.class.getRecordComponents();
        List<String> names = List.of(
                "kernelWidth", "strideWidth", "paddingWidth", "dilationWidth", "ceilMode");
        List<Class<?>> types = List.of(
                long.class, long.class, long.class, long.class, boolean.class);

        assertAll(
                () -> assertTrue(Modifier.isPublic(MaxPool1dAttrs.class.getModifiers())),
                () -> assertTrue(Modifier.isPublic(AveragePool1dAttrs.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(MaxPool1dAttrs.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(AveragePool1dAttrs.class.getModifiers())),
                () -> assertTrue(MaxPool1dAttrs.class.isRecord()),
                () -> assertTrue(AveragePool1dAttrs.class.isRecord()),
                () -> assertEquals(List.of(OperationAttrs.class),
                        List.of(MaxPool1dAttrs.class.getInterfaces())),
                () -> assertEquals(List.of(OperationAttrs.class),
                        List.of(AveragePool1dAttrs.class.getInterfaces())),
                () -> assertEquals(names,
                        Arrays.stream(maxComponents).map(component -> component.getName()).toList()),
                () -> assertEquals(names, Arrays.stream(averageComponents)
                        .map(component -> component.getName()).toList()),
                () -> assertEquals(types,
                        Arrays.stream(maxComponents).map(component -> component.getType()).toList()),
                () -> assertEquals(types, Arrays.stream(averageComponents)
                        .map(component -> component.getType()).toList()),
                () -> assertEquals(0, MaxPool1dAttrs.class.getDeclaredClasses().length),
                () -> assertEquals(0, AveragePool1dAttrs.class.getDeclaredClasses().length));
    }

    @Test
    void retainsEveryComponentInRecordEqualityAndLiteralCeilMode() {
        MaxPool1dAttrs max = new MaxPool1dAttrs(2, 3, 4, 5, true);
        AveragePool1dAttrs average = new AveragePool1dAttrs(2, 3, 4, 5, true);

        assertAll(
                () -> assertEquals(new MaxPool1dAttrs(2, 3, 4, 5, true), max),
                () -> assertNotEquals(new MaxPool1dAttrs(1, 3, 4, 5, true), max),
                () -> assertNotEquals(new MaxPool1dAttrs(2, 1, 4, 5, true), max),
                () -> assertNotEquals(new MaxPool1dAttrs(2, 3, 0, 5, true), max),
                () -> assertNotEquals(new MaxPool1dAttrs(2, 3, 4, 1, true), max),
                () -> assertNotEquals(new MaxPool1dAttrs(2, 3, 4, 5, false), max),
                () -> assertEquals(new AveragePool1dAttrs(2, 3, 4, 5, true), average),
                () -> assertNotEquals(new AveragePool1dAttrs(2, 3, 4, 5, false), average),
                () -> assertTrue(max.ceilMode()),
                () -> assertTrue(average.ceilMode()));
    }

    @Test
    void validatesMaximumComponentsInDeclarationOrderWithExactMessages() {
        assertMaxFailure(0, 0, -1, 0, "kernelWidth must be positive: 0");
        assertMaxFailure(1, 0, -1, 0, "strideWidth must be positive: 0");
        assertMaxFailure(1, 1, -1, 0, "paddingWidth must be non-negative: -1");
        assertMaxFailure(1, 1, 0, 0, "dilationWidth must be positive: 0");
    }

    @Test
    void validatesAverageComponentsInDeclarationOrderWithExactMessages() {
        assertAverageFailure(0, 0, -1, 0, "kernelWidth must be positive: 0");
        assertAverageFailure(1, 0, -1, 0, "strideWidth must be positive: 0");
        assertAverageFailure(1, 1, -1, 0, "paddingWidth must be non-negative: -1");
        assertAverageFailure(1, 1, 0, 0, "dilationWidth must be positive: 0");
    }

    @Test
    void noExistingPoolKindAcceptsPool1dCompositionParameters() {
        MaxPool1dAttrs max = new MaxPool1dAttrs(1, 1, 0, 1, false);
        AveragePool1dAttrs average = new AveragePool1dAttrs(1, 1, 0, 1, false);

        for (Pool2dKind kind : Pool2dKind.values()) {
            assertThrows(IllegalArgumentException.class, () -> new Operation(kind, max));
            assertThrows(IllegalArgumentException.class, () -> new Operation(kind, average));
        }
        assertEquals(List.of(Pool2dKind.MAX_POOL2D, Pool2dKind.AVERAGE_POOL2D),
                List.of(Pool2dKind.values()));
    }

    private static void assertMaxFailure(
            long kernel, long stride, long padding, long dilation, String message) {
        IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
                () -> new MaxPool1dAttrs(kernel, stride, padding, dilation, false));
        assertEquals(message, failure.getMessage());
    }

    private static void assertAverageFailure(
            long kernel, long stride, long padding, long dilation, String message) {
        IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
                () -> new AveragePool1dAttrs(kernel, stride, padding, dilation, false));
        assertEquals(message, failure.getMessage());
    }
}
