package io.github.pho001.synaptik.model.operation.convolution;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.operation.OperationAttrs;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

final class Conv1dAttrsTest {
    @Test
    void exposesExactImmutableAttributeSurfaceAndDefaults() {
        var components = Conv1dAttrs.class.getRecordComponents();
        Conv1dAttrs defaults = Conv1dAttrs.defaults();

        assertAll(
                () -> assertTrue(Conv1dAttrs.class.isRecord()),
                () -> assertEquals(List.of(OperationAttrs.class),
                        List.of(Conv1dAttrs.class.getInterfaces())),
                () -> assertEquals(List.of("stride", "padding", "dilation", "groups"),
                        Arrays.stream(components).map(component -> component.getName()).toList()),
                () -> assertTrue(Arrays.stream(components)
                        .allMatch(component -> component.getType() == long.class)),
                () -> assertEquals(new Conv1dAttrs(1, 0, 1, 1), defaults),
                () -> assertEquals(1, defaults.stride()),
                () -> assertEquals(0, defaults.padding()),
                () -> assertEquals(1, defaults.dilation()),
                () -> assertEquals(1, defaults.groups()));
    }

    @Test
    void validatesComponentsInDeclarationOrderWithExactMessages() {
        assertMessage("stride must be positive: 0", () -> new Conv1dAttrs(0, -1, 0, 0));
        assertMessage("padding must be non-negative: -1", () -> new Conv1dAttrs(1, -1, 0, 0));
        assertMessage("dilation must be positive: 0", () -> new Conv1dAttrs(1, 0, 0, 0));
        assertMessage("groups must be positive: 0", () -> new Conv1dAttrs(1, 0, 1, 0));
        assertMessage("stride must be positive: -1", () -> new Conv1dAttrs(-1, 0, 1, 1));
    }

    private static void assertMessage(String message, Runnable construction) {
        assertEquals(message,
                assertThrows(IllegalArgumentException.class, construction::run).getMessage());
    }
}
