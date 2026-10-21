package io.github.pho001.synaptik.data;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import java.util.Arrays;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

/** Guards the shared padding vocabulary without adding custom enum behavior. */
final class PaddingSideTest {
    @Test
    void hasExactlyTwoSidesAndNoCustomMethods() {
        assertArrayEquals(new PaddingSide[] {PaddingSide.LEFT, PaddingSide.RIGHT}, PaddingSide.values());
        assertEquals(Set.of("values", "valueOf"), Arrays.stream(PaddingSide.class.getDeclaredMethods())
                .filter(method -> !method.isSynthetic()).map(method -> method.getName())
                .collect(Collectors.toSet()));
        assertEquals(PaddingSide.LEFT, PaddingSide.valueOf("LEFT"));
        assertEquals(PaddingSide.RIGHT, PaddingSide.valueOf("RIGHT"));
    }
}
