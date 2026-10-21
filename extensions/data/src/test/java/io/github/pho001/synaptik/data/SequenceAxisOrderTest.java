package io.github.pho001.synaptik.data;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

/** Keeps logical sequence ordering separate from any physical-layout API. */
final class SequenceAxisOrderTest {
    @Test
    void hasExactlyTheTwoApprovedConstantsAndNoCustomMethods() {
        assertTrue(SequenceAxisOrder.class.isEnum());
        assertArrayEquals(new SequenceAxisOrder[] {
            SequenceAxisOrder.TIME_MAJOR, SequenceAxisOrder.BATCH_MAJOR
        }, SequenceAxisOrder.values());
        assertEquals(Set.of("values", "valueOf"),
                Arrays.stream(SequenceAxisOrder.class.getDeclaredMethods())
                        .filter(method -> !method.isSynthetic()).map(method -> method.getName())
                        .collect(Collectors.toSet()));
        assertEquals(SequenceAxisOrder.TIME_MAJOR, SequenceAxisOrder.valueOf("TIME_MAJOR"));
        assertEquals(SequenceAxisOrder.BATCH_MAJOR, SequenceAxisOrder.valueOf("BATCH_MAJOR"));
    }
}
