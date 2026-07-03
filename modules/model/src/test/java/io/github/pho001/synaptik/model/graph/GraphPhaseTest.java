package io.github.pho001.synaptik.model.graph;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import org.junit.jupiter.api.Test;

class GraphPhaseTest {
    @Test
    void exposesExactlyForwardThenBackwardAsTheClosedCurrentVocabulary() {
        assertAll(
                () -> assertTrue(GraphPhase.class.isEnum()),
                () -> assertArrayEquals(
                        new GraphPhase[] {GraphPhase.FORWARD, GraphPhase.BACKWARD},
                        GraphPhase.values()),
                () -> assertEquals(
                        Arrays.asList("FORWARD", "BACKWARD"),
                        Arrays.stream(GraphPhase.values()).map(Enum::name).toList()));
    }

    @Test
    void usesOrdinaryEnumValueBehaviorWithoutAdditionalPhaseState() {
        assertAll(
                () -> assertEquals(GraphPhase.FORWARD, GraphPhase.valueOf("FORWARD")),
                () -> assertEquals(GraphPhase.BACKWARD, GraphPhase.valueOf("BACKWARD")),
                () -> assertTrue(Arrays.stream(GraphPhase.class.getDeclaredFields())
                        .filter(field -> !field.isSynthetic())
                        .allMatch(field -> field.isEnumConstant())));
    }
}
