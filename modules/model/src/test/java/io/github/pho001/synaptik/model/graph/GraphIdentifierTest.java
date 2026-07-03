package io.github.pho001.synaptik.model.graph;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class GraphIdentifierTest {
    @Test
    void acceptsBoundaryValuesAndReturnsStoredValues() {
        assertAll(
                () -> assertEquals(0, new NodeId(0).value()),
                () -> assertEquals(Long.MAX_VALUE, new NodeId(Long.MAX_VALUE).value()),
                () -> assertEquals(0, new ValueId(0).value()),
                () -> assertEquals(Long.MAX_VALUE, new ValueId(Long.MAX_VALUE).value()));
    }

    @Test
    void rejectsNegativeValuesWithoutSentinels() {
        assertAll(
                () -> assertThrows(IllegalArgumentException.class, () -> new NodeId(-1)),
                () -> assertThrows(
                        IllegalArgumentException.class, () -> new NodeId(Long.MIN_VALUE)),
                () -> assertThrows(IllegalArgumentException.class, () -> new ValueId(-1)),
                () -> assertThrows(
                        IllegalArgumentException.class, () -> new ValueId(Long.MIN_VALUE)));
    }

    @Test
    void usesStructuralEqualityWithinEachRecordType() {
        NodeId node = new NodeId(7);
        ValueId value = new ValueId(7);

        assertAll(
                () -> assertTrue(NodeId.class.isRecord()),
                () -> assertTrue(ValueId.class.isRecord()),
                () -> assertEquals(node, new NodeId(7)),
                () -> assertEquals(node.hashCode(), new NodeId(7).hashCode()),
                () -> assertEquals(value, new ValueId(7)),
                () -> assertEquals(value.hashCode(), new ValueId(7).hashCode()),
                () -> assertNotEquals(node, value));
    }

    @Test
    void diagnosticTextIdentifiesConcreteTypeAndValue() {
        String nodeText = new NodeId(7).toString();
        String valueText = new ValueId(7).toString();

        assertAll(
                () -> assertTrue(nodeText.contains("NodeId")),
                () -> assertTrue(nodeText.contains("7")),
                () -> assertTrue(valueText.contains("ValueId")),
                () -> assertTrue(valueText.contains("7")));
    }
}
