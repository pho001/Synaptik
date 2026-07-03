package io.github.pho001.synaptik.model.graph;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class GraphValueTest {
    @Test
    void hasExactlyTheRequiredProducerFreeRecordState() {
        var components = GraphValue.class.getRecordComponents();
        var instanceFields = Arrays.stream(GraphValue.class.getDeclaredFields())
                .filter(field -> !Modifier.isStatic(field.getModifiers()))
                .toList();

        assertAll(
                () -> assertTrue(GraphValue.class.isRecord()),
                () -> assertEquals(2, components.length),
                () -> assertEquals("id", components[0].getName()),
                () -> assertEquals(ValueId.class, components[0].getType()),
                () -> assertEquals("descriptor", components[1].getName()),
                () -> assertEquals(TensorDescriptor.class, components[1].getType()),
                () -> assertEquals(
                        List.of("id", "descriptor"),
                        instanceFields.stream().map(field -> field.getName()).toList()),
                () -> assertTrue(instanceFields.stream().allMatch(
                        field -> Modifier.isPrivate(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers()))),
                () -> assertEquals(1, GraphValue.class.getDeclaredConstructors().length));
    }

    @Test
    void retainsExactImmutableReferencesWithoutAProducingNode() {
        ValueId id = new ValueId(7);
        TensorDescriptor descriptor = descriptor(DataType.FLOAT32, Shape.of(2, 3));

        GraphValue value = new GraphValue(id, descriptor);

        assertAll(
                () -> assertSame(id, value.id()),
                () -> assertSame(descriptor, value.descriptor()),
                () -> assertEquals(id, value.id()),
                () -> assertEquals(descriptor, value.descriptor()));
    }

    @Test
    void rejectsEachNullComponentWithItsExactName() {
        TensorDescriptor descriptor = descriptor(DataType.FLOAT32, Shape.scalar());
        NullPointerException nullId = assertThrows(
                NullPointerException.class, () -> new GraphValue(null, descriptor));
        NullPointerException nullDescriptor = assertThrows(
                NullPointerException.class, () -> new GraphValue(new ValueId(1), null));

        assertAll(
                () -> assertEquals("id", nullId.getMessage()),
                () -> assertEquals("descriptor", nullDescriptor.getMessage()));
    }

    @Test
    void usesStructuralRecordEqualityAndHashingAcrossCompleteState() {
        TensorDescriptor descriptor = descriptor(DataType.FLOAT32, Shape.of(2, 3));
        GraphValue first = new GraphValue(new ValueId(7), descriptor);
        GraphValue equal = new GraphValue(
                new ValueId(7), descriptor(DataType.FLOAT32, Shape.of(2, 3)));
        GraphValue differentId = new GraphValue(new ValueId(8), descriptor);
        GraphValue differentDescriptor = new GraphValue(
                new ValueId(7), descriptor(DataType.FLOAT64, Shape.of(2, 3)));

        assertAll(
                () -> assertEquals(first, equal),
                () -> assertEquals(first.hashCode(), equal.hashCode()),
                () -> assertNotEquals(first, differentId),
                () -> assertNotEquals(first, differentDescriptor));
    }

    @Test
    void diagnosticTextNamesBothComponentsAndRepresentativeValues() {
        String text = new GraphValue(
                        new ValueId(7), descriptor(DataType.FLOAT32, Shape.of(2, 3)))
                .toString();

        assertAll(
                () -> assertTrue(text.contains("GraphValue")),
                () -> assertTrue(text.contains("id=")),
                () -> assertTrue(text.contains("ValueId")),
                () -> assertTrue(text.contains("value=7")),
                () -> assertTrue(text.contains("descriptor=")),
                () -> assertTrue(text.contains("TensorDescriptor")),
                () -> assertTrue(text.contains("dataType=FLOAT32")));
    }

    private static TensorDescriptor descriptor(DataType dataType, Shape shape) {
        return new TensorDescriptor(dataType, shape, Optional.empty(), false);
    }
}
