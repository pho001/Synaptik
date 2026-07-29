package io.github.pho001.synaptik.model.graph;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.tensor.TensorId;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class ForwardPublicationBindingTest {
    @Test
    void hasExactlyTheRequiredTwoComponentRecordState() {
        var components = ForwardPublicationBinding.class.getRecordComponents();
        var instanceFields = Arrays.stream(ForwardPublicationBinding.class.getDeclaredFields())
                .filter(field -> !Modifier.isStatic(field.getModifiers()))
                .toList();

        assertAll(
                () -> assertTrue(ForwardPublicationBinding.class.isRecord()),
                () -> assertEquals(2, components.length),
                () -> assertEquals("tensorId", components[0].getName()),
                () -> assertEquals(TensorId.class, components[0].getType()),
                () -> assertEquals("valueId", components[1].getName()),
                () -> assertEquals(ValueId.class, components[1].getType()),
                () -> assertEquals(
                        List.of("tensorId", "valueId"),
                        instanceFields.stream().map(field -> field.getName()).toList()),
                () -> assertTrue(instanceFields.stream().allMatch(
                        field -> Modifier.isPrivate(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers()))),
                () -> assertEquals(
                        1, ForwardPublicationBinding.class.getDeclaredConstructors().length));
    }

    @Test
    void retainsExactImmutableReferencesAcrossDistinctIdentityDomains() {
        TensorId tensorId = new TensorId(7);
        ValueId valueId = new ValueId(11);

        ForwardPublicationBinding binding = new ForwardPublicationBinding(tensorId, valueId);

        assertAll(
                () -> assertSame(tensorId, binding.tensorId()),
                () -> assertSame(valueId, binding.valueId()),
                () -> assertNotEquals(binding.tensorId(), binding.valueId()));
    }

    @Test
    void rejectsNullComponentsInDeclarationOrderWithExactMessages() {
        NullPointerException nullTensorId = assertThrows(
                NullPointerException.class,
                () -> new ForwardPublicationBinding(null, null));
        NullPointerException nullValueId = assertThrows(
                NullPointerException.class,
                () -> new ForwardPublicationBinding(new TensorId(1), null));

        assertAll(
                () -> assertEquals("tensorId", nullTensorId.getMessage()),
                () -> assertEquals("valueId", nullValueId.getMessage()));
    }

    @Test
    void usesStructuralRecordEqualityHashingAndDiagnosticText() {
        ForwardPublicationBinding first =
                new ForwardPublicationBinding(new TensorId(1), new ValueId(2));
        ForwardPublicationBinding equal =
                new ForwardPublicationBinding(new TensorId(1), new ValueId(2));
        ForwardPublicationBinding differentTensor =
                new ForwardPublicationBinding(new TensorId(3), new ValueId(2));
        ForwardPublicationBinding differentValue =
                new ForwardPublicationBinding(new TensorId(1), new ValueId(4));
        String text = first.toString();

        assertAll(
                () -> assertEquals(first, equal),
                () -> assertEquals(first.hashCode(), equal.hashCode()),
                () -> assertNotEquals(first, differentTensor),
                () -> assertNotEquals(first, differentValue),
                () -> assertTrue(text.contains("ForwardPublicationBinding")),
                () -> assertTrue(text.contains("tensorId=")),
                () -> assertTrue(text.contains("TensorId[value=1]")),
                () -> assertTrue(text.contains("valueId=")),
                () -> assertTrue(text.contains("ValueId[value=2]")));
    }
}
