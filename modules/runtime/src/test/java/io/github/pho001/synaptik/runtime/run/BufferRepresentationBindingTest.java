package io.github.pho001.synaptik.runtime.run;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import java.io.Serializable;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class BufferRepresentationBindingTest {
    @Test
    void ownershipHasTheExactTwoValueSurface() {
        assertAll(
                () -> assertTrue(Modifier.isPublic(RunResourceOwnership.class.getModifiers())),
                () -> assertTrue(RunResourceOwnership.class.isEnum()),
                () -> assertArrayEquals(
                        new RunResourceOwnership[] {
                            RunResourceOwnership.BORROWED, RunResourceOwnership.RUN_OWNED
                        },
                        RunResourceOwnership.values()),
                () -> assertEquals(0, RunResourceOwnership.class.getDeclaredClasses().length));
    }

    @Test
    void bindingHasTheExactPublicRecordShape() throws ReflectiveOperationException {
        var type = BufferRepresentationBinding.class;
        var components = type.getRecordComponents();
        var constructors = type.getDeclaredConstructors();

        assertAll(
                () -> assertTrue(Modifier.isPublic(type.getModifiers())),
                () -> assertTrue(Modifier.isFinal(type.getModifiers())),
                () -> assertTrue(type.isRecord()),
                () -> assertEquals(2, components.length),
                () -> assertEquals("representation", components[0].getName()),
                () -> assertEquals(BufferRepresentation.class, components[0].getType()),
                () -> assertEquals("ownership", components[1].getName()),
                () -> assertEquals(RunResourceOwnership.class, components[1].getType()),
                () -> assertEquals(2, type.getDeclaredFields().length),
                () -> assertEquals(1, constructors.length),
                () -> assertArrayEquals(
                        new Class<?>[] {
                            BufferRepresentation.class, RunResourceOwnership.class
                        },
                        constructors[0].getParameterTypes()),
                () -> assertTrue(Modifier.isPublic(constructors[0].getModifiers())),
                () -> assertEquals(0, type.getInterfaces().length),
                () -> assertEquals(0, type.getDeclaredClasses().length),
                () -> assertFalse(Serializable.class.isAssignableFrom(type)),
                () -> assertEquals(
                        Set.of(
                                "representation",
                                "ownership",
                                "equals",
                                "hashCode",
                                "toString"),
                        Arrays.stream(type.getDeclaredMethods())
                                .filter(method -> Modifier.isPublic(method.getModifiers()))
                                .map(method -> method.getName())
                                .collect(Collectors.toSet())));
    }

    @Test
    void validatesRepresentationBeforeOwnershipWithExactFailures() {
        NullPointerException representationFailure =
                assertThrows(
                        NullPointerException.class,
                        () -> new BufferRepresentationBinding(null, null));
        BufferRepresentation representation = () -> {};
        NullPointerException ownershipFailure =
                assertThrows(
                        NullPointerException.class,
                        () -> new BufferRepresentationBinding(representation, null));

        assertAll(
                () -> assertEquals("representation", representationFailure.getMessage()),
                () -> assertEquals("ownership", ownershipFailure.getMessage()));
    }

    @Test
    void retainsExactReferencesAndUsesOrdinaryRecordSemantics() {
        BufferRepresentation representation = () -> {};
        BufferRepresentationBinding first =
                new BufferRepresentationBinding(
                        representation, RunResourceOwnership.RUN_OWNED);
        BufferRepresentationBinding equal =
                new BufferRepresentationBinding(
                        representation, RunResourceOwnership.RUN_OWNED);
        BufferRepresentationBinding different =
                new BufferRepresentationBinding(
                        representation, RunResourceOwnership.BORROWED);

        assertAll(
                () -> assertSame(representation, first.representation()),
                () -> assertSame(RunResourceOwnership.RUN_OWNED, first.ownership()),
                () -> assertEquals(first, equal),
                () -> assertEquals(first.hashCode(), equal.hashCode()),
                () -> assertNotEquals(first, different));
    }
}
