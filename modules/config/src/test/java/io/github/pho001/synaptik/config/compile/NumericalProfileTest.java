package io.github.pho001.synaptik.config.compile;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class NumericalProfileTest {
    @Test
    void hasTheExactEnumVocabularyOrderAndApiShape() {
        NumericalProfile[] values = NumericalProfile.values();
        Field[] projectFields =
                Arrays.stream(NumericalProfile.class.getDeclaredFields())
                        .filter(field -> !field.isSynthetic())
                        .toArray(Field[]::new);
        Method[] projectMethods =
                Arrays.stream(NumericalProfile.class.getDeclaredMethods())
                        .filter(method -> !method.isSynthetic())
                        .toArray(Method[]::new);
        Constructor<?>[] constructors = NumericalProfile.class.getDeclaredConstructors();

        assertAll(
                () ->
                        assertEquals(
                                "io.github.pho001.synaptik.config.compile",
                                NumericalProfile.class.getPackageName()),
                () -> assertTrue(Modifier.isPublic(NumericalProfile.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(NumericalProfile.class.getModifiers())),
                () -> assertTrue(NumericalProfile.class.isEnum()),
                () ->
                        assertArrayEquals(
                                new NumericalProfile[] {
                                    NumericalProfile.STRICT_IEEE, NumericalProfile.ACCELERATOR
                                },
                                values),
                () ->
                        assertArrayEquals(
                                new String[] {"STRICT_IEEE", "ACCELERATOR"},
                                Arrays.stream(values).map(Enum::name).toArray(String[]::new)),
                () -> assertEquals(2, projectFields.length),
                () ->
                        assertArrayEquals(
                                new String[] {"STRICT_IEEE", "ACCELERATOR"},
                                Arrays.stream(projectFields)
                                        .map(Field::getName)
                                        .toArray(String[]::new)),
                () ->
                        assertTrue(
                                Arrays.stream(projectFields)
                                        .allMatch(
                                                field ->
                                                        field.isEnumConstant()
                                                                && Modifier.isPublic(
                                                                        field.getModifiers())
                                                                && Modifier.isStatic(
                                                                        field.getModifiers())
                                                                && Modifier.isFinal(
                                                                        field.getModifiers()))),
                () ->
                        assertEquals(
                                Set.of("values", "valueOf"),
                                Arrays.stream(projectMethods)
                                        .map(Method::getName)
                                        .collect(Collectors.toSet())),
                () -> assertEquals(2, projectMethods.length),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPrivate(constructors[0].getModifiers())),
                () ->
                        assertArrayEquals(
                                new Class<?>[] {String.class, int.class},
                                constructors[0].getParameterTypes()),
                () -> assertEquals(0, NumericalProfile.class.getInterfaces().length),
                () -> assertEquals(0, NumericalProfile.class.getDeclaredClasses().length));
    }

    @Test
    void usesOrdinaryEnumIdentityWithoutAliases() {
        assertAll(
                () ->
                        assertSame(
                                NumericalProfile.STRICT_IEEE,
                                NumericalProfile.valueOf("STRICT_IEEE")),
                () ->
                        assertSame(
                                NumericalProfile.ACCELERATOR,
                                NumericalProfile.valueOf("ACCELERATOR")),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> NumericalProfile.valueOf("DEFAULT")),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> NumericalProfile.valueOf("FAST")),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> NumericalProfile.valueOf("RELAXED")));
    }
}
