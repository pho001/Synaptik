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

class CompileModeTest {
    @Test
    void hasTheExactEnumVocabularyOrderAndApiShape() {
        CompileMode[] values = CompileMode.values();
        Field[] projectFields =
                Arrays.stream(CompileMode.class.getDeclaredFields())
                        .filter(field -> !field.isSynthetic())
                        .toArray(Field[]::new);
        Method[] projectMethods =
                Arrays.stream(CompileMode.class.getDeclaredMethods())
                        .filter(method -> !method.isSynthetic())
                        .toArray(Method[]::new);
        Constructor<?>[] constructors = CompileMode.class.getDeclaredConstructors();

        assertAll(
                () ->
                        assertEquals(
                                "io.github.pho001.synaptik.config.compile",
                                CompileMode.class.getPackageName()),
                () -> assertTrue(Modifier.isPublic(CompileMode.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(CompileMode.class.getModifiers())),
                () -> assertTrue(CompileMode.class.isEnum()),
                () ->
                        assertArrayEquals(
                                new CompileMode[] {
                                    CompileMode.FORWARD_ONLY,
                                    CompileMode.FORWARD_AND_BACKWARD,
                                    CompileMode.TRAINING_STEP
                                },
                                values),
                () ->
                        assertArrayEquals(
                                new String[] {
                                    "FORWARD_ONLY", "FORWARD_AND_BACKWARD", "TRAINING_STEP"
                                },
                                Arrays.stream(values).map(Enum::name).toArray(String[]::new)),
                () -> assertEquals(3, projectFields.length),
                () ->
                        assertArrayEquals(
                                new String[] {
                                    "FORWARD_ONLY", "FORWARD_AND_BACKWARD", "TRAINING_STEP"
                                },
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
                () -> assertEquals(0, CompileMode.class.getInterfaces().length),
                () -> assertEquals(0, CompileMode.class.getDeclaredClasses().length));
    }

    @Test
    void usesOrdinaryEnumIdentityWithoutAliases() {
        assertAll(
                () ->
                        assertSame(
                                CompileMode.FORWARD_ONLY,
                                CompileMode.valueOf("FORWARD_ONLY")),
                () ->
                        assertSame(
                                CompileMode.FORWARD_AND_BACKWARD,
                                CompileMode.valueOf("FORWARD_AND_BACKWARD")),
                () ->
                        assertSame(
                                CompileMode.TRAINING_STEP,
                                CompileMode.valueOf("TRAINING_STEP")),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> CompileMode.valueOf("INFERENCE")),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> CompileMode.valueOf("BACKWARD_ONLY")),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> CompileMode.valueOf("TRAINING")));
    }
}
