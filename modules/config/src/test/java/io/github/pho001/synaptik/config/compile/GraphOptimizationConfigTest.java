package io.github.pho001.synaptik.config.compile;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.Serializable;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.reflect.RecordComponent;
import java.util.Arrays;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class GraphOptimizationConfigTest {
    @Test
    void hasTheExactPublicRecordAndFactoryShape() throws ReflectiveOperationException {
        RecordComponent[] components = GraphOptimizationConfig.class.getRecordComponents();
        Constructor<?>[] constructors = GraphOptimizationConfig.class.getDeclaredConstructors();
        Field[] fields = GraphOptimizationConfig.class.getDeclaredFields();
        Set<String> publicMethods =
                Arrays.stream(GraphOptimizationConfig.class.getDeclaredMethods())
                        .filter(method -> Modifier.isPublic(method.getModifiers()))
                        .map(Method::getName)
                        .collect(Collectors.toSet());
        Method disabled = GraphOptimizationConfig.class.getDeclaredMethod("disabled");
        Method standard = GraphOptimizationConfig.class.getDeclaredMethod("standard");

        assertAll(
                () ->
                        assertEquals(
                                "io.github.pho001.synaptik.config.compile",
                                GraphOptimizationConfig.class.getPackageName()),
                () -> assertTrue(Modifier.isPublic(GraphOptimizationConfig.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(GraphOptimizationConfig.class.getModifiers())),
                () -> assertTrue(GraphOptimizationConfig.class.isRecord()),
                () ->
                        assertArrayEquals(
                                new String[] {"optionalOptimizationsEnabled"},
                                componentNames(components)),
                () ->
                        assertArrayEquals(
                                new Class<?>[] {boolean.class}, componentTypes(components)),
                () -> assertEquals(1, fields.length),
                () -> assertEquals("optionalOptimizationsEnabled", fields[0].getName()),
                () -> assertEquals(boolean.class, fields[0].getType()),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPublic(constructors[0].getModifiers())),
                () ->
                        assertArrayEquals(
                                new Class<?>[] {boolean.class},
                                constructors[0].getParameterTypes()),
                () -> assertEquals(0, GraphOptimizationConfig.class.getInterfaces().length),
                () ->
                        assertFalse(
                                Serializable.class.isAssignableFrom(
                                        GraphOptimizationConfig.class)),
                () -> assertEquals(0, GraphOptimizationConfig.class.getDeclaredClasses().length),
                () -> assertEquals(6, GraphOptimizationConfig.class.getDeclaredMethods().length),
                () ->
                        assertEquals(
                                Set.of(
                                        "optionalOptimizationsEnabled",
                                        "disabled",
                                        "standard",
                                        "equals",
                                        "hashCode",
                                        "toString"),
                                publicMethods),
                () -> assertPublicStaticFactory(disabled),
                () -> assertArrayEquals(new Class<?>[0], disabled.getParameterTypes()),
                () -> assertPublicStaticFactory(standard),
                () -> assertArrayEquals(new Class<?>[0], standard.getParameterTypes()));
    }

    @Test
    void directConstructionRetainsEitherPrimitiveValue() {
        GraphOptimizationConfig disabled = new GraphOptimizationConfig(false);
        GraphOptimizationConfig enabled = new GraphOptimizationConfig(true);

        assertAll(
                () -> assertFalse(disabled.optionalOptimizationsEnabled()),
                () -> assertTrue(enabled.optionalOptimizationsEnabled()));
    }

    @Test
    void factoriesProduceFreshValuesWithTheirExactPermission() {
        GraphOptimizationConfig firstDisabled = GraphOptimizationConfig.disabled();
        GraphOptimizationConfig secondDisabled = GraphOptimizationConfig.disabled();
        GraphOptimizationConfig firstStandard = GraphOptimizationConfig.standard();
        GraphOptimizationConfig secondStandard = GraphOptimizationConfig.standard();

        assertAll(
                () -> assertNotSame(firstDisabled, secondDisabled),
                () -> assertEquals(firstDisabled, secondDisabled),
                () -> assertFalse(firstDisabled.optionalOptimizationsEnabled()),
                () -> assertNotSame(firstStandard, secondStandard),
                () -> assertEquals(firstStandard, secondStandard),
                () -> assertTrue(firstStandard.optionalOptimizationsEnabled()),
                () -> assertNotEquals(firstDisabled, firstStandard));
    }

    @Test
    void preservesOrdinaryRecordValueBehavior() {
        GraphOptimizationConfig first = new GraphOptimizationConfig(true);
        GraphOptimizationConfig equal = GraphOptimizationConfig.standard();
        GraphOptimizationConfig different = GraphOptimizationConfig.disabled();

        assertAll(
                () -> assertEquals(first, equal),
                () -> assertEquals(first.hashCode(), equal.hashCode()),
                () -> assertNotEquals(first, different),
                () ->
                        assertEquals(
                                "GraphOptimizationConfig[optionalOptimizationsEnabled=true]",
                                first.toString()));
    }

    @Test
    void exposesNoPassNumericalBackendTraceOrLifecycleSurface() {
        Set<String> prohibitedTerms =
                Set.of(
                        "aggressive",
                        "approximate",
                        "autograd",
                        "backend",
                        "budget",
                        "capture",
                        "canonical",
                        "debug",
                        "execute",
                        "infer",
                        "level",
                        "pass",
                        "plan",
                        "prepare",
                        "profile",
                        "publish",
                        "registry",
                        "run",
                        "seed",
                        "serialize",
                        "timeout",
                        "trace",
                        "unsafe",
                        "validate");
        Set<String> declaredNames =
                Arrays.stream(GraphOptimizationConfig.class.getDeclaredMethods())
                        .map(Method::getName)
                        .map(String::toLowerCase)
                        .collect(Collectors.toSet());

        assertAll(
                () ->
                        assertFalse(
                                declaredNames.stream()
                                        .anyMatch(
                                                name ->
                                                        prohibitedTerms.stream()
                                                                .anyMatch(name::contains))),
                () -> assertEquals(1, GraphOptimizationConfig.class.getDeclaredFields().length),
                () -> assertEquals(0, GraphOptimizationConfig.class.getDeclaredClasses().length));
    }

    private static String[] componentNames(RecordComponent[] components) {
        return Arrays.stream(components).map(RecordComponent::getName).toArray(String[]::new);
    }

    private static Class<?>[] componentTypes(RecordComponent[] components) {
        return Arrays.stream(components).map(RecordComponent::getType).toArray(Class<?>[]::new);
    }

    private static void assertPublicStaticFactory(Method method) {
        assertAll(
                () -> assertEquals(GraphOptimizationConfig.class, method.getReturnType()),
                () -> assertTrue(Modifier.isPublic(method.getModifiers())),
                () -> assertTrue(Modifier.isStatic(method.getModifiers())));
    }
}
