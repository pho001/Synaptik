package io.github.pho001.synaptik.config.compile;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.backend.contract.BackendIdRequirement;
import io.github.pho001.synaptik.backend.contract.BackendRequirement;
import java.io.Serializable;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.reflect.RecordComponent;
import java.util.Arrays;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class BackendIntentTest {
    @Test
    void hasTheExactPublicRecordAndFactoryShape() throws ReflectiveOperationException {
        RecordComponent[] components = BackendIntent.class.getRecordComponents();
        Constructor<?>[] constructors = BackendIntent.class.getDeclaredConstructors();
        Field[] fields = BackendIntent.class.getDeclaredFields();
        Set<String> publicMethods =
                Arrays.stream(BackendIntent.class.getDeclaredMethods())
                        .filter(method -> Modifier.isPublic(method.getModifiers()))
                        .map(Method::getName)
                        .collect(Collectors.toSet());
        Method unconstrained = BackendIntent.class.getDeclaredMethod("unconstrained");
        Method requiring =
                BackendIntent.class.getDeclaredMethod("requiring", BackendRequirement.class);

        assertAll(
                () ->
                        assertEquals(
                                "io.github.pho001.synaptik.config.compile",
                                BackendIntent.class.getPackageName()),
                () -> assertTrue(Modifier.isPublic(BackendIntent.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(BackendIntent.class.getModifiers())),
                () -> assertTrue(BackendIntent.class.isRecord()),
                () ->
                        assertArrayEquals(
                                new String[] {"hardRequirement"}, componentNames(components)),
                () ->
                        assertArrayEquals(
                                new Class<?>[] {Optional.class}, componentTypes(components)),
                () ->
                        assertEquals(
                                "java.util.Optional<io.github.pho001.synaptik.backend.contract.BackendRequirement>",
                                components[0].getGenericType().getTypeName()),
                () -> assertEquals(1, fields.length),
                () -> assertEquals("hardRequirement", fields[0].getName()),
                () -> assertEquals(Optional.class, fields[0].getType()),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPublic(constructors[0].getModifiers())),
                () ->
                        assertArrayEquals(
                                new Class<?>[] {Optional.class},
                                constructors[0].getParameterTypes()),
                () -> assertEquals(0, BackendIntent.class.getInterfaces().length),
                () -> assertFalse(Serializable.class.isAssignableFrom(BackendIntent.class)),
                () -> assertEquals(0, BackendIntent.class.getDeclaredClasses().length),
                () -> assertEquals(6, BackendIntent.class.getDeclaredMethods().length),
                () ->
                        assertEquals(
                                Set.of(
                                        "hardRequirement",
                                        "unconstrained",
                                        "requiring",
                                        "equals",
                                        "hashCode",
                                        "toString"),
                                publicMethods),
                () -> assertPublicStaticFactory(unconstrained),
                () -> assertArrayEquals(new Class<?>[0], unconstrained.getParameterTypes()),
                () -> assertPublicStaticFactory(requiring),
                () ->
                        assertArrayEquals(
                                new Class<?>[] {BackendRequirement.class},
                                requiring.getParameterTypes()));
    }

    @Test
    void directConstructionValidatesAndRetainsExactReferences() {
        BackendRequirement requirement =
                new BackendIdRequirement(new BackendId(new String("cuda")));
        Optional<BackendRequirement> present = Optional.of(requirement);
        Optional<BackendRequirement> empty = Optional.empty();

        BackendIntent constrained = new BackendIntent(present);
        BackendIntent unconstrained = new BackendIntent(empty);
        NullPointerException failure =
                assertThrows(NullPointerException.class, () -> new BackendIntent(null));

        assertAll(
                () -> assertEquals("hardRequirement", failure.getMessage()),
                () -> assertSame(present, constrained.hardRequirement()),
                () -> assertSame(requirement, constrained.hardRequirement().orElseThrow()),
                () -> assertSame(empty, unconstrained.hardRequirement()),
                () -> assertTrue(unconstrained.hardRequirement().isEmpty()));
    }

    @Test
    void factoriesProduceFreshValuesWithoutChangingRequirementMeaning() {
        BackendRequirement requirement =
                new BackendIdRequirement(new BackendId(new String("cuda")));

        BackendIntent firstUnconstrained = BackendIntent.unconstrained();
        BackendIntent secondUnconstrained = BackendIntent.unconstrained();
        BackendIntent firstRequired = BackendIntent.requiring(requirement);
        BackendIntent secondRequired = BackendIntent.requiring(requirement);
        NullPointerException failure =
                assertThrows(NullPointerException.class, () -> BackendIntent.requiring(null));

        assertAll(
                () -> assertNotSame(firstUnconstrained, secondUnconstrained),
                () -> assertEquals(firstUnconstrained, secondUnconstrained),
                () -> assertTrue(firstUnconstrained.hardRequirement().isEmpty()),
                () -> assertNotSame(firstRequired, secondRequired),
                () -> assertEquals(firstRequired, secondRequired),
                () -> assertNotSame(firstRequired.hardRequirement(), secondRequired.hardRequirement()),
                () -> assertSame(requirement, firstRequired.hardRequirement().orElseThrow()),
                () -> assertSame(requirement, secondRequired.hardRequirement().orElseThrow()),
                () -> assertEquals("requirement", failure.getMessage()));
    }

    @Test
    void preservesOrdinaryRecordValueBehavior() {
        BackendIntent first =
                BackendIntent.requiring(
                        new BackendIdRequirement(new BackendId(new String("cuda"))));
        BackendIntent equal =
                new BackendIntent(
                        Optional.of(new BackendIdRequirement(new BackendId(new String("cuda")))));
        BackendIntent different = BackendIntent.unconstrained();

        assertAll(
                () -> assertEquals(first, equal),
                () -> assertEquals(first.hashCode(), equal.hashCode()),
                () -> assertNotEquals(first, different),
                () ->
                        assertEquals(
                                "BackendIntent[hardRequirement=Optional[BackendIdRequirement[backendId=BackendId[value=cuda]]]]",
                                first.toString()));
    }

    @Test
    void exposesNoEvaluationPreferenceProfileServiceOrLifecycleSurface() {
        Set<String> prohibitedTerms =
                Set.of(
                        "available",
                        "capable",
                        "discover",
                        "evaluate",
                        "fallback",
                        "match",
                        "ownership",
                        "prefer",
                        "profile",
                        "rank",
                        "score",
                        "service",
                        "prepare",
                        "run",
                        "publish",
                        "serialize");
        Set<String> declaredNames =
                Arrays.stream(BackendIntent.class.getDeclaredMethods())
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
                () -> assertEquals(1, BackendIntent.class.getDeclaredFields().length),
                () -> assertEquals(0, BackendIntent.class.getDeclaredClasses().length));
    }

    private static String[] componentNames(RecordComponent[] components) {
        return Arrays.stream(components).map(RecordComponent::getName).toArray(String[]::new);
    }

    private static Class<?>[] componentTypes(RecordComponent[] components) {
        return Arrays.stream(components).map(RecordComponent::getType).toArray(Class<?>[]::new);
    }

    private static void assertPublicStaticFactory(Method method) {
        assertAll(
                () -> assertEquals(BackendIntent.class, method.getReturnType()),
                () -> assertTrue(Modifier.isPublic(method.getModifiers())),
                () -> assertTrue(Modifier.isStatic(method.getModifiers())));
    }
}
