package io.github.pho001.synaptik.config.tuning;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.Serializable;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.reflect.RecordComponent;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class ModelAutotuningConfigTest {
    @Test
    void hasTheExactTopLevelRecordShape() {
        RecordComponent[] components = ModelAutotuningConfig.class.getRecordComponents();
        Constructor<?>[] constructors = ModelAutotuningConfig.class.getDeclaredConstructors();
        Field[] fields = ModelAutotuningConfig.class.getDeclaredFields();
        Set<String> methods = publicDeclaredMethodNames(ModelAutotuningConfig.class);
        Set<Class<?>> nestedTypes = Set.of(ModelAutotuningConfig.class.getDeclaredClasses());

        assertAll(
                () ->
                        assertEquals(
                                "io.github.pho001.synaptik.config.tuning",
                                ModelAutotuningConfig.class.getPackageName()),
                () -> assertTrue(Modifier.isPublic(ModelAutotuningConfig.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(ModelAutotuningConfig.class.getModifiers())),
                () -> assertTrue(ModelAutotuningConfig.class.isRecord()),
                () -> assertFalse(Serializable.class.isAssignableFrom(ModelAutotuningConfig.class)),
                () -> assertEquals(0, ModelAutotuningConfig.class.getInterfaces().length),
                () ->
                        assertArrayEquals(
                                new String[] {
                                    "objective",
                                    "budget",
                                    "representativeProfile",
                                    "fallbackPolicy",
                                    "workloadCache"
                                },
                                componentNames(components)),
                () ->
                        assertArrayEquals(
                                new Class<?>[] {
                                    ModelAutotuningConfig.Objective.class,
                                    ModelAutotuningConfig.Budget.class,
                                    ModelAutotuningConfig.RepresentativeProfileIdentity.class,
                                    ModelAutotuningConfig.FallbackPolicy.class,
                                    Path.class
                                },
                                componentTypes(components)),
                () -> assertEquals(5, fields.length),
                () -> assertArrayEquals(componentNames(components), fieldNames(fields)),
                () -> assertArrayEquals(componentTypes(components), fieldTypes(fields)),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPublic(constructors[0].getModifiers())),
                () ->
                        assertArrayEquals(
                                componentTypes(components), constructors[0].getParameterTypes()),
                () ->
                        assertEquals(
                                Set.of(
                                        "objective",
                                        "budget",
                                        "representativeProfile",
                                        "fallbackPolicy",
                                        "workloadCache",
                                        "equals",
                                        "hashCode",
                                        "toString"),
                                methods),
                () -> assertEquals(8, ModelAutotuningConfig.class.getDeclaredMethods().length),
                () ->
                        assertEquals(
                                Set.of(
                                        ModelAutotuningConfig.Objective.class,
                                        ModelAutotuningConfig.Budget.class,
                                        ModelAutotuningConfig.RepresentativeProfileIdentity.class,
                                        ModelAutotuningConfig.FallbackPolicy.class),
                                nestedTypes),
                () ->
                        assertTrue(
                                nestedTypes.stream()
                                        .allMatch(type -> Modifier.isPublic(type.getModifiers()))));
    }

    @Test
    void validatesEveryOuterComponentAndRetainsExactValues() {
        ModelAutotuningConfig.Objective objective =
                ModelAutotuningConfig.Objective.MIN_MEDIAN_ELAPSED_NANOS;
        ModelAutotuningConfig.Budget budget = new ModelAutotuningConfig.Budget(1, 2, 0, 3);
        ModelAutotuningConfig.RepresentativeProfileIdentity profile =
                new ModelAutotuningConfig.RepresentativeProfileIdentity(1, new byte[] {4});
        ModelAutotuningConfig.FallbackPolicy fallback =
                ModelAutotuningConfig.FallbackPolicy.REQUIRE_TUNED_RESULT;
        Path cache = Path.of("requested-cache.bin");

        ModelAutotuningConfig config =
                new ModelAutotuningConfig(objective, budget, profile, fallback, cache);

        assertAll(
                () -> assertSame(objective, config.objective()),
                () -> assertSame(budget, config.budget()),
                () -> assertSame(profile, config.representativeProfile()),
                () -> assertSame(fallback, config.fallbackPolicy()),
                () -> assertSame(cache, config.workloadCache()),
                () ->
                        assertNullComponent(
                                "objective",
                                () ->
                                        new ModelAutotuningConfig(
                                                null, budget, profile, fallback, cache)),
                () ->
                        assertNullComponent(
                                "budget",
                                () ->
                                        new ModelAutotuningConfig(
                                                objective, null, profile, fallback, cache)),
                () ->
                        assertNullComponent(
                                "representativeProfile",
                                () ->
                                        new ModelAutotuningConfig(
                                                objective, budget, null, fallback, cache)),
                () ->
                        assertNullComponent(
                                "fallbackPolicy",
                                () ->
                                        new ModelAutotuningConfig(
                                                objective, budget, profile, null, cache)),
                () ->
                        assertNullComponent(
                                "workloadCache",
                                () ->
                                        new ModelAutotuningConfig(
                                                objective, budget, profile, fallback, null)));
    }

    @Test
    void preservesOrdinaryOuterRecordValueBehavior() {
        ModelAutotuningConfig first = config(new byte[] {1, 2});
        ModelAutotuningConfig equal = config(new byte[] {1, 2});
        ModelAutotuningConfig different = config(new byte[] {1, 3});

        assertAll(
                () -> assertEquals(first, equal),
                () -> assertEquals(first.hashCode(), equal.hashCode()),
                () -> assertNotEquals(first, different),
                () -> assertTrue(first.toString().startsWith("ModelAutotuningConfig[objective=")),
                () -> assertTrue(first.toString().contains("workloadCache=requested-cache.bin")));
    }

    @Test
    void objectiveAndFallbackHaveTheExactEnumVocabularies() {
        assertAll(
                () ->
                        assertExactEnum(
                                ModelAutotuningConfig.Objective.class,
                                new String[] {"MIN_MEDIAN_ELAPSED_NANOS"}),
                () ->
                        assertExactEnum(
                                ModelAutotuningConfig.FallbackPolicy.class,
                                new String[] {"REQUIRE_TUNED_RESULT", "ALLOW_SAFE_HEURISTIC"}));
    }

    @Test
    void budgetHasExactShapeAndRetainsBoundaryValues() {
        Class<ModelAutotuningConfig.Budget> type = ModelAutotuningConfig.Budget.class;
        RecordComponent[] components = type.getRecordComponents();
        ModelAutotuningConfig.Budget boundary = new ModelAutotuningConfig.Budget(1, 1, 0, 1);

        assertAll(
                () -> assertTrue(Modifier.isPublic(type.getModifiers())),
                () -> assertTrue(Modifier.isStatic(type.getModifiers())),
                () -> assertTrue(Modifier.isFinal(type.getModifiers())),
                () -> assertTrue(type.isRecord()),
                () -> assertFalse(Serializable.class.isAssignableFrom(type)),
                () -> assertEquals(0, type.getInterfaces().length),
                () -> assertEquals(0, type.getDeclaredClasses().length),
                () ->
                        assertArrayEquals(
                                new String[] {
                                    "maximumDistinctCacheMisses",
                                    "maximumCandidatesPerMiss",
                                    "warmupCount",
                                    "timedSampleCount"
                                },
                                componentNames(components)),
                () ->
                        assertArrayEquals(
                                new Class<?>[] {int.class, int.class, int.class, int.class},
                                componentTypes(components)),
                () -> assertEquals(4, type.getDeclaredFields().length),
                () -> assertEquals(1, type.getDeclaredConstructors().length),
                () ->
                        assertEquals(
                                Set.of(
                                        "maximumDistinctCacheMisses",
                                        "maximumCandidatesPerMiss",
                                        "warmupCount",
                                        "timedSampleCount",
                                        "equals",
                                        "hashCode",
                                        "toString"),
                                publicDeclaredMethodNames(type)),
                () -> assertEquals(1, boundary.maximumDistinctCacheMisses()),
                () -> assertEquals(1, boundary.maximumCandidatesPerMiss()),
                () -> assertEquals(0, boundary.warmupCount()),
                () -> assertEquals(1, boundary.timedSampleCount()));
    }

    @Test
    void budgetRejectsEveryInvalidPrimitiveCondition() {
        assertAll(
                () ->
                        assertBudgetFailure(
                                0, 1, 0, 1, "maximumDistinctCacheMisses must be positive"),
                () ->
                        assertBudgetFailure(
                                -1, 1, 0, 1, "maximumDistinctCacheMisses must be positive"),
                () -> assertBudgetFailure(1, 0, 0, 1, "maximumCandidatesPerMiss must be positive"),
                () -> assertBudgetFailure(1, -1, 0, 1, "maximumCandidatesPerMiss must be positive"),
                () -> assertBudgetFailure(1, 1, -1, 1, "warmupCount must be non-negative"),
                () -> assertBudgetFailure(1, 1, 0, 0, "timedSampleCount must be positive and odd"),
                () -> assertBudgetFailure(1, 1, 0, -1, "timedSampleCount must be positive and odd"),
                () -> assertBudgetFailure(1, 1, 0, 2, "timedSampleCount must be positive and odd"));
    }

    @Test
    void budgetPreservesOrdinaryRecordValueBehavior() {
        ModelAutotuningConfig.Budget first = new ModelAutotuningConfig.Budget(2, 3, 4, 5);
        ModelAutotuningConfig.Budget equal = new ModelAutotuningConfig.Budget(2, 3, 4, 5);
        ModelAutotuningConfig.Budget different = new ModelAutotuningConfig.Budget(2, 3, 4, 7);

        assertAll(
                () -> assertEquals(first, equal),
                () -> assertEquals(first.hashCode(), equal.hashCode()),
                () -> assertNotEquals(first, different),
                () ->
                        assertEquals(
                                "Budget[maximumDistinctCacheMisses=2, maximumCandidatesPerMiss=3, "
                                        + "warmupCount=4, timedSampleCount=5]",
                                first.toString()));
    }

    @Test
    void representativeProfileIdentityHasExactPublicShape() throws ReflectiveOperationException {
        Class<ModelAutotuningConfig.RepresentativeProfileIdentity> type =
                ModelAutotuningConfig.RepresentativeProfileIdentity.class;
        Constructor<?> constructor = type.getDeclaredConstructor(int.class, byte[].class);

        assertAll(
                () -> assertTrue(Modifier.isPublic(type.getModifiers())),
                () -> assertTrue(Modifier.isStatic(type.getModifiers())),
                () -> assertTrue(Modifier.isFinal(type.getModifiers())),
                () -> assertFalse(type.isRecord()),
                () -> assertFalse(type.isEnum()),
                () -> assertFalse(Serializable.class.isAssignableFrom(type)),
                () -> assertEquals(0, type.getInterfaces().length),
                () -> assertEquals(0, type.getDeclaredClasses().length),
                () -> assertEquals(2, type.getDeclaredFields().length),
                () ->
                        assertArrayEquals(
                                new String[] {"schemaVersion", "bytes"},
                                fieldNames(type.getDeclaredFields())),
                () ->
                        assertArrayEquals(
                                new Class<?>[] {int.class, byte[].class},
                                fieldTypes(type.getDeclaredFields())),
                () -> assertEquals(1, type.getDeclaredConstructors().length),
                () -> assertTrue(Modifier.isPublic(constructor.getModifiers())),
                () ->
                        assertEquals(
                                Set.of("schemaVersion", "bytes", "equals", "hashCode", "toString"),
                                publicDeclaredMethodNames(type)),
                () -> assertEquals(5, type.getDeclaredMethods().length));
    }

    @Test
    void representativeProfileIdentitySnapshotsAndComparesByteContent() {
        byte[] source = {1, 2, 3};
        ModelAutotuningConfig.RepresentativeProfileIdentity identity =
                new ModelAutotuningConfig.RepresentativeProfileIdentity(7, source);
        source[0] = 9;
        byte[] firstCopy = identity.bytes();
        firstCopy[1] = 9;
        byte[] secondCopy = identity.bytes();
        ModelAutotuningConfig.RepresentativeProfileIdentity equal =
                new ModelAutotuningConfig.RepresentativeProfileIdentity(7, new byte[] {1, 2, 3});
        ModelAutotuningConfig.RepresentativeProfileIdentity differentVersion =
                new ModelAutotuningConfig.RepresentativeProfileIdentity(8, new byte[] {1, 2, 3});
        ModelAutotuningConfig.RepresentativeProfileIdentity differentBytes =
                new ModelAutotuningConfig.RepresentativeProfileIdentity(7, new byte[] {1, 2, 4});

        assertAll(
                () -> assertEquals(7, identity.schemaVersion()),
                () -> assertArrayEquals(new byte[] {1, 2, 3}, secondCopy),
                () -> assertNotSame(firstCopy, secondCopy),
                () -> assertEquals(identity, equal),
                () -> assertEquals(identity.hashCode(), equal.hashCode()),
                () -> assertNotEquals(identity, differentVersion),
                () -> assertNotEquals(identity, differentBytes),
                () -> assertNotEquals(identity, null),
                () -> assertNotEquals(identity, new Object()),
                () ->
                        assertEquals(
                                "RepresentativeProfileIdentity[schemaVersion=7, byteLength=3]",
                                identity.toString()),
                () -> assertFalse(identity.toString().contains("1, 2, 3")));
    }

    @Test
    void representativeProfileIdentityRejectsInvalidInputs() {
        NullPointerException nullFailure =
                assertThrows(
                        NullPointerException.class,
                        () -> new ModelAutotuningConfig.RepresentativeProfileIdentity(1, null));

        assertAll(
                () -> assertEquals("bytes", nullFailure.getMessage()),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () ->
                                        new ModelAutotuningConfig.RepresentativeProfileIdentity(
                                                0, new byte[] {1})),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () ->
                                        new ModelAutotuningConfig.RepresentativeProfileIdentity(
                                                -1, new byte[] {1})),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () ->
                                        new ModelAutotuningConfig.RepresentativeProfileIdentity(
                                                1, new byte[0])));
    }

    @Test
    void publicSignaturesAndSourcesExposeNoForbiddenLifecycleSurface() throws Exception {
        Set<Class<?>> allowedTypes =
                Set.of(
                        void.class,
                        boolean.class,
                        int.class,
                        byte[].class,
                        Object.class,
                        String.class,
                        Path.class,
                        ModelAutotuningConfig.class,
                        ModelAutotuningConfig.Objective[].class,
                        ModelAutotuningConfig.Objective.class,
                        ModelAutotuningConfig.Budget.class,
                        ModelAutotuningConfig.RepresentativeProfileIdentity.class,
                        ModelAutotuningConfig.FallbackPolicy[].class,
                        ModelAutotuningConfig.FallbackPolicy.class);
        Set<Class<?>> publicTypes =
                Set.of(
                        ModelAutotuningConfig.class,
                        ModelAutotuningConfig.Objective.class,
                        ModelAutotuningConfig.Budget.class,
                        ModelAutotuningConfig.RepresentativeProfileIdentity.class,
                        ModelAutotuningConfig.FallbackPolicy.class);

        for (Class<?> type : publicTypes) {
            for (Constructor<?> constructor : type.getDeclaredConstructors()) {
                assertTrue(
                        Arrays.stream(constructor.getParameterTypes())
                                .allMatch(allowedTypes::contains));
            }
            for (Method method : type.getDeclaredMethods()) {
                if (!Modifier.isPublic(method.getModifiers())) {
                    continue;
                }
                assertTrue(allowedTypes.contains(method.getReturnType()), method.toString());
                assertTrue(
                        Arrays.stream(method.getParameterTypes())
                                .allMatch(allowedTypes::contains));
            }
        }

        Path sourceDirectory = Path.of("src/main/java/io/github/pho001/synaptik/config/tuning");
        String sources;
        try (var files = Files.list(sourceDirectory)) {
            sources =
                    files.filter(path -> path.toString().endsWith(".java"))
                            .sorted()
                            .map(
                                    path -> {
                                        try {
                                            return Files.readString(path);
                                        } catch (java.io.IOException exception) {
                                            throw new RuntimeException(exception);
                                        }
                                    })
                            .collect(Collectors.joining("\n"));
        }

        assertAll(
                () -> assertEquals(2L, countJavaFiles(sourceDirectory)),
                () -> assertFalse(sources.contains("io.github.pho001.synaptik.tools")),
                () -> assertFalse(sources.contains("io.github.pho001.synaptik.backend")),
                () -> assertFalse(sources.contains("io.github.pho001.synaptik.prepare")),
                () -> assertFalse(sources.contains("io.github.pho001.synaptik.runtime")),
                () -> assertFalse(sources.contains("io.github.pho001.synaptik.engine")),
                () -> assertFalse(sources.contains("java.io.Serializable")),
                () -> assertFalse(sources.contains("java.lang.reflect")),
                () -> assertFalse(sources.contains("Map<")),
                () -> assertFalse(sources.contains("Object>")),
                () -> assertFalse(sources.contains("valueOf(")));
    }

    private static ModelAutotuningConfig config(byte[] profileBytes) {
        return new ModelAutotuningConfig(
                ModelAutotuningConfig.Objective.MIN_MEDIAN_ELAPSED_NANOS,
                new ModelAutotuningConfig.Budget(2, 3, 4, 5),
                new ModelAutotuningConfig.RepresentativeProfileIdentity(1, profileBytes),
                ModelAutotuningConfig.FallbackPolicy.ALLOW_SAFE_HEURISTIC,
                Path.of("requested-cache.bin"));
    }

    private static void assertBudgetFailure(
            int misses, int candidates, int warmups, int samples, String expectedMessage) {
        IllegalArgumentException failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                new ModelAutotuningConfig.Budget(
                                        misses, candidates, warmups, samples));
        assertEquals(expectedMessage, failure.getMessage());
    }

    private static void assertNullComponent(String component, Runnable construction) {
        NullPointerException failure = assertThrows(NullPointerException.class, construction::run);
        assertEquals(component, failure.getMessage());
    }

    private static <E extends Enum<E>> void assertExactEnum(
            Class<E> enumType, String[] expectedNames) {
        E[] values = enumType.getEnumConstants();
        Field[] fields =
                Arrays.stream(enumType.getDeclaredFields())
                        .filter(field -> !field.isSynthetic())
                        .toArray(Field[]::new);
        Method[] methods =
                Arrays.stream(enumType.getDeclaredMethods())
                        .filter(method -> !method.isSynthetic())
                        .toArray(Method[]::new);

        assertAll(
                () -> assertTrue(Modifier.isPublic(enumType.getModifiers())),
                () -> assertTrue(Modifier.isStatic(enumType.getModifiers())),
                () -> assertTrue(Modifier.isFinal(enumType.getModifiers())),
                () ->
                        assertArrayEquals(
                                expectedNames, Arrays.stream(values).map(Enum::name).toArray()),
                () ->
                        assertArrayEquals(
                                expectedNames,
                                Arrays.stream(fields).map(Field::getName).toArray()),
                () ->
                        assertEquals(
                                Set.of("values", "valueOf"),
                                Arrays.stream(methods)
                                        .map(Method::getName)
                                        .collect(Collectors.toSet())),
                () -> assertEquals(2, methods.length),
                () -> assertEquals(1, enumType.getDeclaredConstructors().length),
                () -> assertEquals(0, enumType.getDeclaredClasses().length),
                () -> assertEquals(0, enumType.getInterfaces().length));
    }

    private static Set<String> publicDeclaredMethodNames(Class<?> type) {
        return Arrays.stream(type.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .map(Method::getName)
                .collect(Collectors.toSet());
    }

    private static String[] componentNames(RecordComponent[] components) {
        return Arrays.stream(components).map(RecordComponent::getName).toArray(String[]::new);
    }

    private static Class<?>[] componentTypes(RecordComponent[] components) {
        return Arrays.stream(components).map(RecordComponent::getType).toArray(Class<?>[]::new);
    }

    private static String[] fieldNames(Field[] fields) {
        return Arrays.stream(fields).map(Field::getName).toArray(String[]::new);
    }

    private static Class<?>[] fieldTypes(Field[] fields) {
        return Arrays.stream(fields).map(Field::getType).toArray(Class<?>[]::new);
    }

    private static long countJavaFiles(Path directory) {
        try (var files = Files.list(directory)) {
            return files.filter(path -> path.toString().endsWith(".java")).count();
        } catch (java.io.IOException exception) {
            throw new RuntimeException(exception);
        }
    }
}
