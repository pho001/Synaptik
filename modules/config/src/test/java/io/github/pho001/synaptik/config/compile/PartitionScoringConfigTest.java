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

import io.github.pho001.synaptik.backend.contract.DeviceClass;
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

class PartitionScoringConfigTest {
    @Test
    void hasTheExactPublicRecordAndFactoryShape() throws ReflectiveOperationException {
        RecordComponent[] components = PartitionScoringConfig.class.getRecordComponents();
        Constructor<?>[] constructors = PartitionScoringConfig.class.getDeclaredConstructors();
        Field[] fields = PartitionScoringConfig.class.getDeclaredFields();
        Set<String> publicMethods =
                Arrays.stream(PartitionScoringConfig.class.getDeclaredMethods())
                        .filter(method -> Modifier.isPublic(method.getModifiers()))
                        .map(Method::getName)
                        .collect(Collectors.toSet());
        Method neutral = PartitionScoringConfig.class.getDeclaredMethod("neutral");
        Method preferring =
                PartitionScoringConfig.class.getDeclaredMethod("preferring", DeviceClass.class);

        assertAll(
                () ->
                        assertEquals(
                                "io.github.pho001.synaptik.config.compile",
                                PartitionScoringConfig.class.getPackageName()),
                () -> assertTrue(Modifier.isPublic(PartitionScoringConfig.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(PartitionScoringConfig.class.getModifiers())),
                () -> assertTrue(PartitionScoringConfig.class.isRecord()),
                () ->
                        assertArrayEquals(
                                new String[] {"preferredDeviceClass"}, componentNames(components)),
                () ->
                        assertArrayEquals(
                                new Class<?>[] {Optional.class}, componentTypes(components)),
                () ->
                        assertEquals(
                                "java.util.Optional<io.github.pho001.synaptik.backend.contract.DeviceClass>",
                                components[0].getGenericType().getTypeName()),
                () -> assertEquals(1, fields.length),
                () -> assertEquals("preferredDeviceClass", fields[0].getName()),
                () -> assertEquals(Optional.class, fields[0].getType()),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPublic(constructors[0].getModifiers())),
                () ->
                        assertArrayEquals(
                                new Class<?>[] {Optional.class},
                                constructors[0].getParameterTypes()),
                () -> assertEquals(0, PartitionScoringConfig.class.getInterfaces().length),
                () ->
                        assertFalse(
                                Serializable.class.isAssignableFrom(PartitionScoringConfig.class)),
                () -> assertEquals(0, PartitionScoringConfig.class.getDeclaredClasses().length),
                () -> assertEquals(6, PartitionScoringConfig.class.getDeclaredMethods().length),
                () ->
                        assertEquals(
                                Set.of(
                                        "preferredDeviceClass",
                                        "neutral",
                                        "preferring",
                                        "equals",
                                        "hashCode",
                                        "toString"),
                                publicMethods),
                () -> assertPublicStaticFactory(neutral),
                () -> assertArrayEquals(new Class<?>[0], neutral.getParameterTypes()),
                () -> assertPublicStaticFactory(preferring),
                () ->
                        assertArrayEquals(
                                new Class<?>[] {DeviceClass.class},
                                preferring.getParameterTypes()));
    }

    @Test
    void directConstructionValidatesAndRetainsExactReferences() {
        Optional<DeviceClass> present = Optional.of(DeviceClass.ACCELERATOR);
        Optional<DeviceClass> empty = Optional.empty();

        PartitionScoringConfig preferred = new PartitionScoringConfig(present);
        PartitionScoringConfig neutral = new PartitionScoringConfig(empty);
        NullPointerException failure =
                assertThrows(NullPointerException.class, () -> new PartitionScoringConfig(null));

        assertAll(
                () -> assertEquals("preferredDeviceClass", failure.getMessage()),
                () -> assertSame(present, preferred.preferredDeviceClass()),
                () ->
                        assertSame(
                                DeviceClass.ACCELERATOR,
                                preferred.preferredDeviceClass().orElseThrow()),
                () -> assertSame(empty, neutral.preferredDeviceClass()),
                () -> assertTrue(neutral.preferredDeviceClass().isEmpty()));
    }

    @Test
    void factoriesProduceFreshValuesWithOnlyTheirDeclaredPreference() {
        DeviceClass deviceClass = DeviceClass.ACCELERATOR;

        PartitionScoringConfig firstNeutral = PartitionScoringConfig.neutral();
        PartitionScoringConfig secondNeutral = PartitionScoringConfig.neutral();
        PartitionScoringConfig firstPreferred = PartitionScoringConfig.preferring(deviceClass);
        PartitionScoringConfig secondPreferred = PartitionScoringConfig.preferring(deviceClass);
        NullPointerException failure =
                assertThrows(
                        NullPointerException.class, () -> PartitionScoringConfig.preferring(null));

        assertAll(
                () -> assertNotSame(firstNeutral, secondNeutral),
                () -> assertEquals(firstNeutral, secondNeutral),
                () -> assertTrue(firstNeutral.preferredDeviceClass().isEmpty()),
                () -> assertNotSame(firstPreferred, secondPreferred),
                () -> assertEquals(firstPreferred, secondPreferred),
                () ->
                        assertNotSame(
                                firstPreferred.preferredDeviceClass(),
                                secondPreferred.preferredDeviceClass()),
                () -> assertSame(deviceClass, firstPreferred.preferredDeviceClass().orElseThrow()),
                () -> assertSame(deviceClass, secondPreferred.preferredDeviceClass().orElseThrow()),
                () -> assertEquals("deviceClass", failure.getMessage()));
    }

    @Test
    void preservesOrdinaryRecordValueBehavior() {
        PartitionScoringConfig first =
                new PartitionScoringConfig(Optional.of(DeviceClass.ACCELERATOR));
        PartitionScoringConfig equal = PartitionScoringConfig.preferring(DeviceClass.ACCELERATOR);
        PartitionScoringConfig different = PartitionScoringConfig.neutral();

        assertAll(
                () -> assertEquals(first, equal),
                () -> assertEquals(first.hashCode(), equal.hashCode()),
                () -> assertNotEquals(first, different),
                () ->
                        assertEquals(
                                "PartitionScoringConfig[preferredDeviceClass=Optional[ACCELERATOR]]",
                                first.toString()));
    }

    @Test
    void exposesNoEligibilityScoringCandidateServiceRouteProfileOrLifecycleSurface() {
        Set<String> allowedNames =
                Set.of(
                        "preferreddeviceclass",
                        "neutral",
                        "preferring",
                        "equals",
                        "hashcode",
                        "tostring");
        Set<String> declaredNames =
                Arrays.stream(PartitionScoringConfig.class.getDeclaredMethods())
                        .map(Method::getName)
                        .map(String::toLowerCase)
                        .collect(Collectors.toSet());

        assertAll(
                () -> assertEquals(allowedNames, declaredNames),
                () -> assertEquals(1, PartitionScoringConfig.class.getDeclaredFields().length),
                () -> assertEquals(0, PartitionScoringConfig.class.getDeclaredClasses().length));
    }

    private static String[] componentNames(RecordComponent[] components) {
        return Arrays.stream(components).map(RecordComponent::getName).toArray(String[]::new);
    }

    private static Class<?>[] componentTypes(RecordComponent[] components) {
        return Arrays.stream(components).map(RecordComponent::getType).toArray(Class<?>[]::new);
    }

    private static void assertPublicStaticFactory(Method method) {
        assertAll(
                () -> assertEquals(PartitionScoringConfig.class, method.getReturnType()),
                () -> assertTrue(Modifier.isPublic(method.getModifiers())),
                () -> assertTrue(Modifier.isStatic(method.getModifiers())));
    }
}
