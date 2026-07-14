package io.github.pho001.synaptik.backend.contract;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.Serializable;
import java.lang.reflect.Constructor;
import java.lang.reflect.Modifier;
import java.lang.reflect.RecordComponent;
import java.util.Arrays;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class BackendIdentityTest {
    @Test
    void backendIdHasTheExactPublicRecordShape() throws ReflectiveOperationException {
        assertRecordShape(
                BackendId.class,
                new String[] {"value"},
                new Class<?>[] {String.class},
                Set.of("equals", "hashCode", "toString", "value"));
    }

    @Test
    void backendDeviceIdHasTheExactPublicRecordShape() throws ReflectiveOperationException {
        assertRecordShape(
                BackendDeviceId.class,
                new String[] {"backendId", "value"},
                new Class<?>[] {BackendId.class, String.class},
                Set.of("backendId", "equals", "hashCode", "toString", "value"));
    }

    @Test
    void backendIdValidatesExactlyAndRetainsOpenIdentityText() {
        NullPointerException nullFailure =
                assertThrows(NullPointerException.class, () -> new BackendId(null));

        assertEquals("value", nullFailure.getMessage());
        for (String blank : new String[] {"", " ", "\t\n"}) {
            IllegalArgumentException blankFailure =
                    assertThrows(IllegalArgumentException.class, () -> new BackendId(blank));
            assertEquals("value must not be blank", blankFailure.getMessage());
        }

        String exactValue = new String(" CuDa:vendor/β ");
        BackendId identity = new BackendId(exactValue);

        assertAll(
                () -> assertSame(exactValue, identity.value()),
                () -> assertEquals(" CuDa:vendor/β ", identity.value()),
                () -> assertNotEquals(identity, new BackendId("cuda:vendor/β ")),
                () -> assertNotEquals(identity, new BackendId("CuDa:vendor/β")));
    }

    @Test
    void backendDeviceIdValidatesInComponentOrderWithExactMessages() {
        NullPointerException bothNullFailure =
                assertThrows(NullPointerException.class, () -> new BackendDeviceId(null, null));
        NullPointerException backendFirstFailure =
                assertThrows(NullPointerException.class, () -> new BackendDeviceId(null, " "));
        BackendId backendId = new BackendId("cuda");
        NullPointerException nullValueFailure =
                assertThrows(
                        NullPointerException.class, () -> new BackendDeviceId(backendId, null));

        assertAll(
                () -> assertEquals("backendId", bothNullFailure.getMessage()),
                () -> assertEquals("backendId", backendFirstFailure.getMessage()),
                () -> assertEquals("value", nullValueFailure.getMessage()));

        for (String blank : new String[] {"", " ", "\t\n"}) {
            IllegalArgumentException blankFailure =
                    assertThrows(
                            IllegalArgumentException.class,
                            () -> new BackendDeviceId(backendId, blank));
            assertEquals("value must not be blank", blankFailure.getMessage());
        }
    }

    @Test
    void backendDeviceIdRetainsExactReferencesAndScopesEqualityByBackend() {
        BackendId cuda = new BackendId(new String("cuda"));
        BackendId metal = new BackendId(new String("metal"));
        String exactToken = new String(" Device 0/β ");
        BackendDeviceId identity = new BackendDeviceId(cuda, exactToken);
        BackendDeviceId equal = new BackendDeviceId(new BackendId("cuda"), " Device 0/β ");
        BackendDeviceId otherBackend = new BackendDeviceId(metal, " Device 0/β ");

        assertAll(
                () -> assertSame(cuda, identity.backendId()),
                () -> assertSame(exactToken, identity.value()),
                () -> assertEquals(identity, equal),
                () -> assertEquals(identity.hashCode(), equal.hashCode()),
                () -> assertNotEquals(identity, otherBackend),
                () -> assertNotEquals(identity, new BackendDeviceId(cuda, "device 0/β")),
                () ->
                        assertEquals(
                                "BackendDeviceId[backendId=BackendId[value=cuda], value= Device 0/β ]",
                                identity.toString()));
    }

    @Test
    void backendIdUsesOrdinaryRecordValueBehavior() {
        BackendId identity = new BackendId("cpu");
        BackendId equal = new BackendId(new String("cpu"));

        assertAll(
                () -> assertEquals(identity, equal),
                () -> assertEquals(identity.hashCode(), equal.hashCode()),
                () -> assertNotEquals(identity, new BackendId("CPU")),
                () -> assertEquals("BackendId[value=cpu]", identity.toString()));
    }

    private static void assertRecordShape(
            Class<?> type,
            String[] expectedComponentNames,
            Class<?>[] expectedComponentTypes,
            Set<String> expectedPublicMethods)
            throws ReflectiveOperationException {
        RecordComponent[] components = type.getRecordComponents();
        Constructor<?>[] constructors = type.getDeclaredConstructors();

        assertAll(
                type.getSimpleName(),
                () -> assertEquals("io.github.pho001.synaptik.backend.contract", type.getPackageName()),
                () -> assertTrue(Modifier.isPublic(type.getModifiers())),
                () -> assertTrue(Modifier.isFinal(type.getModifiers())),
                () -> assertTrue(type.isRecord()),
                () ->
                        assertArrayEquals(
                                expectedComponentNames,
                                Arrays.stream(components)
                                        .map(RecordComponent::getName)
                                        .toArray(String[]::new)),
                () ->
                        assertArrayEquals(
                                expectedComponentTypes,
                                Arrays.stream(components)
                                        .map(RecordComponent::getType)
                                        .toArray(Class<?>[]::new)),
                () -> assertEquals(expectedComponentNames.length, type.getDeclaredFields().length),
                () ->
                        assertArrayEquals(
                                expectedComponentNames,
                                Arrays.stream(type.getDeclaredFields())
                                        .map(field -> field.getName())
                                        .toArray(String[]::new)),
                () ->
                        assertArrayEquals(
                                expectedComponentTypes,
                                Arrays.stream(type.getDeclaredFields())
                                        .map(field -> field.getType())
                                        .toArray(Class<?>[]::new)),
                () -> assertEquals(1, constructors.length),
                () -> assertArrayEquals(expectedComponentTypes, constructors[0].getParameterTypes()),
                () -> assertTrue(Modifier.isPublic(constructors[0].getModifiers())),
                () -> assertEquals(0, type.getInterfaces().length),
                () -> assertEquals(0, type.getDeclaredClasses().length),
                () -> assertFalse(Serializable.class.isAssignableFrom(type)),
                () ->
                        assertEquals(
                                expectedPublicMethods,
                                Arrays.stream(type.getDeclaredMethods())
                                        .filter(method -> Modifier.isPublic(method.getModifiers()))
                                        .map(method -> method.getName())
                                        .collect(Collectors.toSet())));

        for (int index = 0; index < expectedComponentNames.length; index++) {
            assertEquals(
                    expectedComponentTypes[index],
                    type.getDeclaredMethod(expectedComponentNames[index]).getReturnType());
        }
    }
}
