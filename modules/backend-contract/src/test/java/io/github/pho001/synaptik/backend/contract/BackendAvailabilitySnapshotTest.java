package io.github.pho001.synaptik.backend.contract;

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
import java.lang.reflect.Modifier;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.RecordComponent;
import java.lang.reflect.Type;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class BackendAvailabilitySnapshotTest {
    @Test
    void hasTheExactGenericPublicRecordShape() throws ReflectiveOperationException {
        Class<BackendAvailabilitySnapshot> type = BackendAvailabilitySnapshot.class;
        RecordComponent[] components = type.getRecordComponents();
        Constructor<?>[] constructors = type.getDeclaredConstructors();
        ParameterizedType devicesType = (ParameterizedType) components[1].getGenericType();

        assertAll(
                () ->
                        assertEquals(
                                "io.github.pho001.synaptik.backend.contract",
                                type.getPackageName()),
                () -> assertTrue(Modifier.isPublic(type.getModifiers())),
                () -> assertTrue(Modifier.isFinal(type.getModifiers())),
                () -> assertTrue(type.isRecord()),
                () ->
                        assertArrayEquals(
                                new String[] {"backendId", "devices"},
                                Arrays.stream(components)
                                        .map(RecordComponent::getName)
                                        .toArray(String[]::new)),
                () ->
                        assertArrayEquals(
                                new Class<?>[] {BackendId.class, Map.class},
                                Arrays.stream(components)
                                        .map(RecordComponent::getType)
                                        .toArray(Class<?>[]::new)),
                () -> assertEquals(Map.class, devicesType.getRawType()),
                () ->
                        assertArrayEquals(
                                new Type[] {BackendDeviceId.class, DeviceClass.class},
                                devicesType.getActualTypeArguments()),
                () -> assertEquals(2, type.getDeclaredFields().length),
                () ->
                        assertArrayEquals(
                                new String[] {"backendId", "devices"},
                                Arrays.stream(type.getDeclaredFields())
                                        .map(field -> field.getName())
                                        .toArray(String[]::new)),
                () -> assertEquals(1, constructors.length),
                () ->
                        assertArrayEquals(
                                new Class<?>[] {BackendId.class, Map.class},
                                constructors[0].getParameterTypes()),
                () -> assertTrue(Modifier.isPublic(constructors[0].getModifiers())),
                () -> assertEquals(0, type.getInterfaces().length),
                () -> assertEquals(0, type.getDeclaredClasses().length),
                () -> assertFalse(Serializable.class.isAssignableFrom(type)),
                () -> assertEquals(5, type.getDeclaredMethods().length),
                () ->
                        assertEquals(
                                Set.of("backendId", "devices", "equals", "hashCode", "toString"),
                                Arrays.stream(type.getDeclaredMethods())
                                        .filter(
                                                method ->
                                                        Modifier.isPublic(method.getModifiers()))
                                        .map(method -> method.getName())
                                        .collect(Collectors.toSet())),
                () ->
                        assertEquals(
                                BackendId.class,
                                type.getDeclaredMethod("backendId").getReturnType()),
                () -> assertEquals(Map.class, type.getDeclaredMethod("devices").getReturnType()),
                () ->
                        assertEquals(
                                devicesType,
                                type.getDeclaredMethod("devices").getGenericReturnType()));
    }

    @Test
    void validatesComponentsInExactOrderWithExactMessages() {
        NullPointerException bothNullFailure =
                assertThrows(
                        NullPointerException.class,
                        () -> new BackendAvailabilitySnapshot(null, null));
        NullPointerException devicesFailure =
                assertThrows(
                        NullPointerException.class,
                        () -> new BackendAvailabilitySnapshot(new BackendId("cuda"), null));

        assertAll(
                () -> assertEquals("backendId", bothNullFailure.getMessage()),
                () -> assertEquals("devices", devicesFailure.getMessage()));
    }

    @Test
    void validatesEachEntryKeyThenValueThenBackendBeforeAdvancing() {
        BackendId cuda = new BackendId("cuda");
        BackendDeviceId cudaZero = new BackendDeviceId(cuda, "0");
        BackendDeviceId metalZero = new BackendDeviceId(new BackendId("metal"), "0");

        Map<BackendDeviceId, DeviceClass> nullKeyAndValue = new LinkedHashMap<>();
        nullKeyAndValue.put(cudaZero, DeviceClass.ACCELERATOR);
        nullKeyAndValue.put(null, null);
        NullPointerException nullKeyFailure =
                assertThrows(
                        NullPointerException.class,
                        () -> new BackendAvailabilitySnapshot(cuda, nullKeyAndValue));

        Map<BackendDeviceId, DeviceClass> wrongBackendAndNullValue = new LinkedHashMap<>();
        wrongBackendAndNullValue.put(cudaZero, DeviceClass.ACCELERATOR);
        wrongBackendAndNullValue.put(metalZero, null);
        NullPointerException nullValueFailure =
                assertThrows(
                        NullPointerException.class,
                        () -> new BackendAvailabilitySnapshot(cuda, wrongBackendAndNullValue));

        Map<BackendDeviceId, DeviceClass> wrongBackendBeforeLaterNullKey = new LinkedHashMap<>();
        wrongBackendBeforeLaterNullKey.put(metalZero, DeviceClass.ACCELERATOR);
        wrongBackendBeforeLaterNullKey.put(null, DeviceClass.CPU);
        IllegalArgumentException backendFailure =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                new BackendAvailabilitySnapshot(
                                        cuda, wrongBackendBeforeLaterNullKey));

        assertAll(
                () ->
                        assertEquals(
                                "devices contains null deviceId", nullKeyFailure.getMessage()),
                () ->
                        assertEquals(
                                "devices contains null deviceClass", nullValueFailure.getMessage()),
                () ->
                        assertEquals(
                                "device backendId must match snapshot backendId",
                                backendFailure.getMessage()));
    }

    @Test
    void acceptsEmptySnapshotWithoutAddingAvailabilityState() {
        BackendId backendId = new BackendId("cuda");
        BackendAvailabilitySnapshot snapshot =
                new BackendAvailabilitySnapshot(backendId, Map.of());

        assertAll(
                () -> assertSame(backendId, snapshot.backendId()),
                () -> assertTrue(snapshot.devices().isEmpty()),
                () ->
                        assertThrows(
                                UnsupportedOperationException.class,
                                () -> snapshot.devices().clear()));
    }

    @Test
    void acceptsEqualBackendIdentityAndRetainsExactElementReferences() {
        BackendId snapshotBackendId = new BackendId(new String("cuda"));
        BackendId equalDeviceBackendId = new BackendId(new String("cuda"));
        BackendDeviceId deviceId = new BackendDeviceId(equalDeviceBackendId, new String("0"));
        Map<BackendDeviceId, DeviceClass> source = new LinkedHashMap<>();
        source.put(deviceId, DeviceClass.ACCELERATOR);

        BackendAvailabilitySnapshot snapshot =
                new BackendAvailabilitySnapshot(snapshotBackendId, source);
        BackendDeviceId retainedDeviceId = snapshot.devices().keySet().iterator().next();

        assertAll(
                () -> assertSame(snapshotBackendId, snapshot.backendId()),
                () -> assertSame(deviceId, retainedDeviceId),
                () -> assertSame(DeviceClass.ACCELERATOR, snapshot.devices().get(deviceId)),
                () -> assertSame(equalDeviceBackendId, retainedDeviceId.backendId()),
                () -> assertNotSame(snapshotBackendId, retainedDeviceId.backendId()));
    }

    @Test
    void copiesMutableSourceAndExposesAnImmutableStructuralSnapshot() {
        BackendId cuda = new BackendId("cuda");
        BackendDeviceId zero = new BackendDeviceId(cuda, "0");
        BackendDeviceId one = new BackendDeviceId(cuda, "1");
        Map<BackendDeviceId, DeviceClass> source = new LinkedHashMap<>();
        source.put(zero, DeviceClass.ACCELERATOR);

        BackendAvailabilitySnapshot snapshot = new BackendAvailabilitySnapshot(cuda, source);
        source.clear();
        source.put(one, DeviceClass.ACCELERATOR);

        assertAll(
                () -> assertNotSame(source, snapshot.devices()),
                () -> assertEquals(Map.of(zero, DeviceClass.ACCELERATOR), snapshot.devices()),
                () -> assertFalse(snapshot.devices().containsKey(one)),
                () ->
                        assertThrows(
                                UnsupportedOperationException.class,
                                () -> snapshot.devices().put(one, DeviceClass.ACCELERATOR)),
                () ->
                        assertThrows(
                                UnsupportedOperationException.class,
                                () -> snapshot.devices().remove(zero)));
    }

    @Test
    void preservesOrdinaryRecordValueBehavior() {
        BackendId cuda = new BackendId("cuda");
        BackendDeviceId zero = new BackendDeviceId(cuda, "0");
        BackendAvailabilitySnapshot snapshot =
                new BackendAvailabilitySnapshot(
                        cuda, Map.of(zero, DeviceClass.ACCELERATOR));
        BackendAvailabilitySnapshot equal =
                new BackendAvailabilitySnapshot(
                        new BackendId("cuda"),
                        Map.of(
                                new BackendDeviceId(new BackendId("cuda"), "0"),
                                DeviceClass.ACCELERATOR));

        assertAll(
                () -> assertEquals(snapshot, equal),
                () -> assertEquals(snapshot.hashCode(), equal.hashCode()),
                () ->
                        assertNotEquals(
                                snapshot,
                                new BackendAvailabilitySnapshot(
                                        cuda, Map.of(zero, DeviceClass.CPU))),
                () ->
                        assertNotEquals(
                                snapshot,
                                new BackendAvailabilitySnapshot(new BackendId("metal"), Map.of())),
                () -> assertTrue(snapshot.toString().startsWith("BackendAvailabilitySnapshot[")),
                () -> assertTrue(snapshot.toString().contains("backendId=BackendId[value=cuda]")),
                () -> assertTrue(snapshot.toString().contains("devices={")));
    }
}
