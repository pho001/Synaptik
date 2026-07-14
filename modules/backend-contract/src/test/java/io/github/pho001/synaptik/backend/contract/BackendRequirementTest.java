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
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.reflect.RecordComponent;
import java.util.Arrays;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class BackendRequirementTest {
    @Test
    void markerHasTheExactSealedMethodFreeShape() {
        Class<BackendRequirement> type = BackendRequirement.class;

        assertAll(
                () ->
                        assertEquals(
                                "io.github.pho001.synaptik.backend.contract",
                                type.getPackageName()),
                () -> assertTrue(Modifier.isPublic(type.getModifiers())),
                () -> assertTrue(Modifier.isInterface(type.getModifiers())),
                () -> assertTrue(type.isSealed()),
                () ->
                        assertEquals(
                                Set.of(
                                        BackendIdRequirement.class,
                                        BackendDeviceIdRequirement.class,
                                        DeviceClassRequirement.class),
                                Set.of(type.getPermittedSubclasses())),
                () -> assertEquals(0, type.getDeclaredFields().length),
                () -> assertEquals(0, type.getDeclaredMethods().length),
                () -> assertEquals(0, type.getDeclaredClasses().length),
                () -> assertEquals(0, type.getInterfaces().length));
    }

    @Test
    void variantsHaveTheExactOneComponentPublicRecordShapes()
            throws ReflectiveOperationException {
        assertRequirementRecordShape(BackendIdRequirement.class, "backendId", BackendId.class);
        assertRequirementRecordShape(
                BackendDeviceIdRequirement.class, "deviceId", BackendDeviceId.class);
        assertRequirementRecordShape(
                DeviceClassRequirement.class, "deviceClass", DeviceClass.class);
    }

    @Test
    void variantsRejectNullWithExactComponentMessages() {
        NullPointerException backendFailure =
                assertThrows(NullPointerException.class, () -> new BackendIdRequirement(null));
        NullPointerException deviceFailure =
                assertThrows(
                        NullPointerException.class, () -> new BackendDeviceIdRequirement(null));
        NullPointerException classFailure =
                assertThrows(NullPointerException.class, () -> new DeviceClassRequirement(null));

        assertAll(
                () -> assertEquals("backendId", backendFailure.getMessage()),
                () -> assertEquals("deviceId", deviceFailure.getMessage()),
                () -> assertEquals("deviceClass", classFailure.getMessage()));
    }

    @Test
    void variantsRetainTheExactSuppliedReferences() {
        BackendId backendId = new BackendId(new String("cuda"));
        BackendDeviceId deviceId = new BackendDeviceId(backendId, new String("0"));

        BackendIdRequirement backendRequirement = new BackendIdRequirement(backendId);
        BackendDeviceIdRequirement deviceRequirement =
                new BackendDeviceIdRequirement(deviceId);
        DeviceClassRequirement classRequirement =
                new DeviceClassRequirement(DeviceClass.ACCELERATOR);

        assertAll(
                () -> assertSame(backendId, backendRequirement.backendId()),
                () -> assertSame(deviceId, deviceRequirement.deviceId()),
                () ->
                        assertSame(
                                DeviceClass.ACCELERATOR, classRequirement.deviceClass()));
    }

    @Test
    void variantsPreserveOrdinaryRecordValueBehavior() {
        BackendIdRequirement backend = new BackendIdRequirement(new BackendId("cuda"));
        BackendIdRequirement equalBackend =
                new BackendIdRequirement(new BackendId(new String("cuda")));
        BackendDeviceIdRequirement device =
                new BackendDeviceIdRequirement(
                        new BackendDeviceId(new BackendId("cuda"), "0"));
        BackendDeviceIdRequirement equalDevice =
                new BackendDeviceIdRequirement(
                        new BackendDeviceId(new BackendId("cuda"), new String("0")));
        DeviceClassRequirement deviceClass =
                new DeviceClassRequirement(DeviceClass.ACCELERATOR);
        DeviceClassRequirement equalDeviceClass =
                new DeviceClassRequirement(DeviceClass.ACCELERATOR);

        assertAll(
                () -> assertEquals(backend, equalBackend),
                () -> assertEquals(backend.hashCode(), equalBackend.hashCode()),
                () -> assertNotEquals(backend, new BackendIdRequirement(new BackendId("cpu"))),
                () ->
                        assertEquals(
                                "BackendIdRequirement[backendId=BackendId[value=cuda]]",
                                backend.toString()),
                () -> assertEquals(device, equalDevice),
                () -> assertEquals(device.hashCode(), equalDevice.hashCode()),
                () ->
                        assertNotEquals(
                                device,
                                new BackendDeviceIdRequirement(
                                        new BackendDeviceId(new BackendId("cuda"), "1"))),
                () ->
                        assertEquals(
                                "BackendDeviceIdRequirement[deviceId=BackendDeviceId[backendId=BackendId[value=cuda], value=0]]",
                                device.toString()),
                () -> assertEquals(deviceClass, equalDeviceClass),
                () -> assertEquals(deviceClass.hashCode(), equalDeviceClass.hashCode()),
                () ->
                        assertNotEquals(
                                deviceClass, new DeviceClassRequirement(DeviceClass.CPU)),
                () ->
                        assertEquals(
                                "DeviceClassRequirement[deviceClass=ACCELERATOR]",
                                deviceClass.toString()));
    }

    @Test
    void variantsRemainDistinctExhaustiveTargetsWithoutFactoriesOrSentinels() {
        BackendRequirement[] requirements = {
            new BackendIdRequirement(new BackendId("cuda")),
            new BackendDeviceIdRequirement(
                    new BackendDeviceId(new BackendId("cuda"), "0")),
            new DeviceClassRequirement(DeviceClass.ACCELERATOR)
        };

        assertAll(
                () -> assertEquals(3, requirements.length),
                () ->
                        assertEquals(
                                3,
                                Arrays.stream(requirements)
                                        .map(Object::getClass)
                                        .distinct()
                                        .count()),
                () -> assertTrue(requirements[0] instanceof BackendIdRequirement),
                () -> assertTrue(requirements[1] instanceof BackendDeviceIdRequirement),
                () -> assertTrue(requirements[2] instanceof DeviceClassRequirement),
                () ->
                        assertEquals(
                                Set.of(
                                        "BackendIdRequirement",
                                        "BackendDeviceIdRequirement",
                                        "DeviceClassRequirement"),
                                Arrays.stream(BackendRequirement.class.getPermittedSubclasses())
                                        .map(Class::getSimpleName)
                                        .collect(Collectors.toSet())),
                () ->
                        assertFalse(
                                Arrays.stream(BackendRequirement.class.getDeclaredMethods())
                                        .map(Method::getName)
                                        .anyMatch(
                                                name ->
                                                        Set.of(
                                                                        "of",
                                                                        "any",
                                                                        "auto",
                                                                        "matches",
                                                                        "test",
                                                                        "satisfiedBy")
                                                                .contains(name))),
                () ->
                        assertFalse(
                                Arrays.stream(BackendRequirement.class.getDeclaredFields())
                                        .map(Field::getName)
                                        .anyMatch(
                                                name ->
                                                        Set.of(
                                                                        "ANY",
                                                                        "AUTO",
                                                                        "NONE",
                                                                        "DEFAULT",
                                                                        "PREFER",
                                                                        "AVOID",
                                                                        "REQUIRE",
                                                                        "CPU",
                                                                        "GPU")
                                                                .contains(name))));
    }

    private static void assertRequirementRecordShape(
            Class<?> type, String componentName, Class<?> componentType)
            throws ReflectiveOperationException {
        RecordComponent[] components = type.getRecordComponents();
        Constructor<?>[] constructors = type.getDeclaredConstructors();
        Field[] fields = type.getDeclaredFields();
        Set<String> publicMethods =
                Arrays.stream(type.getDeclaredMethods())
                        .filter(method -> Modifier.isPublic(method.getModifiers()))
                        .map(Method::getName)
                        .collect(Collectors.toSet());

        assertAll(
                type.getSimpleName(),
                () ->
                        assertEquals(
                                "io.github.pho001.synaptik.backend.contract",
                                type.getPackageName()),
                () -> assertTrue(Modifier.isPublic(type.getModifiers())),
                () -> assertTrue(Modifier.isFinal(type.getModifiers())),
                () -> assertTrue(type.isRecord()),
                () -> assertArrayEquals(new String[] {componentName}, componentNames(components)),
                () -> assertArrayEquals(new Class<?>[] {componentType}, componentTypes(components)),
                () -> assertEquals(1, fields.length),
                () -> assertEquals(componentName, fields[0].getName()),
                () -> assertEquals(componentType, fields[0].getType()),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPublic(constructors[0].getModifiers())),
                () ->
                        assertArrayEquals(
                                new Class<?>[] {componentType},
                                constructors[0].getParameterTypes()),
                () ->
                        assertArrayEquals(
                                new Class<?>[] {BackendRequirement.class}, type.getInterfaces()),
                () -> assertFalse(Serializable.class.isAssignableFrom(type)),
                () -> assertEquals(0, type.getDeclaredClasses().length),
                () -> assertEquals(4, type.getDeclaredMethods().length),
                () ->
                        assertEquals(
                                Set.of(componentName, "equals", "hashCode", "toString"),
                                publicMethods),
                () ->
                        assertEquals(
                                componentType,
                                type.getDeclaredMethod(componentName).getReturnType()));
    }

    private static String[] componentNames(RecordComponent[] components) {
        return Arrays.stream(components).map(RecordComponent::getName).toArray(String[]::new);
    }

    private static Class<?>[] componentTypes(RecordComponent[] components) {
        return Arrays.stream(components).map(RecordComponent::getType).toArray(Class<?>[]::new);
    }
}
