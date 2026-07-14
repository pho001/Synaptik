package io.github.pho001.synaptik.backend.contract;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.reflect.RecordComponent;
import java.util.Arrays;
import org.junit.jupiter.api.Test;

class DeviceClassTest {
    @Test
    void hasTheExactPublicEnumShape() {
        Field[] projectFields =
                Arrays.stream(DeviceClass.class.getDeclaredFields())
                        .filter(field -> !field.isEnumConstant())
                        .filter(field -> !field.isSynthetic())
                        .toArray(Field[]::new);
        Method[] declaredEnumMethods =
                Arrays.stream(DeviceClass.class.getDeclaredMethods())
                        .filter(method -> !method.isSynthetic())
                        .toArray(Method[]::new);

        assertAll(
                () ->
                        assertEquals(
                                "io.github.pho001.synaptik.backend.contract",
                                DeviceClass.class.getPackageName()),
                () -> assertTrue(Modifier.isPublic(DeviceClass.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(DeviceClass.class.getModifiers())),
                () -> assertTrue(DeviceClass.class.isEnum()),
                () -> assertArrayEquals(new Field[0], projectFields),
                () -> assertEquals(1, DeviceClass.class.getDeclaredConstructors().length),
                () ->
                        assertTrue(
                                Modifier.isPrivate(
                                        DeviceClass.class
                                                .getDeclaredConstructors()[0]
                                                .getModifiers())),
                () -> assertEquals(2, declaredEnumMethods.length),
                () -> assertGeneratedValueOfMethod(),
                () -> assertGeneratedValuesMethod(),
                () -> assertEquals(0, DeviceClass.class.getInterfaces().length),
                () -> assertEquals(0, DeviceClass.class.getDeclaredClasses().length));
    }

    @Test
    void exposesOnlyCpuAndAcceleratorInStableDeclarationOrder() {
        DeviceClass[] firstValues = DeviceClass.values();
        DeviceClass[] secondValues = DeviceClass.values();

        assertAll(
                () ->
                        assertArrayEquals(
                                new DeviceClass[] {DeviceClass.CPU, DeviceClass.ACCELERATOR},
                                firstValues),
                () -> assertArrayEquals(new String[] {"CPU", "ACCELERATOR"}, names(firstValues)),
                () -> assertEquals(0, DeviceClass.CPU.ordinal()),
                () -> assertEquals(1, DeviceClass.ACCELERATOR.ordinal()),
                () -> assertNotSame(firstValues, secondValues),
                () -> assertSame(DeviceClass.CPU, DeviceClass.valueOf("CPU")),
                () -> assertSame(DeviceClass.ACCELERATOR, DeviceClass.valueOf("ACCELERATOR")),
                () -> assertThrows(IllegalArgumentException.class, () -> DeviceClass.valueOf("GPU")),
                () -> assertThrows(NullPointerException.class, () -> DeviceClass.valueOf(null)));
    }

    @Test
    void remainsSeparateFromBackendAndDeviceIdentityRecords() {
        assertAll(
                () -> assertFalse(BackendId.class.isAssignableFrom(DeviceClass.class)),
                () -> assertFalse(BackendDeviceId.class.isAssignableFrom(DeviceClass.class)),
                () -> assertFalse(DeviceClass.class.isAssignableFrom(BackendId.class)),
                () -> assertFalse(DeviceClass.class.isAssignableFrom(BackendDeviceId.class)),
                () ->
                        assertArrayEquals(
                                new Class<?>[] {String.class}, componentTypes(BackendId.class)),
                () ->
                        assertArrayEquals(
                                new Class<?>[] {BackendId.class, String.class},
                                componentTypes(BackendDeviceId.class)));
    }

    private static String[] names(DeviceClass[] classes) {
        return Arrays.stream(classes).map(Enum::name).toArray(String[]::new);
    }

    private static Class<?>[] componentTypes(Class<?> recordType) {
        return Arrays.stream(recordType.getRecordComponents())
                .map(RecordComponent::getType)
                .toArray(Class<?>[]::new);
    }

    private static void assertGeneratedValueOfMethod() throws ReflectiveOperationException {
        Method valueOf = DeviceClass.class.getDeclaredMethod("valueOf", String.class);

        assertAll(
                () -> assertEquals(DeviceClass.class, valueOf.getReturnType()),
                () -> assertTrue(Modifier.isPublic(valueOf.getModifiers())),
                () -> assertTrue(Modifier.isStatic(valueOf.getModifiers())));
    }

    private static void assertGeneratedValuesMethod() throws ReflectiveOperationException {
        Method values = DeviceClass.class.getDeclaredMethod("values");

        assertAll(
                () -> assertEquals(DeviceClass[].class, values.getReturnType()),
                () -> assertTrue(Modifier.isPublic(values.getModifiers())),
                () -> assertTrue(Modifier.isStatic(values.getModifiers())));
    }
}
