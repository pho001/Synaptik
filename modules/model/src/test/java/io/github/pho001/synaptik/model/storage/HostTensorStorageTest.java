package io.github.pho001.synaptik.model.storage;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class HostTensorStorageTest {
    @Test
    void isExactlyTheRequiredSealedInterface() {
        assertTrue(HostTensorStorage.class.isInterface());
        assertTrue(HostTensorStorage.class.isSealed());
        assertEquals(
                Set.of(MemorySegmentStorage.class),
                Set.of(HostTensorStorage.class.getPermittedSubclasses()));
        assertEquals(Set.of(), Set.of(HostTensorStorage.class.getInterfaces()));

        Map<String, Class<?>> methods = Arrays.stream(HostTensorStorage.class.getDeclaredMethods())
                .collect(Collectors.toMap(
                        method -> method.getName(),
                        method -> method.getReturnType()));
        assertEquals(
                Map.of(
                        "dataType", DataType.class,
                        "elementCapacity", long.class,
                        "byteSize", long.class,
                        "segment", MemorySegment.class,
                        "isReadOnly", boolean.class,
                        "isAlive", boolean.class),
                methods);
        Arrays.stream(HostTensorStorage.class.getDeclaredMethods()).forEach(method -> {
            assertTrue(Modifier.isPublic(method.getModifiers()));
            assertTrue(Modifier.isAbstract(method.getModifiers()));
            assertFalse(method.isDefault());
            assertEquals(0, method.getParameterCount());
        });
    }

    @Test
    void implementationHasNoCrossLayerPublicApi() {
        Set<String> publicMethodNames = Arrays.stream(MemorySegmentStorage.class.getMethods())
                .filter(method -> method.getDeclaringClass() == MemorySegmentStorage.class)
                .map(method -> method.getName())
                .collect(Collectors.toSet());

        assertEquals(
                Set.of("dataType", "elementCapacity", "byteSize", "segment", "isReadOnly", "isAlive"),
                publicMethodNames);
        assertFalse(AutoCloseable.class.isAssignableFrom(HostTensorStorage.class));
        assertFalse(AutoCloseable.class.isAssignableFrom(MemorySegmentStorage.class));
        assertFalse(MemorySegmentStorage.class.isRecord());
        assertTrue(Modifier.isFinal(MemorySegmentStorage.class.getModifiers()));
        assertEquals(Set.of(HostTensorStorage.class), Set.of(MemorySegmentStorage.class.getInterfaces()));
        assertEquals(1, MemorySegmentStorage.class.getConstructors().length);
        assertEquals(
                Arrays.asList(DataType.class, long.class, MemorySegment.class),
                Arrays.asList(MemorySegmentStorage.class.getConstructors()[0].getParameterTypes()));
    }
}
