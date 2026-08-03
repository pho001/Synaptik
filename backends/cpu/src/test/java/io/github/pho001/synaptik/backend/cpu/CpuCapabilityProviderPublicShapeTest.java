package io.github.pho001.synaptik.backend.cpu;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class CpuCapabilityProviderPublicShapeTest {
    @Test void exposesOnlyTheSpecifiedPublicProviderSurface() {
        Class<CpuCapabilityProvider> type = CpuCapabilityProvider.class;
        Set<String> publicMethods = Arrays.stream(type.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .map(method -> method.getName()).collect(Collectors.toSet());
        assertAll(
                () -> assertTrue(Modifier.isPublic(type.getModifiers())),
                () -> assertTrue(Modifier.isFinal(type.getModifiers())),
                () -> assertArrayEquals(new Class<?>[]{BackendCapabilityProvider.class}, type.getInterfaces()),
                () -> assertEquals(Set.of("backendId", "supports"), publicMethods),
                () -> assertEquals(1, type.getDeclaredConstructors().length),
                () -> assertTrue(Modifier.isPublic(type.getDeclaredConstructors()[0].getModifiers())),
                () -> assertEquals(1, type.getDeclaredFields().length),
                () -> assertEquals(BackendId.class, type.getDeclaredField("CPU_BACKEND_ID").getType()),
                () -> assertTrue(Modifier.isPublic(type.getDeclaredField("CPU_BACKEND_ID").getModifiers())),
                () -> assertTrue(Modifier.isStatic(type.getDeclaredField("CPU_BACKEND_ID").getModifiers())),
                () -> assertTrue(Modifier.isFinal(type.getDeclaredField("CPU_BACKEND_ID").getModifiers())));
    }
}
