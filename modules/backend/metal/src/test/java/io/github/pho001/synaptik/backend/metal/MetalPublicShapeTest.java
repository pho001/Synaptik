package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Modifier;
import org.junit.jupiter.api.Test;

/** Locks configuration plus integration as Metal's sole externally callable native opener. */
final class MetalPublicShapeTest {
    @Test
    void onlyIntegrationPublishesTheSelectedNativeOpenBoundary() throws Exception {
        assertTrue(Modifier.isPublic(MetalBackendConfiguration.class.getModifiers()));
        assertTrue(Modifier.isPublic(MetalBackendIntegration.class.getModifiers()));
        assertTrue(Modifier.isPublic(MetalCapabilityProvider.class.getModifiers()));
        assertEquals(MetalBackendIntegration.class,
                MetalBackendIntegration.class
                        .getMethod("open", MetalBackendConfiguration.class)
                        .getReturnType());

        Class<?> runtime = Class.forName(
                "io.github.pho001.synaptik.backend.metal.MetalBackendRuntime");
        assertFalse(Modifier.isPublic(runtime.getModifiers()));
        assertThrows(ClassNotFoundException.class,
                () -> Class.forName(
                        "io.github.pho001.synaptik.backend.metal.internal.MetalBackendRuntime"));
    }
}
