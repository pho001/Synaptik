package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.file.Path;
import org.junit.jupiter.api.Test;

/** Locks explicit absolute native-library configuration without filesystem discovery. */
final class MetalBackendConfigurationTest {
    @Test
    void requiresAbsolutePathAndNormalizesTheSnapshot() {
        assertEquals("nativeLibraryPath must be absolute",
                assertThrows(IllegalArgumentException.class,
                        () -> new MetalBackendConfiguration(Path.of("bridge.dylib")))
                        .getMessage());

        Path configured = Path.of("").toAbsolutePath()
                .resolve("metal").resolve("..").resolve("bridge.dylib");
        assertEquals(configured.normalize(),
                new MetalBackendConfiguration(configured).nativeLibraryPath());
    }
}
