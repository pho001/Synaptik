package backend.metal.bridge;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.lang.foreign.Arena;
import java.lang.foreign.SymbolLookup;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class MetalNativeLibraryResolverTest {
    @Test
    void normalizesMacPlatformResourceIds() {
        assertEquals("macos-arm64", MetalNativeLibraryResolver.platformId("Mac OS X", "aarch64"));
        assertEquals("macos-arm64", MetalNativeLibraryResolver.platformId("macOS", "arm64"));
        assertEquals("macos-x64", MetalNativeLibraryResolver.platformId("Mac OS X", "x86_64"));
        assertEquals("macos-x64", MetalNativeLibraryResolver.platformId("Mac OS X", "amd64"));
    }

    @Test
    void buildsResourcePathFromCurrentPlatform() {
        String resource = MetalNativeLibraryResolver.bundledResourceName();

        assertEquals(
                "native/" + MetalNativeLibraryResolver.currentPlatformId() + "/libsynaptik_apple_mps.dylib",
                resource
        );
    }

    @Test
    void extractsBundledShimAndFindsAvailabilitySymbolWhenResourceExists(@TempDir Path cacheRoot) throws Exception {
        assumeTrue(
                MetalNativeLibraryResolver.class.getClassLoader()
                        .getResource(MetalNativeLibraryResolver.bundledResourceName()) != null,
                "Bundled Metal shim resource is not present on this platform build."
        );

        String previousCacheRoot = System.getProperty(MetalNativeLibraryResolver.CACHE_DIR_PROPERTY);
        System.setProperty(MetalNativeLibraryResolver.CACHE_DIR_PROPERTY, cacheRoot.toString());
        try {
            Path extracted = MetalNativeLibraryResolver.extractBundledLibrary();

            assertNotNull(extracted);
            assertTrue(Files.isRegularFile(extracted));
            try (Arena arena = Arena.ofConfined()) {
                SymbolLookup lookup = SymbolLookup.libraryLookup(extracted.toString(), arena);
                assertTrue(lookup.find("synaptik_apple_mps_available").isPresent());
            }
        } finally {
            if (previousCacheRoot == null) {
                System.clearProperty(MetalNativeLibraryResolver.CACHE_DIR_PROPERTY);
            } else {
                System.setProperty(MetalNativeLibraryResolver.CACHE_DIR_PROPERTY, previousCacheRoot);
            }
        }
    }
}
