package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class MetalCertificationEnvironmentTest {
    private static final String DIGEST = "1".repeat(64);
    private static final String RECORD = String.join("\n",
            MetalCertificationEnvironment.NATIVE_RECORD_MAGIC,
            "APPLE_9",
            "25G83",
            "macos-sdk=27.0;mpsgraph=6.5.1(6.5.1)",
            "host-clang=21;xcode=27;metal=32023",
            "graph-options=NONE;optimization-level=1;wait-for-compilation=YES;"
                    + "reduced-precision-fast-math=NONE");

    @Test
    void exactNativeRecordAndStableBinaryDigestBecomeCertifiable() {
        MetalCertificationEnvironment environment =
                MetalCertificationEnvironment.parse(RECORD, DIGEST, DIGEST);
        assertTrue(environment.certifiable());
        assertEquals("APPLE_9", environment.gpuFamily());
        assertEquals(DIGEST, environment.binaryDigest());
        assertEquals(MetalCertificationEnvironment.CAPABILITY_MANIFEST_HASH,
                environment.capabilityManifestHash());
    }

    @Test
    void malformedOrChangingIdentityFailsClosed() {
        assertThrows(IllegalArgumentException.class,
                () -> MetalCertificationEnvironment.parse(RECORD + "\n", DIGEST, DIGEST));
        assertThrows(IllegalArgumentException.class,
                () -> MetalCertificationEnvironment.parse(
                        RECORD.replace("APPLE_9", "APPLE_9\tforeign"), DIGEST, DIGEST));
        assertThrows(IllegalArgumentException.class,
                () -> MetalCertificationEnvironment.parse(RECORD, DIGEST, "2".repeat(64)));
        assertFalse(MetalCertificationEnvironment.unavailable().certifiable());
    }

    @Test
    void binaryHashBindsExactBytes(@TempDir java.nio.file.Path temporary) throws Exception {
        var library = temporary.resolve("bridge.dylib").toAbsolutePath();
        Files.write(library, new byte[] {0, 1, 2, 3});
        String first = MetalCertificationEnvironment.sha256(library);
        Files.write(library, new byte[] {0, 1, 2, 4});
        String second = MetalCertificationEnvironment.sha256(library);
        assertEquals(64, first.length());
        assertEquals(64, second.length());
        assertFalse(first.equals(second));
    }
}
