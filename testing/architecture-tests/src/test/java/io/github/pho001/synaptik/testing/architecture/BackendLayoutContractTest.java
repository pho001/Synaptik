package io.github.pho001.synaptik.testing.architecture;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Locks physical backend placement independently of the unchanged Gradle identities and edges. */
final class BackendLayoutContractTest {
    /**
     * Requires explicit parent/child mappings and the colocated native tree, without stale roots.
     *
     * @throws IOException if repository settings cannot be read
     */
    @Test
    void backendProjectsRetainTheirIdentitiesAtTheApprovedPhysicalPaths() throws IOException {
        Path root = repositoryRoot();
        String settings = Files.readString(root.resolve("settings.gradle.kts"));
        assertTrue(settings.contains("project(\":backends\").projectDir = file(\"modules/backend\")"));
        for (String child : List.of("cpu", "metal", "cuda", "openblas-provider")) {
            String directory = "modules/backend/" + child;
            assertTrue(settings.contains("project(\":backends:" + child
                    + "\").projectDir = file(\"" + directory + "\")"), child);
            assertTrue(Files.isRegularFile(root.resolve(directory + "/build.gradle.kts")), child);
        }
        assertTrue(Files.isRegularFile(root.resolve("modules/backend-contract/build.gradle.kts")));
        assertTrue(Files.isRegularFile(root.resolve("modules/engine/build.gradle.kts")));
        for (String member : List.of("build.sh", "package-local.sh", "verify-package.sh",
                "src/synaptik_metal_foundation.m")) {
            assertTrue(Files.isRegularFile(root.resolve("modules/backend/metal/macos-arm64/" + member)));
        }
        assertTrue(Files.isRegularFile(root.resolve("modules/backend/cuda/README.md")));
        assertFalse(Files.exists(root.resolve("backends")), "a stale root must not mask missing mappings");
        assertFalse(Files.exists(root.resolve("native")), "native sources belong to their backend");
        assertFalse(Files.exists(root.resolve("modules/backend/metal/native")));
        assertFalse(Files.exists(root.resolve("modules/backend/cuda/native")));
    }

    /**
     * Finds repository settings above the test process, independent of project directory depth.
     *
     * @return the ancestor containing settings, never {@code null}
     * @throws IllegalStateException if no ancestor contains repository settings
     */
    private static Path repositoryRoot() {
        Path directory = Path.of("").toAbsolutePath();
        while (directory != null && !Files.isRegularFile(directory.resolve("settings.gradle.kts"))) {
            directory = directory.getParent();
        }
        if (directory == null) throw new IllegalStateException("could not locate repository root");
        return directory;
    }
}
