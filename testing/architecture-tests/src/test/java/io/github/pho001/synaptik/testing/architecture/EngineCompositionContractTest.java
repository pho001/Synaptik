package io.github.pho001.synaptik.testing.architecture;

import static org.junit.jupiter.api.Assertions.assertFalse;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

/** Guards the inward Engine composition boundary, including nested backends. */
final class EngineCompositionContractTest {

    /**
     * Rejects reverse Engine dependencies while preserving Training's explicit lifecycle role.
     *
     * @throws IOException if project build scripts cannot be read
     */
    @Test
    void inwardModulesDoNotDependOnEngine() throws IOException {
        Path root = repositoryRoot();
        String tuningBuild = Files.readString(root.resolve("tools/tuning/build.gradle.kts"));
        assertFalse(tuningBuild.contains("project(\":modules:engine\")"));

        for (Path build : Files.walk(root)
                .filter(path -> path.getFileName().toString().equals("build.gradle.kts"))
                .toList()) {
            Path relative = root.relativize(build);
            // Generic Checkpoint remains inward and receives no Engine exemption.
            boolean inwardProduction = relative.startsWith("modules");
            if (relative.equals(Path.of("modules/training/build.gradle.kts"))) {
                continue;
            }
            if (inwardProduction && !build.equals(root.resolve("modules/engine/build.gradle.kts"))) {
                assertFalse(Files.readString(build).contains("project(\":modules:engine\")"),
                        () -> "inward dependency on Engine in " + relative);
            }
        }
    }


    private static Path repositoryRoot() {
        Path directory = Path.of("").toAbsolutePath();
        while (directory != null && !Files.isRegularFile(directory.resolve("settings.gradle.kts"))) {
            directory = directory.getParent();
        }
        if (directory == null) throw new IllegalStateException("could not locate repository root");
        return directory;
    }
}
