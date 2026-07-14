package io.github.pho001.synaptik.testing.architecture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Verifies the public config dependency and its concrete-backend boundary. */
final class ConfigDependencyContractTest {
    /**
     * Checks that config publicly exposes only the backend-contract project dependency.
     *
     * @throws IOException if the config build file cannot be read
     */
    @Test
    void configPubliclyDependsOnBackendContractAndNotConcreteBackends() throws IOException {
        Path configBuild = repositoryRoot().resolve("modules/config/build.gradle.kts");
        String buildScript = Files.readString(configBuild);
        List<String> projectDependencyLines =
                buildScript.lines()
                        .map(String::strip)
                        .filter(line -> line.contains("project("))
                        .toList();

        assertEquals(
                List.of("api(project(\":modules:backend-contract\"))"),
                projectDependencyLines,
                "config must expose its backend-contract signature through the sole project edge");
        assertTrue(
                buildScript.contains("api(project(\":modules:backend-contract\"))"),
                "backend-contract must be a public API dependency");
        assertFalse(
                buildScript.contains("implementation(project(\":modules:backend-contract\"))"),
                "backend-contract must not be hidden as an implementation dependency");
        assertFalse(
                buildScript.contains("project(\":backends:"),
                "config must not depend on any concrete backend project");
    }

    /**
     * Locates the repository root from the architecture-test working directory.
     *
     * @return the directory containing {@code settings.gradle.kts}, never {@code null}
     * @throws IllegalStateException if no ancestor is the repository root
     */
    private static Path repositoryRoot() {
        Path directory = Path.of("").toAbsolutePath();
        while (directory != null && !Files.isRegularFile(directory.resolve("settings.gradle.kts"))) {
            directory = directory.getParent();
        }
        if (directory == null) {
            throw new IllegalStateException("could not locate the repository root");
        }
        return directory;
    }
}
