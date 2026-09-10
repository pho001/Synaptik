package io.github.pho001.synaptik.testing.architecture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Locks backend-conformance to its exact provider-agnostic, test-only CPU dependency surface. */
final class BackendConformanceDependencyContractTest {
    /**
     * Checks the ordered project edges and forbidden Engine/OpenBLAS-provider directions.
     *
     * @throws IOException if the backend-conformance build script cannot be read
     */
    @Test void backendConformanceUsesOnlyTheApprovedProjectDependencies() throws IOException {
        String buildScript = Files.readString(repositoryRoot()
                .resolve("testing/backend-conformance/build.gradle.kts"));
        List<String> projectDependencyLines = buildScript.lines().map(String::strip)
                .filter(line -> line.contains("project(")).toList();
        assertEquals(List.of(
                "implementation(project(\":modules:backend-contract\"))",
                "testImplementation(project(\":modules:model\"))",
                "testImplementation(project(\":modules:planning\"))",
                "testImplementation(project(\":modules:runtime\"))",
                "testImplementation(project(\":modules:prepare\"))",
                "testImplementation(project(\":backends:cpu\"))"), projectDependencyLines);
        assertFalse(buildScript.contains("implementation(project(\":backends:cpu\"))"));
        assertFalse(buildScript.contains("api(project(\":backends:cpu\"))"));
        assertFalse(buildScript.contains(":backends:openblas-provider"));
        assertFalse(buildScript.contains(":modules:engine"));
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
