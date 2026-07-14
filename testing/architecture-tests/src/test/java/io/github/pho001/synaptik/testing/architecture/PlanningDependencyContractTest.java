package io.github.pho001.synaptik.testing.architecture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Verifies planning's exact dependency visibility and forbidden outward edges. */
final class PlanningDependencyContractTest {
    /**
     * Checks planning's two public signature dependencies and two internal dependencies.
     *
     * @throws IOException if the planning build file cannot be read
     */
    @Test
    void planningHasOnlyItsExactArchitectureApprovedDependencySurface() throws IOException {
        Path planningBuild = repositoryRoot().resolve("modules/planning/build.gradle.kts");
        String buildScript = Files.readString(planningBuild);
        List<String> projectDependencyLines =
                buildScript.lines()
                        .map(String::strip)
                        .filter(line -> line.contains("project("))
                        .toList();

        assertEquals(
                List.of(
                        "api(project(\":modules:model\"))",
                        "implementation(project(\":modules:config\"))",
                        "api(project(\":modules:backend-contract\"))",
                        "implementation(project(\":modules:trace\"))"),
                projectDependencyLines,
                "planning must expose only model and backend-contract through public signatures");
        assertTrue(
                buildScript.contains("api(project(\":modules:model\"))"),
                "model must be a public API dependency");
        assertTrue(
                buildScript.contains("api(project(\":modules:backend-contract\"))"),
                "backend-contract must be a public API dependency");
        assertTrue(
                buildScript.contains("implementation(project(\":modules:config\"))"),
                "config must remain an internal implementation dependency");
        assertTrue(
                buildScript.contains("implementation(project(\":modules:trace\"))"),
                "trace must remain an internal implementation dependency");
        assertFalse(
                buildScript.contains("implementation(project(\":modules:model\"))"),
                "model must not be hidden as an implementation dependency");
        assertFalse(
                buildScript.contains("implementation(project(\":modules:backend-contract\"))"),
                "backend-contract must not be hidden as an implementation dependency");
        assertFalse(
                buildScript.contains("project(\":backends:"),
                "planning must not depend on a concrete backend project");
        assertFalse(
                buildScript.contains("project(\":modules:runtime\")"),
                "planning must not depend on runtime");
        assertFalse(
                buildScript.contains("project(\":modules:prepare\")"),
                "planning must not depend on prepare");
        assertFalse(
                buildScript.contains("project(\":modules:engine\")"),
                "planning must not depend on engine");
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
