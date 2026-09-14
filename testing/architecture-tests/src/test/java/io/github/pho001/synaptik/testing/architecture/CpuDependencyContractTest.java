package io.github.pho001.synaptik.testing.architecture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Locks the concrete CPU backend's exact direct project dependency boundary. */
final class CpuDependencyContractTest {
    private static final List<String> APPROVED = List.of(
            "implementation(project(\":modules:compiler\"))",
            "implementation(project(\":modules:model\"))",
            "implementation(project(\":modules:config\"))",
            "implementation(project(\":modules:planning\"))",
            "implementation(project(\":modules:runtime\"))",
            "implementation(project(\":modules:prepare\"))",
            "implementation(project(\":modules:backend-contract\"))",
            "implementation(project(\":modules:trace\"))",
            "implementation(project(\":backends:openblas-provider\"))");

    /** Confirms Compiler is direct, Engine is absent, and every other CPU edge remains exact. */
    @Test
    void cpuHasOnlyTheApprovedDirectDependencies() throws IOException {
        String build = Files.readString(repositoryRoot().resolve("backends/cpu/build.gradle.kts"));
        assertEquals(APPROVED, projectDependencyLines(build));
        assertFalse(build.contains(":modules:engine"));
    }

    private static List<String> projectDependencyLines(String buildScript) {
        return buildScript.lines().map(String::strip)
                .filter(line -> line.contains("project("))
                .toList();
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
