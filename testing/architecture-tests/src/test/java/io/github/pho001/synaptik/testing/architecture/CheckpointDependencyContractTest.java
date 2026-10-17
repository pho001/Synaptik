package io.github.pho001.synaptik.testing.architecture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;

/**
 * Protects the model-only Checkpoint project's three public dependencies and the deferred
 * Training Checkpoint adapter gate.
 */
final class CheckpointDependencyContractTest {
    /**
     * Checks that Checkpoint is included alone, has exactly its three public project edges,
     * and remains downstream of all inward modules, concrete backends, tuning, NN, and Training.
     *
     * @throws IOException if settings or a required project build file cannot be read
     */
    @Test
    void checkpointHasOnlyCurrentModelBoundaryEdges() throws IOException {
        Path root = repositoryRoot();
        String settings = Files.readString(root.resolve("settings.gradle.kts"));
        Path checkpoint = root.resolve("extensions/checkpoint");
        Path checkpointBuild = checkpoint.resolve("build.gradle.kts");

        assertTrue(settings.lines().map(String::strip).anyMatch("\":extensions:checkpoint\","::equals),
                "the model-only Checkpoint project must be included");
        assertFalse(settings.contains(":extensions:training-checkpoint"),
                "the optional Training Checkpoint adapter must await its separate gate");
        assertTrue(Files.isRegularFile(checkpointBuild), "Checkpoint must declare its own build");

        List<String> projectEdges = Files.readString(checkpointBuild).lines()
                .map(String::strip)
                .filter(line -> line.contains("project("))
                .toList();
        assertEquals(List.of(
                "api(project(\":extensions:nn\"))",
                "api(project(\":modules:engine\"))",
                "api(project(\":modules:model\"))"), projectEdges,
                "the public model checkpoint must expose exactly NN, Engine, and Model edges");

        for (String area : List.of("modules")) {
            try (Stream<Path> builds = Files.walk(root.resolve(area))) {
                for (Path build : builds.filter(path -> path.getFileName().toString()
                        .equals("build.gradle.kts")).toList()) {
                    assertFalse(Files.readString(build).contains(":extensions:checkpoint"),
                            () -> "inward project must not depend on Checkpoint: " + build);
                }
            }
        }
        for (String project : List.of("extensions/nn", "modules/training", "tools/tuning")) {
            Path build = root.resolve(project + "/build.gradle.kts");
            assertFalse(Files.readString(build).contains(":extensions:checkpoint"),
                    () -> project + " must not depend on Checkpoint");
        }
    }

    /**
     * Locates the build root by its settings file, independent of the test process directory.
     *
     * @return the ancestor containing {@code settings.gradle.kts}, never {@code null}
     * @throws IllegalStateException if no ancestor contains the settings file
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
