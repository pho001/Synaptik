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
 * Protects the current model-only Checkpoint project boundary and the deferred Training
 * Checkpoint adapter gate. A later public Checkpoint API may require a separately approved Model
 * edge; this test records the empty project's present dependency set, not a permanent ban.
 */
final class CheckpointDependencyContractTest {
    /**
     * Checks that Checkpoint is included alone, has exactly its two current direct project edges,
     * and remains downstream of all inward modules, NN, and Training.
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
                "implementation(project(\":extensions:nn\"))",
                "implementation(project(\":modules:engine\"))"), projectEdges,
                "the current empty project must have exactly the NN and public Engine edges");

        try (Stream<Path> moduleBuilds = Files.walk(root.resolve("modules"))) {
            for (Path build : moduleBuilds.filter(path -> path.getFileName().toString()
                    .equals("build.gradle.kts")).toList()) {
                assertFalse(Files.readString(build).contains(":extensions:checkpoint"),
                        () -> "inward module must not depend on Checkpoint: " + build);
            }
        }
        for (String extension : List.of("nn", "training")) {
            Path build = root.resolve("extensions/" + extension + "/build.gradle.kts");
            assertFalse(Files.readString(build).contains(":extensions:checkpoint"),
                    () -> extension + " must not depend on Checkpoint");
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
