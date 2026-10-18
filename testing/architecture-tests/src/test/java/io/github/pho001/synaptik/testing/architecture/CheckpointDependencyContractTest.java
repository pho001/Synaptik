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
 * Protects generic Checkpoint's public Model/private Compiler boundary and the deferred
 * Training Checkpoint adapter gate.
 */
final class CheckpointDependencyContractTest {
    /**
     * Checks physical mapping, exact cold edges and the absence of reverse inward dependencies.
     *
     * @throws IOException if settings or a required project build file cannot be read
     */
    @Test
    void checkpointHasOnlyCurrentModelBoundaryEdges() throws IOException {
        Path root = repositoryRoot();
        String settings = Files.readString(root.resolve("settings.gradle.kts"));
        Path checkpoint = root.resolve("modules/checkpoint");
        Path checkpointBuild = checkpoint.resolve("build.gradle.kts");

        assertTrue(settings.lines().map(String::strip).anyMatch("\":extensions:checkpoint\","::equals),
                "the model-only Checkpoint project must be included");
        assertFalse(settings.contains(":extensions:training-checkpoint"),
                "the optional Training Checkpoint adapter must await its separate gate");
        assertTrue(Files.isRegularFile(checkpointBuild), "Checkpoint must declare its own build");
        assertTrue(settings.contains("project(\":extensions:checkpoint\").projectDir = file(\"modules/checkpoint\")"));
        assertFalse(Files.exists(root.resolve("extensions/checkpoint/src")));
        assertFalse(Files.exists(root.resolve("extensions/checkpoint/build.gradle.kts")));

        List<String> projectEdges = Files.readString(checkpointBuild).lines()
                .map(String::strip)
                .filter(line -> line.contains("project("))
                .toList();
        assertEquals(List.of(
                "api(project(\":modules:model\"))",
                "implementation(project(\":modules:compiler\"))"), projectEdges,
                "generic Checkpoint exposes Model and privately consumes cold Compiler");

        for (String area : List.of("modules")) {
            try (Stream<Path> builds = Files.walk(root.resolve(area))) {
                for (Path build : builds.filter(path -> path.getFileName().toString()
                        .equals("build.gradle.kts")).toList()) {
                    if (build.equals(checkpointBuild)) continue;
                    assertFalse(Files.readString(build).contains(":extensions:checkpoint"),
                            () -> "inward project must not depend on Checkpoint: " + build);
                }
            }
        }
        for (String project : List.of("modules/training", "tools/tuning")) {
            Path build = root.resolve(project + "/build.gradle.kts");
            assertFalse(Files.readString(build).contains(":extensions:checkpoint"),
                    () -> project + " must not depend on Checkpoint");
        }
        try (Stream<Path> sources = Files.walk(checkpoint.resolve("src"))) {
            for (Path source : sources.filter(path -> path.toString().endsWith(".java")).toList()) {
                String text = Files.readString(source);
                for (String forbidden : List.of("nn", "engine", "training")) {
                    assertFalse(text.contains("io.github.pho001.synaptik." + forbidden + "."),
                            () -> "generic source/test must not import " + forbidden + ": " + source);
                }
                assertFalse(text.contains("package io.github.pho001.synaptik.nn.checkpoint"));
            }
        }
        String facade = Files.readString(checkpoint.resolve(
                "src/main/java/io/github/pho001/synaptik/checkpoint/graph/GraphCheckpoint.java"));
        assertFalse(facade.contains("import io.github.pho001.synaptik.compiler"));
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
