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

/** Enforces the coordinated generic Training/NN boundary without directory exemptions. */
final class NnTrainingDependencyContractTest {
    /**
     * Checks physical identity preservation and exact public edges, including test dependencies.
     * @throws IOException if required repository files cannot be read
     */
    @Test
    void nnAndTrainingUseTheDeclaredOneWayDependency() throws IOException {
        Path root = repositoryRoot();
        String settings = Files.readString(root.resolve("settings.gradle.kts"));
        assertTrue(settings.contains("\":extensions:training\","));
        assertTrue(settings.contains(
                "project(\":extensions:training\").projectDir = file(\"modules/training\")"));
        assertFalse(Files.exists(root.resolve("extensions/training/build.gradle.kts")));
        assertFalse(Files.exists(root.resolve("extensions/training/src")));
        assertEquals(List.of("api(project(\":modules:model\"))",
                "api(project(\":modules:engine\"))"),
                projectEdges(root.resolve("modules/training/build.gradle.kts")));
        assertEquals(List.of("api(project(\":modules:model\"))",
                "api(project(\":extensions:training\"))",
                "api(project(\":extensions:checkpoint\"))",
                "api(project(\":modules:engine\"))"),
                projectEdges(root.resolve("extensions/nn/build.gradle.kts")));
    }

    /**
     * Rejects NN imports in all Training sources and locks generic public admission signatures.
     * @throws IOException if a required source file cannot be read
     */
    @Test
    void trainingPublicAdmissionIsGenericAndNnOwnsTheLiveAdapter() throws IOException {
        Path root = repositoryRoot();
        Path training = root.resolve("modules/training/src");
        try (Stream<Path> sources = Files.walk(training)) {
            for (Path source : sources.filter(path -> path.toString().endsWith(".java")).toList()) {
                assertFalse(Files.readString(source).contains("io.github.pho001.synaptik.nn."),
                        () -> "Training must be NN-independent, including tests: " + source);
            }
        }
        Path main = training.resolve("main/java/io/github/pho001/synaptik/training");
        String session = Files.readString(main.resolve("TrainingSession.java"));
        assertEquals(2, session.lines().filter(line ->
                line.contains("public static TrainingSession open(")).count());
        assertEquals(2, java.util.regex.Pattern.compile(
                "public static TrainingSession open\\(\\s*Engine engine,\\s*TrainingParameters parameterSource,")
                .matcher(session).results().count());
        assertFalse(session.contains("import io.github.pho001.synaptik.engine.internal"));
        String storage = Files.readString(main.resolve("TrainingParameterStorage.java"));
        assertTrue(storage.contains("public static void copyToSharedNative(TrainingParameters parameters, Arena arena)"));
        assertFalse(storage.contains("public static void copyToSharedNative(Module"));
        String source = Files.readString(main.resolve("TrainingParameters.java"));
        assertTrue(source.contains("public interface TrainingParameters"));
        assertTrue(source.contains("List<Entry> entries();"));
        assertTrue(source.contains("void validateUnchanged();"));
        assertTrue(source.contains("record Entry(String path, Tensor tensor)"));
        String module = Files.readString(root.resolve(
                "extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Module.java"));
        assertTrue(module.contains("public final TrainingParameters trainingParameters()"));
        assertTrue(module.contains("private static final class LiveTrainingParameters implements TrainingParameters"));
        assertTrue(module.contains("Map<String, Parameter> current = root.parametersRecursively();"));
    }

    /**
     * Reads all project edges to forbid even test-only back-edges.
     * @param build required build script
     * @return ordered normalized project dependency lines
     * @throws IOException if the build script cannot be read
     */
    private static List<String> projectEdges(Path build) throws IOException {
        return Files.readString(build).lines().map(String::strip)
                .filter(line -> line.contains("project(")).toList();
    }

    /**
     * Allows NN's bounded public-owner composition without granting an inward implementation edge.
     * @throws IOException if a required source cannot be read
     */
    @Test
    void ownedModelsComposeOnlyExistingPublicOwnersInTheirBoundedPackage() throws IOException {
        Path nn = repositoryRoot().resolve("extensions/nn/src/main/java/io/github/pho001/synaptik/nn");
        try (Stream<Path> sources = Files.walk(nn)) {
            for (Path path : sources.filter(p -> p.toString().endsWith(".java")).toList()) {
                String text = Files.readString(path);
                for (String forbidden : List.of("compiler.", "runtime.", "prepare.", "backend.", "tuning.",
                        "engine.internal.", "training.internal.", "checkpoint.graph.GraphSource")) {
                    assertFalse(text.contains("import io.github.pho001.synaptik." + forbidden),
                            () -> "NN must compose public owners only: " + path);
                }
                if (text.contains("TrainingSession.open(") || text.contains("TrainingParameterStorage.copyToSharedNative(")) {
                    assertEquals("ModelExecutionState.java", path.getFileName().toString());
                }
            }
        }
        String execution = Files.readString(nn.resolve("module/ModelExecutionState.java"));
        assertTrue(execution.contains("selected.runValues(inference,"));
        assertTrue(execution.contains("session.runValues("));
        assertTrue(execution.contains("GradientMode.RESET_AND_STEP"));
        assertFalse(execution.contains("GraphCheckpoint.rebind(owned.source, stable)"));
        assertTrue(Files.readString(nn.resolve("module/OwnedModelSource.java"))
                .contains("private static final class StateNamespace extends Module"));
    }

    /**
     * Locates settings independently of the test working directory.
     * @return repository root containing settings, never null
     * @throws IllegalStateException if no ancestor contains settings
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
