package io.github.pho001.synaptik.testing.architecture;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Set;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;

/** Verifies NN-to-Training direction and Training's exact public API dependency surface. */
final class NnTrainingDependencyContractTest {
    /**
     * Checks the exact Model/NN/Engine-to-Training extension direction.
     * @throws IOException if a required build file cannot be read
     */
    @Test
    void nnAndTrainingUseTheDeclaredOneWayDependency() throws IOException {
        Path repositoryRoot = repositoryRoot();
        String settings = Files.readString(repositoryRoot.resolve("settings.gradle.kts"));
        Path nnBuild = repositoryRoot.resolve("extensions/nn/build.gradle.kts");
        Path trainingBuild = repositoryRoot.resolve("extensions/training/build.gradle.kts");

        assertTrue(settings.contains("\":extensions:nn\""),
                "extensions/nn must be included explicitly in the Gradle build");
        assertTrue(Files.isRegularFile(nnBuild), "the included NN project must declare its build");
        assertTrue(Files.isRegularFile(trainingBuild), "the training project must declare its build");

        String nnBuildScript = Files.readString(nnBuild);
        String trainingBuildScript = Files.readString(trainingBuild);
        assertTrue(nnBuildScript.contains("implementation(project(\":modules:model\"))"),
                "extensions/nn must depend on modules/model");
        assertTrue(
                Pattern.compile("project\\(\\\"([^\\\"]+)\\\"\\)")
                        .matcher(nnBuildScript)
                        .results()
                        .map(match -> match.group(1))
                        .collect(java.util.stream.Collectors.toUnmodifiableSet())
                        .equals(Set.of(":modules:model")),
                "extensions/nn may have only the modules/model project dependency");
        assertFalse(nnBuildScript.contains(":extensions:training"),
                "extensions/nn must not depend on extensions/training");
        assertFalse(nnBuildScript.contains(":modules:compiler"),
                "extensions/nn must not depend on modules/compiler");
        assertFalse(nnBuildScript.contains(":modules:runtime"),
                "extensions/nn must not depend on modules/runtime");
        assertFalse(nnBuildScript.contains(":modules:prepare"),
                "extensions/nn must not depend on modules/prepare");
        assertFalse(nnBuildScript.contains(":modules:engine"),
                "extensions/nn must not depend on modules/engine");
        assertFalse(nnBuildScript.contains(":backends:"),
                "extensions/nn must not depend on concrete backends");
        assertTrue(trainingBuildScript.contains("api(project(\":extensions:nn\"))"),
                "extensions/training must expose Module and Parameter from extensions/nn");
        assertTrue(trainingBuildScript.contains("api(project(\":modules:model\"))"),
                "extensions/training must expose Tensor, DataType, and Shape from modules/model");
        assertTrue(trainingBuildScript.contains("api(project(\":modules:engine\"))"),
                "extensions/training must expose its intentional public Engine dependency");
        assertTrue(
                Pattern.compile("api\\(project\\(\\\"([^\\\"]+)\\\"\\)\\)")
                        .matcher(trainingBuildScript)
                        .results()
                        .map(match -> match.group(1))
                        .collect(java.util.stream.Collectors.toUnmodifiableSet())
                        .equals(Set.of(
                                ":extensions:nn",
                                ":modules:model",
                                ":modules:engine")),
                "Training API dependencies must be exactly NN, Model, and Engine");
        assertFalse(Pattern.compile("implementation\\(project\\(")
                        .matcher(trainingBuildScript).find(),
                "Training must not hide project types used by its public API");
        assertTrue(
                Pattern.compile("project\\(\\\"([^\\\"]+)\\\"\\)")
                        .matcher(trainingBuildScript)
                        .results()
                        .map(match -> match.group(1))
                        .collect(java.util.stream.Collectors.toUnmodifiableSet())
                        .equals(Set.of(
                                ":extensions:nn",
                                ":modules:model",
                                ":modules:engine")),
                "extensions/training may depend only on NN, Model, and the public Engine facade");
        assertFalse(trainingBuildScript.contains(":modules:runtime"),
                "extensions/training must not depend on Runtime");
        assertFalse(trainingBuildScript.contains(":modules:prepare"),
                "extensions/training must not depend on Prepare");
        assertFalse(trainingBuildScript.contains(":tools:tuning"),
                "extensions/training must not depend on tuning");
        assertFalse(trainingBuildScript.contains(":backends:"),
                "extensions/training must not depend on concrete backends");
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
