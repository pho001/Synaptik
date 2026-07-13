package io.github.pho001.synaptik.testing.architecture;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

/** Verifies the planned one-way dependency between the NN and training extensions. */
final class NnTrainingDependencyContractTest {
    /**
     * Checks the extension build direction when the planned NN Gradle project is introduced.
     *
     * @throws IOException if a required build file cannot be read
     */
    @Test
    void nnAndTrainingUseTheDeclaredOneWayDependencyWhenImplemented() throws IOException {
        Path repositoryRoot = repositoryRoot();
        String settings = Files.readString(repositoryRoot.resolve("settings.gradle.kts"));
        Path nnBuild = repositoryRoot.resolve("extensions/nn/build.gradle.kts");
        Path trainingBuild = repositoryRoot.resolve("extensions/training/build.gradle.kts");

        boolean nnIsIncluded = settings.contains("\":extensions:nn\"");
        if (!nnIsIncluded) {
            assertFalse(Files.exists(nnBuild),
                    "an NN build must be included with its Gradle project rather than added implicitly");
            return;
        }

        assertTrue(Files.isRegularFile(nnBuild), "the included NN project must declare its build");
        assertTrue(Files.isRegularFile(trainingBuild), "the training project must declare its build");

        String nnBuildScript = Files.readString(nnBuild);
        String trainingBuildScript = Files.readString(trainingBuild);
        assertFalse(nnBuildScript.contains(":extensions:training"),
                "extensions/nn must not depend on extensions/training");
        assertTrue(trainingBuildScript.contains("implementation(project(\":extensions:nn\"))"),
                "extensions/training must depend on extensions/nn once both projects exist");
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
