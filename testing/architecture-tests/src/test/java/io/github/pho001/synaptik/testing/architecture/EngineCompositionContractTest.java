package io.github.pho001.synaptik.testing.architecture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Locks Engine's exact direct dependency inventory and one-way composition boundary. */
final class EngineCompositionContractTest {
    private static final List<String> APPROVED_ENGINE = List.of(
            "implementation(project(\":modules:model\"))",
            "implementation(project(\":modules:planning\"))",
            "implementation(project(\":modules:compiler\"))",
            "implementation(project(\":modules:runtime\"))",
            "implementation(project(\":modules:prepare\"))",
            "implementation(project(\":modules:config\"))",
            "implementation(project(\":modules:trace\"))",
            "implementation(project(\":backends:cpu\"))",
            "implementation(project(\":backends:metal\"))",
            "implementation(project(\":backends:cuda\"))",
            "implementation(project(\":tools:tuning\"))");
    private static final List<String> APPROVED_INTEGRATION = List.of(
            "implementation(project(\":modules:engine\"))",
            "testImplementation(project(\":modules:compiler\"))",
            "testImplementation(project(\":modules:runtime\"))",
            "testImplementation(project(\":modules:config\"))",
            "testImplementation(project(\":modules:model\"))",
            "testImplementation(project(\":backends:cpu\"))",
            "testImplementation(project(\":extensions:nn\"))");

    @Test
    void engineHasOnlyTheApprovedOrderedDirectDependencies() throws IOException {
        Path root = repositoryRoot();
        String engineBuild = Files.readString(root.resolve("modules/engine/build.gradle.kts"));
        assertEquals(APPROVED_ENGINE, projectDependencyLines(engineBuild));
        String tuningBuild = Files.readString(root.resolve("tools/tuning/build.gradle.kts"));
        assertFalse(tuningBuild.contains("project(\":modules:engine\")"));

        String integrationBuild = Files.readString(
                root.resolve("testing/integration-tests/build.gradle.kts"));
        assertEquals(APPROVED_INTEGRATION, projectDependencyLines(integrationBuild));

        for (Path build : Files.walk(root)
                .filter(path -> path.getFileName().toString().equals("build.gradle.kts"))
                .toList()) {
            Path relative = root.relativize(build);
            boolean inwardProduction = relative.startsWith("modules")
                    || relative.startsWith("backends");
            if (inwardProduction && !build.equals(root.resolve("modules/engine/build.gradle.kts"))) {
                assertFalse(Files.readString(build).contains("project(\":modules:engine\")"),
                        () -> "inward dependency on Engine in " + relative);
            }
        }
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
