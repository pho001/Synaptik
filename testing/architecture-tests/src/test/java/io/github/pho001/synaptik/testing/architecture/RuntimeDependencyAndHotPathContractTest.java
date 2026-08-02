package io.github.pho001.synaptik.testing.architecture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;

/** Verifies Runtime's exact dependency boundary and direct execution hot-path contract. */
final class RuntimeDependencyAndHotPathContractTest {
    private static final List<String> APPROVED_PROJECT_DEPENDENCIES = List.of(
            "implementation(project(\":modules:config\"))",
            "implementation(project(\":modules:backend-contract\"))",
            "implementation(project(\":modules:trace\"))");

    private static final Set<String> HOT_PATH_SOURCES = Set.of(
            "io/github/pho001/synaptik/runtime/execution/BoundBufferTransfer.java",
            "io/github/pho001/synaptik/runtime/execution/BoundInvocation.java",
            "io/github/pho001/synaptik/runtime/run/BoundPublication.java",
            "io/github/pho001/synaptik/runtime/run/PreparedExecutionRunner.java",
            "io/github/pho001/synaptik/runtime/run/RunState.java");

    private static final Set<String> NON_HOT_PATH_SOURCES = Set.of(
            "io/github/pho001/synaptik/runtime/execution/PreparedBufferTransfer.java",
            "io/github/pho001/synaptik/runtime/execution/PreparedExecutable.java",
            "io/github/pho001/synaptik/runtime/execution/PreparedExecution.java",
            "io/github/pho001/synaptik/runtime/execution/package-info.java",
            "io/github/pho001/synaptik/runtime/memory/BufferSlot.java",
            "io/github/pho001/synaptik/runtime/memory/PreparedMemoryPlan.java",
            "io/github/pho001/synaptik/runtime/memory/WorkspaceSlot.java",
            "io/github/pho001/synaptik/runtime/memory/package-info.java",
            "io/github/pho001/synaptik/runtime/resource/BufferRepresentation.java",
            "io/github/pho001/synaptik/runtime/resource/PreparedRepresentationPlan.java",
            "io/github/pho001/synaptik/runtime/resource/WorkspaceRepresentation.java",
            "io/github/pho001/synaptik/runtime/resource/package-info.java",
            "io/github/pho001/synaptik/runtime/run/BufferRepresentationBinding.java",
            "io/github/pho001/synaptik/runtime/run/PreparedPublication.java",
            "io/github/pho001/synaptik/runtime/run/RunResourceOwnership.java",
            "io/github/pho001/synaptik/runtime/run/RunResult.java",
            "io/github/pho001/synaptik/runtime/run/RunStateCreation.java",
            "io/github/pho001/synaptik/runtime/run/package-info.java",
            "io/github/pho001/synaptik/runtime/schedule/PreparedSchedule.java",
            "io/github/pho001/synaptik/runtime/schedule/package-info.java");

    private static final List<String> FORBIDDEN_IDENTITIES = List.of(
            "io.github.pho001.synaptik.model.operation.Operation",
            "io/github/pho001/synaptik/model/operation/Operation",
            "io.github.pho001.synaptik.model.graph.CompiledNode",
            "io/github/pho001/synaptik/model/graph/CompiledNode");

    /** Checks the approved Runtime dependencies and the complete reviewed source inventory. */
    @Test
    void runtimeHasOnlyApprovedDependenciesAndItsHotPathIsFullyReviewed() throws IOException {
        Path repositoryRoot = repositoryRoot();
        assertEquals(
                APPROVED_PROJECT_DEPENDENCIES,
                projectDependencyLines(Files.readString(repositoryRoot.resolve("modules/runtime/build.gradle.kts"))),
                "runtime must have exactly the approved ordered internal project dependencies");

        Path sourceRoot = repositoryRoot.resolve("modules/runtime/src/main/java");
        Set<String> discoveredSources = discoverSources(sourceRoot);
        assertManifest(discoveredSources, HOT_PATH_SOURCES, NON_HOT_PATH_SOURCES);
        for (String hotPathSource : HOT_PATH_SOURCES) {
            assertNoForbiddenGraphIdentity(hotPathSource, Files.readString(sourceRoot.resolve(hotPathSource)));
        }
    }

    /** Proves the dependency and source checks fail closed for representative contract drift. */
    @Test
    void checkerRejectsDependencyAndManifestDrift() {
        assertThrows(
                AssertionError.class,
                () -> assertEquals(
                        APPROVED_PROJECT_DEPENDENCIES,
                        projectDependencyLines("implementation(project(\":modules:config\"))\n"
                                + "implementation(project(\":modules:backend-contract\"))\n"
                                + "implementation(project(\":modules:trace\"))\n"
                                + "implementation(project(\":modules:engine\"))")),
                "an Engine project edge must be rejected");
        assertThrows(
                AssertionError.class,
                () -> assertEquals(
                        APPROVED_PROJECT_DEPENDENCIES,
                        projectDependencyLines("implementation(project(\":modules:config\"))\n"
                                + "implementation(project(\":modules:backend-contract\"))\n"
                                + "implementation(project(\":modules:trace\"))\n"
                                + "implementation(project(\":backends:cpu\"))")),
                "a concrete-backend project edge must be rejected");
        assertThrows(
                AssertionError.class,
                () -> assertManifest(
                        Set.of("hot.java", "non-hot.java", "unclassified.java"),
                        Set.of("hot.java"),
                        Set.of("non-hot.java")),
                "an unclassified Runtime production source must be rejected");
    }

    /** Proves the hot-path scanner rejects both named Model graph identities. */
    @Test
    void checkerRejectsForbiddenModelGraphIdentitiesInHotSources() {
        assertThrows(
                AssertionError.class,
                () -> assertNoForbiddenGraphIdentity(
                        "hot.java", "import io.github.pho001.synaptik.model.operation.Operation;"),
                "a hot-path Operation reference must be rejected");
        assertThrows(
                AssertionError.class,
                () -> assertNoForbiddenGraphIdentity(
                        "hot.java", "io/github/pho001/synaptik/model/graph/CompiledNode"),
                "a hot-path CompiledNode reference must be rejected");
        assertNoForbiddenGraphIdentity(
                "hot.java", "// Operation and CompiledNode are ordinary words, not qualified identities.");
    }

    private static List<String> projectDependencyLines(String buildScript) {
        return buildScript.lines()
                .map(String::strip)
                .filter(line -> line.contains("project("))
                .toList();
    }

    private static Set<String> discoverSources(Path sourceRoot) throws IOException {
        try (Stream<Path> paths = Files.walk(sourceRoot)) {
            return paths.filter(path -> path.toString().endsWith(".java"))
                    .map(sourceRoot::relativize)
                    .map(path -> path.toString().replace(path.getFileSystem().getSeparator(), "/"))
                    .collect(java.util.stream.Collectors.toUnmodifiableSet());
        }
    }

    private static void assertManifest(
            Set<String> discoveredSources, Set<String> hotPathSources, Set<String> nonHotPathSources) {
        Set<String> overlap = hotPathSources.stream().filter(nonHotPathSources::contains)
                .collect(java.util.stream.Collectors.toUnmodifiableSet());
        assertEquals(Set.of(), overlap, "hot and non-hot Runtime source manifests must be disjoint");

        Set<String> classifiedSources = Stream.concat(hotPathSources.stream(), nonHotPathSources.stream())
                .collect(java.util.stream.Collectors.toUnmodifiableSet());
        assertEquals(
                discoveredSources,
                classifiedSources,
                () -> "Runtime production-source manifest mismatch; unclassified="
                        + difference(discoveredSources, classifiedSources)
                        + ", stale=" + difference(classifiedSources, discoveredSources));
    }

    private static Set<String> difference(Set<String> first, Set<String> second) {
        return first.stream().filter(path -> !second.contains(path))
                .collect(java.util.stream.Collectors.toUnmodifiableSet());
    }

    private static void assertNoForbiddenGraphIdentity(String sourcePath, String source) {
        for (String forbiddenIdentity : FORBIDDEN_IDENTITIES) {
            assertFalse(
                    source.contains(forbiddenIdentity),
                    () -> sourcePath + " must not use Runtime's forbidden hot-path identity: " + forbiddenIdentity);
        }
    }

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
