package io.github.pho001.synaptik.testing.architecture;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;

/** Enforces Text's sole Data export, transitive Model access and external-only integration. */
final class TextDependencyContractTest {
    private static final String PREFIX = "io.github.pho001.synaptik.";
    private static final Pattern REFERENCES = Pattern.compile(
            "\\bio\\.github\\.pho001\\.synaptik(?:\\.[\\w$]+)*");

    /**
     * Checks registration and every production/custom/test dependency configuration.
     * @throws IOException if settings or build files cannot be read
     */
    @Test
    void textIsRegisteredWithExactlyTheDataApiEdge() throws IOException {
        Path root = root();
        String settings = Files.readString(root.resolve("settings.gradle.kts"));
        assertEquals(1, settings.lines().map(String::strip)
                .filter("\":extensions:text\","::equals).count());
        assertFalse(Pattern.compile("project\\s*\\(\\s*\":extensions(?::text)?\"\\s*\\)\\s*\\.projectDir")
                .matcher(settings).find());
        assertBuild(Files.readString(root.resolve("extensions/text/build.gradle.kts")));
    }

    /**
     * Checks all Text Java source sets, including fully qualified and static references.
     * @throws IOException if Text sources cannot be read
     */
    @Test
    void textSourcesUseOnlyTextAndPublicDataModelNamespaces() throws IOException {
        try (var files = Files.walk(root().resolve("extensions/text/src"))) {
            for (Path file : files.filter(path -> path.toString().endsWith(".java")).toList()) {
                assertSource(Files.readString(file));
            }
        }
    }

    /**
     * Prohibits inward/NN Text edges and confines numerical integration to a test dependency.
     * @throws IOException if repository build files cannot be read
     */
    @Test
    void inwardProjectsCannotDependOnTextAndIntegrationUsesOnlyATestEdge() throws IOException {
        Path root = root();
        for (String area : List.of("modules", "extensions")) {
            try (var files = Files.walk(root.resolve(area))) {
                for (Path file : files.filter(path -> path.getFileName().toString()
                        .equals("build.gradle.kts")).toList()) {
                    assertNoTextEdge(Files.readString(file));
                }
            }
        }
        String nn = Files.readString(root.resolve("extensions/nn/build.gradle.kts"));
        assertFalse(nn.contains(":extensions:data"));
        String integration = Files.readString(root.resolve("testing/integration-tests/build.gradle.kts"));
        assertEquals(List.of("testImplementation(project(\":extensions:text\"))"),
                integration.lines().map(String::strip).filter(line -> line.contains(":extensions:text")).toList());
        String consumers = Files.readString(root.resolve("testing/architecture-tests/build.gradle.kts"));
        assertTrue(consumers.contains("create(\"textApiConsumer\")"));
        assertTrue(consumers.contains("add(isolatedTextApiConsumer.implementationConfigurationName, project(\":extensions:text\"))"));
        assertTrue(consumers.contains("dependsOn(tasks.named(isolatedTextApiConsumer.compileJavaTaskName))"));
    }

    /** Proves absent, private, duplicate, test-only and forbidden project edges are rejected. */
    @Test
    void dependencyFixturesRejectEveryAdditionalConfigurationAndReverseEdge() {
        String allowed = "dependencies {\napi(project(\":extensions:data\"))\n}";
        assertDoesNotThrow(() -> assertBuild(allowed));
        assertThrows(AssertionError.class, () -> assertBuild("dependencies {}"));
        assertThrows(AssertionError.class, () -> assertBuild(allowed + allowed));
        assertThrows(AssertionError.class, () -> assertBuild(allowed.replace("api(", "implementation(")));
        for (String configuration : List.of("api", "implementation", "compileOnly", "runtimeOnly", "testImplementation", "testRuntimeOnly")) {
            for (String project : List.of(":modules:model", ":extensions:data", ":extensions:nn", ":modules:engine",
                    ":extensions:training", ":extensions:checkpoint", ":modules:compiler", ":backends:cpu")) {
                assertThrows(AssertionError.class, () -> assertBuild(allowed + "\n"
                        + configuration + "(project(\"" + project + "\"))"));
            }
            assertThrows(AssertionError.class, () -> assertNoTextEdge(configuration + "(project(\":extensions:text\"))"));
        }
    }

    /** Proves ordinary, static and fully qualified forbidden references fail, not just imports. */
    @Test
    void sourceFixturesRejectForbiddenOwnersAndInternalApis() {
        String allowed = "package " + PREFIX + "text;\nimport " + PREFIX + "data.PaddedSequenceBatch;\n"
                + "import " + PREFIX + "model.tensor.Tensor;\nfinal class Fixture {}";
        assertDoesNotThrow(() -> assertSource(allowed));
        assertThrows(AssertionError.class, () -> assertSource("final class Fixture {}"));
        assertThrows(AssertionError.class, () -> assertSource(allowed.replace("package " + PREFIX + "text;",
                "package " + PREFIX + "nn;")));
        for (String owner : List.of("nn", "engine", "training", "checkpoint", "compiler", "runtime", "prepare",
                "backend.cpu", "tuning", "vision", "model.internal", "data.internal", "modelwrong")) {
            String reference = PREFIX + owner + ".Forbidden";
            assertThrows(AssertionError.class, () -> assertSource(allowed + "\nimport " + reference + ";"));
            assertThrows(AssertionError.class, () -> assertSource(allowed + "\nimport static " + reference + ".VALUE;"));
            assertThrows(AssertionError.class, () -> assertSource(allowed + "\nclass Other { " + reference + " value; }"));
        }
    }

    /**
     * Applies the repository's explicit dependency-line convention to all project calls.
     * @param build non-null Kotlin build text
     * @throws AssertionError if the sole literal Data API export is missing or widened
     */
    private static void assertBuild(String build) {
        assertEquals(List.of("api(project(\":extensions:data\"))"), build.lines().map(String::strip)
                .filter(line -> Pattern.compile("\\bproject\\s*\\(").matcher(line).find()).toList());
    }

    /**
     * Rejects reverse Text edges regardless of Gradle configuration.
     * @param build non-null project build text
     * @throws AssertionError if any Text project reference occurs
     */
    private static void assertNoTextEdge(String build) {
        assertFalse(build.contains(":extensions:text"));
    }

    /**
     * Allows only the flat Text package and public Data/Model project references.
     * @param source non-null Java source or fixture
     * @throws AssertionError if a namespace or internal reference violates the boundary
     */
    private static void assertSource(String source) {
        assertTrue(Pattern.compile("\\bpackage\\s+io\\.github\\.pho001\\.synaptik\\.text\\s*;")
                .matcher(source).find());
        REFERENCES.matcher(source).results().map(match -> match.group()).forEach(reference -> {
            assertTrue(List.of("text", "data", "model").stream().anyMatch(owner ->
                    reference.equals(PREFIX + owner) || reference.startsWith(PREFIX + owner + ".")), reference);
            assertFalse(reference.contains(".internal."), reference);
        });
    }

    /**
     * Finds settings independently of the test process working directory.
     * @return non-null repository root
     * @throws IllegalStateException if settings cannot be located
     */
    private static Path root() {
        Path path = Path.of("").toAbsolutePath();
        while (path != null && !Files.isRegularFile(path.resolve("settings.gradle.kts"))) path = path.getParent();
        if (path == null) throw new IllegalStateException("repository settings not found");
        return path;
    }
}
