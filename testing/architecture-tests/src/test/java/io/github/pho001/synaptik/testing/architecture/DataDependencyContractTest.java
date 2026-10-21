package io.github.pho001.synaptik.testing.architecture;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;

/**
 * Enforces Data's Model-only boundary across all source sets and dependency configurations.
 * Local fixtures also exercise the rules independently of the current Data Java sources.
 */
final class DataDependencyContractTest {
    private static final String DATA_NAMESPACE = "io.github.pho001.synaptik.data";
    private static final String MODEL_NAMESPACE = "io.github.pho001.synaptik.model";
    private static final Pattern PACKAGE = Pattern.compile("\\bpackage\\s+([\\w.]+)\\s*;");
    private static final Pattern PROJECT_CALL = Pattern.compile("\\bproject\\s*\\(");
    private static final Pattern PROJECT_REFERENCE = Pattern.compile(
            "\\bio\\.github\\.pho001\\.synaptik(?:\\.[\\w$]+)*");

    /**
     * Requires Data's physical build and default project path; approved Text is optional.
     *
     * @throws IOException if repository settings cannot be read
     */
    @Test
    void dataIsRegisteredAtItsDefaultPhysicalPathOnly() throws IOException {
        Path root = repositoryRoot();
        assertSettings(Files.readString(root.resolve("settings.gradle.kts")));
        assertTrue(Files.isRegularFile(root.resolve("extensions/data/build.gradle.kts")),
                "Data must declare its own build at the default physical path");
    }

    /**
     * Positively requires the Model API edge regardless of Data's source-set contents.
     *
     * @throws IOException if Data's build cannot be read
     */
    @Test
    void dataExportsExactlyModelAcrossAllConfigurations() throws IOException {
        assertDataBuild(Files.readString(repositoryRoot().resolve("extensions/data/build.gradle.kts")));
    }

    /**
     * Rejects reverse Data edges in inward projects and extensions other than approved Text.
     * Text's exact sole edge is enforced by its own boundary tests when registered.
     *
     * @throws IOException if project build scripts cannot be traversed or read
     */
    @Test
    void inwardModulesAndOtherExtensionsCannotDependOnData() throws IOException {
        Path root = repositoryRoot();
        Path dataBuild = root.resolve("extensions/data/build.gradle.kts");
        Path textBuild = root.resolve("extensions/text/build.gradle.kts");
        for (String area : List.of("modules", "extensions")) {
            try (Stream<Path> builds = Files.walk(root.resolve(area))) {
                for (Path build : builds.filter(path -> path.getFileName().toString()
                        .equals("build.gradle.kts")).toList()) {
                    if (build.equals(dataBuild) || build.equals(textBuild)) continue;
                    assertNoDataEdge(Files.readString(build), build.toString());
                }
            }
        }
    }

    /**
     * Checks every Java file beneath Data's source root, not only the main or test source set.
     * Fixtures also test the rules independently when that root is absent.
     *
     * @throws IOException if an existing source tree cannot be traversed or read
     */
    @Test
    void allDataJavaSourceSetsUseOnlyTheirNamespaceAndModel() throws IOException {
        Path sources = repositoryRoot().resolve("extensions/data/src");
        if (!Files.exists(sources)) return;
        try (Stream<Path> files = Files.walk(sources)) {
            for (Path source : files.filter(path -> path.toString().endsWith(".java")).toList()) {
                assertDataSource(Files.readString(source), source.toString());
            }
        }
    }

    /** Proves allowed local/Model references pass and wrong namespaces or project imports fail. */
    @Test
    void sourceFixturesExerciseTheDataModulesBoundary() {
        String allowed = """
                package io.github.pho001.synaptik.data;
                import io.github.pho001.synaptik.model.tensor.Tensor;
                import io.github.pho001.synaptik.model.shape.Shape;
                import io.github.pho001.synaptik.model.datatype.DataType;
                import java.util.List;
                import static org.junit.jupiter.api.Assertions.assertEquals;
                final class Fixture { Tensor tensor; Shape shape; DataType type; }
                """;
        assertDoesNotThrow(() -> assertDataSource(allowed, "allowed root fixture"));
        assertDoesNotThrow(() -> assertDataSource(allowed.replace("import java.util.List;",
                "import " + DATA_NAMESPACE + ".local.*;\nimport " + MODEL_NAMESPACE + ".*;"),
                "allowed local and Model wildcard fixture"));
        assertDoesNotThrow(() -> assertDataSource(allowed.replace(
                "package " + DATA_NAMESPACE + ";", "package " + DATA_NAMESPACE + ".batch;"),
                "allowed nested fixture"));
        for (String namespace : List.of("io.github.pho001.synaptik.nn", DATA_NAMESPACE + "wrong")) {
            assertThrows(AssertionError.class, () -> assertDataSource(allowed.replace(
                    "package " + DATA_NAMESPACE + ";", "package " + namespace + ";"), namespace));
        }
        assertThrows(AssertionError.class, () -> assertDataSource("final class Fixture {}", "no package"));
        for (String forbidden : List.of("compiler", "runtime", "prepare", "engine", "checkpoint",
                "backend.cpu", "tuning", "nn", "training", "text", "vision", "modelwrong")) {
            String reference = "io.github.pho001.synaptik." + forbidden + ".Forbidden";
            assertThrows(AssertionError.class, () -> assertDataSource(
                    allowed.replace("import java.util.List;", "import " + reference + ";"), forbidden));
            assertThrows(AssertionError.class, () -> assertDataSource(
                    allowed.replace("import java.util.List;", "import static " + reference + ".VALUE;"),
                    forbidden));
            assertThrows(AssertionError.class, () -> assertDataSource(
                    allowed + "\nfinal class Other { " + reference + " value; }", forbidden));
        }
        assertThrows(AssertionError.class, () -> assertDataSource(allowed.replace(
                "import java.util.List;", "import io.github.pho001.synaptik.*;"), "project root wildcard"));
    }

    /** Proves the exact API export cannot be omitted, hidden, duplicated or widened in tests. */
    @Test
    void buildFixturesRejectForbiddenConfigurationsAndReverseEdges() {
        String allowed = "dependencies {\n    api(project(\":modules:model\"))\n}\n";
        assertDoesNotThrow(() -> assertDataBuild(allowed));
        assertThrows(AssertionError.class, () -> assertDataBuild("dependencies {}"));
        assertThrows(AssertionError.class, () -> assertDataBuild(allowed + allowed));
        assertThrows(AssertionError.class, () -> assertDataBuild(allowed.replace("api(", "implementation(")));
        for (String configuration : List.of("api", "implementation", "compileOnly", "runtimeOnly",
                "testImplementation", "testRuntimeOnly", "integrationTestImplementation")) {
            for (String project : List.of(":modules:model", ":modules:engine", ":extensions:nn",
                    ":extensions:training", ":extensions:checkpoint", ":backends:cpu",
                    ":extensions:text", ":extensions:vision")) {
                String edge = configuration + "(project(\"" + project + "\"))";
                assertThrows(AssertionError.class, () -> assertDataBuild(allowed + edge));
            }
            assertThrows(AssertionError.class, () -> assertNoDataEdge(
                    configuration + "(project(\":extensions:data\"))", "reverse fixture"));
        }
        assertThrows(AssertionError.class, () -> assertDataBuild(
                allowed + "add(\"customConfiguration\", project (\":modules:engine\"))"));
        assertDoesNotThrow(() -> assertNoDataEdge(allowed, "allowed inward fixture"));
    }

    /** Proves optional approved Text registration passes while missing Data and Vision fail. */
    @Test
    void settingsFixturesRejectUnapprovedRegistrationsAndMappings() {
        String allowed = "include(\n    \":extensions:data\",\n)";
        assertDoesNotThrow(() -> assertSettings(allowed));
        assertThrows(AssertionError.class, () -> assertSettings("include()"));
        assertDoesNotThrow(() -> assertSettings(allowed + "\ninclude(\":extensions:text\")"));
        assertThrows(AssertionError.class, () -> assertSettings(allowed + "\ninclude(\":extensions:vision\")"));
        for (String project : List.of(":extensions:data", ":extensions")) {
            assertThrows(AssertionError.class, () -> assertSettings(allowed
                    + "\nproject(\"" + project + "\").projectDir = file(\"elsewhere\")"));
        }
    }

    /**
     * Applies the repository's explicit include/default-path convention, without parsing Kotlin.
     *
     * @param settings non-null settings text using the current line-based include convention
     * @throws AssertionError if Data is absent/remapped or Vision is mentioned
     */
    private static void assertSettings(String settings) {
        assertEquals(1, settings.lines().map(String::strip).filter("\":extensions:data\","::equals).count(),
                "Data must be explicitly included once");
        assertFalse(Pattern.compile("project\\s*\\(\\s*\":extensions(?::data)?\"\\s*\\)\\s*\\.projectDir")
                .matcher(settings).find(), "Data must retain its default physical path");
        assertFalse(settings.contains(":extensions:vision"), "Vision requires separate approval");
    }

    /**
     * Requires the single literal Model API declaration in any dependency configuration.
     *
     * @param build non-null build text using the repository's explicit dependency-line convention
     * @throws AssertionError if the exact Model API edge is absent or any other project call exists
     */
    private static void assertDataBuild(String build) {
        List<String> edges = build.lines().map(String::strip)
                .filter(line -> PROJECT_CALL.matcher(line).find()).toList();
        assertEquals(List.of("api(project(\":modules:model\"))"), edges,
                "Data must export only Model, including across test/custom configurations");
    }

    /**
     * Forbids Data references in inward/other-extension builds regardless of configuration.
     *
     * @param build non-null build script to inspect
     * @param location non-null diagnostic path or fixture label
     * @throws AssertionError if the build mentions the Data project
     */
    private static void assertNoDataEdge(String build, String location) {
        assertFalse(build.contains(":extensions:data"), () -> "reverse Data edge in " + location);
    }

    /**
     * Checks a Java package and project-qualified references, including static imports and fully
     * qualified uses. Data's own namespace and Model are allowed; JDK/JUnit imports are unaffected.
     *
     * @param source non-null Java text using ordinary package and qualified-name spelling
     * @param location non-null diagnostic path or fixture label
     * @throws AssertionError if the package is outside Data or a project reference is not Data/Model
     */
    private static void assertDataSource(String source, String location) {
        var declaration = PACKAGE.matcher(source);
        assertTrue(declaration.find(), () -> "missing Data package: " + location);
        String namespace = declaration.group(1);
        assertTrue(namespace.equals(DATA_NAMESPACE) || namespace.startsWith(DATA_NAMESPACE + "."),
                () -> "wrong Data package: " + location);
        PROJECT_REFERENCE.matcher(source).results().map(match -> match.group()).forEach(reference ->
                assertTrue(reference.equals(DATA_NAMESPACE) || reference.startsWith(DATA_NAMESPACE + ".")
                        || reference.equals(MODEL_NAMESPACE) || reference.startsWith(MODEL_NAMESPACE + "."),
                        () -> "forbidden project reference " + reference + " in " + location));
    }

    /**
     * Locates repository settings independently of the architecture-test process directory.
     *
     * @return the non-null ancestor containing the repository settings file
     * @throws IllegalStateException if no ancestor contains the settings file
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
