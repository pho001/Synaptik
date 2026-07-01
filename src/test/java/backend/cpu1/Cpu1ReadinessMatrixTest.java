package backend.cpu1;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class Cpu1ReadinessMatrixTest {
    private static final Path CPU1_TEST_ROOT = Path.of("src/test/java/backend/cpu1");

    @Test
    void benchmarkMatrixHasCheckedInTaggedOwnersWithoutRunningBenchmarks() throws IOException {
        Cpu1BenchmarkMatrixReport report = Cpu1BenchmarkMatrixReport.current();

        assertEquals(List.of(), report.coveredEntriesWithoutOwners(), report::gateReport);
        for (Cpu1BenchmarkMatrixReport.BenchmarkEntry entry : report.coveredEntries()) {
            assertBenchmarkOwnerSource(entry, report.gateReport());
        }
    }

    @Test
    void benchmarkMatrixInventoriesCanonicalCpu1BenchmarkSources() throws IOException {
        Cpu1BenchmarkMatrixReport report = Cpu1BenchmarkMatrixReport.current();

        assertEquals(canonicalTaggedCpu1BenchmarkClasses(), report.canonicalBenchmarkOwnerClassNames(), report::gateReport);
        assertFalse(
                report.canonicalBenchmarkOwnerClassNames().contains("backend.cpu1.Cpu1FusedParityBenchmarkTest"),
                report::gateReport
        );
    }

    @Test
    void benchmarkMatrixDocumentsDeferredScope() {
        Cpu1BenchmarkMatrixReport report = Cpu1BenchmarkMatrixReport.current();
        String text = report.gateReport();

        assertFalse(report.deferredEntries().isEmpty(), text);
        assertTrue(text.contains("todo/118-cpu1-graph-input-materialization-plan.md"), text);
        assertTrue(text.contains("todo/119-general-matmul-epilogue-ir-plan.md"), text);
        assertTrue(text.contains("MATMUL_EPILOGUE"), text);
        assertTrue(text.contains("BF16 attention backward"), text);
        assertTrue(text.contains("blocked/tiled attention"), text);
        assertTrue(text.contains("deterministic parallel scatter"), text);
    }

    @Test
    void targetedParityMatrixCoversEveryCpu1CoverageFamilyRoute() {
        Cpu1CoverageReport coverageReport = Cpu1CoverageReport.current();
        Cpu1TargetedParityTestMatrixReport report = Cpu1TargetedParityTestMatrixReport.from(coverageReport);

        assertEquals(
                List.of(),
                report.missingRequiredRouteFamilies(),
                () -> coverageReport.gateReport() + "\n" + report.gateReport()
        );
    }

    @Test
    void targetedParityMatrixInventoriesCheckedInContractSources() throws IOException {
        Cpu1TargetedParityTestMatrixReport report = Cpu1TargetedParityTestMatrixReport.current();
        List<String> ownerClasses = report.targetedOwnerClassNames();

        for (String ownerClass : ownerClasses) {
            assertTestOwnerSource(ownerClass, report.gateReport());
        }
        assertTrue(ownerClasses.containsAll(canonicalCpu1ExecutionContractClasses()), report::gateReport);
    }

    @Test
    void targetedParityMatrixDocumentsDeferredAndNonGoalScope() {
        Cpu1TargetedParityTestMatrixReport report = Cpu1TargetedParityTestMatrixReport.current();
        String text = report.gateReport();

        assertFalse(report.deferredEntries().isEmpty(), text);
        assertFalse(report.nonGoalEntries().isEmpty(), text);
        assertTrue(text.contains("todo/118-cpu1-graph-input-materialization-plan.md"), text);
        assertTrue(text.contains("todo/119-general-matmul-epilogue-ir-plan.md"), text);
        assertTrue(text.contains("MATMUL_EPILOGUE"), text);
        assertTrue(text.contains("BF16 attention backward"), text);
        assertTrue(text.contains("blocked/tiled attention"), text);
        assertTrue(text.contains("deterministic parallel scatter"), text);
        assertTrue(text.contains("actual benchmark performance numbers/default-route enablement"), text);
    }

    private static void assertBenchmarkOwnerSource(
            Cpu1BenchmarkMatrixReport.BenchmarkEntry entry,
            String reportText
    ) throws IOException {
        Path sourcePath = sourcePathForClass(entry.ownerClassName());
        assertTrue(Files.exists(sourcePath), () -> reportText + "\nMissing benchmark owner source: " + sourcePath);

        String source = Files.readString(sourcePath);
        assertTrue(source.contains("class " + simpleClassName(entry.ownerClassName())),
                () -> reportText + "\nMissing benchmark owner class declaration: " + entry.ownerClassName());
        assertTrue(source.contains("@Tag(\"benchmark\")"),
                () -> reportText + "\nBenchmark owner is not tagged @Tag(\"benchmark\"): " + entry.ownerClassName());
        assertTrue(source.contains("void " + entry.ownerMethodName() + "("),
                () -> reportText + "\nMissing benchmark owner method: " + entry.ownerClassName()
                        + "#" + entry.ownerMethodName());
    }

    private static void assertTestOwnerSource(String ownerClassName, String reportText) throws IOException {
        Path sourcePath = sourcePathForClass(ownerClassName);
        assertTrue(Files.exists(sourcePath), () -> reportText + "\nMissing targeted parity owner source: " + sourcePath);

        String source = Files.readString(sourcePath);
        assertTrue(source.contains("class " + simpleClassName(ownerClassName)),
                () -> reportText + "\nMissing targeted parity owner class declaration: " + ownerClassName);
        assertTrue(source.contains("@Test"),
                () -> reportText + "\nTargeted parity owner has no JUnit @Test methods: " + ownerClassName);
    }

    private static List<String> canonicalTaggedCpu1BenchmarkClasses() throws IOException {
        try (var stream = Files.list(CPU1_TEST_ROOT)) {
            return stream
                    .filter(path -> path.getFileName().toString().endsWith("BenchmarkTest.java"))
                    .filter(Cpu1ReadinessMatrixTest::containsBenchmarkTag)
                    .map(Cpu1ReadinessMatrixTest::cpu1ClassNameForSource)
                    .sorted()
                    .toList();
        }
    }

    private static List<String> canonicalCpu1ExecutionContractClasses() throws IOException {
        try (var stream = Files.list(CPU1_TEST_ROOT)) {
            return stream
                    .filter(path -> path.getFileName().toString().endsWith("ExecutionContractTest.java"))
                    .filter(Cpu1ReadinessMatrixTest::containsJUnitTest)
                    .map(Cpu1ReadinessMatrixTest::cpu1ClassNameForSource)
                    .sorted()
                    .toList();
        }
    }

    private static boolean containsBenchmarkTag(Path path) {
        return sourceContains(path, "@Tag(\"benchmark\")");
    }

    private static boolean containsJUnitTest(Path path) {
        return sourceContains(path, "@Test");
    }

    private static boolean sourceContains(Path path, String token) {
        try {
            return Files.readString(path).contains(token);
        } catch (IOException ex) {
            throw new IllegalStateException("Unable to read source file " + path, ex);
        }
    }

    private static String cpu1ClassNameForSource(Path path) {
        String fileName = path.getFileName().toString();
        return "backend.cpu1." + fileName.substring(0, fileName.length() - ".java".length());
    }

    private static Path sourcePathForClass(String className) {
        return Path.of("src/test/java", className.replace('.', '/') + ".java");
    }

    private static String simpleClassName(String className) {
        int index = className.lastIndexOf('.');
        return index < 0 ? className : className.substring(index + 1);
    }
}
