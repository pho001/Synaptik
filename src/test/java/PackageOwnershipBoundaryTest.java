import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Predicate;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class PackageOwnershipBoundaryTest {
    private static final Path MAIN = Path.of("src/main/java");

    @Test
    void operationsDoNotDependOnGraphRuntimeOrBackendInternals() throws IOException {
        assertNoImports(
                "operations",
                importedType -> startsWithAny(importedType, List.of("graph.", "runtime.", "backend.")),
                "operations must not import graph, runtime, or backend internals"
        );
    }

    @Test
    void backendContractIsJdkOnlyDependencyLeaf() throws IOException {
        assertNoImports(
                "backend/contract",
                importedType -> !isJdkImport(importedType),
                "backend.contract may import only JDK types"
        );
    }

    @Test
    void graphModelDoesNotDependOnCompilePlanningRuntimeTraceOrConcreteBackends() throws IOException {
        List<String> allowedPackages = List.of("backend.contract.", "operations.", "tensor.");
        assertNoImports(
                "graph/model",
                importedType -> !isJdkImport(importedType) && !startsWithAny(importedType, allowedPackages),
                "graph.model may import only JDK, backend.contract, operations, and tensor types"
        );
    }

    @Test
    void graphOptimizerDoesNotDependOnConcreteBackendKernelPackages() throws IOException {
        List<String> concreteBackends = List.of(
                "backend.cpu.",
                "backend.cpu1.",
                "backend.metal.",
                "backend.cuda.",
                "backend.opencl."
        );
        assertNoImports(
                "graph/optimizer",
                importedType -> startsWithAny(importedType, concreteBackends)
                        && (importedType.contains(".kernel.") || importedType.contains(".kernels.")),
                "graph.optimizer must not import concrete backend kernel packages"
        );
    }

    @Test
    void traceContainsOnlySnapshotsAndRuntimeContracts() throws IOException {
        List<String> allowedPackages = List.of("runtime.contract.", "trace.");
        assertNoImports(
                "trace",
                importedType -> !isJdkImport(importedType) && !startsWithAny(importedType, allowedPackages),
                "trace DTOs may import only JDK, trace-owned DTOs, and runtime contracts"
        );
    }

    @Test
    void tensorDoesNotDependOnRuntimeResidencyImplementations() throws IOException {
        assertNoImports(
                "tensor",
                importedType -> startsWithAny(importedType, List.of(
                        "graph.execution.residency.",
                        "runtime.residency."
                )),
                "tensor must not import current or target runtime residency implementations"
        );
    }

    private static void assertNoImports(
            String relativeRoot,
            Predicate<String> forbiddenImport,
            String rule
    ) throws IOException {
        Path root = MAIN.resolve(relativeRoot);
        assertTrue(Files.isDirectory(root), () -> "Required source root is missing: " + root);

        List<String> offenders = importsUnder(root, forbiddenImport);
        assertTrue(offenders.isEmpty(), () -> rule + ": " + offenders);
    }

    private static List<String> importsUnder(Path root, Predicate<String> forbiddenImport) throws IOException {
        List<String> offenders = new ArrayList<>();
        try (Stream<Path> paths = Files.walk(root)) {
            for (Path path : paths.filter(Files::isRegularFile)
                    .filter(file -> file.toString().endsWith(".java"))
                    .sorted()
                    .toList()) {
                List<String> lines = Files.readAllLines(path);
                for (int lineNumber = 0; lineNumber < lines.size(); lineNumber++) {
                    String importedType = importedType(lines.get(lineNumber));
                    if (importedType != null && forbiddenImport.test(importedType)) {
                        offenders.add(path + ":" + (lineNumber + 1) + ": " + importedType);
                    }
                }
            }
        }
        return offenders;
    }

    private static String importedType(String sourceLine) {
        String trimmed = sourceLine.trim();
        if (!trimmed.startsWith("import ")) {
            return null;
        }
        String importedType = trimmed.substring("import ".length());
        if (importedType.startsWith("static ")) {
            importedType = importedType.substring("static ".length());
        }
        return importedType.endsWith(";")
                ? importedType.substring(0, importedType.length() - 1)
                : importedType;
    }

    private static boolean startsWithAny(String value, List<String> prefixes) {
        return prefixes.stream().anyMatch(value::startsWith);
    }

    private static boolean isJdkImport(String importedType) {
        return startsWithAny(importedType, List.of("java.", "javax.", "jdk."));
    }
}
