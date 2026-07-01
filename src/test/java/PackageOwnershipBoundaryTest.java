import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Predicate;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class PackageOwnershipBoundaryTest {
    private static final Path MAIN = Path.of("src/main/java");

    @Test
    void compiledGraphIsTheOnlyGraphLifecycleFacade() throws IOException {
        List<String> offenders = importsUnder(
                MAIN.resolve("graph"),
                importedType -> importedType.startsWith("prepare.") || importedType.startsWith("runtime.")
        ).stream()
                .filter(line -> !line.startsWith("src/main/java/graph/CompiledGraph.java:"))
                .toList();
        assertTrue(offenders.isEmpty(), () -> "graph lifecycle imports outside CompiledGraph: " + offenders);
    }

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
    void externalOpenBlasProviderIsLowLevelAndBackendNeutral() throws IOException {
        assertNoImports(
                "backend/provider/blas/openblas",
                importedType -> !importedType.startsWith("java."),
                "the shared OpenBLAS provider may import only JDK java.* and static java.* types"
        );
    }

    @Test
    void removedPackageTreesStayRemoved() throws IOException {
        List<Path> removedTrees = List.of(
                MAIN.resolve("graph/execution"),
                MAIN.resolve("graph/compile/descriptor"),
                MAIN.resolve("graph/compile/intent"),
                MAIN.resolve("graph/compile/planning"),
                MAIN.resolve("backend/prepare"),
                MAIN.resolve("backend/runtime"),
                MAIN.resolve("backend/memory"),
                MAIN.resolve("backend/blas"),
                MAIN.resolve("backend/ComputeEngine.java")
        );
        List<Path> offenders = new ArrayList<>();
        for (Path removedTree : removedTrees) {
            if (containsJavaSources(removedTree)) {
                offenders.add(removedTree);
            }
        }
        assertTrue(offenders.isEmpty(), () -> "removed package source trees remain: " + offenders);
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
    void planningDoesNotDependOnCompileRuntimeOrConcreteBackends() throws IOException {
        assertNoImports(
                "planning",
                importedType -> startsWithAny(importedType, List.of(
                        "graph.CompiledGraph",
                        "graph.compile.",
                        "prepare.",
                        "runtime.",
                        "backend.cpu.",
                        "backend.cpu1.",
                        "backend.metal.",
                        "backend.cuda.",
                        "backend.opencl.",
                        "backend.accelerator.",
                        "backend.partition.",
                        "backend.lowering."
                )),
                "planning must not depend on graph compile orchestration, prepare, runtime, or concrete backends"
        );
    }

    @Test
    void legacyCompilePlanningTreesAreRemoved() {
        List<Path> legacyTrees = List.of(
                MAIN.resolve("graph/compile/descriptor"),
                MAIN.resolve("graph/compile/intent"),
                MAIN.resolve("graph/compile/planning")
        );
        assertTrue(legacyTrees.stream().noneMatch(Files::exists),
                () -> "legacy compile planning trees remain: "
                        + legacyTrees.stream().filter(Files::exists).toList());
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

    @Test
    void runtimeDoesNotDependOnConcreteBackends() throws IOException {
        assertNoImports(
                "runtime",
                importedType -> startsWithAny(importedType, List.of(
                        "backend.cpu.",
                        "backend.cpu1.",
                        "backend.metal.",
                        "backend.cuda.",
                        "backend.opencl."
                )),
                "runtime must not import concrete backend implementations"
        );
    }

    @Test
    void prepareContextAndValidationDoNotDependOnOrchestrationOrConcreteBackends() throws IOException {
        List<String> forbiddenPackages = List.of(
                "prepare.orchestration.",
                "backend.cpu.",
                "backend.cpu1.",
                "backend.metal.",
                "backend.cuda.",
                "backend.opencl."
        );
        assertNoImports(
                "prepare/context",
                importedType -> startsWithAny(importedType, forbiddenPackages),
                "prepare.context must not depend on orchestration or concrete backends"
        );
        assertNoImports(
                "prepare/validation",
                importedType -> startsWithAny(importedType, forbiddenPackages),
                "prepare.validation must not depend on orchestration or concrete backends"
        );
    }

    @Test
    void concreteBackendPreparersDoNotDependOnPrepareOrchestration() throws IOException {
        for (String backend : List.of("cpu", "cpu1", "metal", "cuda")) {
            assertNoImports(
                    "backend/" + backend + "/prepare",
                    importedType -> importedType.startsWith("prepare.orchestration."),
                    "backend." + backend + ".prepare must not depend on prepare orchestration"
            );
        }
    }

    @Test
    void legacyBackendPrepareTreeContainsNoSources() throws IOException {
        Path legacyTree = MAIN.resolve("backend/prepare");
        List<String> sources = containsJavaSources(legacyTree) ? List.of(legacyTree.toString()) : List.of();
        assertTrue(sources.isEmpty(), () -> "legacy backend.prepare Java sources remain: " + sources);
    }

    @Test
    void tensorRuntimeImportsAreLimitedToLifecycleAndStorageContracts() throws IOException {
        Map<String, Set<String>> allowedByFile = Map.of(
                "src/main/java/tensor/Tensor.java", Set.of(
                        "runtime.contract.ExecutionMode",
                        "runtime.execution.PreparedExecution"
                ),
                "src/main/java/tensor/internal/TensorExecution.java", Set.of(
                        "runtime.contract.ExecutionMode",
                        "runtime.execution.PreparedExecution"
                ),
                "src/main/java/tensor/storage/NativeMemoryAllocation.java", Set.of(
                        "runtime.memory.ExecutionResource"
                )
        );
        List<String> offenders = importsUnder(MAIN.resolve("tensor"), importedType -> importedType.startsWith("runtime."))
                .stream()
                .filter(line -> allowedByFile.entrySet().stream().noneMatch(entry ->
                        line.startsWith(entry.getKey() + ":")
                                && entry.getValue().stream().anyMatch(line::endsWith)))
                .toList();
        assertTrue(offenders.isEmpty(), () -> "tensor runtime imports outside lifecycle boundary: " + offenders);
    }

    @Test
    void legacyRuntimeMemoryOwnershipTreesAreRemoved() {
        List<Path> legacyTrees = List.of(
                MAIN.resolve("backend/memory"),
                MAIN.resolve("backend/accelerator/buffer")
        );
        assertTrue(legacyTrees.stream().noneMatch(Files::exists),
                () -> "legacy runtime memory ownership trees remain: "
                        + legacyTrees.stream().filter(Files::exists).toList());
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

    private static boolean containsJavaSources(Path root) throws IOException {
        if (!Files.exists(root)) {
            return false;
        }
        try (Stream<Path> paths = Files.walk(root)) {
            return paths.anyMatch(path -> Files.isRegularFile(path) && path.toString().endsWith(".java"));
        }
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
