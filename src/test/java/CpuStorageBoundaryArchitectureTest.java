import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertTrue;

class CpuStorageBoundaryArchitectureTest {
    private static final String LEGACY_STORAGE_PACKAGE = "backend.cpu.kernels." + "storage";

    @Test
    void cpuStorageBoundaryLivesOutsideKernelPackage() throws IOException {
        Path storageRoot = Path.of("src/main/java/backend/cpu/storage");
        Path legacyStorageRoot = Path.of("src/main/java/backend/cpu/kernels", "storage");
        assertTrue(Files.isDirectory(storageRoot), "CPU storage descriptors must live under backend.cpu.storage.");
        assertTrue(!Files.exists(legacyStorageRoot),
                LEGACY_STORAGE_PACKAGE + " must not remain as a compatibility package.");

        List<String> requiredFiles = List.of(
                "CpuStorageBindings.java",
                "CpuStorageKind.java",
                "CpuStorageResolver.java",
                "CpuStorageView.java"
        );
        List<String> missing = requiredFiles.stream()
                .filter(name -> !Files.exists(storageRoot.resolve(name)))
                .toList();
        assertTrue(missing.isEmpty(), () -> "Missing CPU storage boundary files: " + missing);
    }

    @Test
    void sourceDoesNotImportLegacyKernelStoragePackage() throws IOException {
        List<Path> roots = List.of(Path.of("src/main/java"), Path.of("src/test/java"));
        String legacyImport = "import " + LEGACY_STORAGE_PACKAGE;
        String legacyPackageDeclaration = "package " + LEGACY_STORAGE_PACKAGE;
        try (Stream<Path> paths = roots.stream().flatMap(root -> {
            try {
                return Files.walk(root);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        })) {
            List<String> offenders = paths
                    .filter(Files::isRegularFile)
                    .filter(path -> path.toString().endsWith(".java"))
                    .flatMap(path -> {
                        try {
                            return Files.readAllLines(path).stream()
                                    .filter(line -> line.contains(legacyImport)
                                            || line.contains(legacyPackageDeclaration))
                                    .map(line -> path + ": " + line.trim());
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                    })
                    .sorted()
                    .toList();
            assertTrue(offenders.isEmpty(), () -> "Legacy " + LEGACY_STORAGE_PACKAGE + " imports remain: " + offenders);
        }
    }
}
