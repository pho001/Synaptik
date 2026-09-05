package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuLossIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScatterLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutable;
import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuWorkerGroup;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBorrowedBuffer;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.loss.LossKind;
import io.github.pho001.synaptik.model.operation.loss.LossReduction;
import io.github.pho001.synaptik.model.operation.loss.MeanSquaredErrorAttrs;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import io.github.pho001.synaptik.runtime.run.BufferRepresentationBinding;
import io.github.pho001.synaptik.runtime.run.RunResourceOwnership;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.lang.classfile.ClassFile;
import java.lang.classfile.Instruction;
import java.lang.classfile.Opcode;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.lang.classfile.instruction.InvokeInstruction;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.concurrent.atomic.AtomicReference;
import java.lang.reflect.AccessFlag;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Assumptions;

/** Focused schema-62 selection and Class-File evidence for direct vector MSE {@code NONE}. */
class CpuVectorMseEvidenceTest {
    private static final Path TASK_START_SCALAR_BEFORE = Path.of(
            "/private/tmp/synaptik-cpu-0008m-3WYHkRA5/scalar-before");
    private static final String TASK_START_SCALAR_INVENTORY_SHA256 =
            "ce863c65922a5de76527eeaf48f7d1d969914765c3450508859846d39354fe2b";
    private static final String TASK_START_SCALAR_TREE_SHA256 =
            "04d49430747cd1991005363c8b4b973f2a42b8a16074a0abed7bc002889cee99";
    private static final String STRUCTURAL_MANIFEST = "vector-mse-structural-manifest.sha256";
    private static final String STRUCTURAL_COMMANDS = "vector-mse-structural-commands.txt";
    private static final String SOURCE_RELATIVE =
            "src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/"
                    + "CpuVectorMseEvidenceTest.java";
    private static final ValueLayout.OfFloat FLOAT = ValueLayout.JAVA_FLOAT_UNALIGNED
            .withOrder(ByteOrder.nativeOrder());
    private static final ValueLayout.OfDouble DOUBLE = ValueLayout.JAVA_DOUBLE_UNALIGNED
            .withOrder(ByteOrder.nativeOrder());

    @Test void executesAllFortyEightIdentityAndStrategyScenarios() throws Throwable {
        int cases = 0;
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            for (int pattern = 0; pattern < 8; pattern++) {
                execute(type, List.of(0, 1), carriers(type, 3, pattern), false);
                execute(type, List.of(0, 1), carriers(type, 3, pattern), true);
                cases += 2;
            }
            for (int pattern = 0; pattern < 4; pattern++) {
                execute(type, List.of(0, 0), carriers(type, 2, pattern), false);
                execute(type, List.of(0, 0), carriers(type, 2, pattern), true);
                cases += 2;
            }
        }
        assertEquals(48, cases);
    }

    @Test void executesTwelveSelectionAndFallbackCases() {
        int cases = 0;
        for (DataType type : floatingTypes()) {
            int lanes = lanes(type);
            for (LossReduction reduction : List.of(LossReduction.SUM, LossReduction.MEAN)) {
                assertScalar(route(type, List.of(0, 1), carriers(type, 3, 0), 1, reduction));
                cases++;
            }
            Shape shape = Shape.of(lanes * 2);
            TensorDescriptor dense = desc(type, shape);
            TensorDescriptor strided = new TensorDescriptor(type, shape,
                    java.util.Optional.of(LayoutDescriptor.of(shape, new long[] {2}, 0, true)), false);
            assertScalar(route(List.of(strided, dense), dense, List.of(0, 1),
                    carriers(type, 3, 0), 1)); cases++;
            assertScalar(route(List.of(dense, strided), dense, List.of(0, 1),
                    carriers(type, 3, 0), 1)); cases++;
            assertScalar(route(List.of(dense, dense), strided, List.of(0, 1),
                    carriers(type, 3, 0), 1)); cases++;
            assertScalar(route(type, Shape.of(lanes - 1), List.of(0, 1),
                    carriers(type, 3, 0), 1)); cases++;
        }
        assertEquals(12, cases);
    }

    @Test void executesTenShapeAndRangeCases() throws Throwable {
        int cases = 0;
        for (DataType type : floatingTypes()) {
            assertScalar(route(type, Shape.of(2, 0, 3), List.of(0, 1),
                    carriers(type, 3, 0), 1)); cases++;
            var scalar = route(type, Shape.scalar(), List.of(0, 1), carriers(type, 3, 0), 1);
            assertScalar(scalar); executeRange(scalar, type, 1, 0, 1); cases++;
            var vector = route(type, Shape.of(lanes(type) * 3 + 5), List.of(0, 1),
                    carriers(type, 3, 0), 1);
            executeRange(vector, type, lanes(type) * 3 + 5, 2, 2); cases++;
            executeRange(vector, type, lanes(type) * 3 + 5, 3, 4); cases++;
            executeRange(vector, type, lanes(type) * 3 + 5, lanes(type) + 1,
                    lanes(type) * 3 + 4); cases++;
        }
        assertEquals(10, cases);
    }

    @Test void executesFourExceptionalValueCases() throws Throwable {
        int cases = 0;
        for (DataType type : floatingTypes()) for (boolean segment : List.of(false, true)) {
            executeExceptional(type, segment); cases++;
        }
        assertEquals(4, cases);
    }

    @Test void executesTwelveLegalAliasCases() throws Throwable {
        int cases = 0;
        for (DataType type : floatingTypes()) {
            executeLegalAlias(type, AliasCase.SHARED_ARRAY); cases++;
            executeLegalAlias(type, AliasCase.SHARED_SEGMENT); cases++;
            executeLegalAlias(type, AliasCase.OVERLAPPING_READ_ARRAYS); cases++;
            executeLegalAlias(type, AliasCase.OVERLAPPING_READ_SEGMENTS); cases++;
            executeLegalAlias(type, AliasCase.DISJOINT_OUTPUT_ARRAY); cases++;
            executeLegalAlias(type, AliasCase.DISJOINT_OUTPUT_SEGMENT); cases++;
        }
        assertEquals(12, cases);
    }

    @Test void executesSixteenPreWriteRejections() throws Throwable {
        int cases = 0;
        for (DataType type : floatingTypes()) {
            for (RejectionCase scenario : RejectionCase.values()) {
                executeRejection(type, scenario); cases++;
            }
        }
        assertEquals(16, cases);
    }

    @Test void executesFourConcurrentRunIsolationCases() throws Throwable {
        int cases = 0;
        for (DataType type : floatingTypes()) for (boolean parallel : List.of(false, true)) {
            executeConcurrent(type, parallel); cases++;
        }
        assertEquals(4, cases);
        assertEquals(48 + 12 + 10 + 4 + 12 + 16 + 4, 106);
    }

    /** Writes the retained schema-62 dossiers and compares pre/post scalar inventories on demand. */
    @Test void retainsStructuralDossiersAndScalarInventoryEqualityWhenMarked() throws Exception {
        String configured = System.getProperty("synaptik.cpu.vectorMse.structuralEvidenceRoot");
        Assumptions.assumeTrue(configured != null && Files.exists(Path.of(configured)
                .resolve("RUN-VECTOR-MSE-STRUCTURAL-EVIDENCE")));
        Path root = Path.of(configured).toAbsolutePath().normalize();
        assertTrue(root.getNameCount() > 2, "structural evidence root must be narrow");
        cleanStructuralEvidence(root);
        copyTaskStartScalarBaseline(root.resolve("scalar-before"));
        Path classes = root.resolve("vector-mse-classes");
        Files.createDirectories(classes);
        var lines = new ArrayList<String>();
        lines.add("key,type,roles,carriers,descriptor,sha256,normalized,witnesses");
        Set<String> keys = new HashSet<>();
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) for (int shared = 0;
                shared < 2; shared++) for (int pattern = 0; pattern < (shared == 0 ? 8 : 4); pattern++) {
            List<Integer> roles = shared == 0 ? List.of(0, 1) : List.of(0, 0);
            var route = route(type, roles, carriers(type, shared == 0 ? 3 : 2, pattern), 1);
            byte[] bytes = new CpuClassFileKernelGenerator().generateClassBytes(route.specialization(), route.kernelIr());
            byte[] second = new CpuClassFileKernelGenerator().generateClassBytes(
                    route.specialization(), route.kernelIr());
            assertArrayEquals(bytes, second, "deterministic second emission " + key(route));
            String key = route.specialization().structuralKey();
            assertTrue(keys.add(key), "duplicate dossier " + key);
            Files.write(classes.resolve(key + ".class"), bytes);
            StructuralWitness witness = inspectVectorClass(bytes, route, type);
            lines.add(csvRecord(key, type.toString(), roles.toString(),
                    route.specialization().carrierPattern().toString(), witness.descriptor(),
                    hash(bytes), witness.normalizedHash(), witness.text()));
        }
        assertEquals(24, keys.size());
        Path dossiers = root.resolve("vector-mse-dossiers.csv");
        Files.write(dossiers, lines, StandardCharsets.UTF_8);
        validateCsv(dossiers,
                "key,type,roles,carriers,descriptor,sha256,normalized,witnesses", 8, 24);
        Path javap = root.resolve("vector-mse-javap.txt");
        List<Path> classFiles;
        try (var listed = Files.list(classes)) { classFiles = listed.sorted().toList(); }
        assertEquals(24, classFiles.size());
        for (Path classFile : classFiles) {
            Process process = new ProcessBuilder("javap", "-c", "-v", "-p", classFile.toString())
                    .redirectErrorStream(true).redirectOutput(ProcessBuilder.Redirect.appendTo(javap.toFile())).start();
            assertEquals(0, process.waitFor(), "javap " + classFile);
        }
        Path beforeRoot = root.resolve("scalar-before");
        Path afterRoot = root.resolve("scalar-after");
        verifyTaskStartScalarBaseline(beforeRoot);
        Files.createDirectories(afterRoot);
        Files.writeString(afterRoot.resolve("RUN-STRUCTURAL-EVIDENCE"), "");
        String previous = System.getProperty("synaptik.cpu.loss.structuralEvidenceRoot");
        try {
            System.setProperty("synaptik.cpu.loss.structuralEvidenceRoot", afterRoot.toString());
            new CpuLossEvidenceTest().emitsExactSevenHundredNinetyTwoInventory();
        } finally {
            if (previous == null) System.clearProperty("synaptik.cpu.loss.structuralEvidenceRoot");
            else System.setProperty("synaptik.cpu.loss.structuralEvidenceRoot", previous);
        }
        Path before = beforeRoot.resolve("inventory.csv"), after = afterRoot.resolve("inventory.csv");
        Map<String, String> beforeInventory = inventory(before), afterInventory = inventory(after);
        assertEquals(beforeInventory.keySet(), afterInventory.keySet(), "exact scalar keys");
        assertEquals(beforeInventory, afterInventory, "all 792 scalar schema-58 hashes must match");
        writeScalarEqualityEvidence(root, before, after, beforeInventory, afterInventory);
        Files.writeString(root.resolve("scalar-before-provenance.txt"),
                "source=" + TASK_START_SCALAR_BEFORE + "\n"
                        + "files=797\n"
                        + "tree_sha256=" + TASK_START_SCALAR_TREE_SHA256 + "\n"
                        + "inventory_sha256=" + TASK_START_SCALAR_INVENTORY_SHA256 + "\n",
                StandardCharsets.UTF_8);
        Files.writeString(root.resolve("vector-mse-structural-marker.txt"),
                "marker=RUN-VECTOR-MSE-STRUCTURAL-EVIDENCE\nstate=present-and-consumed\n",
                StandardCharsets.UTF_8);
        Path source = locateEvidenceSource();
        Path sourceSnapshot = root.resolve("vector-mse-structural-sources")
                .resolve("CpuVectorMseEvidenceTest.java");
        Files.createDirectories(sourceSnapshot.getParent());
        Files.copy(source, sourceSnapshot, StandardCopyOption.REPLACE_EXISTING);
        writeCommandRecord(root);
        Set<String> evidenceFiles = structuralEvidenceFiles(root);
        writeManifest(root, evidenceFiles);
        verifyManifest(root, evidenceFiles);
        Files.delete(root.resolve("RUN-VECTOR-MSE-STRUCTURAL-EVIDENCE"));
    }

    @Test void structuralEvidenceValidatorsRejectMalformedOrIncompleteRecords() throws Exception {
        Path root = Files.createTempDirectory("vector-mse-evidence-validation-");
        try {
            Path csv = root.resolve("dossiers.csv");
            Files.writeString(csv, "a,b\n\"unterminated,b\n", StandardCharsets.UTF_8);
            assertThrows(AssertionError.class, () -> validateCsv(csv, "a,b", 2, 1));

            Files.writeString(root.resolve(STRUCTURAL_COMMANDS), "commands\n", StandardCharsets.UTF_8);
            Files.writeString(root.resolve("payload.txt"), "payload\n", StandardCharsets.UTF_8);
            Set<String> expected = Set.of(STRUCTURAL_COMMANDS, "payload.txt");
            Path manifest = root.resolve(STRUCTURAL_MANIFEST);
            String commandLine = manifestLine(root, STRUCTURAL_COMMANDS);
            String payloadLine = manifestLine(root, "payload.txt");
            Files.writeString(manifest, commandLine, StandardCharsets.UTF_8);
            assertThrows(AssertionError.class, () -> verifyManifest(root, expected));
            Files.writeString(manifest, commandLine + payloadLine + payloadLine,
                    StandardCharsets.UTF_8);
            assertThrows(AssertionError.class, () -> verifyManifest(root, expected));
            Files.writeString(manifest, commandLine + "0".repeat(64) + "  payload.txt\n",
                    StandardCharsets.UTF_8);
            assertThrows(AssertionError.class, () -> verifyManifest(root, expected));
            Files.delete(root.resolve(STRUCTURAL_COMMANDS));
            Files.writeString(manifest, payloadLine, StandardCharsets.UTF_8);
            assertThrows(AssertionError.class, () -> verifyManifest(root, Set.of("payload.txt")));

            Path baseline = root.resolve("inventory.csv");
            Files.writeString(baseline, "not the task-start baseline\n", StandardCharsets.UTF_8);
            assertThrows(AssertionError.class, () -> verifyTaskStartScalarInventory(baseline));
        } finally {
            deleteTree(root);
        }
    }

    private static void cleanStructuralEvidence(Path root) throws Exception {
        for (String directory : List.of("vector-mse-classes", "scalar-before", "scalar-after",
                "vector-mse-structural-sources"))
            deleteTree(root.resolve(directory));
        for (String file : List.of("vector-mse-dossiers.csv", "vector-mse-javap.txt",
                "scalar-792-equality.txt", "scalar-792-key-map.csv",
                "scalar-before-provenance.txt", "vector-mse-structural-marker.txt",
                STRUCTURAL_COMMANDS, STRUCTURAL_MANIFEST))
            Files.deleteIfExists(root.resolve(file));
    }

    private static void copyTaskStartScalarBaseline(Path destination) throws Exception {
        verifyTaskStartScalarBaseline(TASK_START_SCALAR_BEFORE);
        try (var paths = Files.walk(TASK_START_SCALAR_BEFORE)) {
            for (Path source : paths.sorted().toList()) {
                Path target = destination.resolve(TASK_START_SCALAR_BEFORE.relativize(source));
                if (Files.isDirectory(source)) Files.createDirectories(target);
                else Files.copy(source, target, StandardCopyOption.REPLACE_EXISTING);
            }
        }
        verifyTaskStartScalarBaseline(destination);
    }

    private static void verifyTaskStartScalarBaseline(Path baseline) throws Exception {
        assertTrue(Files.isDirectory(baseline), "task-start scalar-before directory is mandatory");
        verifyTaskStartScalarInventory(baseline.resolve("inventory.csv"));
        List<Path> files;
        try (var paths = Files.walk(baseline)) {
            files = paths.filter(Files::isRegularFile).sorted().toList();
        }
        assertEquals(797, files.size(), "complete task-start scalar-before file count");
        assertEquals(TASK_START_SCALAR_TREE_SHA256, treeHash(baseline, files),
                "immutable task-start scalar-before tree");
        inventory(baseline.resolve("inventory.csv"));
    }

    private static void verifyTaskStartScalarInventory(Path inventory) throws Exception {
        assertTrue(Files.isRegularFile(inventory), "task-start scalar-before inventory is mandatory");
        assertEquals(TASK_START_SCALAR_INVENTORY_SHA256, hash(Files.readAllBytes(inventory)),
                "immutable task-start scalar-before inventory");
    }

    private static String treeHash(Path root, List<Path> files) throws Exception {
        var lines = new StringBuilder();
        for (Path file : files) lines.append(hash(Files.readAllBytes(file))).append("  ")
                .append(root.relativize(file).toString().replace('\\', '/')).append('\n');
        return hash(lines.toString().getBytes(StandardCharsets.UTF_8));
    }

    private static void writeScalarEqualityEvidence(Path root, Path before, Path after,
            Map<String, String> beforeInventory, Map<String, String> afterInventory) throws Exception {
        String keyHash = hash(String.join("\n", beforeInventory.keySet())
                .getBytes(StandardCharsets.UTF_8));
        Files.writeString(root.resolve("scalar-792-equality.txt"),
                "equal=true\nentries=792\n"
                        + "before_inventory_sha256=" + hash(Files.readAllBytes(before)) + "\n"
                        + "after_inventory_sha256=" + hash(Files.readAllBytes(after)) + "\n"
                        + "key_sha256=" + keyHash + "\n", StandardCharsets.UTF_8);
        var keyMap = new ArrayList<String>();
        keyMap.add("key,before_sha256,after_sha256");
        for (String key : beforeInventory.keySet()) keyMap.add(csvRecord(key,
                beforeInventory.get(key), afterInventory.get(key)));
        Path keyMapFile = root.resolve("scalar-792-key-map.csv");
        Files.write(keyMapFile, keyMap, StandardCharsets.UTF_8);
        validateCsv(keyMapFile, "key,before_sha256,after_sha256", 3, 792);
    }

    private static Path locateEvidenceSource() {
        Path current = Path.of(System.getProperty("user.dir")).toAbsolutePath().normalize();
        while (current != null) {
            for (Path candidate : List.of(current.resolve(SOURCE_RELATIVE),
                    current.resolve("backends/cpu").resolve(SOURCE_RELATIVE)))
                if (Files.isRegularFile(candidate)) return candidate;
            current = current.getParent();
        }
        throw new AssertionError("cannot locate CpuVectorMseEvidenceTest.java source");
    }

    private static void writeCommandRecord(Path root) throws Exception {
        Map<String, String> commands = expectedCommands(root);
        var lines = new ArrayList<String>();
        commands.forEach((name, command) -> lines.add(name + '=' + command));
        Files.write(root.resolve(STRUCTURAL_COMMANDS), lines, StandardCharsets.UTF_8);
        verifyCommandRecord(root, commands);
    }

    private static Map<String, String> expectedCommands(Path root) {
        String base = "io.github.pho001.synaptik.backend.cpu.internal.";
        var commands = new LinkedHashMap<String, String>();
        commands.put("normal-evidence", "./gradlew :backends:cpu:test --tests " + base
                + "codegen.emit.CpuVectorMseEvidenceTest --rerun-tasks");
        commands.put("task-start-scalar-before", "./gradlew :backends:cpu:test --tests " + base
                + "codegen.emit.CpuLossEvidenceTest -Dsynaptik.cpu.loss.structuralEvidenceRoot=\""
                + TASK_START_SCALAR_BEFORE + "\" --rerun-tasks");
        commands.put("structural-marker", "touch \"" + root
                + "/RUN-VECTOR-MSE-STRUCTURAL-EVIDENCE\"");
        commands.put("structural-evidence", "./gradlew :backends:cpu:test --tests " + base
                + "codegen.emit.CpuVectorMseEvidenceTest -Dsynaptik.cpu.vectorMse.structuralEvidenceRoot=\""
                + root + "\" --rerun-tasks");
        commands.put("task-focused", focusedCommand(base));
        commands.put("full-cpu", "./gradlew :backends:cpu:test --rerun-tasks");
        return commands;
    }

    private static String focusedCommand(String base) {
        return "./gradlew :backends:cpu:test " + String.join(" ", List.of(
                "--tests " + base + "cache.CpuGeneratedKernelArtifactStoreTest",
                "--tests " + base + "cache.CpuKernelSpecializationTest",
                "--tests " + base + "codegen.emit.CpuBatchNormTrainingEvidenceTest",
                "--tests " + base + "codegen.emit.CpuClassFileKernelGeneratorTest",
                "--tests " + base + "codegen.emit.CpuConv2dEvidenceTest",
                "--tests " + base + "codegen.emit.CpuConv3dEvidenceTest",
                "--tests " + base + "codegen.emit.CpuLossGeneratedKernelTest",
                "--tests " + base + "codegen.emit.CpuPartitionDagGeneratedEvidenceTest",
                "--tests " + base + "codegen.emit.CpuPointwiseLedgerEvidenceTest",
                "--tests " + base + "codegen.emit.CpuPointwiseMaskEvidenceTest",
                "--tests " + base + "codegen.emit.CpuVectorMseEvidenceTest",
                "--tests " + base + "executable.CpuPreparedExecutableTest",
                "--tests " + base + "prepare.CpuPartitionFinalizerTest",
                "--tests " + base + "prepare.CpuPartitionPreparerTest"));
    }

    private static void verifyCommandRecord(Path root, Map<String, String> expected) throws Exception {
        Path record = root.resolve(STRUCTURAL_COMMANDS);
        assertTrue(Files.isRegularFile(record), "structural command record is mandatory");
        var actual = new LinkedHashMap<String, String>();
        for (String line : Files.readAllLines(record, StandardCharsets.UTF_8)) {
            int separator = line.indexOf('=');
            assertTrue(separator > 0, "command record label");
            String name = line.substring(0, separator);
            assertEquals(null, actual.put(name, line.substring(separator + 1)),
                    "duplicate command record " + name);
        }
        assertEquals(expected, actual, "exact structural/task validation commands");
    }

    private static Set<String> structuralEvidenceFiles(Path root) throws Exception {
        var files = new TreeSet<String>();
        for (String directory : List.of("vector-mse-classes", "scalar-before", "scalar-after",
                "vector-mse-structural-sources")) {
            Path owned = root.resolve(directory);
            try (var paths = Files.walk(owned)) {
                paths.filter(Files::isRegularFile).forEach(path -> files.add(
                        root.relativize(path).toString().replace('\\', '/')));
            }
        }
        files.addAll(List.of("vector-mse-dossiers.csv", "vector-mse-javap.txt",
                "scalar-792-equality.txt", "scalar-792-key-map.csv",
                "scalar-before-provenance.txt", "vector-mse-structural-marker.txt",
                STRUCTURAL_COMMANDS));
        assertEquals(1626, files.size(), "complete structural evidence file count");
        for (String file : files)
            assertTrue(Files.isRegularFile(root.resolve(file)), "missing structural evidence " + file);
        return files;
    }

    private static void writeManifest(Path root, Set<String> files) throws Exception {
        var lines = new ArrayList<String>();
        for (String file : files) lines.add(manifestLine(root, file).stripTrailing());
        Files.write(root.resolve(STRUCTURAL_MANIFEST), lines, StandardCharsets.UTF_8);
    }

    private static String manifestLine(Path root, String file) throws Exception {
        return hash(Files.readAllBytes(root.resolve(file))) + "  " + file + "\n";
    }

    private static void verifyManifest(Path root, Set<String> expectedFiles) throws Exception {
        Path manifest = root.resolve(STRUCTURAL_MANIFEST);
        assertTrue(Files.isRegularFile(manifest), "structural manifest is mandatory");
        var records = new TreeMap<String, String>();
        for (String line : Files.readAllLines(manifest, StandardCharsets.UTF_8)) {
            assertTrue(line.matches("[0-9a-f]{64}  [^/].*"), "valid manifest record: " + line);
            String file = line.substring(66);
            assertFalse(file.contains("..") || Path.of(file).isAbsolute(),
                    "manifest path stays below evidence root");
            assertEquals(null, records.put(file, line.substring(0, 64)),
                    "duplicate manifest record " + file);
        }
        assertTrue(records.containsKey(STRUCTURAL_COMMANDS),
                "structural command record must be manifested");
        assertTrue(Files.isRegularFile(root.resolve(STRUCTURAL_COMMANDS)),
                "structural command record is mandatory");
        assertEquals(expectedFiles, records.keySet(), "complete structural manifest records");
        for (Map.Entry<String, String> record : records.entrySet())
            assertEquals(record.getValue(), hash(Files.readAllBytes(root.resolve(record.getKey()))),
                    "structural manifest checksum " + record.getKey());
    }

    private static void deleteTree(Path root) throws Exception {
        if (!Files.exists(root)) return;
        try (var paths = Files.walk(root)) {
            for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
        }
    }

    private static Map<String, String> inventory(Path file) throws Exception {
        var values = new TreeMap<String, String>();
        List<String> lines = Files.readAllLines(file, StandardCharsets.UTF_8);
        for (String line : lines.subList(1, lines.size())) {
            List<String> fields = csvFields(line);
            assertEquals(10, fields.size(), "scalar inventory columns");
            assertEquals(null, values.put(fields.get(0), fields.get(7)),
                    "duplicate scalar key " + fields.get(0));
            Path classFile = file.getParent().resolve("classes").resolve(fields.get(0) + ".class");
            assertTrue(Files.isRegularFile(classFile), "missing scalar class " + fields.get(0));
            assertEquals(fields.get(7), hash(Files.readAllBytes(classFile)),
                    "scalar Class-File hash " + fields.get(0));
        }
        assertEquals(792, values.size());
        Set<String> classKeys;
        try (var listed = Files.list(file.getParent().resolve("classes"))) {
            classKeys = listed.filter(path -> path.getFileName().toString().endsWith(".class"))
                    .map(path -> path.getFileName().toString().replaceFirst("\\.class$", ""))
                    .collect(java.util.stream.Collectors.toSet());
        }
        assertEquals(values.keySet(), classKeys, "no missing or extra scalar Class-File");
        return values;
    }

    private static List<String> csvFields(String line) {
        var fields = new ArrayList<String>();
        var value = new StringBuilder();
        boolean quoted = false, closedQuote = false;
        for (int index = 0; index < line.length(); index++) {
            char current = line.charAt(index);
            if (current == '"') {
                if (quoted && index + 1 < line.length() && line.charAt(index + 1) == '"') {
                    value.append('"'); index++;
                } else if (quoted) {
                    quoted = false;
                    closedQuote = true;
                } else {
                    assertTrue(value.isEmpty() && !closedQuote,
                            "CSV quote must begin a field");
                    quoted = true;
                }
            } else if (current == ',' && !quoted) {
                fields.add(value.toString());
                value.setLength(0);
                closedQuote = false;
            } else {
                assertFalse(closedQuote, "only a comma may follow a closing CSV quote");
                value.append(current);
            }
        }
        assertFalse(quoted, "balanced CSV quotes");
        fields.add(value.toString());
        return fields;
    }

    private static String csvRecord(String... fields) {
        var encoded = new ArrayList<String>(fields.length);
        for (String field : fields) {
            if (field.indexOf(',') >= 0 || field.indexOf('"') >= 0
                    || field.indexOf('\n') >= 0 || field.indexOf('\r') >= 0)
                encoded.add('"' + field.replace("\"", "\"\"") + '"');
            else encoded.add(field);
        }
        return String.join(",", encoded);
    }

    private static void validateCsv(Path file, String expectedHeader, int columns, int rows)
            throws Exception {
        List<String> lines = Files.readAllLines(file, StandardCharsets.UTF_8);
        assertEquals(rows + 1, lines.size(), "CSV row count " + file.getFileName());
        assertEquals(expectedHeader, lines.getFirst(), "CSV header " + file.getFileName());
        assertEquals(columns, csvFields(lines.getFirst()).size(),
                "CSV header columns " + file.getFileName());
        for (int row = 1; row < lines.size(); row++)
            assertEquals(columns, csvFields(lines.get(row)).size(),
                    "CSV columns at row " + row + " in " + file.getFileName());
    }

    private static StructuralWitness inspectVectorClass(byte[] bytes,
            io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan route,
            DataType type) throws Exception {
        assertTrue(ClassFile.of().verify(bytes).isEmpty());
        var model = ClassFile.of().parse(bytes);
        assertTrue(model.flags().has(AccessFlag.FINAL));
        assertTrue(model.fields().isEmpty());
        assertTrue(model.interfaces().isEmpty());
        assertEquals(3, model.methods().size());
        var entry = model.methods().stream().filter(method -> method.flags().has(AccessFlag.PUBLIC))
                .findFirst().orElseThrow();
        assertEquals("invoke", entry.methodName().stringValue());
        assertTrue(entry.flags().has(AccessFlag.STATIC));
        assertEquals(route.specialization().entryType().descriptorString(),
                entry.methodTypeSymbol().descriptorString());
        assertFalse(entry.methodTypeSymbol().descriptorString().contains("Ljava/lang/Object;"));
        var body = model.methods().stream().filter(method -> method.methodName().stringValue()
                .equals(CpuLossEmitter.CONTIGUOUS_INT_NAME)).findFirst().orElseThrow();
        assertTrue(body.flags().has(AccessFlag.PRIVATE));
        assertTrue(body.flags().has(AccessFlag.STATIC));
        List<Instruction> instructions = body.code().orElseThrow().elementStream()
                .filter(Instruction.class::isInstance).map(Instruction.class::cast).toList();
        String vector = type == DataType.FLOAT32 ? "jdk/incubator/vector/FloatVector"
                : "jdk/incubator/vector/DoubleVector";
        int loads = 0, stores = 0, sub = 0, mul = 0, branches = 0;
        StringBuilder normalized = new StringBuilder();
        for (Instruction instruction : instructions) {
            Opcode opcode = instruction.opcode();
            assertFalse(opcode.name().startsWith("NEW"), "allocation " + opcode);
            assertNotEquals(Opcode.MONITORENTER, opcode); assertNotEquals(Opcode.MONITOREXIT, opcode);
            if (opcode.name().startsWith("IF") || opcode == Opcode.GOTO) branches++;
            if (instruction instanceof InvokeInstruction call) {
                String owner = call.owner().asInternalName(), name = call.name().stringValue();
                if (owner.equals(vector) && (name.equals("fromArray")
                        || name.equals("fromMemorySegment"))) loads++;
                if (owner.equals(vector) && (name.equals("intoArray")
                        || name.equals("intoMemorySegment"))) stores++;
                if (owner.equals(vector) && name.equals("sub")) sub++;
                if (owner.equals(vector) && name.equals("mul")) mul++;
                assertFalse(owner.startsWith("io/github/pho001/synaptik")
                        && !owner.equals(model.thisClass().asInternalName()), owner + '.' + name);
                assertFalse(owner.startsWith("java/util/") || owner.startsWith("java/lang/reflect")
                        || owner.startsWith("java/lang/invoke"), owner + '.' + name);
                assertFalse(Set.of("reduceLanes", "fma", "lane", "withLane", "toArray")
                        .contains(name), owner + '.' + name);
                normalized.append("CALL:").append(owner).append('.').append(name).append('\n');
            } else normalized.append(opcode.name()).append('\n');
        }
        String pool = java.util.stream.StreamSupport.stream(model.constantPool().spliterator(), false)
                .filter(MemberRefEntry.class::isInstance).map(MemberRefEntry.class::cast)
                .map(ref -> ref.owner().asInternalName() + '.' + ref.nameAndType().name().stringValue())
                .reduce("", (left, right) -> left + '\n' + right);
        for (String forbidden : List.of("java/util/", "java/lang/reflect", "java/lang/invoke",
                "CpuLossReferenceKernel", "CpuPortable", "CpuKernel", "Worker", "Cache"))
            assertFalse(pool.contains(forbidden), forbidden);
        assertEquals(2, loads, "one prediction and one target vector load");
        assertEquals(1, sub); assertEquals(1, mul); assertEquals(1, stores);
        assertTrue(branches >= 4, "vector and scalar tail loops");
        assertTrue(instructions.stream().anyMatch(i -> i.opcode() == (type == DataType.FLOAT32
                ? Opcode.FSUB : Opcode.DSUB)), "typed scalar-tail subtraction");
        assertTrue(instructions.stream().anyMatch(i -> i.opcode() == (type == DataType.FLOAT32
                ? Opcode.FMUL : Opcode.DMUL)), "typed scalar-tail multiplication");
        String text = "loads=" + loads + ";sub=" + sub + ";mul=" + mul + ";stores=" + stores
                + ";branches=" + branches + ";members=3;scalarTail=true";
        return new StructuralWitness(entry.methodTypeSymbol().descriptorString(),
                hash(normalized.toString().getBytes(StandardCharsets.UTF_8)), text);
    }

    @Test void retainsExactlyTwentyFourOrderedCarrierDossiers() throws Exception {
        Set<String> keys = new HashSet<>();
        int dossiers = 0;
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            for (int pattern = 0; pattern < 8; pattern++) {
                var route = route(type, List.of(0, 1), carriers(type, 3, pattern), 1);
                assertVectorRoute(route, type);
                keys.add(route.specialization().structuralKey());
                dossiers++;
            }
            for (int pattern = 0; pattern < 4; pattern++) {
                var route = route(type, List.of(0, 0), carriers(type, 2, pattern), 1);
                assertVectorRoute(route, type);
                keys.add(route.specialization().structuralKey());
                dossiers++;
            }
        }
        assertEquals(24, dossiers);
        assertEquals(24, keys.size());
    }

    @Test void vectorAndParallelVectorShareSchema62ArtifactAndScalarControlsStaySchema58() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            var vector = route(type, List.of(0, 1), carriers(type, 3, 0), 1);
            var parallel = route(type, List.of(0, 1), carriers(type, 3, 0), 2);
            assertEquals(vector.specialization(), parallel.specialization());
            var scalar = route(type, List.of(0, 1), carriers(type, 3, 0), 1,
                    LossReduction.SUM);
            assertEquals(58, scalar.specialization().classIdentitySchema());
        }
    }

    private static void assertVectorRoute(io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan route,
            DataType type) throws Exception {
        assertEquals(62, route.specialization().classIdentitySchema());
        assertEquals(type == DataType.FLOAT32 ? FloatVector.SPECIES_PREFERRED.vectorBitSize()
                : DoubleVector.SPECIES_PREFERRED.vectorBitSize(), route.specialization().vectorSpeciesBitSize());
        byte[] bytes = new CpuClassFileKernelGenerator().generateClassBytes(route.specialization(), route.kernelIr());
        var model = ClassFile.of().parse(bytes);
        List<String> owners = model.methods().stream().flatMap(method -> method.code().orElseThrow()
                .elementStream().filter(Instruction.class::isInstance).map(Instruction.class::cast)
                .filter(InvokeInstruction.class::isInstance).map(InvokeInstruction.class::cast)
                .map(call -> call.owner().asInternalName() + "." + call.name().stringValue())
        ).toList();
        String vector = type == DataType.FLOAT32 ? "jdk/incubator/vector/FloatVector" : "jdk/incubator/vector/DoubleVector";
        assertTrue(owners.stream().anyMatch(call -> call.equals(vector + ".fromArray") || call.equals(vector + ".fromMemorySegment")));
        assertTrue(owners.contains(vector + ".sub"));
        assertTrue(owners.contains(vector + ".mul"));
        assertFalse(owners.stream().anyMatch(call -> call.contains("reduceLanes") || call.contains("fma")));
    }

    private static void assertScalar(
            io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan route) {
        assertEquals(io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionStrategy.Compute.SCALAR,
                route.specialization().executionStrategy().compute());
        assertEquals(58, route.specialization().classIdentitySchema());
        assertEquals(0, route.specialization().vectorSpeciesBitSize());
    }

    private static void executeRange(
            io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan route,
            DataType type, int count, int start, int end) throws Throwable {
        int offset = 2, capacity = offset + count + lanes(type);
        Object prediction = array(type, capacity), target = array(type, capacity), output = array(type, capacity);
        fill(output, type, capacity, -91);
        for (int index = 0; index < count; index++) {
            write(prediction, type, offset + index, value(type, index, 0));
            write(target, type, offset + index, value(type, index, 1));
        }
        invoke(entry(route), List.of(prediction, target, output),
                ((CpuLossIr) route.portableKernelIr()).geometry().pack(
                        new long[] {offset, offset, offset}), start, end);
        for (int index = 0; index < count; index++) {
            double expected = index >= start && index < end
                    ? mse(type, value(type, index, 0), value(type, index, 1)) : -91;
            assertBits(type, expected, read(output, type, offset + index));
        }
        for (int index = 0; index < offset; index++) assertBits(type, -91, read(output, type, index));
        for (int index = offset + count; index < capacity; index++)
            assertBits(type, -91, read(output, type, index));
    }

    private static void executeExceptional(DataType type, boolean segment) throws Throwable {
        int count = lanes(type) * 2 + 3, offset = 3, capacity = offset + count + 4;
        CarrierAccess access = segment ? CarrierAccess.MEMORY_SEGMENT : arrayAccess(type);
        var route = route(type, Shape.of(count), List.of(0, 1), List.of(access, access, access), 1);
        try (Arena arena = Arena.ofConfined()) {
            Object prediction = carrier(type, capacity, segment, arena);
            Object target = carrier(type, capacity, segment, arena);
            Object output = carrier(type, capacity, segment, arena);
            fill(output, type, capacity, -91);
            double[] special = type == DataType.FLOAT32
                    ? new double[] {1.25f, 0f, -0f, Float.POSITIVE_INFINITY,
                            Float.NEGATIVE_INFINITY, Float.intBitsToFloat(0x7fc01234),
                            Float.MAX_VALUE, Float.MIN_NORMAL, Float.MIN_VALUE}
                    : new double[] {1.25, 0d, -0d, Double.POSITIVE_INFINITY,
                            Double.NEGATIVE_INFINITY, Double.longBitsToDouble(0x7ff8000000001234L),
                            Double.MAX_VALUE, Double.MIN_NORMAL, Double.MIN_VALUE};
            for (int index = 0; index < count; index++) {
                double p = special[index % special.length];
                double t = special[(index * 5 + 2) % special.length];
                write(prediction, type, offset + index, p); write(target, type, offset + index, t);
            }
            invoke(entry(route), List.of(prediction, target, output),
                    ((CpuLossIr) route.portableKernelIr()).geometry().pack(
                            new long[] {offset, offset, offset}), 0, count);
            for (int index = 0; index < count; index++) {
                double p = special[index % special.length];
                double t = special[(index * 5 + 2) % special.length];
                assertBits(type, mse(type, p, t), read(output, type, offset + index));
            }
        }
    }

    private static void executeLegalAlias(DataType type, AliasCase scenario) throws Throwable {
        int count = lanes(type) * 2 + 3, gap = 4, capacity = count * 2 + gap;
        boolean segment = scenario.segment;
        boolean shared = scenario.shared;
        List<Integer> roles = shared ? List.of(0, 0) : List.of(0, 1);
        List<CarrierAccess> accesses = java.util.Collections.nCopies(shared ? 2 : 3,
                segment ? CarrierAccess.MEMORY_SEGMENT : arrayAccess(type));
        var route = route(type, Shape.of(count), roles, accesses, 1);
        try (Arena arena = Arena.ofConfined()) {
            Object common = carrier(type, capacity, segment, arena);
            Object prediction = common, target = shared ? common
                    : scenario.overlappingReads ? common : carrier(type, capacity, segment, arena);
            Object output = scenario.disjointOutput ? common : carrier(type, capacity, segment, arena);
            int predictionBase = 0, targetBase = shared ? 0 : scenario.overlappingReads ? 1 : 0;
            int outputBase = scenario.disjointOutput ? count + gap : 0;
            fill(output, type, capacity, -91);
            for (int index = 0; index < count; index++)
                write(prediction, type, predictionBase + index, value(type, index, 0));
            if (!shared) for (int index = 0; index < count; index++)
                write(target, type, targetBase + index, value(type, index, 1));
            var boundaries = shared ? List.of(prediction, output) : List.of(prediction, target, output);
            long[] bases = new long[] {predictionBase, targetBase, outputBase};
            invoke(entry(route), boundaries, ((CpuLossIr) route.portableKernelIr()).geometry().pack(bases),
                    0, count);
            for (int index = 0; index < count; index++) {
                double p = read(prediction, type, predictionBase + index);
                double t = read(target, type, targetBase + index);
                // Disjoint output can overwrite neither input; shared-role output is exactly zero.
                assertBits(type, shared ? 0 : mse(type, p, t),
                        read(output, type, outputBase + index));
            }
        }
    }

    private static void executeRejection(DataType type, RejectionCase scenario) throws Throwable {
        int count = lanes(type) * 2 + 3;
        boolean segment = scenario.segment;
        List<CarrierAccess> accesses = java.util.Collections.nCopies(3,
                segment ? CarrierAccess.MEMORY_SEGMENT : arrayAccess(type));
        var workers = new CpuWorkerGroup(4);
        Arena arena = scenario == RejectionCase.INACCESSIBLE_SEGMENT
                ? Arena.ofConfined() : Arena.ofShared();
        try {
            CpuPreparedExecutable executable = prepared(type, Shape.of(count), List.of(0, 1),
                    accesses, 4, workers);
            int width = type.byteWidth(), capacity = count * 3 + 8;
            Object predictionBacking = carrier(type, capacity, segment, arena);
            Object targetBacking = carrier(type, capacity, segment, arena);
            Object outputBacking = carrier(type, capacity + 1, segment, arena);
            int predictionBase = 0, targetBase = 0, outputBase = 0, outputCount = count;
            if (scenario.predictionOverlap) outputBacking = predictionBacking;
            if (scenario.targetOverlap) outputBacking = targetBacking;
            if (scenario.predictionOverlap || scenario.targetOverlap) outputBase = 1;
            if (scenario == RejectionCase.INSUFFICIENT_SPAN) outputCount = count - 1;
            MemorySegment prediction = slice(predictionBacking, type, predictionBase, count);
            MemorySegment target = slice(targetBacking, type, targetBase, count);
            MemorySegment output;
            if (scenario == RejectionCase.MISALIGNED_SEGMENT_OFFSET) {
                MemorySegment raw = arena.allocate((long) count * width + 1, 1);
                output = raw.asSlice(1, (long) count * width);
            } else output = slice(outputBacking, type, outputBase, outputCount);
            fillSegment(output, type, -91);
            if (scenario == RejectionCase.READ_ONLY_OUTPUT) output = output.asReadOnly();
            MemorySegment boundOutput = output;
            long[] unchanged = rawBits(boundOutput, type);
            List<CpuBorrowedBuffer> resources = List.of(borrowed(type, count, prediction),
                    borrowed(type, count, target), borrowed(type, outputCount, boundOutput));
            RunState state = state(executable, resources);
            try {
                IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
                        () -> executable.bind(state), scenario.toString());
                String expected = scenario.predictionOverlap || scenario.targetOverlap
                        ? "output accessed span must not overlap an input"
                        : scenario == RejectionCase.INACCESSIBLE_SEGMENT
                                ? "segment is not accessible to every CPU worker"
                                : "bufferSelections[2] is incompatible with prepared executable";
                assertEquals(expected, failure.getMessage(), scenario.toString());
                assertArrayEquals(unchanged, rawBits(boundOutput, type),
                        scenario + " must reject before mutation/submission");
            } finally { state.close(); }
        } finally {
            arena.close(); workers.close();
        }
    }

    private static void executeConcurrent(DataType type, boolean parallel) throws Throwable {
        int count = lanes(type) * 5 + 3;
        CpuWorkerGroup workers = parallel ? new CpuWorkerGroup(4) : null;
        try {
            List<CarrierAccess> accesses = java.util.Collections.nCopies(3, arrayAccess(type));
            CpuPreparedExecutable executable = prepared(type, Shape.of(count), List.of(0, 1),
                    accesses, parallel ? 4 : 1, workers);
            Object p1 = array(type, count), t1 = array(type, count), o1 = array(type, count);
            Object p2 = array(type, count), t2 = array(type, count), o2 = array(type, count);
            for (int i = 0; i < count; i++) {
                write(p1, type, i, value(type, i, 0)); write(t1, type, i, value(type, i, 1));
                write(p2, type, i, value(type, i + 7, 0)); write(t2, type, i, value(type, i + 3, 1));
            }
            RunState first = state(executable, List.of(borrowed(type, count, slice(p1, type, 0, count)),
                    borrowed(type, count, slice(t1, type, 0, count)),
                    borrowed(type, count, slice(o1, type, 0, count))));
            RunState second = state(executable, List.of(borrowed(type, count, slice(p2, type, 0, count)),
                    borrowed(type, count, slice(t2, type, 0, count)),
                    borrowed(type, count, slice(o2, type, 0, count))));
            AtomicReference<Throwable> failure = new AtomicReference<>();
            try {
                Thread a = Thread.ofVirtual().start(() -> run(executable, first, failure));
                Thread b = Thread.ofVirtual().start(() -> run(executable, second, failure));
                a.join(); b.join();
                if (failure.get() != null) throw failure.get();
                for (int i = 0; i < count; i++) {
                    assertBits(type, mse(type, read(p1, type, i), read(t1, type, i)), read(o1, type, i));
                    assertBits(type, mse(type, read(p2, type, i), read(t2, type, i)), read(o2, type, i));
                }
            } finally { first.close(); second.close(); }
        } finally { if (workers != null) workers.close(); }
    }

    private static io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan route(
            DataType type, List<Integer> roles, List<CarrierAccess> carriers, int workers) {
        return route(type, roles, carriers, workers, LossReduction.NONE);
    }

    private static io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan route(
            DataType type, Shape shape, List<Integer> roles, List<CarrierAccess> carriers, int workers) {
        List<TensorDescriptor> inputs = roles.equals(List.of(0, 0))
                ? List.of(desc(type, shape)) : List.of(desc(type, shape), desc(type, shape));
        return route(inputs, desc(type, shape), roles, carriers, workers);
    }

    private static io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan route(
            List<TensorDescriptor> inputs, TensorDescriptor output, List<Integer> roles,
            List<CarrierAccess> carriers, int workers) {
        return analysis(inputs, output, roles, carriers, workers).plan().units().getFirst().portablePlan();
    }

    private static io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis<
            io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan> analysis(
            List<TensorDescriptor> inputs, TensorDescriptor output, List<Integer> roles,
            List<CarrierAccess> carriers, int workers) {
        PrepareContext<CpuPartitionAnalysisInputs> base = CpuScatterLoweringTest.context(
                new Operation(LossKind.MEAN_SQUARED_ERROR,
                        new MeanSquaredErrorAttrs(LossReduction.NONE)), roles, inputs, output);
        var config = new PortableExecutionConfig(ComputePreference.VECTOR_IF_ELIGIBLE,
                workers, workers, 1);
        return new CpuPartitionPreparer().analyze(new PrepareContext<>(base.partition(), base.nodes(),
                base.values(), base.memoryRequirements(), Map.of(),
                new CpuPartitionAnalysisInputs(false, carriers, config)));
    }

    private static CpuPreparedExecutable prepared(DataType type, Shape shape, List<Integer> roles,
            List<CarrierAccess> carriers, int workerCount, CpuWorkerGroup workers) {
        List<TensorDescriptor> inputs = roles.equals(List.of(0, 0))
                ? List.of(desc(type, shape)) : List.of(desc(type, shape), desc(type, shape));
        var analyzed = analysis(inputs, desc(type, shape), roles, carriers, workerCount);
        return CpuPartitionFinalizerTest.finalizeExecutable(analyzed, java.util.Optional.empty(),
                java.util.Optional.ofNullable(workers));
    }

    private static io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan route(
            DataType type, List<Integer> roles, List<CarrierAccess> carriers, int workers,
            LossReduction reduction) {
        int lanes = type == DataType.FLOAT32 ? FloatVector.SPECIES_PREFERRED.length()
                : DoubleVector.SPECIES_PREFERRED.length();
        Shape shape = Shape.of(lanes * 4);
        List<TensorDescriptor> inputs = roles.equals(List.of(0, 0))
                ? List.of(CpuScatterLoweringTest.desc(type, shape))
                : List.of(CpuScatterLoweringTest.desc(type, shape), CpuScatterLoweringTest.desc(type, shape));
        Shape output = reduction == LossReduction.NONE ? shape : Shape.scalar();
        PrepareContext<CpuPartitionAnalysisInputs> base = CpuScatterLoweringTest.context(
                new Operation(LossKind.MEAN_SQUARED_ERROR, new MeanSquaredErrorAttrs(reduction)), roles,
                inputs, CpuScatterLoweringTest.desc(type, output));
        var config = new PortableExecutionConfig(ComputePreference.VECTOR_IF_ELIGIBLE, workers, workers, 1);
        var plan = new CpuPartitionPreparer().analyze(new PrepareContext<>(base.partition(), base.nodes(),
                base.values(), base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false, carriers, config))).plan();
        return plan.units().getFirst().portablePlan();
    }

    private static TensorDescriptor desc(DataType type, Shape shape) {
        return CpuScatterLoweringTest.desc(type, shape);
    }

    private static List<DataType> floatingTypes() { return List.of(DataType.FLOAT32, DataType.FLOAT64); }
    private static int lanes(DataType type) { return type == DataType.FLOAT32
            ? FloatVector.SPECIES_PREFERRED.length() : DoubleVector.SPECIES_PREFERRED.length(); }
    private static CarrierAccess arrayAccess(DataType type) { return type == DataType.FLOAT32
            ? CarrierAccess.FLOAT_ARRAY : CarrierAccess.DOUBLE_ARRAY; }
    private static Object array(DataType type, int count) { return type == DataType.FLOAT32
            ? new float[count] : new double[count]; }
    private static Object carrier(DataType type, int count, boolean segment, Arena arena) {
        return segment ? arena.allocate((long) count * type.byteWidth(), type.byteWidth())
                : array(type, count);
    }
    private static MethodHandle entry(
            io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan route) {
        byte[] bytes = new CpuClassFileKernelGenerator().generateClassBytes(
                route.specialization(), route.kernelIr());
        return new CpuClassFileKernelGenerator().defineClassBytes(route.specialization(), bytes).entryPoint();
    }
    private static String key(
            io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan route) {
        return route.specialization().structuralKey();
    }
    private static double mse(DataType type, double prediction, double target) {
        if (type == DataType.FLOAT32) {
            float difference = (float) prediction - (float) target;
            return difference * difference;
        }
        double difference = prediction - target;
        return difference * difference;
    }
    private static MemorySegment slice(Object carrier, DataType type, int offset, int count) {
        MemorySegment segment = carrier instanceof MemorySegment value ? value
                : carrier instanceof float[] values ? MemorySegment.ofArray(values)
                : MemorySegment.ofArray((double[]) carrier);
        return segment.asSlice((long) offset * type.byteWidth(), (long) count * type.byteWidth());
    }
    private static CpuBorrowedBuffer borrowed(DataType type, int count, MemorySegment segment) {
        return CpuBorrowedBuffer.borrow(new MemorySegmentStorage(type, count, segment));
    }
    private static RunState state(CpuPreparedExecutable executable,
            List<? extends io.github.pho001.synaptik.runtime.resource.BufferRepresentation> resources) {
        var bindings = resources.stream().map(resource -> List.of(new BufferRepresentationBinding(
                resource, RunResourceOwnership.BORROWED))).toList();
        return new RunState(executable.memoryPlan(), bindings, List.of());
    }
    private static void run(CpuPreparedExecutable executable, RunState state,
            AtomicReference<Throwable> failure) {
        try { executable.bind(state).execute(); }
        catch (Throwable thrown) { failure.compareAndSet(null, thrown); }
    }
    private static void fillSegment(MemorySegment segment, DataType type, double value) {
        for (int index = 0; index < segment.byteSize() / type.byteWidth(); index++) {
            if (type == DataType.FLOAT32) segment.setAtIndex(FLOAT, index, (float) value);
            else segment.setAtIndex(DOUBLE, index, value);
        }
    }
    private static long[] rawBits(MemorySegment segment, DataType type) {
        int count = Math.toIntExact(segment.byteSize() / type.byteWidth());
        long[] bits = new long[count];
        for (int index = 0; index < count; index++) bits[index] = type == DataType.FLOAT32
                ? Float.floatToRawIntBits(segment.getAtIndex(FLOAT, index))
                : Double.doubleToRawLongBits(segment.getAtIndex(DOUBLE, index));
        return bits;
    }
    private static String hash(byte[] bytes) throws Exception { return HexFormat.of().formatHex(
            MessageDigest.getInstance("SHA-256").digest(bytes)); }

    private enum AliasCase {
        SHARED_ARRAY(false, true, false, false), SHARED_SEGMENT(true, true, false, false),
        OVERLAPPING_READ_ARRAYS(false, false, true, false),
        OVERLAPPING_READ_SEGMENTS(true, false, true, false),
        DISJOINT_OUTPUT_ARRAY(false, false, false, true),
        DISJOINT_OUTPUT_SEGMENT(true, false, false, true);
        final boolean segment, shared, overlappingReads, disjointOutput;
        AliasCase(boolean segment, boolean shared, boolean overlappingReads, boolean disjointOutput) {
            this.segment = segment; this.shared = shared; this.overlappingReads = overlappingReads;
            this.disjointOutput = disjointOutput;
        }
    }

    private enum RejectionCase {
        OUTPUT_PREDICTION_ARRAY(false, true, false), OUTPUT_TARGET_ARRAY(false, false, true),
        OUTPUT_PREDICTION_SEGMENT(true, true, false), OUTPUT_TARGET_SEGMENT(true, false, true),
        INSUFFICIENT_SPAN(false, false, false), MISALIGNED_SEGMENT_OFFSET(true, false, false),
        READ_ONLY_OUTPUT(false, false, false), INACCESSIBLE_SEGMENT(true, false, false);
        final boolean segment, predictionOverlap, targetOverlap;
        RejectionCase(boolean segment, boolean predictionOverlap, boolean targetOverlap) {
            this.segment = segment; this.predictionOverlap = predictionOverlap;
            this.targetOverlap = targetOverlap;
        }
    }

    private record StructuralWitness(String descriptor, String normalizedHash, String text) { }

    private static List<CarrierAccess> carriers(DataType type, int count, int bits) {
        CarrierAccess array = type == DataType.FLOAT32 ? CarrierAccess.FLOAT_ARRAY : CarrierAccess.DOUBLE_ARRAY;
        var result = new ArrayList<CarrierAccess>(count);
        for (int index = 0; index < count; index++)
            result.add((bits & (1 << index)) == 0 ? array : CarrierAccess.MEMORY_SEGMENT);
        return result;
    }

    private static void execute(DataType type, List<Integer> roles,
            List<CarrierAccess> accesses, boolean parallel) throws Throwable {
        int lanes = type == DataType.FLOAT32 ? FloatVector.SPECIES_PREFERRED.length()
                : DoubleVector.SPECIES_PREFERRED.length();
        int count = parallel ? 9 * lanes + 3 : 4 * lanes;
        int[] offsets = roles.equals(List.of(0, 0)) ? new int[] {3, 7}
                : new int[] {3, 5, 7};
        var route = route(type, Shape.of(count), roles, accesses, parallel ? 4 : 1);
        byte[] bytes = new CpuClassFileKernelGenerator().generateClassBytes(
                route.specialization(), route.kernelIr());
        MethodHandle entry = new CpuClassFileKernelGenerator().defineClassBytes(
                route.specialization(), bytes).entryPoint();
        int capacity = 7 + count + lanes;
        try (Arena arena = Arena.ofConfined()) {
            List<Object> boundaries = new ArrayList<>();
            for (int boundary = 0; boundary < accesses.size(); boundary++) {
                Object carrier = accesses.get(boundary) == CarrierAccess.MEMORY_SEGMENT
                        ? arena.allocate((long) capacity * type.byteWidth(), type.byteWidth())
                        : type == DataType.FLOAT32 ? new float[capacity] : new double[capacity];
                fill(carrier, type, capacity, boundary == accesses.size() - 1 ? -91.0 : 0.0);
                boundaries.add(carrier);
            }
            boolean shared = roles.equals(List.of(0, 0));
            Object prediction = boundaries.get(roles.getFirst());
            Object target = boundaries.get(roles.getLast());
            for (int i = 0; i < count; i++) {
                write(prediction, type, offsets[0] + i, value(type, i, 0));
                if (!shared) write(target, type, offsets[1] + i, value(type, i, 1));
            }
            long[] bases = shared ? new long[] {offsets[0], offsets[0], offsets[1]}
                    : new long[] {offsets[0], offsets[1], offsets[2]};
            long[] geometry = ((CpuLossIr) route.portableKernelIr()).geometry().pack(bases);
            if (parallel) {
                int quotient = count / 4, remainder = count % 4, start = 0;
                for (int worker = 0; worker < 4; worker++) {
                    int end = start + quotient + (worker < remainder ? 1 : 0);
                    invoke(entry, boundaries, geometry, start, end);
                    start = end;
                }
            } else invoke(entry, boundaries, geometry, 0, count);
            Object output = boundaries.getLast();
            int outputOffset = shared ? offsets[1] : offsets[2];
            for (int i = 0; i < count; i++) {
                double expected = shared ? 0.0 : square(type,
                        value(type, i, 0) - value(type, i, 1));
                assertBits(type, expected, read(output, type, outputOffset + i));
            }
            for (int i = 0; i < outputOffset; i++)
                assertBits(type, -91.0, read(output, type, i));
            for (int i = outputOffset + count; i < capacity; i++)
                assertBits(type, -91.0, read(output, type, i));
        }
    }

    private static void invoke(MethodHandle entry, List<Object> boundaries, long[] geometry,
            long start, long end) throws Throwable {
        var arguments = new ArrayList<>(boundaries);
        arguments.add(geometry);
        arguments.add(start);
        arguments.add(end);
        entry.invokeWithArguments(arguments);
    }

    private static double value(DataType type, int index, int role) {
        double value = ((index * 17 + role * 11) % 29 - 14) * 0.25;
        return type == DataType.FLOAT32 ? (float) value : value;
    }

    private static double square(DataType type, double difference) {
        if (type == DataType.FLOAT32) {
            float value = (float) difference;
            return value * value;
        }
        return difference * difference;
    }

    private static void fill(Object carrier, DataType type, int count, double value) {
        for (int index = 0; index < count; index++) write(carrier, type, index, value);
    }

    private static void write(Object carrier, DataType type, int index, double value) {
        if (carrier instanceof float[] array) array[index] = (float) value;
        else if (carrier instanceof double[] array) array[index] = value;
        else if (type == DataType.FLOAT32) ((MemorySegment) carrier).setAtIndex(FLOAT, index,
                (float) value);
        else ((MemorySegment) carrier).setAtIndex(DOUBLE, index, value);
    }

    private static double read(Object carrier, DataType type, int index) {
        if (carrier instanceof float[] array) return array[index];
        if (carrier instanceof double[] array) return array[index];
        if (type == DataType.FLOAT32) return ((MemorySegment) carrier).getAtIndex(FLOAT, index);
        return ((MemorySegment) carrier).getAtIndex(DOUBLE, index);
    }

    private static void assertBits(DataType type, double expected, double actual) {
        if (type == DataType.FLOAT32) assertEquals(Float.floatToRawIntBits((float) expected),
                Float.floatToRawIntBits((float) actual));
        else assertEquals(Double.doubleToRawLongBits(expected), Double.doubleToRawLongBits(actual));
    }
}
