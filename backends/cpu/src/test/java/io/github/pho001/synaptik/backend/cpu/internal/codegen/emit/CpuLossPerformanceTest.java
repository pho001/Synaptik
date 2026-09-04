package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuLossIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScatterLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.loss.DenseCategoricalCrossEntropyWithLogitsAttrs;
import io.github.pho001.synaptik.model.operation.loss.IndexCategoricalCrossEntropyWithLogitsAttrs;
import io.github.pho001.synaptik.model.operation.loss.LossKind;
import io.github.pho001.synaptik.model.operation.loss.LossReduction;
import io.github.pho001.synaptik.model.operation.loss.MeanSquaredErrorAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.lang.classfile.ClassFile;
import java.lang.classfile.Instruction;
import java.lang.classfile.Opcode;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Random;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Retained, deliberately opt-in generated-versus-direct loss timing protocol.
 *
 * <p>The parent process only coordinates five fresh Java forks.  A fork constructs every exact
 * schema-58 identity, defines its Class-File, alternates generated and frozen-oracle trials, and
 * records all raw values without retry or discard.  The ordinary CPU suite skips this test unless
 * either {@code SYNAPTIK_CPU_LOSS_PERFORMANCE=true} or {@code -Dsynaptik.cpu.loss.performance=true}
 * and a caller-supplied evidence root are present.
 * The oracle selection occurs before timing; the measured oracle call is one family/type-specific
 * primitive loop, never a production dispatcher or generated fallback. Every retained fork row
 * and every median-of-fork row is a binding {@code <= 1.15x} acceptance gate; the protocol never
 * retries or discards a sample.</p>
 */
class CpuLossPerformanceTest {
    private static final List<DataType> FLOATS = List.of(DataType.BFLOAT16, DataType.FLOAT32,
            DataType.FLOAT64);
    private static final String ENABLE = "SYNAPTIK_CPU_LOSS_PERFORMANCE";
    private static final String ENABLE_PROPERTY = "synaptik.cpu.loss.performance";
    private static final String ROOT = "synaptik.cpu.loss.performanceEvidenceRoot";
    private static final String C2_EVIDENCE = "synaptik.cpu.loss.c2Evidence";
    private static final double THRESHOLD = 1.15d;
    private static final int FORKS = 5;
    private static final int WARMUP_ROUNDS = 5;
    private static final int MEASUREMENTS = 9;
    /*
     * A timed side adapts to elapsed time.  Without an equal-count warmup, the much faster
     * oracle reaches C2 compilation while the generated entry receives only a few hundred
     * invocations.  Keep five warmup rounds, but make every round compilation-ready for both
     * already-bound, statically typed calls before collecting any timing evidence.
     */
    private static final int WARMUP_INVOCATIONS_PER_SIDE = 2_048;
    /*
     * The per-entry calibration chooses one shared batch count for both timed sides.  A larger
     * floor than the task minimum gives the tiny contiguous MSE controls enough elapsed time to
     * suppress process-level clock and scheduling noise without changing their generated/direct
     * algorithm, carrier binding, or number of calls in a measured pair.
     */
    private static final long MINIMUM_SIDE_BATCH_NANOS = 100_000_000L;
    private static volatile long checksum;
    private static volatile CpuLossPerformanceOracle oracle;

    /** Runs the retained parent protocol from a command line. */
    public static void main(String[] args) throws Exception {
        if (args.length == 2 && args[0].equals("--fork")) runFork(Path.of(System.getProperty(ROOT)),
                Integer.parseInt(args[1]));
        else if (args.length == 2 && args[0].equals("--targeted-fork"))
            runTargetedFork(Path.of(System.getProperty(ROOT)), Integer.parseInt(args[1]));
        else if (args.length == 2 && args[0].equals("--retained-outlier-fork"))
            runRetainedOutlierFork(Path.of(System.getProperty(ROOT)), Integer.parseInt(args[1]));
        else if (args.length == 1 && args[0].equals("--representative"))
            runRetainedOutlierParent(Path.of(System.getProperty(ROOT)));
        else runParent(Path.of(System.getProperty(ROOT)));
    }

    @Test
    void retainedFiveForkPerformanceProtocol() throws Exception {
        Assumptions.assumeTrue("true".equals(System.getenv(ENABLE))
                || Boolean.getBoolean(ENABLE_PROPERTY));
        String root = System.getProperty(ROOT);
        Assumptions.assumeTrue(root != null && !root.isBlank(), "explicit evidence root required");
        runParent(Path.of(root));
    }

    @Test
    void boundedRepresentativePerformancePreflight() throws Exception {
        Assumptions.assumeTrue(Boolean.getBoolean("synaptik.cpu.loss.performanceRepresentative"));
        String root = System.getProperty(ROOT);
        Assumptions.assumeTrue(root != null && !root.isBlank(), "explicit evidence root required");
        runRetainedOutlierParent(Path.of(root));
    }

    @Test
    void ratioUsesGeneratedNanosecondsDividedByDirectNanoseconds(@TempDir Path directory)
            throws IOException {
        Path raw = directory.resolve("raw.csv");
        Files.writeString(raw, "row,key,iterations,generated_ns,direct_ns,ratio,checksum\n"
                + "0,example,8192,115,100,1.15,42\n");

        assertEquals(1.15d, ratio(raw, 0));
    }

    @Test
    void failedFirstFullForkPreventsLaunchingTheSecond(@TempDir Path directory) throws Exception {
        var launches = new ArrayList<Integer>();
        assertThrows(AssertionError.class, () -> runFullForks(directory, fork -> {
            launches.add(fork);
            writeSyntheticFork(directory, fork, fork == 0 ? 1.16d : 1.0d, null);
        }));
        assertEquals(List.of(0), launches);
    }

    @Test
    void fullForkRejectsACanonicalKeyMismatch(@TempDir Path directory) throws Exception {
        writeSyntheticFork(directory, 0, 1.0d, "not-a-canonical-key");
        assertThrows(AssertionError.class, () -> validateFork(directory.resolve("raw-fork-0.csv"),
                directory.resolve("checksum-fork-0.txt"), 0, rows()));
    }

    /**
     * Constructs and prebinds every timed specialization without running its timing protocol.
     *
     * <p>This keeps fixture carrier/type correspondence in the ordinary suite, so a later opt-in
     * fork cannot first expose a malformed row after it has begun collecting evidence.</p>
     */
    @Test
    void allExactPerformanceRowsConstructAndPrebindWithoutTiming() throws Exception {
        int constructed = 0;
        for (Row row : rows()) {
            assertEquals(1, row.key().split(",", -1).length,
                    "CSV key must remain one column: " + row.key());
            try {
                Case value = create(row);
                assertEquals(carriers(List.of(row.result()), row.bits() >> (row.roles().equals(List.of(0, 0)) ? 1 : 2)).getFirst()
                                == CarrierAccess.MEMORY_SEGMENT,
                        value.generated().carrier() instanceof MemorySegment,
                        "shared output carrier must match the specialization: " + row.key());
                assertSame(value.generated().carrier(), value.direct().carrier(),
                        "both timed peers must bind exactly the same output carrier: " + row.key());
            } catch (Exception failure) {
                throw new AssertionError("could not construct performance row " + row.key(),
                        failure);
            }
            constructed++;
        }
        assertEquals(792, constructed);
    }

    @Test
    void compiledOracleSourceHasOnlyTypedCarrierLoops() {
        String source = oracle().source();
        assertEquals(792, source.split("public static void m", -1).length - 1);
        assertTrue(source.contains("MemorySegment"));
        assertTrue(source.contains("StrictMath.exp"));
        assertTrue(source.contains("StrictMath.log"));
        assertFalse(source.contains("Float32Source"));
        assertFalse(source.contains("Float64Source"));
        assertFalse(source.contains("IndexSource"));
        assertFalse(source.contains("Sink"));
        assertFalse(source.contains("Object"));
        assertFalse(source.contains("DataType"));
        assertFalse(source.contains("invokeinterface"));
        assertFalse(source.contains("new "));
        assertTrue(source.contains("output,long[] geometry,long start,long end"),
                "every oracle entry must receive the generated helper's cold payload and range");
        assertTrue(source.contains("geometry[9]"));
        assertTrue(source.contains("geometry[10+axis]"));
        assertFalse(source.contains("4096"), "fixture extent must not shape the oracle loop");
        assertFalse(source.contains("<128"), "fixture sample count must not shape the oracle loop");
        assertFalse(source.contains("<32"), "fixture class count must not shape the oracle loop");
        assertFalse(source.contains("*64"), "fixture stride must not shape the oracle loop");
        assertFalse(source.contains("2048"), "fixture base must not shape the oracle loop");
        assertTrue(oracle().classBytes().length > 0, "compiler must emit a concrete Class-File");
    }

    @Test
    void compiledOracleBytecodeUsesOnlyDirectCarriers(@TempDir Path directory) throws Exception {
        Path classFile = directory.resolve("LossPerformanceOracleGenerated.class");
        Files.write(classFile, oracle().classBytes());
        Process process = new ProcessBuilder(Path.of(System.getProperty("java.home"), "bin", "javap")
                .toString(), "-c", "-p", classFile.toString()).redirectErrorStream(true).start();
        String bytecode = new String(process.getInputStream().readAllBytes());
        assertEquals(0, process.waitFor(), bytecode);
        assertTrue(bytecode.contains("faload") || bytecode.contains("daload"), bytecode);
        assertTrue(bytecode.contains("MemorySegment.get"), bytecode);
        assertTrue(bytecode.contains(", long[], long, long);"), bytecode);
        assertFalse(bytecode.contains("Float32Source"), bytecode);
        assertFalse(bytecode.contains("Float64Source"), bytecode);
        assertFalse(bytecode.contains("IndexSource"), bytecode);
        assertFalse(bytecode.contains("Float32Sink"), bytecode);
        assertFalse(bytecode.contains("Float64Sink"), bytecode);
        assertFalse(bytecode.contains("new "), bytecode);
    }

    @Test
    void compiledOracleMatchesGeneratedRepresentativeOutputs() throws Exception {
        for (Row row : representativeRows()) {
            Case value = create(row);
            invoke(value.generated().entry());
            long generatedResult = checksum(value.generated().carrier(), row.result()).read();
            invoke(value.direct().entry());
            long directResult = checksum(value.direct().carrier(), row.result()).read();
            assertEquals(generatedResult, directResult, row.key());
        }
    }

    @Test
    void timedLoopUsesOnlyPreboundMethodHandlesWithoutASideSelectorOrInterfaceInvocation()
            throws Exception {
        try (InputStream resource = CpuLossPerformanceTest.class
                .getResourceAsStream("CpuLossPerformanceTest.class")) {
            assertTrue(resource != null, "test class bytes must be available");
            var elapsed = ClassFile.of().parse(resource.readAllBytes()).methods().stream()
                    .filter(method -> method.methodName().stringValue().equals("elapsed"))
                    .findFirst().orElseThrow();
            var code = elapsed.code().orElseThrow();
            assertAll(
                    () -> assertFalse(elapsed.methodTypeSymbol().descriptorString().contains("Z"),
                            "elapsed must receive one concrete pre-bound side, not a boolean selector"),
                    () -> assertFalse(code.elementStream().filter(Instruction.class::isInstance)
                            .map(Instruction.class::cast).map(Instruction::opcode)
                            .anyMatch(opcode -> opcode == Opcode.INVOKEINTERFACE),
                            "timed loop must not dispatch through Action or Checksum interfaces"),
                    () -> assertTrue(code.elementStream().filter(Instruction.class::isInstance)
                            .map(Instruction.class::cast).map(Instruction::opcode)
                            .anyMatch(opcode -> opcode == Opcode.INVOKEVIRTUAL),
                            "timed loop must invoke its pre-bound exact MethodHandles"));
        }
    }

    /** Keeps every retained full-matrix failure represented exactly once before timed preflight. */
    @Test
    void retainedOutlierPreflightContainsEveryRetainedFailureExactlyOnce() {
        List<String> keys = retainedOutlierPreflightRows().stream().map(Row::key).toList();
        assertTrue(keys.contains("MEAN_SQUARED_ERROR-FLOAT64-FLOAT32-NONE-false-roles0_1-0"));
        assertEquals(keys.size(), keys.stream().distinct().count(),
                "retained preflight rows must have unique canonical keys");
    }

    @Test
    void timingProtocolRetainsItsRequiredConstantsAndExactInventory() {
        assertAll(
                () -> assertEquals(5, FORKS),
                () -> assertEquals(5, WARMUP_ROUNDS),
                () -> assertEquals(2_048, WARMUP_INVOCATIONS_PER_SIDE),
                () -> assertEquals(9, MEASUREMENTS),
                () -> assertEquals(100_000_000L, MINIMUM_SIDE_BATCH_NANOS),
                () -> assertEquals(1.15d, THRESHOLD),
                () -> assertEquals(792, rows().size()),
                () -> assertEquals(24, retainedOutlierPreflightRows().size()));
    }

    private static void runParent(Path root) throws Exception {
        Files.createDirectories(root);
        writeOracleEvidence(root);
        Files.writeString(root.resolve("protocol.txt"), "forks=" + FORKS + "\nwarmups=" + WARMUP_ROUNDS + "\n"
                + "warmup_invocations_per_side=" + WARMUP_INVOCATIONS_PER_SIDE + "\nmeasurements=" + MEASUREMENTS + "\n"
                + "minimum_side_batch_ns=" + MINIMUM_SIDE_BATCH_NANOS
                + "\nshared_adaptive_batch_count=true\nrows_per_fork=792\norder=seeded-alternation\n"
                + "threshold=1.15\noracle=frozen allocation-free typed direct loops\n");
        Files.writeString(root.resolve("machine.txt"), "java=" + System.getProperty("java.version")
                + "\nvm=" + System.getProperty("java.vm.name") + "\nos="
                + System.getProperty("os.name") + ' ' + System.getProperty("os.version")
                + "\narchitecture=" + System.getProperty("os.arch") + "\nheap=-Xms1g,-Xmx1g\n");
        var forkChecksums = runFullForks(root, fork -> {
            List<String> command = List.of(Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                    "-Xms1g", "-Xmx1g", "--add-modules", "jdk.incubator.vector", "-cp",
                    System.getProperty("java.class.path"), "-D" + ROOT + '=' + root,
                    CpuLossPerformanceTest.class.getName(), "--fork", Integer.toString(fork));
            Files.writeString(root.resolve("progress.txt"), "starting full fork " + fork + "\n",
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND);
            Process process = new ProcessBuilder(command)
                    .redirectOutput(root.resolve("fork-" + fork + ".stdout").toFile())
                    .redirectError(root.resolve("fork-" + fork + ".stderr").toFile()).start();
            assertEquals(0, process.waitFor(), "fork " + fork);
            Files.writeString(root.resolve("progress.txt"), "finished full fork " + fork + "\n",
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        });
        var ratios = new ArrayList<Double>();
        for (int row = 0; row < 792; row++) {
            var values = new ArrayList<Double>();
            for (int fork = 0; fork < FORKS; fork++) values.add(ratio(root.resolve("raw-fork-" + fork + ".csv"), row));
            values.sort(Comparator.naturalOrder());
            double median = values.get(2);
            ratios.add(median);
            assertTrue(median <= THRESHOLD, "median ratio row " + row + " = " + median);
        }
        Files.writeString(root.resolve("report.csv"), "row,median_of_fork_ratios,accepted\n");
        for (int row = 0; row < ratios.size(); row++) Files.writeString(root.resolve("report.csv"),
                row + "," + ratios.get(row) + ",true\n", StandardOpenOption.APPEND);
        Files.writeString(root.resolve("checksums.csv"), "fork,checksum\n");
        for (int fork = 0; fork < forkChecksums.size(); fork++) Files.writeString(
                root.resolve("checksums.csv"), fork + "," + forkChecksums.get(fork) + "\n",
                StandardOpenOption.APPEND);
        Files.writeString(root.resolve("manifest.txt"), "forks=" + FORKS + "\nrows_per_fork=792\n"
                + "row_fork_records=3960\nunique_row_fork_records=3960\nreport_rows=792\n"
                + "threshold=1.15\nall_row_fork_gates=passed\nall_median_gates=passed\n");
    }

    private static double ratio(Path raw, int row) throws IOException {
        try (BufferedReader reader = Files.newBufferedReader(raw)) {
            assertEquals("row,key,iterations,generated_ns,direct_ns,ratio,checksum", reader.readLine());
            for (int current = 0; current <= row; current++) {
                String line = reader.readLine();
                if (line == null) break;
                RawRow rawRow = parseRawRow(line, raw, current);
                if (current == row) return rawRow.generatedToDirectRatio();
            }
        }
        throw new IOException("missing row " + row + " in " + raw);
    }

    private static void runFork(Path root, int fork) throws Exception {
        Files.createDirectories(root);
        StringBuilder raw = new StringBuilder("row,key,iterations,generated_ns,direct_ns,ratio,checksum\n");
        Random random = new Random(0x0008_1L + fork);
        int ordinal = 0;
        for (Row row : rows()) {
            Case value = create(row);
            for (int warmup = 0; warmup < WARMUP_ROUNDS; warmup++) {
                if (random.nextBoolean()) { warmup(value.generated()); warmup(value.direct()); }
                else { warmup(value.direct()); warmup(value.generated()); }
            }
            int iterations = calibratedIterations(value);
            long generated = 0L, direct = 0L;
            for (int measurement = 0; measurement < MEASUREMENTS; measurement++) {
                boolean generatedFirst = random.nextBoolean();
                if (generatedFirst) { generated += measure(value.generated(), iterations); direct += measure(value.direct(), iterations); }
                else { direct += measure(value.direct(), iterations); generated += measure(value.generated(), iterations); }
            }
            raw.append(ordinal++).append(',').append(row.key()).append(',').append(iterations).append(',').append(generated)
                    .append(',').append(direct).append(',').append((double) generated / direct).append(',')
                    .append(checksum).append('\n');
        }
        assertEquals(792, ordinal);
        Files.writeString(root.resolve("raw-fork-" + fork + ".csv"), raw);
        Files.writeString(root.resolve("checksum-fork-" + fork + ".txt"), Long.toString(checksum)
                + "\n");
    }

    /** Runs the five-independent-JVM preflight over retained outliers and representative controls. */
    private static void runTargetedParent(Path root) throws Exception {
        Files.createDirectories(root);
        writeOracleEvidence(root);
        List<Row> rows = indexMeanSegmentRows();
        Files.writeString(root.resolve("targeted-protocol.txt"), "forks=" + FORKS + "\nwarmups=" + WARMUP_ROUNDS + "\n"
                + "measurements=" + MEASUREMENTS + "\nminimum_side_batch_ns=" + MINIMUM_SIDE_BATCH_NANOS
                + "\nshared_adaptive_batch_count=true\norder=seeded-alternation\n"
                + "threshold=1.15\nrows=" + rows.size() + "\nrequired_rows=index-mean-segment\n");
        for (int fork = 0; fork < FORKS; fork++) {
            var command = new ArrayList<String>(List.of(Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                    "-Xms1g", "-Xmx1g"));
            if (Boolean.getBoolean(C2_EVIDENCE)) {
                command.addAll(List.of("-XX:+UnlockDiagnosticVMOptions", "-XX:+PrintCompilation",
                        "-XX:+LogCompilation", "-XX:LogFile="
                                + root.resolve("targeted-fork-" + fork + ".hotspot.log")));
            }
            command.addAll(List.of("--add-modules", "jdk.incubator.vector", "-cp",
                    System.getProperty("java.class.path"), "-D" + ROOT + '=' + root,
                    CpuLossPerformanceTest.class.getName(), "--targeted-fork", Integer.toString(fork)));
            Files.writeString(root.resolve("progress.txt"), "starting targeted fork " + fork + "\n",
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND);
            Process process = new ProcessBuilder(command)
                    .redirectOutput(root.resolve("targeted-fork-" + fork + ".stdout").toFile())
                    .redirectError(root.resolve("targeted-fork-" + fork + ".stderr").toFile()).start();
            assertEquals(0, process.waitFor(), "targeted fork " + fork);
            Files.writeString(root.resolve("progress.txt"), "finished targeted fork " + fork + "\n",
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        }
        StringBuilder report = new StringBuilder("row,key,fork0,fork1,fork2,fork3,fork4,median,accepted\n");
        for (int row = 0; row < rows.size(); row++) {
            var ratios = new ArrayList<Double>();
            for (int fork = 0; fork < FORKS; fork++) {
                double value = ratio(root.resolve("targeted-raw-fork-" + fork + ".csv"), row);
                assertTrue(value <= THRESHOLD, "targeted fork " + fork + " row " + row + " = " + value);
                ratios.add(value);
            }
            ratios.sort(Comparator.naturalOrder());
            double median = ratios.get(2);
            assertTrue(median <= THRESHOLD, "targeted median row " + row + " = " + median);
            report.append(row).append(',').append(rows.get(row).key());
            for (double value : ratios) report.append(',').append(value);
            report.append(',').append(median).append(",true\n");
        }
        Files.writeString(root.resolve("targeted-report.csv"), report.toString());
    }

    private static void runTargetedFork(Path root, int fork) throws Exception {
        Files.createDirectories(root);
        StringBuilder raw = new StringBuilder("row,key,iterations,generated_ns,direct_ns,ratio,checksum\n");
        Random random = new Random(0x0008_1L + fork);
        int ordinal = 0;
        for (Row row : indexMeanSegmentRows()) {
            Case value = create(row);
            for (int warmup = 0; warmup < WARMUP_ROUNDS; warmup++) {
                if (random.nextBoolean()) { warmup(value.generated()); warmup(value.direct()); }
                else { warmup(value.direct()); warmup(value.generated()); }
            }
            int iterations = calibratedIterations(value);
            long generated = 0L, direct = 0L;
            for (int measurement = 0; measurement < MEASUREMENTS; measurement++) {
                if (random.nextBoolean()) { generated += measure(value.generated(), iterations); direct += measure(value.direct(), iterations); }
                else { direct += measure(value.direct(), iterations); generated += measure(value.generated(), iterations); }
            }
            raw.append(ordinal++).append(',').append(row.key()).append(',').append(iterations).append(',').append(generated)
                    .append(',').append(direct).append(',').append((double) generated / direct).append(',')
                    .append(checksum).append('\n');
        }
        Files.writeString(root.resolve("targeted-raw-fork-" + fork + ".csv"), raw.toString());
    }

    /** Runs all retained full-matrix failures and matched controls in fresh isolated JVMs. */
    private static void runRetainedOutlierParent(Path root) throws Exception {
        Files.createDirectories(root);
        writeOracleEvidence(root);
        List<Row> rows = retainedOutlierPreflightRows();
        Files.writeString(root.resolve("retained-outlier-protocol.txt"), "forks=" + FORKS + "\nwarmups=" + WARMUP_ROUNDS + "\n"
                + "measurements=" + MEASUREMENTS + "\nminimum_side_batch_ns=" + MINIMUM_SIDE_BATCH_NANOS
                + "\nshared_adaptive_batch_count=true\norder=seeded-alternation\n"
                + "threshold=1.15\nrows=" + rows.size()
                + "\nrequired_rows=all-retained-fork-0-outliers-plus-controls\n");
        for (int fork = 0; fork < FORKS; fork++) {
            List<String> command = List.of(Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                    "-Xms1g", "-Xmx1g", "--add-modules", "jdk.incubator.vector", "-cp",
                    System.getProperty("java.class.path"), "-D" + ROOT + '=' + root,
                    CpuLossPerformanceTest.class.getName(), "--retained-outlier-fork", Integer.toString(fork));
            Process process = new ProcessBuilder(command)
                    .redirectOutput(root.resolve("retained-outlier-fork-" + fork + ".stdout").toFile())
                    .redirectError(root.resolve("retained-outlier-fork-" + fork + ".stderr").toFile()).start();
            assertEquals(0, process.waitFor(), "retained outlier fork " + fork);
            validateRows(root.resolve("retained-outlier-raw-fork-" + fork + ".csv"), fork, rows);
        }
    }

    private static void runRetainedOutlierFork(Path root, int fork) throws Exception {
        runRowsFork(root.resolve("retained-outlier-raw-fork-" + fork + ".csv"), fork,
                retainedOutlierPreflightRows());
    }

    private static void runRowsFork(Path rawPath, int fork, List<Row> selectedRows) throws Exception {
        Files.createDirectories(rawPath.getParent());
        StringBuilder raw = new StringBuilder("row,key,iterations,generated_ns,direct_ns,ratio,checksum\n");
        Random random = new Random(0x0008_1L + fork);
        int ordinal = 0;
        for (Row row : selectedRows) {
            Case value = create(row);
            for (int warmup = 0; warmup < WARMUP_ROUNDS; warmup++) {
                if (random.nextBoolean()) { warmup(value.generated()); warmup(value.direct()); }
                else { warmup(value.direct()); warmup(value.generated()); }
            }
            int iterations = calibratedIterations(value);
            long generated = 0L, direct = 0L;
            for (int measurement = 0; measurement < MEASUREMENTS; measurement++) {
                if (random.nextBoolean()) { generated += measure(value.generated(), iterations); direct += measure(value.direct(), iterations); }
                else { direct += measure(value.direct(), iterations); generated += measure(value.generated(), iterations); }
            }
            raw.append(ordinal++).append(',').append(row.key()).append(',').append(iterations).append(',')
                    .append(generated).append(',').append(direct).append(',')
                    .append((double) generated / direct).append(',').append(checksum).append('\n');
        }
        Files.writeString(rawPath, raw.toString());
    }

    private static void validateRows(Path raw, int fork, List<Row> expectedRows) throws IOException {
        try (BufferedReader reader = Files.newBufferedReader(raw)) {
            assertEquals("row,key,iterations,generated_ns,direct_ns,ratio,checksum", reader.readLine());
            for (int row = 0; row < expectedRows.size(); row++) {
                RawRow actual = parseRawRow(reader.readLine(), raw, row);
                assertEquals(row, actual.ordinal());
                assertEquals(expectedRows.get(row).key(), actual.key());
                assertTrue(actual.generatedToDirectRatio() <= THRESHOLD,
                        "retained outlier fork " + fork + " row " + row);
            }
            assertEquals(null, reader.readLine(), "extra retained outlier record");
        }
    }

    /**
     * Runs a bounded preflight without weakening the retained five-fork acceptance protocol.
     *
     * <p>The rows deliberately cover all families and reductions, binary32/binary64 accumulator
     * domains, heap and segment output roles, and both index widths. It includes the seven
     * retained index-loss failures and the separate MSE comparator, and is intended only to
     * diagnose a candidate emitter before the caller elects to spend the full matrix budget.</p>
     */
    private static void runRepresentative(Path root) throws Exception {
        Files.createDirectories(root);
        writeOracleEvidence(root);
        StringBuilder measurements = new StringBuilder("key,measurement,generated_first,generated_ns,direct_ns,ratio\n");
        StringBuilder report = new StringBuilder("key,generated_median_ns,direct_median_ns,median_ratio\n");
        for (Row row : representativeRows()) {
            Case value = create(row);
            Files.write(root.resolve(row.key() + ".class"), value.generatedClassBytes());
            for (int warmup = 0; warmup < WARMUP_ROUNDS; warmup++) {
                if ((warmup & 1) == 0) { warmup(value.generated()); warmup(value.direct()); }
                else { warmup(value.direct()); warmup(value.generated()); }
            }
            int iterations = calibratedIterations(value);
            var generatedSamples = new ArrayList<Long>();
            var directSamples = new ArrayList<Long>();
            for (int measurement = 0; measurement < MEASUREMENTS; measurement++) {
                boolean generatedFirst = (measurement & 1) == 0;
                long generated;
                long direct;
                if (generatedFirst) { generated = measure(value.generated(), iterations); direct = measure(value.direct(), iterations); }
                else { direct = measure(value.direct(), iterations); generated = measure(value.generated(), iterations); }
                generatedSamples.add(generated);
                directSamples.add(direct);
                measurements.append(row.key()).append(',').append(measurement).append(',')
                        .append(generatedFirst).append(',').append(generated).append(',').append(direct)
                        .append(',').append((double) generated / direct).append('\n');
            }
            generatedSamples.sort(Comparator.naturalOrder());
            directSamples.sort(Comparator.naturalOrder());
            long generated = generatedSamples.get(4);
            long direct = directSamples.get(4);
            double ratio = (double) generated / direct;
            report.append(row.key()).append(',').append(generated).append(',').append(direct)
                    .append(',').append(ratio).append('\n');
        }
        Files.writeString(root.resolve("representative-measurements.csv"), measurements);
        Files.writeString(root.resolve("representative.csv"), report);
        Files.writeString(root.resolve("representative-protocol.txt"), "warmups=" + WARMUP_ROUNDS + "\n"
                + "warmup_invocations_per_side=" + WARMUP_INVOCATIONS_PER_SIDE + "\nmeasurements=" + MEASUREMENTS + "\n"
                + "minimum_side_batch_ns=" + MINIMUM_SIDE_BATCH_NANOS
                + "\nshared_adaptive_batch_count=true\norder=alternating-generated-first\n"
                + "statistic=independent-side-medians\nthreshold=1.15\nrows="
                + representativeRows().size() + "\n");
    }

    /** Launches one full fork at a time and accepts it before the next launch is permitted. */
    private static List<Long> runFullForks(Path root, ForkLauncher launcher) throws Exception {
        var checksums = new ArrayList<Long>();
        List<Row> expectedRows = rows();
        for (int fork = 0; fork < FORKS; fork++) {
            launcher.launch(fork);
            checksums.add(validateFork(root.resolve("raw-fork-" + fork + ".csv"),
                    root.resolve("checksum-fork-" + fork + ".txt"), fork, expectedRows));
        }
        return checksums;
    }

    private static long validateFork(Path raw, Path checksumFile, int fork, List<Row> expectedRows)
            throws IOException {
        try (BufferedReader reader = Files.newBufferedReader(raw)) {
            assertEquals("row,key,iterations,generated_ns,direct_ns,ratio,checksum", reader.readLine());
            long checksum = 0L;
            for (int row = 0; row < 792; row++) {
                String line = reader.readLine();
                assertTrue(line != null, "missing fork " + fork + " row " + row);
                RawRow rawRow = parseRawRow(line, raw, row);
                assertEquals(row, rawRow.ordinal(), "fork " + fork + " row ordinal");
                assertEquals(expectedRows.get(row).key(), rawRow.key(),
                        "fork " + fork + " row canonical key");
                double ratio = rawRow.generatedToDirectRatio();
                assertTrue(ratio <= THRESHOLD, "fork " + fork + " row " + row + " = " + ratio);
                checksum = rawRow.checksum();
            }
            assertEquals(null, reader.readLine(), "extra fork " + fork + " record");
            assertEquals(Long.toString(checksum), Files.readString(checksumFile).trim(),
                    "fork " + fork + " checksum evidence");
            return checksum;
        }
    }

    private static RawRow parseRawRow(String line, Path raw, int expectedRow) {
        String[] columns = line.split(",", -1);
        assertEquals(7, columns.length, raw + " row " + expectedRow + " column count");
        int ordinal = Integer.parseInt(columns[0]);
        int iterations = Integer.parseInt(columns[2]);
        long generatedNanoseconds = Long.parseLong(columns[3]);
        long directNanoseconds = Long.parseLong(columns[4]);
        double reportedRatio = Double.parseDouble(columns[5]);
        assertTrue(iterations > 0, raw + " row " + expectedRow + " iterations must be positive");
        assertTrue(generatedNanoseconds >= 0L, raw + " row " + expectedRow
                + " generated nanoseconds must be non-negative");
        assertTrue(directNanoseconds > 0L, raw + " row " + expectedRow
                + " direct nanoseconds must be positive");
        double generatedToDirectRatio = (double) generatedNanoseconds / directNanoseconds;
        assertTrue(Double.isFinite(reportedRatio), raw + " row " + expectedRow
                + " ratio must be finite");
        assertEquals(generatedToDirectRatio, reportedRatio, raw + " row " + expectedRow
                + " ratio must equal generated_ns/direct_ns");
        return new RawRow(ordinal, columns[1], generatedToDirectRatio, Long.parseLong(columns[6]));
    }

    /** Writes a complete cheap CSV fixture for protocol-control tests only. */
    private static void writeSyntheticFork(Path root, int fork, double firstRatio, String firstKey)
            throws IOException {
        StringBuilder raw = new StringBuilder("row,key,iterations,generated_ns,direct_ns,ratio,checksum\n");
        for (int row = 0; row < rows().size(); row++) {
            double ratio = row == 0 ? firstRatio : 1.0d;
            raw.append(row).append(',')
                    .append(row == 0 && firstKey != null ? firstKey : rows().get(row).key())
                    .append(",1,").append((long) (ratio * 100)).append(",100,")
                    .append(ratio).append(",0\n");
        }
        Files.writeString(root.resolve("raw-fork-" + fork + ".csv"), raw.toString());
        Files.writeString(root.resolve("checksum-fork-" + fork + ".txt"), "0\n");
    }

    private static int calibratedIterations(Case value) throws Exception {
        int iterations = 1;
        do {
            long generated = elapsed(value.generated(), iterations);
            long direct = elapsed(value.direct(), iterations);
            if (generated >= MINIMUM_SIDE_BATCH_NANOS && direct >= MINIMUM_SIDE_BATCH_NANOS) {
                return Math.multiplyExact(iterations, 2);
            }
            iterations = Math.multiplyExact(iterations, 2);
        } while (true);
    }

    private static long measure(TimedSide side, int iterations) throws Exception {
        return elapsed(side, iterations) / iterations;
    }

    /**
     * Times one already-bound typed kernel and consumes its bound output after every invocation.
     *
     * <p>The side is selected before entering this loop. Both calls use exact {@code void()} and
     * {@code long()} MethodHandles, so the measured loop contains neither a generated/direct
     * boolean branch nor an {@code Action}/{@code Checksum} interface dispatch.</p>
     *
     * @param side non-null pre-bound kernel and output-consumer pair
     * @param iterations positive equal invocation count selected for both sides
     * @return elapsed wall-clock nanoseconds for the complete consumed invocation batch
     * @throws Exception if a bound kernel or output consumer fails
     */
    private static long elapsed(TimedSide side, int iterations) throws Exception {
        MethodHandle entry = side.entry();
        MethodHandle output = side.output();
        long started = System.nanoTime();
        long observed = checksum;
        try {
            for (int iteration = 0; iteration < iterations; iteration++) {
                entry.invokeExact();
                observed ^= (long) output.invokeExact();
            }
        } catch (Throwable failure) {
            rethrow(failure);
        }
        checksum = observed;
        return System.nanoTime() - started;
    }

    /** Performs one equal-count, output-consuming warmup round for one already-bound entry. */
    private static void warmup(TimedSide side) throws Exception {
        elapsed(side, WARMUP_INVOCATIONS_PER_SIDE);
    }

    private static void invoke(MethodHandle entry) throws Exception {
        try {
            entry.invokeExact();
        } catch (Throwable failure) {
            rethrow(failure);
        }
    }

    private static void rethrow(Throwable failure) throws Exception {
        if (failure instanceof Exception exception) throw exception;
        if (failure instanceof Error error) throw error;
        throw new AssertionError("typed loss benchmark invocation failed", failure);
    }

    /*
     * Each direct entry is an already-bound call to an isolated compiled method. Its exact
     * primitive carrier signature is selected while the Case is created; timed loops contain no
     * source/sink interface call, carrier/type/reduction dispatch, lookup, allocation, or copy.
     */
    private static MethodHandle directEntry(CpuLossPerformanceOracle.Spec spec, Object left, Object right,
            Object output, long[] geometry, long start, long end) throws NoSuchMethodException,
            IllegalAccessException {
        return oracle().bind(spec, left, right, output, geometry, start, end);
    }

    private static List<Row> rows() {
        var rows = new ArrayList<Row>();
        for (LossKind family : List.of(LossKind.MEAN_SQUARED_ERROR,
                LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS))
            for (DataType left : FLOATS) for (DataType right : FLOATS) for (LossReduction reduction : LossReduction.values()) {
                for (int bits = 0; bits < 8; bits++) rows.add(new Row(family, left, right, reduction, false,
                        List.of(0, 1), bits));
                if (left == right) for (int bits = 0; bits < 4; bits++) rows.add(new Row(family, left, right,
                        reduction, false, List.of(0, 0), bits));
            }
        for (DataType logits : FLOATS) for (DataType index : List.of(DataType.INT32, DataType.INT64))
            for (LossReduction reduction : LossReduction.values()) for (boolean ignore : List.of(false, true))
                for (int bits = 0; bits < 8; bits++) rows.add(new Row(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                        logits, index, reduction, ignore, List.of(0, 1), bits));
        assertEquals(792, rows.size());
        return rows;
    }

    private static List<Row> representativeRows() {
        return List.of(
                new Row(LossKind.MEAN_SQUARED_ERROR, DataType.FLOAT32, DataType.FLOAT32,
                        LossReduction.NONE, false, List.of(0, 1), 0),
                new Row(LossKind.MEAN_SQUARED_ERROR, DataType.FLOAT64, DataType.FLOAT32,
                        LossReduction.NONE, false, List.of(0, 1), 0),
                new Row(LossKind.MEAN_SQUARED_ERROR, DataType.FLOAT64, DataType.FLOAT64,
                        LossReduction.MEAN, false, List.of(0, 1), 7),
                new Row(LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, DataType.BFLOAT16,
                        DataType.FLOAT32, LossReduction.SUM, false, List.of(0, 1), 5),
                new Row(LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, DataType.FLOAT64,
                        DataType.FLOAT64, LossReduction.NONE, false, List.of(0, 0), 2),
                new Row(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, DataType.FLOAT32,
                        DataType.INT32, LossReduction.MEAN, true, List.of(0, 1), 7),
                new Row(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, DataType.FLOAT64,
                        DataType.INT64, LossReduction.SUM, false, List.of(0, 1), 4),
                // Keep the known index-loss outliers in the bounded preflight.  This preserves
                // the full 792-row gate while making a candidate change cheaply falsifiable.
                new Row(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, DataType.FLOAT32,
                        DataType.INT32, LossReduction.SUM, false, List.of(0, 1), 4),
                new Row(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, DataType.FLOAT32,
                        DataType.INT32, LossReduction.MEAN, false, List.of(0, 1), 4),
                new Row(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, DataType.FLOAT32,
                        DataType.INT64, LossReduction.MEAN, false, List.of(0, 1), 4),
                new Row(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, DataType.FLOAT64,
                        DataType.INT32, LossReduction.MEAN, false, List.of(0, 1), 4),
                new Row(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, DataType.FLOAT64,
                        DataType.INT64, LossReduction.MEAN, false, List.of(0, 1), 4),
                new Row(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, DataType.BFLOAT16,
                        DataType.INT64, LossReduction.MEAN, true, List.of(0, 1), 0),
                new Row(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, DataType.FLOAT32,
                        DataType.INT64, LossReduction.SUM, false, List.of(0, 1), 4),
                // The full five-fork gate found these contiguous heap-carrier forms at the
                // threshold boundary. Retain the exact rows so a bounded preflight exercises
                // the same selected body before another full-matrix run.
                new Row(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, DataType.BFLOAT16,
                        DataType.INT64, LossReduction.MEAN, true, List.of(0, 1), 4),
                new Row(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, DataType.FLOAT32,
                        DataType.INT32, LossReduction.NONE, true, List.of(0, 1), 0),
                new Row(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, DataType.FLOAT64,
                        DataType.INT32, LossReduction.NONE, true, List.of(0, 1), 0));
    }

    /** All retained full-matrix failures, followed by non-failing controls for each affected family. */
    private static List<Row> retainedOutlierPreflightRows() {
        return List.of(
                new Row(LossKind.MEAN_SQUARED_ERROR, DataType.FLOAT64, DataType.FLOAT32, LossReduction.NONE, false, List.of(0, 1), 0),
                new Row(LossKind.MEAN_SQUARED_ERROR, DataType.FLOAT64, DataType.FLOAT64, LossReduction.SUM, false, List.of(0, 1), 6),
                new Row(LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, DataType.BFLOAT16, DataType.BFLOAT16, LossReduction.NONE, false, List.of(0, 1), 5),
                new Row(LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, DataType.BFLOAT16, DataType.BFLOAT16, LossReduction.MEAN, false, List.of(0, 1), 2),
                new Row(LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, DataType.BFLOAT16, DataType.BFLOAT16, LossReduction.MEAN, false, List.of(0, 1), 6),
                new Row(LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, DataType.BFLOAT16, DataType.FLOAT32, LossReduction.MEAN, false, List.of(0, 1), 2),
                new Row(LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, DataType.BFLOAT16, DataType.FLOAT32, LossReduction.MEAN, false, List.of(0, 1), 6),
                new Row(LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, DataType.BFLOAT16, DataType.FLOAT64, LossReduction.MEAN, false, List.of(0, 1), 2),
                new Row(LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, DataType.BFLOAT16, DataType.FLOAT64, LossReduction.MEAN, false, List.of(0, 1), 6),
                new Row(LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, DataType.FLOAT64, DataType.FLOAT64, LossReduction.MEAN, false, List.of(0, 1), 6),
                new Row(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, DataType.BFLOAT16, DataType.INT32, LossReduction.NONE, true, List.of(0, 1), 1),
                new Row(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, DataType.BFLOAT16, DataType.INT32, LossReduction.MEAN, true, List.of(0, 1), 6),
                new Row(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, DataType.BFLOAT16, DataType.INT64, LossReduction.MEAN, false, List.of(0, 1), 2),
                new Row(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, DataType.BFLOAT16, DataType.INT64, LossReduction.MEAN, false, List.of(0, 1), 6),
                new Row(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, DataType.FLOAT32, DataType.INT32, LossReduction.NONE, true, List.of(0, 1), 3),
                new Row(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, DataType.FLOAT32, DataType.INT32, LossReduction.SUM, false, List.of(0, 1), 0),
                new Row(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, DataType.FLOAT32, DataType.INT32, LossReduction.MEAN, false, List.of(0, 1), 6),
                new Row(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, DataType.FLOAT32, DataType.INT64, LossReduction.NONE, true, List.of(0, 1), 4),
                new Row(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, DataType.FLOAT64, DataType.INT32, LossReduction.MEAN, false, List.of(0, 1), 2),
                new Row(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, DataType.FLOAT64, DataType.INT32, LossReduction.MEAN, false, List.of(0, 1), 6),
                new Row(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, DataType.FLOAT64, DataType.INT64, LossReduction.NONE, true, List.of(0, 1), 4),
                new Row(LossKind.MEAN_SQUARED_ERROR, DataType.FLOAT32, DataType.FLOAT32, LossReduction.NONE, false, List.of(0, 1), 0),
                new Row(LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, DataType.FLOAT32, DataType.FLOAT32, LossReduction.SUM, false, List.of(0, 1), 0),
                new Row(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, DataType.FLOAT32, DataType.INT32, LossReduction.MEAN, false, List.of(0, 1), 0));
    }

    /** Returns every exact array-input/segment-output INDEX/MEAN carrier identity. */
    private static List<Row> indexMeanSegmentRows() {
        var rows = new ArrayList<Row>();
        for (DataType logits : FLOATS) for (DataType index : List.of(DataType.INT32, DataType.INT64))
            for (boolean ignore : List.of(false, true)) rows.add(new Row(
                    LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, logits, index,
                    LossReduction.MEAN, ignore, List.of(0, 1), 4));
        return rows;
    }

    private static Case create(Row row) throws Exception {
        Shape logits = Shape.of(2, 32, 64), target = row.family() == LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS
                ? Shape.of(2, 64) : logits;
        Shape output = row.reduction() == LossReduction.NONE ? switch (row.family()) {
            case MEAN_SQUARED_ERROR -> logits;
            case DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                    INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS -> Shape.of(2, 64);
        } : Shape.scalar();
        Operation operation = operation(row);
        List<TensorDescriptor> inputs = row.roles().equals(List.of(0, 0)) ? List.of(descriptor(row.left(), logits))
                : List.of(descriptor(row.left(), logits), descriptor(row.right(), target));
        List<DataType> types = new ArrayList<>(inputs.stream().map(TensorDescriptor::dataType).toList()); types.add(row.result());
        List<CarrierAccess> carriers = carriers(types, row.bits());
        var base = CpuScatterLoweringTest.context(operation, row.roles(), inputs, descriptor(row.result(), output));
        var plan = new CpuPartitionPreparer().analyze(new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false, carriers))).plan();
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        var kernel = generator.defineClassBytes(route.specialization(), generator.generateClassBytes(route.specialization(), route.kernelIr()));
        long domain = row.family() == LossKind.MEAN_SQUARED_ERROR ? 4_096L : 128L;
        Object[] storage = arrays(types);
        populateInputs(storage, types, inputs.size(), row.ignore());
        Object[] arguments = arguments(storage, carriers, ((CpuLossIr) route.portableKernelIr()).geometry().pack(new long[] {0, 0, 0}),
                0L, row.reduction() == LossReduction.NONE ? domain : 1L);
        Object outputCarrier = arguments[types.size() - 1];
        // The benchmark binds the identical cold-selected private entry used by production loss
        // binding. Both timed peers share this output object, while representative semantic tests
        // snapshot the generated result before the direct peer overwrites it. These static
        // representative fixtures are all contiguous by construction.
        MethodHandle generated = exactVoid(kernel.lossEntryPointFor(true), arguments);
        MethodHandle direct = directEntry(oracleSpec(row, carriers), arguments[row.roles().getFirst()],
                arguments[row.roles().getLast()], outputCarrier, (long[]) arguments[storage.length],
                (long) arguments[storage.length + 1], (long) arguments[storage.length + 2]);
        return new Case(new TimedSide(generated, checksumHandle(outputCarrier, row.result()), outputCarrier),
                new TimedSide(direct, checksumHandle(outputCarrier, row.result()), outputCarrier),
                kernel.classBytes());
    }

    /** Binds all cold carriers, geometry, and range once; the timed call site is exact void(). */
    private static MethodHandle exactVoid(MethodHandle entryPoint, Object[] arguments) {
        return MethodHandles.insertArguments(entryPoint, 0, arguments)
                .asType(MethodType.methodType(void.class));
    }

    private static CpuLossPerformanceOracle oracle() {
        CpuLossPerformanceOracle current = oracle;
        if (current != null) return current;
        synchronized (CpuLossPerformanceTest.class) {
            current = oracle;
            if (current == null) {
                var specs = new ArrayList<CpuLossPerformanceOracle.Spec>();
                for (Row row : rows()) {
                    List<DataType> types = row.roles().equals(List.of(0, 0))
                            ? List.of(row.left(), row.result())
                            : List.of(row.left(), row.right(), row.result());
                    specs.add(oracleSpec(row, carriers(types, row.bits())));
                }
                oracle = current = CpuLossPerformanceOracle.compile(specs);
            }
        }
        return current;
    }

    private static CpuLossPerformanceOracle.Spec oracleSpec(Row row,
            List<CarrierAccess> carriers) {
        int ordinal = rows().indexOf(row);
        if (ordinal < 0) throw new AssertionError("unknown performance row " + row.key());
        return new CpuLossPerformanceOracle.Spec("m" + ordinal, oracleFamily(row.family()),
                oracleFloating(row.left()), row.family() == LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS
                        ? CpuLossPerformanceOracle.Floating.F32 : oracleFloating(row.right()),
                CpuLossPerformanceOracle.Reduction.valueOf(row.reduction().name()), row.ignore(),
                oracleCarrier(carriers.get(row.roles().getFirst())),
                oracleCarrier(carriers.get(row.roles().getLast())), oracleCarrier(carriers.getLast()),
                row.family() == LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS
                        ? (row.right() == DataType.INT32 ? CpuLossPerformanceOracle.Index.I32
                                : CpuLossPerformanceOracle.Index.I64)
                        : CpuLossPerformanceOracle.Index.UNUSED);
    }

    private static CpuLossPerformanceOracle.Family oracleFamily(LossKind family) {
        return switch (family) {
            case MEAN_SQUARED_ERROR -> CpuLossPerformanceOracle.Family.MSE;
            case DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS -> CpuLossPerformanceOracle.Family.DENSE;
            case INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS -> CpuLossPerformanceOracle.Family.INDEX;
        };
    }

    private static CpuLossPerformanceOracle.Floating oracleFloating(DataType type) {
        return switch (type) {
            case BFLOAT16 -> CpuLossPerformanceOracle.Floating.BF16;
            case FLOAT32 -> CpuLossPerformanceOracle.Floating.F32;
            case FLOAT64 -> CpuLossPerformanceOracle.Floating.F64;
            default -> throw new AssertionError(type);
        };
    }

    private static CpuLossPerformanceOracle.Carrier oracleCarrier(CarrierAccess carrier) {
        return switch (carrier) {
            case SHORT_ARRAY -> CpuLossPerformanceOracle.Carrier.SHORT_ARRAY;
            case FLOAT_ARRAY -> CpuLossPerformanceOracle.Carrier.FLOAT_ARRAY;
            case DOUBLE_ARRAY -> CpuLossPerformanceOracle.Carrier.DOUBLE_ARRAY;
            case INT_ARRAY -> CpuLossPerformanceOracle.Carrier.INT_ARRAY;
            case LONG_ARRAY -> CpuLossPerformanceOracle.Carrier.LONG_ARRAY;
            case MEMORY_SEGMENT -> CpuLossPerformanceOracle.Carrier.SEGMENT;
            default -> throw new AssertionError(carrier);
        };
    }

    private static void writeOracleEvidence(Path root) throws IOException {
        CpuLossPerformanceOracle current = oracle();
        Files.writeString(root.resolve("loss-performance-oracle.java"), current.source());
        Files.write(root.resolve("LossPerformanceOracleGenerated.class"), current.classBytes());
    }

    private static Object[] arguments(Object[] storage, List<CarrierAccess> carriers, long[] geometry,
            long start, long end) {
        Object[] values = new Object[storage.length + 3];
        for (int i = 0; i < storage.length; i++)
            values[i] = carriers.get(i) == CarrierAccess.MEMORY_SEGMENT ? segment(storage[i]) : storage[i];
        values[storage.length] = geometry;
        values[storage.length + 1] = start;
        values[storage.length + 2] = end;
        return values;
    }
    private static Object[] arrays(List<DataType> types) {
        Object[] storage = new Object[types.size()];
        for (int i = 0; i < storage.length; i++) storage[i] = array(types.get(i));
        return storage;
    }
    private static void populateInputs(Object[] storage, List<DataType> types, int inputCount,
            boolean ignore) {
        for (int input = 0; input < inputCount; input++) switch (types.get(input)) {
            case BFLOAT16 -> {
                short[] values = (short[]) storage[input];
                for (int i = 0; i < values.length; i++) values[i] = (short) (Float.floatToRawIntBits(
                        (i % 31) * .03125f) >>> 16);
            }
            case FLOAT32 -> {
                float[] values = (float[]) storage[input];
                for (int i = 0; i < values.length; i++) values[i] = (i % 31) * .03125f;
            }
            case FLOAT64 -> {
                double[] values = (double[]) storage[input];
                for (int i = 0; i < values.length; i++) values[i] = (i % 31) * .03125d;
            }
            case INT32 -> {
                int[] values = (int[]) storage[input];
                for (int i = 0; i < values.length; i++) values[i] = ignore && i == 0 ? -1 : i % 32;
            }
            case INT64 -> {
                long[] values = (long[]) storage[input];
                for (int i = 0; i < values.length; i++) values[i] = ignore && i == 0 ? -1L : i % 32;
            }
            default -> throw new AssertionError(types.get(input));
        }
    }
    private static Object array(DataType type) { return switch (type) { case BFLOAT16 -> new short[4096]; case FLOAT32 -> new float[4096]; case FLOAT64 -> new double[4096]; case INT32 -> new int[128]; case INT64 -> new long[128]; default -> throw new AssertionError(type); }; }
    private static MemorySegment segment(Object value) { return value instanceof short[] x ? MemorySegment.ofArray(x) : value instanceof float[] x ? MemorySegment.ofArray(x) : value instanceof double[] x ? MemorySegment.ofArray(x) : value instanceof int[] x ? MemorySegment.ofArray(x) : MemorySegment.ofArray((long[]) value); }
    private static Operation operation(Row row) { return switch (row.family()) { case MEAN_SQUARED_ERROR -> new Operation(row.family(), new MeanSquaredErrorAttrs(row.reduction())); case DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS -> new Operation(row.family(), new DenseCategoricalCrossEntropyWithLogitsAttrs(1, row.reduction())); case INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS -> new Operation(row.family(), new IndexCategoricalCrossEntropyWithLogitsAttrs(1, row.reduction(), row.ignore() ? Optional.of(row.right() == DataType.INT32 ? ScalarValue.int32(-1) : ScalarValue.int64(-1)) : Optional.empty())); default -> throw new AssertionError(row.family()); }; }
    private static TensorDescriptor descriptor(DataType type, Shape shape) { return CpuScatterLoweringTest.desc(type, shape); }
    private static List<CarrierAccess> carriers(List<DataType> types, int bits) { var result = new ArrayList<CarrierAccess>(); for (int i = 0; i < types.size(); i++) result.add((bits & (1 << i)) != 0 ? CarrierAccess.MEMORY_SEGMENT : carrier(types.get(i))); return List.copyOf(result); }
    private static CarrierAccess carrier(DataType type) { return switch (type) { case BFLOAT16 -> CarrierAccess.SHORT_ARRAY; case FLOAT32 -> CarrierAccess.FLOAT_ARRAY; case FLOAT64 -> CarrierAccess.DOUBLE_ARRAY; case INT32 -> CarrierAccess.INT_ARRAY; case INT64 -> CarrierAccess.LONG_ARRAY; default -> throw new AssertionError(type); }; }

    private record Row(LossKind family, DataType left, DataType right, LossReduction reduction, boolean ignore, List<Integer> roles, int bits) {
        DataType result() {
            return family == LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS
                    ? left : DataTypePromotion.promoteFloating(left, right);
        }

        /* CSV deliberately uses an identifier-safe role spelling.  The former List.toString()
           spelling contained a comma, turning one retained record into seven columns. */
        String key() {
            return family + "-" + left + "-" + right + "-" + reduction + "-" + ignore
                    + "-roles" + roles.getFirst() + '_' + roles.getLast() + "-" + bits;
        }
    }
    @FunctionalInterface private interface ForkLauncher { void launch(int fork) throws Exception; }
    private record RawRow(int ordinal, String key, double generatedToDirectRatio, long checksum) { }

    /**
     * Binds a typed output reader once for use by the symmetric timed sides.
     *
     * @param output non-null typed output carrier owned by one benchmark side
     * @param result non-null stored floating result type
     * @return exact non-null {@code long()} handle reading the carrier's first output value
     * @throws NoSuchMethodException if the fixed test-reader declaration is unavailable
     * @throws IllegalAccessException if this test class cannot resolve its private reader
     */
    private static MethodHandle checksumHandle(Object output, DataType result)
            throws NoSuchMethodException, IllegalAccessException {
        if (output instanceof double[] values) return outputReader("readDoubleArray", double[].class, values);
        if (output instanceof float[] values) return outputReader("readFloatArray", float[].class, values);
        if (output instanceof short[] values) return outputReader("readShortArray", short[].class, values);
        MemorySegment values = (MemorySegment) output;
        return switch (result) {
            case FLOAT64 -> outputReader("readDoubleSegment", MemorySegment.class, values);
            case FLOAT32 -> outputReader("readFloatSegment", MemorySegment.class, values);
            case BFLOAT16 -> outputReader("readShortSegment", MemorySegment.class, values);
            default -> throw new AssertionError(result);
        };
    }

    private static MethodHandle outputReader(String name, Class<?> carrier, Object output)
            throws NoSuchMethodException, IllegalAccessException {
        return MethodHandles.lookup().findStatic(CpuLossPerformanceTest.class, name,
                MethodType.methodType(long.class, carrier)).bindTo(output);
    }

    private static long readDoubleArray(double[] values) {
        return Double.doubleToRawLongBits(values[0]);
    }

    private static long readFloatArray(float[] values) {
        return Float.floatToRawIntBits(values[0]);
    }

    private static long readShortArray(short[] values) {
        return values[0];
    }

    private static long readDoubleSegment(MemorySegment values) {
        return Double.doubleToRawLongBits(values.get(ValueLayout.JAVA_DOUBLE_UNALIGNED, 0));
    }

    private static long readFloatSegment(MemorySegment values) {
        return Float.floatToRawIntBits(values.get(ValueLayout.JAVA_FLOAT_UNALIGNED, 0));
    }

    private static long readShortSegment(MemorySegment values) {
        return values.get(ValueLayout.JAVA_SHORT_UNALIGNED, 0);
    }

    private static Checksum checksum(Object output, DataType result) {
        if (output instanceof double[] values) return () -> Double.doubleToRawLongBits(values[0]);
        if (output instanceof float[] values) return () -> Float.floatToRawIntBits(values[0]);
        if (output instanceof short[] values) return () -> values[0];
        MemorySegment values = (MemorySegment) output;
        return switch (result) {
            case FLOAT64 -> () -> Double.doubleToRawLongBits(values.get(ValueLayout.JAVA_DOUBLE_UNALIGNED, 0));
            case FLOAT32 -> () -> Float.floatToRawIntBits(values.get(ValueLayout.JAVA_FLOAT_UNALIGNED, 0));
            case BFLOAT16 -> () -> values.get(ValueLayout.JAVA_SHORT_UNALIGNED, 0);
            default -> throw new AssertionError(result);
        };
    }
    @FunctionalInterface private interface Checksum { long read(); }
    private record TimedSide(MethodHandle entry, MethodHandle output, Object carrier) {
        TimedSide {
            if (!entry.type().equals(MethodType.methodType(void.class)))
                throw new IllegalArgumentException("entry must be exact void()");
            if (!output.type().equals(MethodType.methodType(long.class)))
                throw new IllegalArgumentException("output must be exact long()");
            if (carrier == null) throw new NullPointerException("carrier");
        }
    }
    private record Case(TimedSide generated, TimedSide direct, byte[] generatedClassBytes) { }
}
