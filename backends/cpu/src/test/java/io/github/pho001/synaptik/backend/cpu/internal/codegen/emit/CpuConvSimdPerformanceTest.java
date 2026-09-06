package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratorSchema;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuConv2dLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuConv3dLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionStrategy;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.classfile.ClassFile;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/**
 * Opt-in, fresh-VM Stage B protocol for the provisional schema-63 Conv SIMD candidate.
 *
 * <p>This owner intentionally contains the complete measurement contract rather than accepting
 * ad-hoc benchmark output: every retained pair is symmetric, calibrated with one shared count,
 * retained once, and independently checked before fork and cross-fork medians are accepted.</p>
 */
class CpuConvSimdPerformanceTest {
    private static final String ENABLE = "synaptik.cpu.convSimd.performance";
    private static final String ROOT = "synaptik.cpu.convSimd.performanceEvidenceRoot";
    private static final String RUN_ROOT = "synaptik.cpu.convSimd.performanceRunRoot";
    private static final int FORKS = 5, WARMUPS = 5, PAIRS = 9, ROWS = 16;
    private static final long CALIBRATION_NANOS = 50_000_000L, SIDE_NANOS = 25_000_000L;
    private static final double VECTOR_SCALAR_LIMIT = .90d, VECTOR_DIRECT_LIMIT = 1.15d;
    private static volatile long sink;

    enum Rank { CONV2D, CONV3D }
    enum Mode { FULL, DIAGNOSTIC }
    enum Branch { ARRAY_EXACT_VECTOR, SEGMENT_TAIL_OFFSET_PARALLEL_VECTOR,
        MIXED_PADDING_BORDER_VECTOR, GROUPED_DEPTHWISE_PARALLEL_VECTOR }
    enum Row {
        C2_F32_ARRAY(Rank.CONV2D, true, Branch.ARRAY_EXACT_VECTOR),
        C2_F32_SEGMENT(Rank.CONV2D, true, Branch.SEGMENT_TAIL_OFFSET_PARALLEL_VECTOR),
        C2_F32_MIXED(Rank.CONV2D, true, Branch.MIXED_PADDING_BORDER_VECTOR),
        C2_F32_GROUPED(Rank.CONV2D, true, Branch.GROUPED_DEPTHWISE_PARALLEL_VECTOR),
        C2_F64_ARRAY(Rank.CONV2D, false, Branch.ARRAY_EXACT_VECTOR),
        C2_F64_SEGMENT(Rank.CONV2D, false, Branch.SEGMENT_TAIL_OFFSET_PARALLEL_VECTOR),
        C2_F64_MIXED(Rank.CONV2D, false, Branch.MIXED_PADDING_BORDER_VECTOR),
        C2_F64_GROUPED(Rank.CONV2D, false, Branch.GROUPED_DEPTHWISE_PARALLEL_VECTOR),
        C3_F32_ARRAY(Rank.CONV3D, true, Branch.ARRAY_EXACT_VECTOR),
        C3_F32_SEGMENT(Rank.CONV3D, true, Branch.SEGMENT_TAIL_OFFSET_PARALLEL_VECTOR),
        C3_F32_MIXED(Rank.CONV3D, true, Branch.MIXED_PADDING_BORDER_VECTOR),
        C3_F32_GROUPED(Rank.CONV3D, true, Branch.GROUPED_DEPTHWISE_PARALLEL_VECTOR),
        C3_F64_ARRAY(Rank.CONV3D, false, Branch.ARRAY_EXACT_VECTOR),
        C3_F64_SEGMENT(Rank.CONV3D, false, Branch.SEGMENT_TAIL_OFFSET_PARALLEL_VECTOR),
        C3_F64_MIXED(Rank.CONV3D, false, Branch.MIXED_PADDING_BORDER_VECTOR),
        C3_F64_GROUPED(Rank.CONV3D, false, Branch.GROUPED_DEPTHWISE_PARALLEL_VECTOR);
        final Rank rank; final boolean f32; final Branch branch;
        Row(Rank rank, boolean f32, Branch branch) { this.rank = rank; this.f32 = f32; this.branch = branch; }
        DataType type() { return f32 ? DataType.FLOAT32 : DataType.FLOAT64; }
        boolean parallel() { return branch == Branch.SEGMENT_TAIL_OFFSET_PARALLEL_VECTOR
                || branch == Branch.GROUPED_DEPTHWISE_PARALLEL_VECTOR; }
        boolean padded() { return branch == Branch.MIXED_PADDING_BORDER_VECTOR; }
        boolean grouped() { return branch == Branch.GROUPED_DEPTHWISE_PARALLEL_VECTOR; }
        List<CarrierAccess> carriers() {
            CarrierAccess array = f32 ? CarrierAccess.FLOAT_ARRAY : CarrierAccess.DOUBLE_ARRAY;
            return switch (branch) {
                case ARRAY_EXACT_VECTOR, GROUPED_DEPTHWISE_PARALLEL_VECTOR ->
                    List.of(array, array, array);
                case SEGMENT_TAIL_OFFSET_PARALLEL_VECTOR -> List.of(
                    CarrierAccess.MEMORY_SEGMENT, CarrierAccess.MEMORY_SEGMENT,
                    CarrierAccess.MEMORY_SEGMENT);
                case MIXED_PADDING_BORDER_VECTOR -> List.of(
                    array, CarrierAccess.MEMORY_SEGMENT, CarrierAccess.MEMORY_SEGMENT);
            };
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length == 3 && args[0].equals("--fork")) {
            fork(Path.of(System.getProperty(RUN_ROOT)), Integer.parseInt(args[1]),
                    Mode.valueOf(args[2]));
        } else parent(root(), Mode.FULL);
    }

    @Test void lockedStageBInventoryAndIndependentOracleAreExecutable() {
        assertEquals(ROWS, Row.values().length);
        assertEquals(4, Arrays.stream(Row.values()).filter(r -> r.rank == Rank.CONV2D && r.f32).count());
        assertEquals(4, Arrays.stream(Row.values()).filter(r -> r.rank == Rank.CONV3D && !r.f32).count());
        for (Rank rank : Rank.values()) for (boolean f32 : List.of(false, true))
            for (Branch branch : Branch.values()) assertEquals(1,
                    Arrays.stream(Row.values()).filter(row -> row.rank == rank
                            && row.f32 == f32 && row.branch == branch).count());
        assertEquals(1440, ROWS * FORKS * PAIRS * 2);
        assertEquals(5760, ROWS * FORKS * PAIRS * 2 * 4);
        assertEquals(160, ROWS * FORKS * 2);
        assertEquals(32, ROWS * 2);
        assertEquals(5, WARMUPS); assertEquals(50_000_000L, CALIBRATION_NANOS);
        assertEquals(25_000_000L, SIDE_NANOS); assertEquals(.90d, VECTOR_SCALAR_LIMIT);
        String source = CpuConvSimdPerformanceOracle.source(specs());
        assertTrue(source.contains("SPECIES_PREFERRED") && source.contains("broadcast")
                && source.contains("while(ow<outputWidth"));
        assertTrue(source.contains("for(long batch=firstN")
                && source.contains("for(long oc=ocStart")
                && source.contains("for(long od=odStart")
                && source.contains("for(long oh=ohStart"));
        assertFalse(source.contains("long q=cell") || source.contains("cell%outputWidth"));
        assertOracleWidthLoopsDoNotDecode(source);
        assertFalse(source.contains("invokeWithArguments") || source.contains("reduceLanes"));
        byte[] bytes = CpuConvSimdPerformanceOracle.compile(specs());
        assertFalse(bytes.length == 0); assertTrue(ClassFile.of().verify(bytes).isEmpty());
    }

    @Test void semanticPreflightUsesEveryCarrierSignatureBeforeTiming() throws Throwable {
        for (Row row : Row.values()) try (OracleWork work = new OracleWork(row)) { work.verify(); }
    }

    @Test void evidencePreflightRetainsIndependentOracleAndGeneratedDossiers() throws Exception {
        Assumptions.assumeTrue(System.getProperty(ROOT) != null);
        Path out = root().resolve("cpu-conv-simd-performance")
                .resolve("preflight-" + System.currentTimeMillis()
                        + "-pid" + ProcessHandle.current().pid());
        Files.createDirectories(out);
        Files.writeString(out.resolve("protocol.txt"),
                "mode=PREFLIGHT\nfinal_protocol=false\ntiming=false\nrows=16\n");
        prepareEvidence(out, rows(Mode.FULL));
        manifest(out);
    }

    @Test void retainedFiveFreshForkEvidence() throws Exception {
        Assumptions.assumeTrue("true".equals(System.getProperty(ENABLE)));
        parent(root(), Mode.FULL);
    }

    @Test void diagnosticFreshForkRunsRepresentativeBranchesBeforeFullMode() throws Exception {
        Assumptions.assumeTrue("diagnostic".equals(System.getProperty(ENABLE)));
        parent(root(), Mode.DIAGNOSTIC);
    }

    private static Path root() {
        String value = System.getProperty(ROOT);
        if (value == null || value.isBlank()) throw new IllegalArgumentException(ROOT + " is required");
        Path root = Path.of(value).toAbsolutePath().normalize();
        Path checkout = Path.of("").toAbsolutePath().normalize();
        if (root.startsWith(checkout)) {
            throw new IllegalArgumentException(ROOT + " must be outside the checkout");
        }
        return root;
    }
    private static Path evidence(Path root, Mode mode) {
        return root.resolve("cpu-conv-simd-performance")
                .resolve(mode.name().toLowerCase() + "-" + System.currentTimeMillis()
                        + "-pid" + ProcessHandle.current().pid());
    }

    private static void parent(Path root, Mode mode) throws Exception {
        int forks = mode == Mode.FULL ? FORKS : 1;
        List<Row> rows = rows(mode);
        Path out = evidence(root, mode);
        Files.createDirectories(out);
        protocol(out, mode, rows.size(), forks);
        try {
            prepareEvidence(out, rows);
        } catch (Exception | AssertionError failure) {
            Files.writeString(out.resolve("stop.txt"), "status=FAILED\nmode=" + mode
                    + "\nstage=evidence-preflight\nfailure=" + failure + "\n");
            manifest(out);
            throw failure;
        }
        String java = Path.of(System.getProperty("java.home"), "bin", "java").toString(); StringBuilder commands = new StringBuilder();
        for (int fork = 0; fork < forks; fork++) {
            List<String> command = List.of(java, "-Xms1g", "-Xmx1g", "-XX:-TieredCompilation",
                    "-Xbatch", "--add-modules", "jdk.incubator.vector", "-cp",
                    System.getProperty("java.class.path"), "-D" + ROOT + '=' + root,
                    "-D" + RUN_ROOT + '=' + out, CpuConvSimdPerformanceTest.class.getName(),
                    "--fork", Integer.toString(fork), mode.name());
            commands.append(String.join(" ", command)).append('\n');
            Process process = new ProcessBuilder(command).redirectOutput(out.resolve("fork-" + fork + ".stdout").toFile()).redirectError(out.resolve("fork-" + fork + ".stderr").toFile()).start();
            int exit = process.waitFor();
            if (exit != 0) {
                Files.writeString(out.resolve("stop.txt"), "status=FAILED\nmode=" + mode
                        + "\nfirst_failed_fork=" + fork + "\nexit=" + exit + "\n");
                Files.writeString(out.resolve("commands.txt"), commands.toString());
                manifest(out);
                throw new AssertionError("fork " + fork + " failed; partial evidence retained at " + out);
            }
        }
        Files.writeString(out.resolve("commands.txt"), commands.toString());
        aggregate(out, mode, rows, forks);
        manifest(out);
    }

    private static void prepareEvidence(Path out, List<Row> rows) throws Exception {
        Files.copy(source(CpuConvSimdPerformanceTest.class), out.resolve("CpuConvSimdPerformanceTest.java"), StandardCopyOption.REPLACE_EXISTING);
        Files.copy(source(CpuConvSimdPerformanceOracle.class), out.resolve("CpuConvSimdPerformanceOracle.java"), StandardCopyOption.REPLACE_EXISTING);
        compileOracle(out, rows);
        for (Row row : rows) try (OracleWork work = new OracleWork(row)) {
            work.retainGeneratedDossier(out);
        }
    }

    private static void protocol(Path out, Mode mode, int rowCount, int forks) throws Exception {
        int pairCount = rowCount * forks * PAIRS * 2;
        Files.writeString(out.resolve("protocol.txt"), "mode=" + mode
                + "\nfinal_protocol=" + (mode == Mode.FULL) + "\nrows=" + rowCount
                + "\nforks=" + forks + "\nwarmup_pairs_vs=5\nwarmup_pairs_vd=5"
                + "\nretained_pairs_per_comparison=9\ncalibration_shared_doubling=true"
                + "\ncalibration_two_invocation_each_ns=50000000"
                + "\nretained_individual_side_ns=25000000\nretry=false\ndiscard=false"
                + "\nreplacement=false\nvs_limit=0.90\nvd_limit=1.15\npairs=" + pairCount
                + "\nsides=" + pairCount * 4 + "\nfork_medians=" + rowCount * forks * 2
                + "\naggregates=" + rowCount * 2
                + "\ndispatch=prebound_zero_argument_MethodHandle_invokeExact_equal_for_V_S_D"
                + "\ntimed_argument_allocation=false\n");
        Files.writeString(out.resolve("environment.txt"),
                "java=" + System.getProperty("java.version")
                        + "\nvm=" + System.getProperty("java.vm.name")
                        + "\nvm_version=" + System.getProperty("java.vm.version")
                        + "\nos=" + System.getProperty("os.name")
                        + "\nos_version=" + System.getProperty("os.version")
                        + "\narch=" + System.getProperty("os.arch")
                        + "\navailable_processors=" + Runtime.getRuntime().availableProcessors()
                        + "\npreferred_float_lanes="
                        + jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.length()
                        + "\npreferred_double_lanes="
                        + jdk.incubator.vector.DoubleVector.SPECIES_PREFERRED.length()
                        + "\nheap=-Xms1g,-Xmx1g\nc2=-XX:-TieredCompilation,-Xbatch\n");
    }

    private static List<Row> rows(Mode mode) {
        return mode == Mode.FULL ? List.of(Row.values()) : List.of(
                Row.C2_F32_ARRAY, Row.C2_F32_SEGMENT,
                Row.C2_F32_MIXED, Row.C2_F32_GROUPED);
    }

    private static void compileOracle(Path out, List<Row> rows) throws Exception {
        List<CpuConvSimdPerformanceOracle.Spec> selected = rows.stream()
                .map(CpuConvSimdPerformanceTest::oracleSpec).toList();
        String actualSource = CpuConvSimdPerformanceOracle.source(selected);
        assertOracleWidthLoopsDoNotDecode(actualSource);
        Path sourceDirectory = out.resolve("direct-source");
        Path classes = out.resolve("direct-classes");
        Files.createDirectories(sourceDirectory);
        Files.createDirectories(classes);
        Path source = sourceDirectory.resolve(CpuConvSimdPerformanceOracle.SIMPLE_NAME + ".java");
        Files.writeString(source, actualSource);
        String javac = Path.of(System.getProperty("java.home"), "bin", "javac").toString();
        List<String> command = List.of(javac, "--release", "26", "--add-modules",
                "jdk.incubator.vector", "-d", classes.toString(), source.toString());
        Process process = new ProcessBuilder(command).redirectErrorStream(true).start();
        Files.write(out.resolve("direct-javac.txt"), process.getInputStream().readAllBytes());
        if (process.waitFor() != 0) throw new AssertionError("independent direct-oracle javac failed");
        Path compiled = classes.resolve("io/github/pho001/synaptik/backend/cpu/internal/codegen/emit")
                .resolve(CpuConvSimdPerformanceOracle.SIMPLE_NAME + ".class");
        Path retained = out.resolve(CpuConvSimdPerformanceOracle.SIMPLE_NAME + ".class");
        Files.copy(compiled, retained, StandardCopyOption.REPLACE_EXISTING);
        Path javap = out.resolve("direct-oracle.javap.txt");
        javap(retained, javap);
        assertOracleJavapWidthLoops(Files.readString(javap), rows.size());
        Files.writeString(out.resolve("direct-javac-command.txt"), String.join(" ", command) + '\n');
    }

    private static void assertOracleWidthLoopsDoNotDecode(String source) {
        int cursor = 0;
        int loops = 0;
        while ((cursor = source.indexOf("while(ow<outputWidth", cursor)) >= 0) {
            int end = source.indexOf("\n}\n}\n", cursor);
            assertTrue(end > cursor, "oracle width loop source boundary");
            String widthLoop = source.substring(cursor, end);
            assertFalse(widthLoop.contains("geometry[") || widthLoop.contains("%")
                    || widthLoop.contains("/="), "oracle width loop must reuse decoded locals");
            loops++;
            cursor = end + 1;
        }
        assertTrue(loops > 0, "oracle must contain explicit width loops");
    }

    private static void assertOracleJavapWidthLoops(String javap, int methodCount) {
        int cursor = javap.indexOf("  public static void");
        assertTrue(cursor >= 0, "oracle javap method inventory");
        int inspected = 0;
        while (true) {
            int array = javap.indexOf("Vector.fromArray", cursor);
            int segment = javap.indexOf("Vector.fromMemorySegment", cursor);
            int vectorCall = array < 0 ? segment : segment < 0 ? array : Math.min(array, segment);
            if (vectorCall < 0) break;
            int nextMethod = javap.indexOf("  public static void", vectorCall);
            String vectorLoopSuffix = javap.substring(vectorCall,
                    nextMethod < 0 ? javap.length() : nextMethod);
            assertFalse(vectorLoopSuffix.contains(" lrem") || vectorLoopSuffix.contains(" ldiv")
                    || vectorLoopSuffix.contains(" laload"),
                    "oracle vector-loop suffix must contain no decode or geometry load");
            inspected++;
            cursor = nextMethod < 0 ? javap.length() : nextMethod;
            if (nextMethod < 0) break;
        }
        assertEquals(methodCount, inspected, "one inspected vector loop per direct method");
    }

    private static void fork(Path out, int fork, Mode mode) throws Exception {
        if (fork < 0 || fork >= FORKS) throw new IllegalArgumentException("fork");
        Files.createDirectories(out); Random random = new Random(0x0008_1B0L + fork);
        StringBuilder summary = new StringBuilder("row,comparison,iterations,fork_median\n");
        for (Row row : rows(mode)) try (OracleWork work = new OracleWork(row)) {
            try {
                work.verify();
                for (int warmup = 0; warmup < WARMUPS; warmup++) {
                    pair(work, 1, random, true); pair(work, 1, random, false);
                }
                int iterations = calibrate(out, fork, row, work);
                double[] vs = retain(out, fork, row, work, iterations, random, true);
                double[] vd = retain(out, fork, row, work, iterations, random, false);
                summary.append(row).append(",V/S,").append(iterations).append(',')
                        .append(median(vs)).append('\n');
                summary.append(row).append(",V/D,").append(iterations).append(',')
                        .append(median(vd)).append('\n');
                Files.writeString(out.resolve("raw-fork-" + fork + ".csv"), summary.toString());
            } catch (Exception | AssertionError failure) {
                Files.writeString(out.resolve("fork-" + fork + "-stop.txt"),
                        "row=" + row + "\nfailure=" + failure + "\n");
                Files.writeString(out.resolve("raw-fork-" + fork + ".csv"), summary.toString());
                throw failure;
            }
        }
        Files.writeString(out.resolve("raw-fork-" + fork + ".csv"), summary.toString());
    }

    private static int calibrate(Path out, int fork, Row row, OracleWork work) throws Exception {
        int iterations = 1;
        StringBuilder log = new StringBuilder("iterations,s_total,v_total,d_total\n");
        for (;;) {
            Measurement scalar = pair(work, iterations, new Random(0), true);
            Measurement direct = pair(work, iterations, new Random(1), false);
            log.append(iterations).append(',').append(scalar.otherTotal()).append(',')
                    .append(scalar.vectorTotal()).append(',').append(direct.otherTotal()).append('\n');
            if (scalar.otherTotal() >= CALIBRATION_NANOS
                    && scalar.vectorTotal() >= CALIBRATION_NANOS
                    && direct.otherTotal() >= CALIBRATION_NANOS) {
                Files.writeString(out.resolve("calibration-fork-" + fork + '-' + row + ".csv"),
                        log.toString());
                return iterations;
            }
            iterations = Math.multiplyExact(iterations, 2);
        }
    }
    private static double[] retain(Path out, int fork, Row row, OracleWork work, int n,
            Random random, boolean scalar) throws Exception {
        double[] ratios = new double[PAIRS];
        String comparison = scalar ? "vs" : "vd";
        StringBuilder raw = new StringBuilder(
                "pair,order,v_before,other_after,other_before,v_after,ratio\n");
        for (int pair = 0; pair < PAIRS; pair++) {
            Measurement measurement = pair(work, n, random, scalar);
            String record = "pair,order,v_before,other_after,other_before,v_after,ratio\n"
                    + pair + ',' + measurement.order + ',' + measurement.vb + ','
                    + measurement.oa + ',' + measurement.ob + ',' + measurement.va + ','
                    + measurement.ratio() + '\n';
            Files.writeString(out.resolve("pair-fork-" + fork + '-' + row + '-'
                    + comparison + '-' + pair + ".csv"), record);
            raw.append(record.substring(record.indexOf('\n') + 1));
            assertTrue(measurement.allAtLeast(SIDE_NANOS), row + " side below 25ms");
            double limit = scalar ? VECTOR_SCALAR_LIMIT : VECTOR_DIRECT_LIMIT;
            assertTrue(measurement.ratio() <= limit, row + " retained pair " + pair);
            ratios[pair] = measurement.ratio();
        }
        Files.writeString(out.resolve("samples-fork-" + fork + '-' + row + '-'
                + comparison + ".csv"), raw.toString());
        assertTrue(median(ratios) <= (scalar ? VECTOR_SCALAR_LIMIT : VECTOR_DIRECT_LIMIT));
        return ratios;
    }
    private static Measurement pair(OracleWork work, int iterations, Random random,
            boolean scalarComparison) throws Exception {
        boolean vectorFirst = random.nextBoolean();
        int other = scalarComparison ? 1 : 2;
        long vectorBefore;
        long otherAfter;
        long otherBefore;
        long vectorAfter;
        if (vectorFirst) {
            vectorBefore = time(work, 0, iterations);
            otherAfter = time(work, other, iterations);
            otherBefore = time(work, other, iterations);
            vectorAfter = time(work, 0, iterations);
        } else {
            otherBefore = time(work, other, iterations);
            vectorAfter = time(work, 0, iterations);
            vectorBefore = time(work, 0, iterations);
            otherAfter = time(work, other, iterations);
        }
        return new Measurement(vectorFirst ? "V-O-O-V" : "O-V-V-O", vectorBefore,
                otherAfter, otherBefore, vectorAfter, scalarComparison);
    }

    private static long time(OracleWork work, int implementation, int iterations) throws Exception {
        long start = System.nanoTime();
        for (int i = 0; i < iterations; i++) sink ^= work.run(implementation);
        return System.nanoTime() - start;
    }

    private static double median(double[] values) {
        double[] copy = values.clone();
        Arrays.sort(copy);
        return copy[copy.length / 2];
    }

    private record Measurement(String order, long vb, long oa, long ob, long va,
            boolean scalarComparison) {
        long vectorTotal() { return vb + va; }
        long otherTotal() { return oa + ob; }
        boolean allAtLeast(long minimum) {
            return vb >= minimum && oa >= minimum && ob >= minimum && va >= minimum;
        }
        double ratio() { return Math.sqrt((double) vb / oa * ((double) va / ob)); }
    }

    /** One exact row with distinct generated-vector, generated-scalar, and direct-oracle code. */
    private static final class OracleWork implements AutoCloseable {
        private static final int OUTPUT_SENTINEL = 0x7fa1_23bc;
        private static final long DOUBLE_OUTPUT_SENTINEL = 0x7ff4_1234_5678_9abCL;

        final Row row;
        final Arena arena = Arena.ofConfined();
        final CpuKernelSpecialization scalarSpecialization;
        final CpuKernelSpecialization vectorSpecialization;
        final Class<?> scalarClass;
        final Class<?> vectorClass;
        final Class<?> directClass;
        final byte[] scalarBytes;
        final byte[] vectorBytes;
        final BoundImplementation vector;
        final BoundImplementation scalar;
        final BoundImplementation direct;
        final long outputElements;
        final Object[] inputs;
        final byte[][] inputSnapshots;

        OracleWork(Row row) throws Exception {
            this.row = row;
            var plan = new CpuPartitionPreparer().analyze(context(row)).plan();
            var unit = plan.units().getFirst();
            var route = unit.portablePlan();
            scalarSpecialization = route.specialization();
            vectorSpecialization = provisionalVector(scalarSpecialization, row);
            assertEquals(ExecutionStrategy.SCALAR, scalarSpecialization.executionStrategy(),
                    row.name() + " generated scalar artifact");
            assertEquals(row.parallel() ? ExecutionStrategy.PARALLEL_SCALAR
                    : ExecutionStrategy.SCALAR, unit.executionStrategy(),
                    row.name() + " selected scalar plan");
            assertEquals(row.parallel() ? ExecutionStrategy.PARALLEL_VECTOR
                    : ExecutionStrategy.VECTOR, vectorSpecialization.executionStrategy(), row.name());
            assertEquals(row.carriers(), scalarSpecialization.carrierPattern(), row.name());

            int rank = row.rank == Rank.CONV2D ? 4 : 5;
            long[] geometry = row.rank == Rank.CONV2D
                    ? plan.conv2dGeometry().orElseThrow().pack(new long[3])
                    : unit.conv3dGeometry().orElseThrow().pack(new long[3]);
            outputElements = product(geometry, outputExtentStart(3, rank), rank);
            List<Range> ranges = ranges(row, outputElements);
            assertEquals(row.parallel() ? 4 : 1, ranges.size(), row + " range strategy");
            int outputWidthIndex = outputExtentStart(3, rank) + rank - 1;
            int lanes = row.f32 ? jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.length()
                    : jdk.incubator.vector.DoubleVector.SPECIES_PREFERRED.length();
            if (row.branch == Branch.ARRAY_EXACT_VECTOR) {
                assertEquals(0, geometry[outputWidthIndex] % lanes, row + " exact width");
            } else if (row.branch == Branch.SEGMENT_TAIL_OFFSET_PARALLEL_VECTOR) {
                assertNotEquals(0, geometry[outputWidthIndex] % lanes, row + " tail width");
                assertTrue(geometry[0] > 0 && geometry[1] > 0 && geometry[2] > 0,
                        row + " positive carrier offsets");
                assertTrue(ranges.getFirst().start() > 0
                        && ranges.getLast().end() < outputElements, row + " positive subrange");
            }
            inputs = allocateAndFillInputs(row, geometry, rank, arena);
            inputSnapshots = Arrays.stream(inputs).map(CpuConvSimdPerformanceTest::bytes)
                    .toArray(byte[][]::new);

            CpuClassFileKernelGenerator generator = new CpuClassFileKernelGenerator();
            scalarBytes = generator.generateClassBytes(scalarSpecialization, route.kernelIr());
            vectorBytes = generator.generateClassBytes(vectorSpecialization, route.kernelIr());
            var scalarArtifact = generator.defineClassBytes(scalarSpecialization, scalarBytes);
            var vectorArtifact = generator.defineClassBytes(vectorSpecialization, vectorBytes);
            scalarClass = scalarArtifact.hiddenClass();
            vectorClass = vectorArtifact.hiddenClass();

            CpuConvSimdPerformanceOracle.Spec oracleSpec = oracleSpec(row);
            String retainedRoot = System.getProperty(RUN_ROOT);
            directClass = retainedRoot == null ? define(
                    CpuConvSimdPerformanceOracle.compile(List.of(oracleSpec)))
                    : define(Files.readAllBytes(Path.of(retainedRoot)
                            .resolve(CpuConvSimdPerformanceOracle.SIMPLE_NAME + ".class")));
            MethodHandle directHandle = MethodHandles.lookup().findStatic(directClass, oracleSpec.name(),
                    scalarSpecialization.entryType());

            Object scalarOutput = allocateCarrier(row.carriers().getLast(),
                    carrierElements(geometry, 2, 3, rank), row, arena);
            Object vectorOutput = allocateCarrier(row.carriers().getLast(),
                    carrierElements(geometry, 2, 3, rank), row, arena);
            Object directOutput = allocateCarrier(row.carriers().getLast(),
                    carrierElements(geometry, 2, 3, rank), row, arena);
            fillOutput(scalarOutput, row);
            fillOutput(vectorOutput, row);
            fillOutput(directOutput, row);
            scalar = bind(row, scalarArtifact.entryPoint(), inputs, scalarOutput, geometry, ranges);
            vector = bind(row, vectorArtifact.entryPoint(), inputs, vectorOutput, geometry, ranges);
            direct = bind(row, directHandle, inputs, directOutput, geometry, ranges);
        }

        long run(int which) throws Exception {
            return switch (which) {
                case 0 -> vector.run();
                case 1 -> scalar.run();
                case 2 -> direct.run();
                default -> throw new IllegalArgumentException("implementation " + which);
            };
        }

        void verify() throws Exception {
            assertNotSame(vectorClass, scalarClass, row.name());
            assertNotSame(vectorClass, directClass, row.name());
            assertNotSame(scalarClass, directClass, row.name());
            assertNotEquals(vectorSpecialization.structuralKey(),
                    scalarSpecialization.structuralKey(), row.name());
            scalar.run();
            vector.run();
            direct.run();
            assertArrayEquals(bytes(scalar.output), bytes(vector.output), row + " V/S raw output");
            assertArrayEquals(bytes(scalar.output), bytes(direct.output), row + " V/D raw output");
            for (int boundary = 0; boundary < inputs.length; boundary++) {
                assertArrayEquals(inputSnapshots[boundary], bytes(inputs[boundary]),
                        row + " immutable input boundary " + boundary);
            }
            assertSentinels(row, scalar.output, scalar.ranges, scalar.geometry, outputElements);
            assertSentinels(row, vector.output, vector.ranges, vector.geometry, outputElements);
            assertSentinels(row, direct.output, direct.ranges, direct.geometry, outputElements);
        }

        void retainGeneratedDossier(Path out) throws Exception {
            Path directory = out.resolve("generated").resolve(row.name());
            Files.createDirectories(directory);
            retainGenerated(directory, "scalar", scalarSpecialization, scalarBytes);
            retainGenerated(directory, "vector", vectorSpecialization, vectorBytes);
        }

        @Override public void close() { arena.close(); }
    }

    private record Range(long start, long end) { }

    private static final class BoundImplementation {
        final MethodHandle[] calls;
        final Object output;
        final long[] geometry;
        final List<Range> ranges;
        final Row row;

        BoundImplementation(MethodHandle[] calls, Object output, long[] geometry,
                List<Range> ranges, Row row) {
            this.calls = calls;
            this.output = output;
            this.geometry = geometry;
            this.ranges = ranges;
            this.row = row;
        }

        long run() throws Exception {
            try {
                for (MethodHandle call : calls) call.invokeExact();
                return checksum(output, row);
            } catch (Throwable failure) {
                throw new Exception(failure);
            }
        }
    }

    private static BoundImplementation bind(Row row, MethodHandle handle, Object[] inputs,
            Object output, long[] geometry, List<Range> ranges) {
        MethodHandle[] calls = new MethodHandle[ranges.size()];
        for (int i = 0; i < ranges.size(); i++) {
            Range range = ranges.get(i);
            Object[] arguments = new Object[inputs.length + 4];
            System.arraycopy(inputs, 0, arguments, 0, inputs.length);
            arguments[inputs.length] = output;
            arguments[inputs.length + 1] = geometry;
            arguments[inputs.length + 2] = range.start();
            arguments[inputs.length + 3] = range.end();
            calls[i] = MethodHandles.insertArguments(handle, 0, arguments);
            assertEquals(MethodType.methodType(void.class), calls[i].type());
        }
        return new BoundImplementation(calls, output, geometry, ranges, row);
    }

    private static CpuKernelSpecialization provisionalVector(CpuKernelSpecialization scalar,
            Row row) {
        int bits = row.f32 ? jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.vectorBitSize()
                : jdk.incubator.vector.DoubleVector.SPECIES_PREFERRED.vectorBitSize();
        return new CpuKernelSpecialization(scalar.loweringFingerprint(), scalar.numericalMode(),
                row.parallel() ? ExecutionStrategy.PARALLEL_VECTOR : ExecutionStrategy.VECTOR,
                scalar.boundaryDataTypes(), scalar.carrierPattern(), bits, -1,
                scalar.scalarPowerRealizations(), false, 63);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> context(Row row) {
        return row.rank == Rank.CONV2D ? context2d(row) : context3d(row);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> context2d(Row row) {
        int lanes = row.f32 ? jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.length()
                : jdk.incubator.vector.DoubleVector.SPECIES_PREFERRED.length();
        long outputWidth = 8192L * lanes;
        if (row.branch == Branch.SEGMENT_TAIL_OFFSET_PARALLEL_VECTOR) outputWidth += lanes + 3L;
        if (row.padded()) outputWidth += 2L;
        int channels = row.grouped() ? 4 : 3;
        int outputChannels = row.grouped() ? 4 : 1;
        int groups = row.grouped() ? channels : 1;
        Shape input = Shape.of(1, channels, 1,
                row.padded() ? outputWidth : outputWidth + 2L);
        Shape weight = Shape.of(outputChannels, channels / groups, 1, 3);
        Shape output = Shape.of(1, outputChannels, 1, outputWidth);
        Conv2dAttrs attrs = new Conv2dAttrs(1, 1, 0, row.padded() ? 1 : 0, 1, 1, groups);
        List<LayoutDescriptor> layouts = row.branch
                == Branch.SEGMENT_TAIL_OFFSET_PARALLEL_VECTOR
                ? List.of(offset(input, 3), offset(weight, 5), offset(output, 7))
                : null;
        var base = CpuConv2dLoweringTest.context(List.of(row.type(), row.type()), input,
                weight, output, attrs, layouts);
        int rangeCount = row.parallel() ? 4 : 1;
        var inputs = new CpuPartitionAnalysisInputs(false, row.carriers(),
                new CpuPartitionAnalysisInputs.PortableExecutionConfig(
                        CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.SCALAR,
                        rangeCount, rangeCount, 1));
        return new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), base.constants(), inputs);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> context3d(Row row) {
        int lanes = row.f32 ? jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.length()
                : jdk.incubator.vector.DoubleVector.SPECIES_PREFERRED.length();
        long outputWidth = 8192L * lanes;
        if (row.branch == Branch.SEGMENT_TAIL_OFFSET_PARALLEL_VECTOR) outputWidth += lanes + 3L;
        if (row.padded()) outputWidth += 2L;
        int channels = row.grouped() ? 4 : 3;
        int outputChannels = row.grouped() ? 4 : 1;
        int groups = row.grouped() ? channels : 1;
        Shape input = Shape.of(1, channels, 1, 1,
                row.padded() ? outputWidth : outputWidth + 2L);
        Shape weight = Shape.of(outputChannels, channels / groups, 1, 1, 3);
        Shape output = Shape.of(1, outputChannels, 1, 1, outputWidth);
        Conv3dAttrs attrs = new Conv3dAttrs(1, 1, 1, 0, 0, row.padded() ? 1 : 0,
                1, 1, 1, groups);
        List<LayoutDescriptor> layouts = row.branch
                == Branch.SEGMENT_TAIL_OFFSET_PARALLEL_VECTOR
                ? List.of(offset(input, 3), offset(weight, 5), offset(output, 7))
                : null;
        var base = CpuConv3dLoweringTest.context(List.of(row.type(), row.type()), input,
                weight, output, attrs, layouts);
        int rangeCount = row.parallel() ? 4 : 1;
        var inputs = new CpuPartitionAnalysisInputs(false, row.carriers(),
                new CpuPartitionAnalysisInputs.PortableExecutionConfig(
                        CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.SCALAR,
                        rangeCount, rangeCount, 1));
        return new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), base.constants(), inputs);
    }

    private static LayoutDescriptor offset(Shape shape, long offset) {
        long[] extents = shape.toLongArray();
        long[] strides = new long[extents.length];
        long stride = 1;
        for (int axis = extents.length - 1; axis >= 0; axis--) {
            strides[axis] = stride;
            stride = Math.multiplyExact(stride, extents[axis]);
        }
        return LayoutDescriptor.of(shape, strides, offset, true);
    }

    private static List<Range> ranges(Row row, long count) {
        if (!row.parallel()) return List.of(new Range(0, count));
        long start = row.branch == Branch.SEGMENT_TAIL_OFFSET_PARALLEL_VECTOR ? 1 : 0;
        long end = row.branch == Branch.SEGMENT_TAIL_OFFSET_PARALLEL_VECTOR ? count - 1 : count;
        List<Range> result = new java.util.ArrayList<>(4);
        for (int part = 0; part < 4; part++) {
            long left = start + (end - start) * part / 4;
            long right = start + (end - start) * (part + 1) / 4;
            result.add(new Range(left, right));
        }
        return List.copyOf(result);
    }

    private static Object[] allocateAndFillInputs(Row row, long[] geometry, int rank, Arena arena) {
        Object[] result = new Object[2];
        for (int boundary = 0; boundary < result.length; boundary++) {
            long elements = carrierElements(geometry, boundary, 3, rank);
            result[boundary] = allocateCarrier(row.carriers().get(boundary), elements, row, arena);
            fillInput(result[boundary], row);
        }
        return result;
    }

    private static Object allocateCarrier(CarrierAccess access, long elements, Row row,
            Arena arena) {
        if (access == CarrierAccess.MEMORY_SEGMENT) {
            int bytes = row.f32 ? Float.BYTES : Double.BYTES;
            return arena.allocate(Math.multiplyExact(elements, bytes), bytes);
        }
        return row.f32 ? new float[Math.toIntExact(elements)]
                : new double[Math.toIntExact(elements)];
    }

    private static void fillInput(Object carrier, Row row) {
        long elements = carrier instanceof float[] values ? values.length
                : carrier instanceof double[] values ? values.length
                : ((MemorySegment) carrier).byteSize() / (row.f32 ? Float.BYTES : Double.BYTES);
        for (long index = 0; index < elements; index++) {
            double value = (index % 29L - 14L) * .03125;
            put(carrier, row, index, value);
        }
    }

    private static void fillOutput(Object carrier, Row row) {
        long elements = carrier instanceof float[] values ? values.length
                : carrier instanceof double[] values ? values.length
                : ((MemorySegment) carrier).byteSize() / (row.f32 ? Float.BYTES : Double.BYTES);
        for (long index = 0; index < elements; index++) {
            if (row.f32) {
                float value = Float.intBitsToFloat(OracleWork.OUTPUT_SENTINEL);
                if (carrier instanceof float[] values) values[Math.toIntExact(index)] = value;
                else ((MemorySegment) carrier).setAtIndex(
                        ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.nativeOrder()),
                        index, value);
            } else {
                double value = Double.longBitsToDouble(OracleWork.DOUBLE_OUTPUT_SENTINEL);
                if (carrier instanceof double[] values) values[Math.toIntExact(index)] = value;
                else ((MemorySegment) carrier).setAtIndex(
                        ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.nativeOrder()),
                        index, value);
            }
        }
    }

    private static void put(Object value, Row row, long index, double x) {
        if (value instanceof float[] values) values[Math.toIntExact(index)] = (float) x;
        else if (value instanceof double[] values) values[Math.toIntExact(index)] = x;
        else if (row.f32) ((MemorySegment) value).setAtIndex(
                ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.nativeOrder()), index, (float) x);
        else ((MemorySegment) value).setAtIndex(
                ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.nativeOrder()), index, x);
    }

    private static long carrierElements(long[] geometry, int boundary, int boundaries, int rank) {
        int extentStart = boundaries + boundary * rank * 2;
        long last = geometry[boundary];
        for (int axis = 0; axis < rank; axis++) {
            last = Math.addExact(last, Math.multiplyExact(geometry[extentStart + axis] - 1,
                    geometry[extentStart + rank + axis]));
        }
        return Math.addExact(last, 1);
    }

    private static int outputExtentStart(int boundaries, int rank) {
        return boundaries + (boundaries - 1) * rank * 2;
    }

    private static long product(long[] values, int start, int count) {
        long result = 1;
        for (int i = 0; i < count; i++) result = Math.multiplyExact(result, values[start + i]);
        return result;
    }

    private static byte[] bytes(Object carrier) {
        if (carrier instanceof float[] values) return MemorySegment.ofArray(values)
                .toArray(ValueLayout.JAVA_BYTE);
        if (carrier instanceof double[] values) return MemorySegment.ofArray(values)
                .toArray(ValueLayout.JAVA_BYTE);
        return ((MemorySegment) carrier).toArray(ValueLayout.JAVA_BYTE);
    }

    private static long checksum(Object carrier, Row row) {
        long elements = carrier instanceof float[] values ? values.length
                : carrier instanceof double[] values ? values.length
                : ((MemorySegment) carrier).byteSize() / (row.f32 ? Float.BYTES : Double.BYTES);
        long result = 1;
        long step = Math.max(1L, elements / 67L);
        for (long index = 0; index < elements; index += step) {
            result = 31 * result + (row.f32
                    ? Float.floatToRawIntBits(readFloat(carrier, index))
                    : Double.doubleToRawLongBits(readDouble(carrier, index)));
        }
        return result;
    }

    private static void assertSentinels(Row row, Object output, List<Range> ranges,
            long[] geometry, long outputElements) {
        boolean[] written = new boolean[Math.toIntExact(outputElements)];
        for (Range range : ranges) {
            for (long ordinal = range.start(); ordinal < range.end(); ordinal++) {
                written[Math.toIntExact(ordinal)] = true;
            }
        }
        long base = geometry[2];
        for (int ordinal = 0; ordinal < written.length; ordinal++) {
            if (written[ordinal]) continue;
            long bits = row.f32
                    ? Float.floatToRawIntBits(readFloat(output, base + ordinal))
                    : Double.doubleToRawLongBits(readDouble(output, base + ordinal));
            long expected = row.f32 ? Integer.toUnsignedLong(OracleWork.OUTPUT_SENTINEL)
                    : OracleWork.DOUBLE_OUTPUT_SENTINEL;
            if (row.f32) bits = Integer.toUnsignedLong((int) bits);
            assertEquals(expected, bits, row + " sentinel " + ordinal);
        }
    }

    private static float readFloat(Object carrier, long index) {
        return carrier instanceof float[] values ? values[Math.toIntExact(index)]
                : ((MemorySegment) carrier).getAtIndex(
                        ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.nativeOrder()), index);
    }

    private static double readDouble(Object carrier, long index) {
        return carrier instanceof double[] values ? values[Math.toIntExact(index)]
                : ((MemorySegment) carrier).getAtIndex(
                        ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.nativeOrder()), index);
    }
    private static Class<?> define(byte[] bytes) { return new OracleLoader().define(bytes); }
    private static final class OracleLoader extends ClassLoader {
        OracleLoader() { super(CpuConvSimdPerformanceTest.class.getClassLoader()); }
        Class<?> define(byte[] bytes) { return defineClass(null, bytes, 0, bytes.length); }
    }
    private static CpuConvSimdPerformanceOracle.Spec oracleSpec(Row row) {
        List<CarrierAccess> carriers = row.carriers();
        return new CpuConvSimdPerformanceOracle.Spec(row.name().toLowerCase(),
                row.rank == Rank.CONV3D, row.f32,
                carriers.get(0) == CarrierAccess.MEMORY_SEGMENT,
                carriers.get(1) == CarrierAccess.MEMORY_SEGMENT,
                carriers.get(2) == CarrierAccess.MEMORY_SEGMENT,
                row.padded() ? 1L : 0L, row.grouped() ? 4L : 1L);
    }
    private static List<CpuConvSimdPerformanceOracle.Spec> specs() {
        return Arrays.stream(Row.values()).map(CpuConvSimdPerformanceTest::oracleSpec).toList();
    }
    private static void aggregate(Path out, Mode mode, List<Row> rows, int forks) throws Exception {
        StringBuilder result = new StringBuilder("row,comparison,median_of_fork_medians\n");
        for (Row row : rows) for (String comparison : List.of("V/S", "V/D")) {
            double[] values = new double[forks];
            for (int fork = 0; fork < forks; fork++) {
                String[] lines = Files.readString(out.resolve("raw-fork-" + fork + ".csv"))
                        .split("\\R");
                int matches = 0;
                for (String line : lines) {
                    String[] columns = line.split(",", -1);
                    if (columns.length == 4 && columns[0].equals(row.name())
                            && columns[1].equals(comparison)) {
                        matches++;
                        values[fork] = Double.parseDouble(columns[3]);
                    }
                }
                assertEquals(1, matches, row + " " + comparison + " fork " + fork);
                assertTrue(Double.isFinite(values[fork]) && values[fork] > 0.0,
                        row + " " + comparison + " finite positive median");
            }
            double aggregate = median(values);
            assertTrue(aggregate <= (comparison.equals("V/S")
                    ? VECTOR_SCALAR_LIMIT : VECTOR_DIRECT_LIMIT));
            result.append(row).append(',').append(comparison).append(',')
                    .append(aggregate).append('\n');
        }
        Files.writeString(out.resolve(mode == Mode.FULL ? "aggregate.csv"
                : "diagnostic-aggregate.csv"), result.toString());
    }

    private static void retainGenerated(Path directory, String form,
            CpuKernelSpecialization specialization, byte[] classBytes) throws Exception {
        Path classFile = directory.resolve(form + ".class");
        Files.write(classFile, classBytes);
        Files.writeString(directory.resolve(form + ".identity.txt"),
                "schema=" + specialization.classIdentitySchema()
                        + "\nkey=" + specialization.structuralKey()
                        + "\nbinaryName=" + CpuGeneratorSchema.generatedBinaryName(specialization)
                        + "\ndescriptor=" + specialization.entryType().descriptorString() + "\n");
        javap(classFile, directory.resolve(form + ".javap.txt"));
    }
    private static void javap(Path clazz, Path target) throws Exception { Process p=new ProcessBuilder(Path.of(System.getProperty("java.home"),"bin","javap").toString(),"-c","-v","-p",clazz.toString()).redirectErrorStream(true).start(); Files.write(target,p.getInputStream().readAllBytes()); if(p.waitFor()!=0)throw new AssertionError("javap"); }
    private static void manifest(Path out) throws Exception { MessageDigest sha=MessageDigest.getInstance("SHA-256"); StringBuilder lines=new StringBuilder(); try(var paths=Files.walk(out)){for(Path path:paths.filter(Files::isRegularFile).sorted(Comparator.naturalOrder()).toList())if(!path.getFileName().toString().equals("manifest.sha256"))lines.append(java.util.HexFormat.of().formatHex(sha.digest(Files.readAllBytes(path)))).append("  ").append(out.relativize(path)).append('\n');} Files.writeString(out.resolve("manifest.sha256"),lines.toString()); }
    private static Path source(Class<?> type) {
        String relative = "io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/"
                + type.getSimpleName() + ".java";
        for (Path directory = Path.of(System.getProperty("user.dir")).toAbsolutePath();
                directory != null; directory = directory.getParent()) {
            Path moduleSource = directory.resolve("src/test/java").resolve(relative);
            if (Files.isRegularFile(moduleSource)) return moduleSource;
            Path repositorySource = directory.resolve("backends/cpu/src/test/java").resolve(relative);
            if (Files.isRegularFile(repositorySource)) return repositorySource;
        }
        throw new IllegalStateException("source not found for " + type.getName());
    }
}
