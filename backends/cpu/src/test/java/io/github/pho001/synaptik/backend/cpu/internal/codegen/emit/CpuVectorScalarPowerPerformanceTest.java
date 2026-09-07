package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.ClassFile;
import java.lang.classfile.constantpool.MemberRefEntry;
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
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.stream.StreamSupport;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/** Opt-in immutable five-fork generated/direct gate for CPU 0008Q1A. */
class CpuVectorScalarPowerPerformanceTest {
    private static final String ENABLE = "synaptik.cpu.vectorScalarPower.performance";
    private static final String ROOT = "synaptik.cpu.vectorScalarPower.performanceEvidenceRoot";
    private static final int FORKS = 5, WARMUPS = 5, SAMPLES = 9;
    private static final int COUNT = (1 << 18) + 3;
    private static final long MINIMUM_NANOS = 25_000_000L;
    private static final long CALIBRATION_NANOS = 50_000_000L;
    private static final double LIMIT = 1.15d;
    private static final ByteOrder ORDER = ByteOrder.nativeOrder();
    private static final ValueLayout.OfFloat FLOAT =
            ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ORDER);
    private static final ValueLayout.OfDouble DOUBLE =
            ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ORDER);
    private static volatile long sink;

    enum Row {
        F32_ONE(DataType.FLOAT32, CpuKernelIr.PowerRealization.POSITIVE_ONE),
        F32_IDENTITY(DataType.FLOAT32, CpuKernelIr.PowerRealization.IDENTITY),
        F32_SQUARE(DataType.FLOAT32, CpuKernelIr.PowerRealization.SQUARE),
        F32_RECIPROCAL(DataType.FLOAT32, CpuKernelIr.PowerRealization.RECIPROCAL),
        F64_ONE(DataType.FLOAT64, CpuKernelIr.PowerRealization.POSITIVE_ONE),
        F64_IDENTITY(DataType.FLOAT64, CpuKernelIr.PowerRealization.IDENTITY),
        F64_SQUARE(DataType.FLOAT64, CpuKernelIr.PowerRealization.SQUARE),
        F64_RECIPROCAL(DataType.FLOAT64, CpuKernelIr.PowerRealization.RECIPROCAL);

        final DataType type;
        final CpuKernelIr.PowerRealization realization;
        Row(DataType type, CpuKernelIr.PowerRealization realization) {
            this.type = type;
            this.realization = realization;
        }
    }

    enum CarrierShape { AA, SS, AS, SA }

    private record Case(Row row, CarrierShape carrier) {
        String id() { return row + "-" + carrier; }
    }

    private static final List<Case> CASES = Arrays.stream(Row.values())
            .flatMap(row -> Arrays.stream(CarrierShape.values()).map(carrier -> new Case(row, carrier)))
            .toList();

    public static void main(String[] args) throws Throwable {
        if (args.length == 2 && args[0].equals("--fork")) fork(root(), Integer.parseInt(args[1]));
        else parent(root());
    }

    @Test void matrixAndProtocolAreFixed() {
        assertEquals(32, CASES.size());
        assertEquals(5, FORKS);
        assertEquals(5, WARMUPS);
        assertEquals(9, SAMPLES);
        assertEquals(25_000_000L, MINIMUM_NANOS);
        assertEquals(50_000_000L, CALIBRATION_NANOS);
        assertEquals(1.15d, LIMIT);
    }

    @Test void everyAggregateCarrierRowMatchesDirectBeforeTiming() throws Throwable {
        for (Case benchmarkCase : CASES) try (Work work = new Work(benchmarkCase)) {
            work.generated();
            long[] generated = work.outputBits();
            work.direct();
            assertArrayEquals(generated, work.outputBits(), benchmarkCase.id());
        }
    }

    @Test void retainedFiveFreshForkEvidence() throws Throwable {
        Assumptions.assumeTrue(Boolean.getBoolean(ENABLE));
        parent(root());
    }

    private static Path root() {
        String value = System.getProperty(ROOT);
        if (value == null || value.isBlank()) throw new IllegalArgumentException(ROOT + " is required");
        return Path.of(value).toAbsolutePath();
    }

    private static Path evidence(Path root) { return root.resolve("vector-scalar-power"); }

    private static void parent(Path root) throws Throwable {
        Path evidence = evidence(root);
        if (Files.exists(evidence) && Files.list(evidence).findAny().isPresent())
            throw new IllegalStateException("evidence directory must be absent or empty: " + evidence);
        Files.createDirectories(evidence.resolve("classes"));
        Files.createDirectories(evidence.resolve("javap"));
        Files.writeString(evidence.resolve("protocol.txt"),
                "rows=32;timed_carrier_shapes=AA,SS,AS,SA;"
                + "orchestration=VECTOR,PARALLEL_VECTOR;projection=orchestration-only-byte-identity\n"
                + "forks=5;warmups=5;samples=9;adaptive_minimum_ns=25000000;"
                + "calibration_target_ns=50000000\n"
                + "heap=-Xms1g,-Xmx1g;c2_only=true;randomized_order=true;retry=false;discard=false\n"
                + "threshold=1.15;count=" + COUNT + "\n");
        Files.writeString(evidence.resolve("environment.txt"),
                System.getProperties() + "\nvectorBitsF32="
                        + FloatVector.SPECIES_PREFERRED.vectorBitSize() + "\nvectorBitsF64="
                        + DoubleVector.SPECIES_PREFERRED.vectorBitSize() + "\n");
        Path source = Path.of("src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/"
                + "CpuVectorScalarPowerPerformanceTest.java").toAbsolutePath();
        Files.copy(source, evidence.resolve(source.getFileName()), StandardCopyOption.REPLACE_EXISTING);
        retainClasses(evidence);
        runJavap(evidence);
        String java = Path.of(System.getProperty("java.home"), "bin", "java").toString();
        StringBuilder commands = new StringBuilder();
        List<Integer> exitCodes = new ArrayList<>();
        for (int fork = 0; fork < FORKS; fork++) {
            List<String> command = List.of(java, "-Xms1g", "-Xmx1g", "-XX:-TieredCompilation",
                    "-Xbatch", "--add-modules", "jdk.incubator.vector", "-cp",
                    System.getProperty("java.class.path"), "-D" + ROOT + "=" + root,
                    CpuVectorScalarPowerPerformanceTest.class.getName(), "--fork",
                    Integer.toString(fork));
            commands.append(String.join(" ", command)).append('\n');
            Process process = new ProcessBuilder(command)
                    .redirectOutput(evidence.resolve("fork-" + fork + ".stdout").toFile())
                    .redirectError(evidence.resolve("fork-" + fork + ".stderr").toFile()).start();
            exitCodes.add(process.waitFor());
            validateFork(evidence.resolve("fork-" + fork + ".csv"), fork);
        }
        Files.writeString(evidence.resolve("commands.txt"), commands);
        aggregate(evidence);
        writeManifest(evidence);
        verifyManifest(evidence);
        for (int fork = 0; fork < FORKS; fork++) assertEquals(0, exitCodes.get(fork), "fork " + fork);
    }

    private static void fork(Path root, int fork) throws Throwable {
        if (fork < 0 || fork >= FORKS) throw new IllegalArgumentException("fork");
        Path evidence = evidence(root);
        Random random = new Random(0x8000_01aL + fork);
        StringBuilder result = new StringBuilder("row,iterations,median_ratio,checksum\n");
        for (Case benchmarkCase : CASES) try (Work work = new Work(benchmarkCase)) {
            work.generated();
            long[] expected = work.outputBits();
            work.direct();
            assertArrayEquals(expected, work.outputBits(), benchmarkCase.id());
            long expectedSink = expected[expected.length - 1];
            int iterations = calibrate(work, 1);
            for (int i = 0; i < WARMUPS; i++) pair(work, iterations, random);
            double[] ratios = new double[SAMPLES];
            StringBuilder samples = new StringBuilder(
                    "sample,iterations,generated_ns,direct_ns,ratio\n");
            for (int sample = 0; sample < SAMPLES; sample++) {
                Measurement measurement = pair(work, iterations, random);
                assertTrue(measurement.generated >= MINIMUM_NANOS);
                assertTrue(measurement.direct >= MINIMUM_NANOS);
                ratios[sample] = measurement.ratio();
                samples.append(sample).append(',').append(iterations).append(',')
                        .append(measurement.generated).append(',').append(measurement.direct)
                        .append(',').append(ratios[sample]).append('\n');
            }
            Files.writeString(evidence.resolve("samples-" + fork + "-" + benchmarkCase.id()
                    + ".csv"), samples);
            Arrays.sort(ratios);
            double median = ratios[SAMPLES / 2];
            result.append(benchmarkCase.id()).append(',').append(iterations).append(',')
                    .append(median).append(',').append(expectedSink).append('\n');
        }
        Files.writeString(evidence.resolve("fork-" + fork + ".csv"), result);
        for (String line : Files.readAllLines(evidence.resolve("fork-" + fork + ".csv")).subList(1,
                CASES.size() + 1)) {
            String[] columns = line.split(",", -1);
            assertTrue(Double.parseDouble(columns[2]) <= LIMIT,
                    columns[0] + " fork=" + fork + " median_ratio=" + columns[2]);
        }
    }

    private static int calibrate(Work work, int initial) throws Throwable {
        int iterations = initial;
        while (true) {
            long generated = timed(work::generated, iterations);
            long direct = timed(work::direct, iterations);
            if (generated >= CALIBRATION_NANOS && direct >= CALIBRATION_NANOS) return iterations;
            if (iterations > 1 << 20) throw new AssertionError("calibration did not converge");
            iterations *= 2;
        }
    }

    private static Measurement pair(Work work, int iterations, Random random) throws Throwable {
        long generated;
        long direct;
        if (random.nextBoolean()) {
            generated = timed(work::generated, iterations);
            direct = timed(work::direct, iterations);
        } else {
            direct = timed(work::direct, iterations);
            generated = timed(work::generated, iterations);
        }
        sink ^= work.lastBits();
        return new Measurement(generated, direct);
    }

    private static long timed(Throwing action, int iterations) throws Throwable {
        long start = System.nanoTime();
        for (int i = 0; i < iterations; i++) action.run();
        return System.nanoTime() - start;
    }

    private static void retainClasses(Path evidence) throws Exception {
        for (Row row : Row.values()) for (var carriers : carrierPairs(row.type)) {
            byte[] bytes = bytes(row, carriers);
            String name = row + "-" + pattern(row.type, carriers);
            Files.write(evidence.resolve("classes").resolve(name + ".class"), bytes);
            var model = ClassFile.of().parse(bytes);
            String members = StreamSupport.stream(model.constantPool().spliterator(), false)
                    .filter(MemberRefEntry.class::isInstance).map(MemberRefEntry.class::cast)
                    .map(member -> member.owner().asInternalName() + "." + member.name().stringValue()
                            + member.type().stringValue()).sorted().reduce("", (a, b) -> a + b + "\n");
            assertFalse(members.contains("io/github/pho001/synaptik"));
            Files.writeString(evidence.resolve("classes").resolve(name + ".members"), members);
            Files.writeString(evidence.resolve("classes").resolve(name + ".sha256"),
                    sha256(bytes) + "  " + name + ".class\n");
        }
    }

    private static void runJavap(Path evidence) throws Exception {
        String javap = Path.of(System.getProperty("java.home"), "bin", "javap").toString();
        for (Path path : Files.list(evidence.resolve("classes"))
                .filter(value -> value.toString().endsWith(".class")).sorted().toList()) {
            String name = path.getFileName().toString().replace(".class", "");
            run(List.of(javap, "-c", "-p", path.toString()), evidence.resolve("javap").resolve(name + ".c.txt"));
            run(List.of(javap, "-v", "-p", path.toString()), evidence.resolve("javap").resolve(name + ".v.txt"));
        }
        String classPath = System.getProperty("java.class.path");
        run(List.of(javap, "-c", "-p", "-classpath", classPath,
                CpuVectorScalarPowerPerformanceTest.class.getName()),
                evidence.resolve("javap").resolve("direct.c.txt"));
        run(List.of(javap, "-v", "-p", "-classpath", classPath,
                CpuVectorScalarPowerPerformanceTest.class.getName()),
                evidence.resolve("javap").resolve("direct.v.txt"));
    }

    private static void run(List<String> command, Path output) throws Exception {
        Process process = new ProcessBuilder(command).redirectErrorStream(true)
                .redirectOutput(output.toFile()).start();
        assertEquals(0, process.waitFor(), command.toString());
    }

    private static void validateFork(Path path, int fork) throws Exception {
        List<String> lines = Files.readAllLines(path);
        assertEquals(CASES.size() + 1, lines.size(), "fork " + fork);
        for (int i = 1; i < lines.size(); i++) {
            String[] columns = lines.get(i).split(",", -1);
            assertEquals(4, columns.length);
            assertEquals(CASES.get(i - 1).id(), columns[0]);
        }
    }

    private static void aggregate(Path evidence) throws Exception {
        StringBuilder summary = new StringBuilder("row,fork_ratios,median_of_fork_medians\n");
        for (int row = 0; row < CASES.size(); row++) {
            double[] ratios = new double[FORKS];
            for (int fork = 0; fork < FORKS; fork++) ratios[fork] = Double.parseDouble(
                    Files.readAllLines(evidence.resolve("fork-" + fork + ".csv"))
                            .get(row + 1).split(",")[2]);
            double[] sorted = ratios.clone();
            Arrays.sort(sorted);
            assertTrue(sorted[FORKS / 2] <= LIMIT);
            summary.append(CASES.get(row).id()).append(',').append(Arrays.toString(ratios).replace(',', ';'))
                    .append(',').append(sorted[FORKS / 2]).append('\n');
        }
        Files.writeString(evidence.resolve("summary.csv"), summary);
    }

    private static void writeManifest(Path evidence) throws Exception {
        List<Path> files = Files.walk(evidence).filter(Files::isRegularFile)
                .filter(path -> !path.getFileName().toString().equals("manifest.sha256"))
                .sorted(Comparator.comparing(path -> evidence.relativize(path).toString())).toList();
        StringBuilder manifest = new StringBuilder();
        for (Path file : files) manifest.append(sha256(Files.readAllBytes(file))).append("  ")
                .append(evidence.relativize(file)).append('\n');
        Files.writeString(evidence.resolve("manifest.sha256"), manifest);
        Files.writeString(evidence.resolve("seal.sha256"), sha256(manifest.toString().getBytes(
                StandardCharsets.UTF_8)) + "  manifest.sha256\n");
    }

    private static void verifyManifest(Path evidence) throws Exception {
        String manifest = Files.readString(evidence.resolve("manifest.sha256"));
        String seal = Files.readString(evidence.resolve("seal.sha256")).split("  ")[0];
        assertEquals(seal, sha256(manifest.getBytes(StandardCharsets.UTF_8)));
        for (String line : manifest.lines().toList()) {
            String[] columns = line.split("  ", 2);
            assertEquals(columns[0], sha256(Files.readAllBytes(evidence.resolve(columns[1]))));
        }
    }

    private record Measurement(long generated, long direct) {
        double ratio() { return (double) generated / direct; }
    }
    @FunctionalInterface private interface Throwing { void run() throws Throwable; }

    private static final class Work implements AutoCloseable {
        final Case benchmarkCase;
        final Arena arena = Arena.ofConfined();
        final Object inputArray, outputArrayA, outputArrayB;
        final MemorySegment inputSegment, outputSegmentA, outputSegmentB;
        final List<MethodHandle> handles = new ArrayList<>();
        final long[] geometry = geometry(2, COUNT);

        Work(Case benchmarkCase) {
            this.benchmarkCase = benchmarkCase;
            Row row = benchmarkCase.row();
            inputArray = row.type == DataType.FLOAT32 ? new float[COUNT] : new double[COUNT];
            inputSegment = arena.allocate((long) COUNT * row.type.byteWidth(), row.type.byteWidth());
            outputArrayA = row.type == DataType.FLOAT32 ? new float[COUNT] : new double[COUNT];
            outputArrayB = row.type == DataType.FLOAT32 ? new float[COUNT] : new double[COUNT];
            outputSegmentA = arena.allocate((long) COUNT * row.type.byteWidth(), row.type.byteWidth());
            outputSegmentB = arena.allocate((long) COUNT * row.type.byteWidth(), row.type.byteWidth());
            for (int i = 0; i < COUNT; i++) {
                double value = (i % 1009 - 504.5d) / 37.0d;
                write(row.type, inputArray, i, value);
                write(row.type, inputSegment, i, value);
            }
            handles.add(artifact(row, carriers(row.type, benchmarkCase.carrier())).entryPoint());
        }

        void generated() throws Throwable {
            boolean inputArrayCarrier = benchmarkCase.carrier() == CarrierShape.AA
                    || benchmarkCase.carrier() == CarrierShape.AS;
            boolean outputArrayCarrier = benchmarkCase.carrier() == CarrierShape.AA
                    || benchmarkCase.carrier() == CarrierShape.SA;
            handles.get(0).invokeWithArguments(inputArrayCarrier ? inputArray : inputSegment,
                    outputArrayCarrier ? outputArrayA : outputSegmentA, geometry, 0L, (long) COUNT);
        }

        void direct() {
            Row row = benchmarkCase.row();
            if (row.type == DataType.FLOAT32) directFloat(row.realization,
                    benchmarkCase.carrier(), (float[]) inputArray, inputSegment,
                    (float[]) outputArrayA, outputSegmentA);
            else directDouble(row.realization, benchmarkCase.carrier(), (double[]) inputArray,
                    inputSegment, (double[]) outputArrayA, outputSegmentA);
        }

        long lastBits() {
            int last = COUNT - 1;
            Row row = benchmarkCase.row();
            boolean outputArrayCarrier = benchmarkCase.carrier() == CarrierShape.AA
                    || benchmarkCase.carrier() == CarrierShape.SA;
            return bits(row.type, outputArrayCarrier ? outputArrayA : outputSegmentA, last);
        }

        long[] outputBits() {
            Row row = benchmarkCase.row();
            boolean outputArrayCarrier = benchmarkCase.carrier() == CarrierShape.AA
                    || benchmarkCase.carrier() == CarrierShape.SA;
            Object output = outputArrayCarrier ? outputArrayA : outputSegmentA;
            long[] result = new long[COUNT];
            for (int index = 0; index < COUNT; index++) {
                result[index] = bits(row.type, output, index);
            }
            return result;
        }

        @Override public void close() { arena.close(); }
    }

    private static void directFloat(CpuKernelIr.PowerRealization realization, CarrierShape carrier,
            float[] ia, MemorySegment is, float[] oa, MemorySegment os) {
        switch (realization) {
            case POSITIVE_ONE -> directFloatOne(carrier, oa, os);
            case IDENTITY -> directFloatIdentity(carrier, ia, is, oa, os);
            case SQUARE -> directFloatSquare(carrier, ia, is, oa, os);
            case RECIPROCAL -> directFloatReciprocal(carrier, ia, is, oa, os);
            case DIRECT -> throw new AssertionError();
        }
    }

    private static void directDouble(CpuKernelIr.PowerRealization realization, CarrierShape carrier,
            double[] ia, MemorySegment is, double[] oa, MemorySegment os) {
        switch (realization) {
            case POSITIVE_ONE -> directDoubleOne(carrier, oa, os);
            case IDENTITY -> directDoubleIdentity(carrier, ia, is, oa, os);
            case SQUARE -> directDoubleSquare(carrier, ia, is, oa, os);
            case RECIPROCAL -> directDoubleReciprocal(carrier, ia, is, oa, os);
            case DIRECT -> throw new AssertionError();
        }
    }

    private static void directFloatOne(CarrierShape carrier, float[] oa, MemorySegment os) {
        FloatVector one = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, 1.0f);
        if (carrier == CarrierShape.AA || carrier == CarrierShape.SA) {
            int bound = FloatVector.SPECIES_PREFERRED.loopBound(COUNT), i = 0;
            for (; i < bound; i += FloatVector.SPECIES_PREFERRED.length()) one.intoArray(oa, i);
            for (; i < COUNT; i++) oa[i] = 1.0f;
        } else {
            long bound = COUNT - COUNT % FloatVector.SPECIES_PREFERRED.length(), i = 0;
            for (; i < bound; i += FloatVector.SPECIES_PREFERRED.length())
                one.intoMemorySegment(os, i * 4, ORDER);
            for (; i < COUNT; i++) os.set(FLOAT, i * 4, 1.0f);
        }
    }

    private static void directFloatIdentity(CarrierShape carrier, float[] ia, MemorySegment is,
            float[] oa, MemorySegment os) {
        switch (carrier) {
            case AA -> { int i = 0, bound = FloatVector.SPECIES_PREFERRED.loopBound(COUNT); for (; i < bound; i += FloatVector.SPECIES_PREFERRED.length()) FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, ia, i).intoArray(oa, i); for (; i < COUNT; i++) oa[i] = ia[i]; }
            case SS -> { long i = 0, bound = COUNT - COUNT % FloatVector.SPECIES_PREFERRED.length(); for (; i < bound; i += FloatVector.SPECIES_PREFERRED.length()) FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, is, i * 4, ORDER).intoMemorySegment(os, i * 4, ORDER); for (; i < COUNT; i++) os.set(FLOAT, i * 4, is.get(FLOAT, i * 4)); }
            case AS -> { long i = 0, bound = COUNT - COUNT % FloatVector.SPECIES_PREFERRED.length(); for (; i < bound; i += FloatVector.SPECIES_PREFERRED.length()) FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, ia, (int) i).intoMemorySegment(os, i * 4, ORDER); for (; i < COUNT; i++) os.set(FLOAT, i * 4, ia[(int) i]); }
            case SA -> { long i = 0, bound = COUNT - COUNT % FloatVector.SPECIES_PREFERRED.length(); for (; i < bound; i += FloatVector.SPECIES_PREFERRED.length()) FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, is, i * 4, ORDER).intoArray(oa, (int) i); for (; i < COUNT; i++) oa[(int) i] = is.get(FLOAT, i * 4); }
        }
    }

    private static void directFloatSquare(CarrierShape carrier, float[] ia, MemorySegment is,
            float[] oa, MemorySegment os) {
        switch (carrier) {
            case AA -> { int i = 0, bound = FloatVector.SPECIES_PREFERRED.loopBound(COUNT); for (; i < bound; i += FloatVector.SPECIES_PREFERRED.length()) { var v = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, ia, i); v.mul(v).intoArray(oa, i); } for (; i < COUNT; i++) oa[i] = ia[i] * ia[i]; }
            case SS -> { long i = 0, bound = COUNT - COUNT % FloatVector.SPECIES_PREFERRED.length(); for (; i < bound; i += FloatVector.SPECIES_PREFERRED.length()) { var v = FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, is, i * 4, ORDER); v.mul(v).intoMemorySegment(os, i * 4, ORDER); } for (; i < COUNT; i++) { float v = is.get(FLOAT, i * 4); os.set(FLOAT, i * 4, v * v); } }
            case AS -> { long i = 0, bound = COUNT - COUNT % FloatVector.SPECIES_PREFERRED.length(); for (; i < bound; i += FloatVector.SPECIES_PREFERRED.length()) { var v = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, ia, (int) i); v.mul(v).intoMemorySegment(os, i * 4, ORDER); } for (; i < COUNT; i++) os.set(FLOAT, i * 4, ia[(int) i] * ia[(int) i]); }
            case SA -> { long i = 0, bound = COUNT - COUNT % FloatVector.SPECIES_PREFERRED.length(); for (; i < bound; i += FloatVector.SPECIES_PREFERRED.length()) { var v = FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, is, i * 4, ORDER); v.mul(v).intoArray(oa, (int) i); } for (; i < COUNT; i++) { float v = is.get(FLOAT, i * 4); oa[(int) i] = v * v; } }
        }
    }

    private static void directFloatReciprocal(CarrierShape carrier, float[] ia, MemorySegment is,
            float[] oa, MemorySegment os) {
        FloatVector one = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, 1.0f);
        switch (carrier) {
            case AA -> { int i = 0, bound = FloatVector.SPECIES_PREFERRED.loopBound(COUNT); for (; i < bound; i += FloatVector.SPECIES_PREFERRED.length()) one.div(FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, ia, i)).intoArray(oa, i); for (; i < COUNT; i++) oa[i] = 1.0f / ia[i]; }
            case SS -> { long i = 0, bound = COUNT - COUNT % FloatVector.SPECIES_PREFERRED.length(); for (; i < bound; i += FloatVector.SPECIES_PREFERRED.length()) one.div(FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, is, i * 4, ORDER)).intoMemorySegment(os, i * 4, ORDER); for (; i < COUNT; i++) os.set(FLOAT, i * 4, 1.0f / is.get(FLOAT, i * 4)); }
            case AS -> { long i = 0, bound = COUNT - COUNT % FloatVector.SPECIES_PREFERRED.length(); for (; i < bound; i += FloatVector.SPECIES_PREFERRED.length()) one.div(FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, ia, (int) i)).intoMemorySegment(os, i * 4, ORDER); for (; i < COUNT; i++) os.set(FLOAT, i * 4, 1.0f / ia[(int) i]); }
            case SA -> { long i = 0, bound = COUNT - COUNT % FloatVector.SPECIES_PREFERRED.length(); for (; i < bound; i += FloatVector.SPECIES_PREFERRED.length()) one.div(FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, is, i * 4, ORDER)).intoArray(oa, (int) i); for (; i < COUNT; i++) oa[(int) i] = 1.0f / is.get(FLOAT, i * 4); }
        }
    }

    private static void directDoubleOne(CarrierShape carrier, double[] oa, MemorySegment os) {
        DoubleVector one = DoubleVector.broadcast(DoubleVector.SPECIES_PREFERRED, 1.0d);
        if (carrier == CarrierShape.AA || carrier == CarrierShape.SA) {
            int bound = DoubleVector.SPECIES_PREFERRED.loopBound(COUNT), i = 0;
            for (; i < bound; i += DoubleVector.SPECIES_PREFERRED.length()) one.intoArray(oa, i);
            for (; i < COUNT; i++) oa[i] = 1.0d;
        } else {
            long bound = COUNT - COUNT % DoubleVector.SPECIES_PREFERRED.length(), i = 0;
            for (; i < bound; i += DoubleVector.SPECIES_PREFERRED.length()) one.intoMemorySegment(os, i * 8, ORDER);
            for (; i < COUNT; i++) os.set(DOUBLE, i * 8, 1.0d);
        }
    }

    private static void directDoubleIdentity(CarrierShape carrier, double[] ia, MemorySegment is,
            double[] oa, MemorySegment os) {
        switch (carrier) {
            case AA -> { int i = 0, bound = DoubleVector.SPECIES_PREFERRED.loopBound(COUNT); for (; i < bound; i += DoubleVector.SPECIES_PREFERRED.length()) DoubleVector.fromArray(DoubleVector.SPECIES_PREFERRED, ia, i).intoArray(oa, i); for (; i < COUNT; i++) oa[i] = ia[i]; }
            case SS -> { long i = 0, bound = COUNT - COUNT % DoubleVector.SPECIES_PREFERRED.length(); for (; i < bound; i += DoubleVector.SPECIES_PREFERRED.length()) DoubleVector.fromMemorySegment(DoubleVector.SPECIES_PREFERRED, is, i * 8, ORDER).intoMemorySegment(os, i * 8, ORDER); for (; i < COUNT; i++) os.set(DOUBLE, i * 8, is.get(DOUBLE, i * 8)); }
            case AS -> { long i = 0, bound = COUNT - COUNT % DoubleVector.SPECIES_PREFERRED.length(); for (; i < bound; i += DoubleVector.SPECIES_PREFERRED.length()) DoubleVector.fromArray(DoubleVector.SPECIES_PREFERRED, ia, (int) i).intoMemorySegment(os, i * 8, ORDER); for (; i < COUNT; i++) os.set(DOUBLE, i * 8, ia[(int) i]); }
            case SA -> { long i = 0, bound = COUNT - COUNT % DoubleVector.SPECIES_PREFERRED.length(); for (; i < bound; i += DoubleVector.SPECIES_PREFERRED.length()) DoubleVector.fromMemorySegment(DoubleVector.SPECIES_PREFERRED, is, i * 8, ORDER).intoArray(oa, (int) i); for (; i < COUNT; i++) oa[(int) i] = is.get(DOUBLE, i * 8); }
        }
    }

    private static void directDoubleSquare(CarrierShape carrier, double[] ia, MemorySegment is,
            double[] oa, MemorySegment os) {
        switch (carrier) {
            case AA -> { int i = 0, bound = DoubleVector.SPECIES_PREFERRED.loopBound(COUNT); for (; i < bound; i += DoubleVector.SPECIES_PREFERRED.length()) { var v = DoubleVector.fromArray(DoubleVector.SPECIES_PREFERRED, ia, i); v.mul(v).intoArray(oa, i); } for (; i < COUNT; i++) oa[i] = ia[i] * ia[i]; }
            case SS -> { long i = 0, bound = COUNT - COUNT % DoubleVector.SPECIES_PREFERRED.length(); for (; i < bound; i += DoubleVector.SPECIES_PREFERRED.length()) { var v = DoubleVector.fromMemorySegment(DoubleVector.SPECIES_PREFERRED, is, i * 8, ORDER); v.mul(v).intoMemorySegment(os, i * 8, ORDER); } for (; i < COUNT; i++) { double v = is.get(DOUBLE, i * 8); os.set(DOUBLE, i * 8, v * v); } }
            case AS -> { long i = 0, bound = COUNT - COUNT % DoubleVector.SPECIES_PREFERRED.length(); for (; i < bound; i += DoubleVector.SPECIES_PREFERRED.length()) { var v = DoubleVector.fromArray(DoubleVector.SPECIES_PREFERRED, ia, (int) i); v.mul(v).intoMemorySegment(os, i * 8, ORDER); } for (; i < COUNT; i++) os.set(DOUBLE, i * 8, ia[(int) i] * ia[(int) i]); }
            case SA -> { long i = 0, bound = COUNT - COUNT % DoubleVector.SPECIES_PREFERRED.length(); for (; i < bound; i += DoubleVector.SPECIES_PREFERRED.length()) { var v = DoubleVector.fromMemorySegment(DoubleVector.SPECIES_PREFERRED, is, i * 8, ORDER); v.mul(v).intoArray(oa, (int) i); } for (; i < COUNT; i++) { double v = is.get(DOUBLE, i * 8); oa[(int) i] = v * v; } }
        }
    }

    private static void directDoubleReciprocal(CarrierShape carrier, double[] ia, MemorySegment is,
            double[] oa, MemorySegment os) {
        DoubleVector one = DoubleVector.broadcast(DoubleVector.SPECIES_PREFERRED, 1.0d);
        switch (carrier) {
            case AA -> { int i = 0, bound = DoubleVector.SPECIES_PREFERRED.loopBound(COUNT); for (; i < bound; i += DoubleVector.SPECIES_PREFERRED.length()) one.div(DoubleVector.fromArray(DoubleVector.SPECIES_PREFERRED, ia, i)).intoArray(oa, i); for (; i < COUNT; i++) oa[i] = 1.0d / ia[i]; }
            case SS -> { long i = 0, bound = COUNT - COUNT % DoubleVector.SPECIES_PREFERRED.length(); for (; i < bound; i += DoubleVector.SPECIES_PREFERRED.length()) one.div(DoubleVector.fromMemorySegment(DoubleVector.SPECIES_PREFERRED, is, i * 8, ORDER)).intoMemorySegment(os, i * 8, ORDER); for (; i < COUNT; i++) os.set(DOUBLE, i * 8, 1.0d / is.get(DOUBLE, i * 8)); }
            case AS -> { long i = 0, bound = COUNT - COUNT % DoubleVector.SPECIES_PREFERRED.length(); for (; i < bound; i += DoubleVector.SPECIES_PREFERRED.length()) one.div(DoubleVector.fromArray(DoubleVector.SPECIES_PREFERRED, ia, (int) i)).intoMemorySegment(os, i * 8, ORDER); for (; i < COUNT; i++) os.set(DOUBLE, i * 8, 1.0d / ia[(int) i]); }
            case SA -> { long i = 0, bound = COUNT - COUNT % DoubleVector.SPECIES_PREFERRED.length(); for (; i < bound; i += DoubleVector.SPECIES_PREFERRED.length()) one.div(DoubleVector.fromMemorySegment(DoubleVector.SPECIES_PREFERRED, is, i * 8, ORDER)).intoArray(oa, (int) i); for (; i < COUNT; i++) oa[(int) i] = 1.0d / is.get(DOUBLE, i * 8); }
        }
    }

    private static CpuGeneratedKernel artifact(Row row,
            List<CpuKernelSpecialization.CarrierAccess> carriers) {
        CpuKernelIr ir = ir(row);
        var specialization = specialization(row, carriers, ir);
        var generator = new CpuClassFileKernelGenerator();
        return generator.defineClassBytes(specialization,
                generator.generateClassBytes(specialization, ir));
    }

    private static byte[] bytes(Row row, List<CpuKernelSpecialization.CarrierAccess> carriers) {
        CpuKernelIr ir = ir(row);
        return new CpuClassFileKernelGenerator().generateClassBytes(
                specialization(row, carriers, ir), ir);
    }

    private static CpuKernelSpecialization specialization(Row row,
            List<CpuKernelSpecialization.CarrierAccess> carriers, CpuKernelIr ir) {
        int species = row.type == DataType.FLOAT32 ? FloatVector.SPECIES_PREFERRED.vectorBitSize()
                : DoubleVector.SPECIES_PREFERRED.vectorBitSize();
        return new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR, List.of(row.type, row.type),
                carriers, species, -1, List.of(row.realization), false, 52);
    }

    private static CpuKernelIr ir(Row row) {
        long bits = switch (row.realization) {
            case POSITIVE_ONE -> 0L;
            case IDENTITY -> row.type == DataType.FLOAT32 ? 0x3f80_0000L : 0x3ff0_0000_0000_0000L;
            case SQUARE -> row.type == DataType.FLOAT32 ? 0x4000_0000L : 0x4000_0000_0000_0000L;
            case RECIPROCAL -> row.type == DataType.FLOAT32 ? 0xbf80_0000L : 0xbff0_0000_0000_0000L;
            case DIRECT -> throw new AssertionError();
        };
        return new CpuKernelIr(List.of(
                new CpuKernelIr.Value(0, row.type, CpuKernelIr.Value.Kind.INPUT,
                        dense(CpuAccessPlan.AccessKind.READ)),
                new CpuKernelIr.Value(1, row.type, CpuKernelIr.Value.Kind.OUTPUT,
                        dense(CpuAccessPlan.AccessKind.WRITE))),
                List.of(new CpuKernelIr.Instruction(CpuPointwiseOpcode.SCALAR_POW, List.of(0), 1,
                        new CpuKernelIr.ScalarImmediate(row.type, bits), row.realization)),
                new CpuKernelIr.Loop("start", "end"), List.of(new CpuKernelIr.Store(1, 0)));
    }

    private static CpuAccessPlan dense(CpuAccessPlan.AccessKind kind) {
        return new CpuAccessPlan(kind, CpuAccessPlan.Regime.DENSE_LINEAR, 1,
                List.of(CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
    }

    private static List<List<CpuKernelSpecialization.CarrierAccess>> carrierPairs(DataType type) {
        var array = type == DataType.FLOAT32 ? CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY
                : CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY;
        var segment = CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT;
        return List.of(List.of(array, array), List.of(segment, segment), List.of(array, segment),
                List.of(segment, array));
    }

    private static List<CpuKernelSpecialization.CarrierAccess> carriers(
            DataType type, CarrierShape shape) {
        return carrierPairs(type).get(shape.ordinal());
    }

    private static String pattern(DataType type,
            List<CpuKernelSpecialization.CarrierAccess> carriers) {
        var array = type == DataType.FLOAT32 ? CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY
                : CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY;
        return (carriers.get(0) == array ? "A" : "S") + (carriers.get(1) == array ? "A" : "S");
    }

    private static long[] geometry(int count, long extent) {
        long[] result = new long[2 + count + count + 2 * count];
        result[0] = extent;
        for (int i = 0; i < count; i++) {
            result[2 + count + i] = 1;
            result[2 + count + count + count + i] = extent;
        }
        return result;
    }

    private static void write(DataType type, Object carrier, int index, double value) {
        if (carrier instanceof float[] array) array[index] = (float) value;
        else if (carrier instanceof double[] array) array[index] = value;
        else if (type == DataType.FLOAT32) ((MemorySegment) carrier).set(FLOAT, (long) index * 4,
                (float) value);
        else ((MemorySegment) carrier).set(DOUBLE, (long) index * 8, value);
    }

    private static long bits(DataType type, Object carrier, int index) {
        if (carrier instanceof float[] array) return Integer.toUnsignedLong(
                Float.floatToRawIntBits(array[index]));
        if (carrier instanceof double[] array) return Double.doubleToRawLongBits(array[index]);
        if (type == DataType.FLOAT32) return Integer.toUnsignedLong(Float.floatToRawIntBits(
                ((MemorySegment) carrier).get(FLOAT, (long) index * 4)));
        return Double.doubleToRawLongBits(((MemorySegment) carrier).get(DOUBLE, (long) index * 8));
    }

    private static String sha256(byte[] bytes) throws Exception {
        return java.util.HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
    }
}
