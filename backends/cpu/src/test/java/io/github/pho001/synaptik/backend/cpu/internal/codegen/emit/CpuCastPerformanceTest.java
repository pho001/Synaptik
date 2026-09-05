package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/** Opt-in retained generated-versus-direct performance gate for CPU 0008K CAST. */
class CpuCastPerformanceTest {
    private static final String ENABLE = "synaptik.cpu.cast.performance";
    private static final String ROOT = "synaptik.cpu.cast.performanceEvidenceRoot";
    private static final int FORKS = 5, WARMUPS = 32, SAMPLES = 9, CHUNKS = 64, COUNT = 65_536;
    private static final int COMPILE_STABILIZATION_INVOCATIONS = 12_000;
    private static final long MINIMUM_NANOS = 25_000_000L;
    /*
     * The mandatory per-sample limit makes a short scheduling or frequency excursion part of the
     * result; it cannot be hidden by the fork median. Retained attempts showed roughly 20 ms of
     * opposite-side excursion in an otherwise stable dense row, enough to dominate a 100 ms
     * sample. Calibrating each side to at least 250 ms and measuring symmetric generated/direct/
     * direct/generated (or reverse) blocks gives both sides the same mean temporal position. The
     * geometric mean of the two directional ratios cancels multiplicative linear drift in log
     * time while retaining every timed invocation and every sample.
     */
    private static final long CALIBRATION_MINIMUM_NANOS = 250_000_000L;
    private static final double LIMIT = 1.15d;
    private static volatile long sink;

    enum Row {
        DENSE_ARRAY_F64_TO_BF16, GENERAL_MIXED_I64_TO_BF16, OFFSET_DENSE_MIXED_F64_TO_I64,
        BLOCK_OUTER_BF16_TO_F64, DENSE_SEGMENT_I64_TO_F32, GENERAL_ARRAY_F32_TO_BOOL,
        DENSE_MIXED_BOOL_TO_F64, SAME_TYPE_RAW_IDENTITY, ROUNDING_SENSITIVE_TWO_CAST_CHAIN
    }

    public static void main(String[] args) throws Throwable {
        if (args.length == 2 && args[0].equals("--fork")) runFork(root(), Integer.parseInt(args[1]));
        else runParent(root());
    }

    @Test void retainedFiveForkEvidence() throws Exception {
        Assumptions.assumeTrue(Boolean.getBoolean(ENABLE));
        runParent(root());
    }

    @Test void exactNineRowInventory() { assertEquals(9, Row.values().length); }

    @Test void retainedProtocolKeepsStrictLongPairedSamples() {
        assertEquals(5, FORKS);
        assertEquals(9, SAMPLES);
        assertEquals(64, CHUNKS);
        assertEquals(0, CHUNKS % 2);
        assertEquals(12_000, COMPILE_STABILIZATION_INVOCATIONS);
        assertEquals(250_000_000L, CALIBRATION_MINIMUM_NANOS);
        assertTrue(CALIBRATION_MINIMUM_NANOS >= 10L * MINIMUM_NANOS);
        assertEquals(1.15d, LIMIT);
    }

    @Test void symmetricEstimatorCancelsMultiplicativeLinearDrift() {
        // True generated/direct cost is 1.10. Successive positions slow by 2x: G,D,D,G.
        assertEquals(1.10d, symmetricRatio(110L, 200L, 880L, 400L), 1.0e-12d);
    }

    @Test void directBinary64Bfloat16UsesRneAndCanonicalNaN() {
        assertEquals(0x7fc0, Short.toUnsignedInt(bf16(Double.longBitsToDouble(0xfff0_0000_0000_0001L))));
        assertEquals(0x3f80, Short.toUnsignedInt(bf16(1.0d + Math.scalb(1.0d, -8))));
        assertEquals(0x3f82, Short.toUnsignedInt(bf16(1.0d + Math.scalb(3.0d, -8))));
        assertEquals(0, Short.toUnsignedInt(bf16(Math.scalb(1.0d, -134))));
        assertEquals(2, Short.toUnsignedInt(bf16(Math.scalb(3.0d, -134))));
        assertEquals(0x7f80, Short.toUnsignedInt(bf16(Double.MAX_VALUE)));
        assertEquals(0xff80, Short.toUnsignedInt(bf16(Double.NEGATIVE_INFINITY)));
    }

    @Test void twoCastChainCanonicalizesBinary64NaNAtFloat32Boundary() {
        float intermediate = narrowFloat(Double.longBitsToDouble(0xfff0_0000_0000_0001L));
        assertEquals(0x7fc00000, Float.floatToRawIntBits(intermediate));
        assertEquals(0x7fc0, Short.toUnsignedInt(BFloat16Bits.fromFloat(intermediate)));
    }

    @Test void directSidesMatchGeneratedBeforeTiming() throws Throwable {
        for (Row row : Row.values()) verify(row, prepare(row));
    }

    private static Path root() {
        String value = System.getProperty(ROOT);
        if (value == null || value.isBlank()) throw new IllegalArgumentException(ROOT + " is required");
        return Path.of(value).toAbsolutePath();
    }

    private static void runParent(Path root) throws Exception {
        Path evidence = performanceRoot(root);
        Files.createDirectories(evidence);
        Files.writeString(evidence.resolve("protocol.txt"), "forks=" + FORKS + "\nwarmups=" + WARMUPS
                + "\nmeasurements=" + SAMPLES + "\nchunks=" + CHUNKS + "\n"
                + "minimum_side_ns=25000000\ncalibration_minimum_side_ns="
                + CALIBRATION_MINIMUM_NANOS + "\ncompile_stabilization_invocations="
                + COMPILE_STABILIZATION_INVOCATIONS
                + "\norder=seeded-randomized-symmetric-ab-ba\nestimator=geometric-mean-directional-ratios\nretry=false\n"
                + "discard=false\nfixed_heap=-Xms1g,-Xmx1g\nc2_only=true\nthreshold=1.15\nrows="
                + Arrays.toString(Row.values()) + "\n");
        Files.copy(sourceFile(), evidence.resolve("CpuCastPerformanceTest.java"),
                StandardCopyOption.REPLACE_EXISTING);
        Files.writeString(evidence.resolve("environment.txt"), System.getProperties().toString());
        String java = Path.of(System.getProperty("java.home"), "bin", "java").toString();
        StringBuilder commands = new StringBuilder();
        for (int fork = 0; fork < FORKS; fork++) {
            List<String> command = List.of(java, "-Xms1g", "-Xmx1g", "-XX:-TieredCompilation",
                    "-Xbatch", "--add-modules", "jdk.incubator.vector", "-cp",
                    System.getProperty("java.class.path"), "-D" + ROOT + "=" + root,
                    CpuCastPerformanceTest.class.getName(), "--fork", Integer.toString(fork));
            commands.append(String.join(" ", command)).append('\n');
            Process process = new ProcessBuilder(command).redirectOutput(evidence.resolve("fork-" + fork
                    + ".stdout").toFile()).redirectError(evidence.resolve("fork-" + fork + ".stderr").toFile()).start();
            assertEquals(0, process.waitFor(), "fork " + fork);
            validateFork(evidence.resolve("raw-fork-" + fork + ".csv"), fork);
        }
        Files.writeString(evidence.resolve("commands.txt"), commands);
        aggregate(evidence); manifest(evidence);
    }

    private static void runFork(Path root, int fork) throws Throwable {
        if (fork < 0 || fork >= FORKS) throw new IllegalArgumentException("fork");
        Random random = new Random(0x0008_000b_cafeL + fork);
        StringBuilder raw = new StringBuilder("row,iterations,median_ratio,checksum\n");
        for (Row row : Row.values()) {
            Prepared p = prepare(row); verify(row, p);
            int iterations = calibrate(p);
            assertTrue(iterations >= CHUNKS && iterations % CHUNKS == 0,
                    row + " calibration must assign equal positive work to every paired chunk");
            stabilizeMeasurementActions(p);
            for (int warmup = 0; warmup < WARMUPS; warmup++) measurePair(p, iterations, random);
            double[] ratios = new double[SAMPLES];
            Measurement[] measurements = new Measurement[SAMPLES];
            StringBuilder samples = new StringBuilder("row,sample,iterations,generated_ns,direct_ns,"
                    + "generated_before_ns,direct_after_ns,generated_after_ns,direct_before_ns,ratio\n");
            for (int sample = 0; sample < SAMPLES; sample++) {
                Measurement measured = measurePair(p, iterations, random);
                measurements[sample] = measured;
                ratios[sample] = measured.ratio();
                samples.append(row).append(',').append(sample).append(',').append(iterations).append(',')
                        .append(measured.generatedNanos()).append(',').append(measured.directNanos()).append(',')
                        .append(measured.generatedBeforeNanos()).append(',').append(measured.directAfterNanos()).append(',')
                        .append(measured.generatedAfterNanos()).append(',').append(measured.directBeforeNanos()).append(',')
                        .append(ratios[sample]).append('\n');
            }
            Files.writeString(performanceRoot(root).resolve("measurements-fork-" + fork + "-" + row + ".csv"), samples);
            for (int sample = 0; sample < SAMPLES; sample++)
                assertTrue(measurements[sample].generatedNanos() >= MINIMUM_NANOS
                                && measurements[sample].directNanos() >= MINIMUM_NANOS,
                        row + " sample below minimum duration");
            for (int sample = 0; sample < SAMPLES; sample++)
                assertTrue(ratios[sample] <= LIMIT, row + " fork " + fork + " sample " + sample
                        + "=" + ratios[sample]);
            Arrays.sort(ratios); assertTrue(ratios[4] <= LIMIT, row + " fork " + fork + "=" + ratios[4]);
            raw.append(row).append(',').append(iterations).append(',').append(ratios[4]).append(',').append(sink).append('\n');
        }
        Files.writeString(performanceRoot(root).resolve("raw-fork-" + fork + ".csv"), raw);
    }

    private static Path performanceRoot(Path root) { return root.resolve("cpu-cast-performance"); }

    private static int calibrate(Prepared p) throws Throwable {
        int count = CHUNKS;
        while (true) { Measurement pair = measurePair(p, count, new Random(1));
            if (pair.generatedNanos() >= CALIBRATION_MINIMUM_NANOS
                    && pair.directNanos() >= CALIBRATION_MINIMUM_NANOS) return count;
            count = Math.multiplyExact(count, 2);
        }
    }

    private static void stabilizeMeasurementActions(Prepared p) throws Throwable {
        Invocation generatedAction = p.generatedAction;
        Invocation directAction = p.directAction;
        for (int invocation = 0; invocation < COMPILE_STABILIZATION_INVOCATIONS; invocation++) {
            generatedAction.elapsed(0);
            directAction.elapsed(0);
        }
    }

    private static Measurement measurePair(Prepared p, int count, Random random) throws Throwable {
        Invocation generatedAction = p.generatedAction;
        Invocation directAction = p.directAction;
        long generatedBeforeNanos = 0L, directAfterNanos = 0L;
        long generatedAfterNanos = 0L, directBeforeNanos = 0L;
        int baseChunkCount = count / CHUNKS, remainder = count % CHUNKS;
        for (int block = 0; block < CHUNKS / 2; block++) {
            int firstChunk = block * 2;
            int firstCount = baseChunkCount + (firstChunk < remainder ? 1 : 0);
            int secondCount = baseChunkCount + (firstChunk + 1 < remainder ? 1 : 0);
            if (random.nextBoolean()) {
                generatedBeforeNanos += generatedAction.elapsed(firstCount);
                directAfterNanos += directAction.elapsed(firstCount);
                directBeforeNanos += directAction.elapsed(secondCount);
                generatedAfterNanos += generatedAction.elapsed(secondCount);
            } else {
                directBeforeNanos += directAction.elapsed(firstCount);
                generatedAfterNanos += generatedAction.elapsed(firstCount);
                generatedBeforeNanos += generatedAction.elapsed(secondCount);
                directAfterNanos += directAction.elapsed(secondCount);
            }
        }
        return new Measurement(generatedBeforeNanos, directAfterNanos,
                generatedAfterNanos, directBeforeNanos);
    }

    private static double symmetricRatio(long generatedBeforeNanos, long directAfterNanos,
                                         long generatedAfterNanos, long directBeforeNanos) {
        return Math.sqrt(((double) generatedBeforeNanos / directAfterNanos)
                * ((double) generatedAfterNanos / directBeforeNanos));
    }

    private static void generated(Row row, Prepared p) throws Throwable {
        switch (row) {
            case DENSE_ARRAY_F64_TO_BF16, ROUNDING_SENSITIVE_TWO_CAST_CHAIN ->
                    p.entry.invokeExact((double[]) p.input, (short[]) p.output, p.geometry, 0L,
                            (long) COUNT);
            case GENERAL_MIXED_I64_TO_BF16 -> p.entry.invokeExact((long[]) p.input,
                    (MemorySegment) p.output, p.geometry, 0L, (long) COUNT);
            case OFFSET_DENSE_MIXED_F64_TO_I64 -> p.entry.invokeExact((double[]) p.input,
                    (MemorySegment) p.output, p.geometry, 0L, (long) COUNT);
            case BLOCK_OUTER_BF16_TO_F64 -> p.entry.invokeExact((short[]) p.input,
                    (double[]) p.output, p.geometry, 0L, (long) COUNT);
            case DENSE_SEGMENT_I64_TO_F32 -> p.entry.invokeExact((MemorySegment) p.input,
                    (MemorySegment) p.output, p.geometry, 0L, (long) COUNT);
            case GENERAL_ARRAY_F32_TO_BOOL -> p.entry.invokeExact((float[]) p.input,
                    (byte[]) p.output, p.geometry, 0L, (long) COUNT);
            case DENSE_MIXED_BOOL_TO_F64 -> p.entry.invokeExact((byte[]) p.input,
                    (MemorySegment) p.output, p.geometry, 0L, (long) COUNT);
            case SAME_TYPE_RAW_IDENTITY -> p.entry.invokeExact((float[]) p.input,
                    (float[]) p.output, p.geometry, 0L, (long) COUNT);
        }
        sink += observe(p);
    }

    private static void direct(Row row, Prepared p) {
        long[] g = p.geometry;
        switch (row) {
            case DENSE_ARRAY_F64_TO_BF16 -> { double[] a=(double[])p.input; short[] o=(short[])p.output; for(int i=0;i<COUNT;i++)o[(int)g[3]+i]=bf16(a[(int)g[2]+i]); }
            case GENERAL_MIXED_I64_TO_BF16 -> directGeneralI64ToBfloat((long[]) p.input,
                    (MemorySegment) p.output, g);
            case OFFSET_DENSE_MIXED_F64_TO_I64 -> { double[] a=(double[])p.input; MemorySegment o=(MemorySegment)p.output; for(int i=0;i<COUNT;i++)o.setAtIndex(longLayout(),g[3]+i,toLong(a[(int)g[2]+i])); }
            case BLOCK_OUTER_BF16_TO_F64 -> directBlockOuterBfloatToF64Kernel((short[]) p.input,
                    (double[]) p.output, g);
            case DENSE_SEGMENT_I64_TO_F32 -> { MemorySegment a=(MemorySegment)p.input,o=(MemorySegment)p.output;for(int i=0;i<COUNT;i++)o.setAtIndex(floatLayout(),g[3]+i,(float)a.getAtIndex(longLayout(),g[2]+i)); }
            case GENERAL_ARRAY_F32_TO_BOOL -> directGeneralF32ToBool((float[]) p.input,
                    (byte[]) p.output, g);
            case DENSE_MIXED_BOOL_TO_F64 -> { byte[] a=(byte[])p.input;MemorySegment o=(MemorySegment)p.output;for(int i=0;i<COUNT;i++)o.setAtIndex(doubleLayout(),g[3]+i,a[(int)g[2]+i]==0?0d:1d); }
            case SAME_TYPE_RAW_IDENTITY -> {
                float[] a = (float[]) p.input, o = (float[]) p.output;
                int inputAddress = (int) g[2], outputAddress = (int) g[3];
                int end = COUNT, index = 0;
                do {
                    o[outputAddress] = a[inputAddress];
                    inputAddress++;
                    outputAddress++;
                    index++;
                } while (index < end);
            }
            case ROUNDING_SENSITIVE_TWO_CAST_CHAIN -> {
                double[] a = (double[]) p.input;
                short[] o = (short[]) p.output;
                int inputAddress = (int) g[2], outputAddress = (int) g[3];
                int end = COUNT, index = 0;
                do {
                    float intermediate = narrowFloat(a[inputAddress]);
                    o[outputAddress] = BFloat16Bits.fromFloat(intermediate);
                    inputAddress++;
                    outputAddress++;
                    index++;
                } while (index < end);
            }
        } sink += observe(p);
    }

    /**
     * Narrows one binary64 value with the Model-required canonical NaN result for this lossy
     * FLOAT64-to-FLOAT32 boundary.
     *
     * @param value the binary64 input value
     * @return the RNE binary32 value, or positive canonical quiet NaN for a binary64 NaN
     */
    private static float narrowFloat(double value) {
        long bits = Double.doubleToRawLongBits(value);
        boolean nan = (bits & 0x7ff0_0000_0000_0000L) == 0x7ff0_0000_0000_0000L
                && (bits & 0x000f_ffff_ffff_ffffL) != 0L;
        return nan ? Float.intBitsToFloat(0x7fc0_0000) : (float) value;
    }

    private static Prepared prepare(Row row) {
        DataType source = source(row), target = target(row); CpuKernelIr ir = ir(row, source, target);
        List<CpuKernelIr.Value> boundary = ir.values().stream().filter(v -> v.kind()!=CpuKernelIr.Value.Kind.VIRTUAL).toList();
        var specialization = new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT, CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,
                boundary.stream().map(CpuKernelIr.Value::dataType).toList(), carriers(row), 0, -1,
                List.of(), false, source == target ? 52 : 60);
        var generator = new CpuClassFileKernelGenerator(); MethodHandle entry;
        try { entry = generator.defineClassBytes(specialization, generator.generateClassBytes(specialization, ir)).entryPoint(); }
        catch (IllegalArgumentException failure) { throw new IllegalArgumentException(row.toString(), failure); }
        Object input = input(row), output = output(row);
        return bind(row, entry, input, output, geometry(row));
    }

    private static Prepared bind(Row row, MethodHandle entry, Object input, Object output, long[] geometry) {
        return switch (row) {
            case DENSE_ARRAY_F64_TO_BF16 -> new Prepared(entry, input, output, geometry,
                    count -> elapsedGeneratedDenseArrayF64ToBfloat(count, entry, (double[]) input, (short[]) output, geometry),
                    count -> elapsedDirectDenseArrayF64ToBfloat(count, (double[]) input, (short[]) output, geometry));
            case GENERAL_MIXED_I64_TO_BF16 -> new Prepared(entry, input, output, geometry,
                    count -> elapsedGeneratedGeneralMixedI64ToBfloat(count, entry, (long[]) input, (MemorySegment) output, geometry),
                    count -> elapsedDirectGeneralMixedI64ToBfloat(count, (long[]) input, (MemorySegment) output, geometry));
            case OFFSET_DENSE_MIXED_F64_TO_I64 -> new Prepared(entry, input, output, geometry,
                    count -> elapsedGeneratedOffsetDenseMixedF64ToI64(count, entry, (double[]) input, (MemorySegment) output, geometry),
                    count -> elapsedDirectOffsetDenseMixedF64ToI64(count, (double[]) input, (MemorySegment) output, geometry));
            case BLOCK_OUTER_BF16_TO_F64 -> new Prepared(entry, input, output, geometry,
                    count -> elapsedGeneratedBlockOuterBfloatToF64(count, entry, (short[]) input, (double[]) output, geometry),
                    count -> elapsedDirectBlockOuterBfloatToF64(count, (short[]) input, (double[]) output, geometry));
            case DENSE_SEGMENT_I64_TO_F32 -> new Prepared(entry, input, output, geometry,
                    count -> elapsedGeneratedDenseSegmentI64ToF32(count, entry, (MemorySegment) input, (MemorySegment) output, geometry),
                    count -> elapsedDirectDenseSegmentI64ToF32(count, (MemorySegment) input, (MemorySegment) output, geometry));
            case GENERAL_ARRAY_F32_TO_BOOL -> new Prepared(entry, input, output, geometry,
                    count -> elapsedGeneratedGeneralArrayF32ToBool(count, entry, (float[]) input, (byte[]) output, geometry),
                    count -> elapsedDirectGeneralArrayF32ToBool(count, (float[]) input, (byte[]) output, geometry));
            case DENSE_MIXED_BOOL_TO_F64 -> new Prepared(entry, input, output, geometry,
                    count -> elapsedGeneratedDenseMixedBoolToF64(count, entry, (byte[]) input, (MemorySegment) output, geometry),
                    count -> elapsedDirectDenseMixedBoolToF64(count, (byte[]) input, (MemorySegment) output, geometry));
            case SAME_TYPE_RAW_IDENTITY -> new Prepared(entry, input, output, geometry,
                    count -> elapsedGeneratedSameTypeRawIdentity(count, entry, (float[]) input, (float[]) output, geometry),
                    count -> elapsedDirectSameTypeRawIdentity(count, (float[]) input, (float[]) output, geometry));
            case ROUNDING_SENSITIVE_TWO_CAST_CHAIN -> new Prepared(entry, input, output, geometry,
                    count -> elapsedGeneratedRoundingSensitiveTwoCastChain(count, entry, (double[]) input, (short[]) output, geometry),
                    count -> elapsedDirectRoundingSensitiveTwoCastChain(count, (double[]) input, (short[]) output, geometry));
        };
    }

    private static long elapsedGeneratedDenseArrayF64ToBfloat(int count, MethodHandle e, double[] i, short[] o, long[] g) throws Throwable { long s=System.nanoTime(); for(int n=0;n<count;n++)generatedDenseArrayF64ToBfloat(e,i,o,g); return System.nanoTime()-s; }
    private static long elapsedDirectDenseArrayF64ToBfloat(int count, double[] i, short[] o, long[] g) { long s=System.nanoTime(); for(int n=0;n<count;n++)directDenseArrayF64ToBfloat(i,o,g); return System.nanoTime()-s; }
    private static long elapsedGeneratedGeneralMixedI64ToBfloat(int count, MethodHandle e, long[] i, MemorySegment o, long[] g) throws Throwable { long s=System.nanoTime(); for(int n=0;n<count;n++)generatedGeneralMixedI64ToBfloat(e,i,o,g); return System.nanoTime()-s; }
    private static long elapsedDirectGeneralMixedI64ToBfloat(int count, long[] i, MemorySegment o, long[] g) { long s=System.nanoTime(); for(int n=0;n<count;n++)directGeneralMixedI64ToBfloat(i,o,g); return System.nanoTime()-s; }
    private static long elapsedGeneratedOffsetDenseMixedF64ToI64(int count, MethodHandle e, double[] i, MemorySegment o, long[] g) throws Throwable { long s=System.nanoTime(); for(int n=0;n<count;n++)generatedOffsetDenseMixedF64ToI64(e,i,o,g); return System.nanoTime()-s; }
    private static long elapsedDirectOffsetDenseMixedF64ToI64(int count, double[] i, MemorySegment o, long[] g) { long s=System.nanoTime(); for(int n=0;n<count;n++)directOffsetDenseMixedF64ToI64(i,o,g); return System.nanoTime()-s; }
    private static long elapsedGeneratedBlockOuterBfloatToF64(int count, MethodHandle e, short[] i, double[] o, long[] g) throws Throwable { long s=System.nanoTime(); for(int n=0;n<count;n++)generatedBlockOuterBfloatToF64(e,i,o,g); return System.nanoTime()-s; }
    private static long elapsedDirectBlockOuterBfloatToF64(int count, short[] i, double[] o, long[] g) { long s=System.nanoTime(); for(int n=0;n<count;n++)directBlockOuterBfloatToF64(i,o,g); return System.nanoTime()-s; }
    private static long elapsedGeneratedDenseSegmentI64ToF32(int count, MethodHandle e, MemorySegment i, MemorySegment o, long[] g) throws Throwable { long s=System.nanoTime(); for(int n=0;n<count;n++)generatedDenseSegmentI64ToF32(e,i,o,g); return System.nanoTime()-s; }
    private static long elapsedDirectDenseSegmentI64ToF32(int count, MemorySegment i, MemorySegment o, long[] g) { long s=System.nanoTime(); for(int n=0;n<count;n++)directDenseSegmentI64ToF32(i,o,g); return System.nanoTime()-s; }
    private static long elapsedGeneratedGeneralArrayF32ToBool(int count, MethodHandle e, float[] i, byte[] o, long[] g) throws Throwable { long s=System.nanoTime(); for(int n=0;n<count;n++)generatedGeneralArrayF32ToBool(e,i,o,g); return System.nanoTime()-s; }
    private static long elapsedDirectGeneralArrayF32ToBool(int count, float[] i, byte[] o, long[] g) { long s=System.nanoTime(); for(int n=0;n<count;n++)directGeneralArrayF32ToBool(i,o,g); return System.nanoTime()-s; }
    private static long elapsedGeneratedDenseMixedBoolToF64(int count, MethodHandle e, byte[] i, MemorySegment o, long[] g) throws Throwable { long s=System.nanoTime(); for(int n=0;n<count;n++)generatedDenseMixedBoolToF64(e,i,o,g); return System.nanoTime()-s; }
    private static long elapsedDirectDenseMixedBoolToF64(int count, byte[] i, MemorySegment o, long[] g) { long s=System.nanoTime(); for(int n=0;n<count;n++)directDenseMixedBoolToF64(i,o,g); return System.nanoTime()-s; }
    private static long elapsedGeneratedSameTypeRawIdentity(int count, MethodHandle e, float[] i, float[] o, long[] g) throws Throwable { long s=System.nanoTime(); for(int n=0;n<count;n++)generatedSameTypeRawIdentity(e,i,o,g); return System.nanoTime()-s; }
    private static long elapsedDirectSameTypeRawIdentity(int count, float[] i, float[] o, long[] g) { long s=System.nanoTime(); for(int n=0;n<count;n++)directSameTypeRawIdentity(i,o,g); return System.nanoTime()-s; }
    private static long elapsedGeneratedRoundingSensitiveTwoCastChain(int count, MethodHandle e, double[] i, short[] o, long[] g) throws Throwable { long s=System.nanoTime(); for(int n=0;n<count;n++)generatedRoundingSensitiveTwoCastChain(e,i,o,g); return System.nanoTime()-s; }
    private static long elapsedDirectRoundingSensitiveTwoCastChain(int count, double[] i, short[] o, long[] g) { long s=System.nanoTime(); for(int n=0;n<count;n++)directRoundingSensitiveTwoCastChain(i,o,g); return System.nanoTime()-s; }

    private static void generatedDenseArrayF64ToBfloat(MethodHandle entry, double[] input, short[] output, long[] geometry) throws Throwable { entry.invokeExact(input, output, geometry, 0L, (long) COUNT); sink += output[0]; }
    private static void generatedGeneralMixedI64ToBfloat(MethodHandle entry, long[] input, MemorySegment output, long[] geometry) throws Throwable { entry.invokeExact(input, output, geometry, 0L, (long) COUNT); sink += output.getAtIndex(shortLayout(), 0); }
    private static void generatedOffsetDenseMixedF64ToI64(MethodHandle entry, double[] input, MemorySegment output, long[] geometry) throws Throwable { entry.invokeExact(input, output, geometry, 0L, (long) COUNT); sink += output.getAtIndex(longLayout(), 0); }
    private static void generatedBlockOuterBfloatToF64(MethodHandle entry, short[] input, double[] output, long[] geometry) throws Throwable { entry.invokeExact(input, output, geometry, 0L, (long) COUNT); sink += Double.doubleToRawLongBits(output[0]); }
    private static void generatedDenseSegmentI64ToF32(MethodHandle entry, MemorySegment input, MemorySegment output, long[] geometry) throws Throwable { entry.invokeExact(input, output, geometry, 0L, (long) COUNT); sink += Float.floatToRawIntBits(output.getAtIndex(floatLayout(), 0)); }
    private static void generatedGeneralArrayF32ToBool(MethodHandle entry, float[] input, byte[] output, long[] geometry) throws Throwable { entry.invokeExact(input, output, geometry, 0L, (long) COUNT); sink += output[0]; }
    private static void generatedDenseMixedBoolToF64(MethodHandle entry, byte[] input, MemorySegment output, long[] geometry) throws Throwable { entry.invokeExact(input, output, geometry, 0L, (long) COUNT); sink += Double.doubleToRawLongBits(output.getAtIndex(doubleLayout(), 0)); }
    private static void generatedSameTypeRawIdentity(MethodHandle entry, float[] input, float[] output, long[] geometry) throws Throwable { entry.invokeExact(input, output, geometry, 0L, (long) COUNT); sink += Float.floatToRawIntBits(output[0]); }
    private static void generatedRoundingSensitiveTwoCastChain(MethodHandle entry, double[] input, short[] output, long[] geometry) throws Throwable { entry.invokeExact(input, output, geometry, 0L, (long) COUNT); sink += output[0]; }

    private static void directDenseArrayF64ToBfloat(double[] input, short[] output, long[] geometry) { for (int i = 0; i < COUNT; i++) output[(int) geometry[3] + i] = bf16(input[(int) geometry[2] + i]); sink += output[0]; }
    private static void directGeneralMixedI64ToBfloat(long[] input, MemorySegment output, long[] geometry) { directGeneralI64ToBfloat(input, output, geometry); sink += output.getAtIndex(shortLayout(), 0); }
    private static void directOffsetDenseMixedF64ToI64(double[] input, MemorySegment output, long[] geometry) { for (int i = 0; i < COUNT; i++) output.setAtIndex(longLayout(), geometry[3] + i, toLong(input[(int) geometry[2] + i])); sink += output.getAtIndex(longLayout(), 0); }
    private static void directBlockOuterBfloatToF64(short[] input, double[] output, long[] geometry) { directBlockOuterBfloatToF64Kernel(input, output, geometry); sink += Double.doubleToRawLongBits(output[0]); }
    private static void directDenseSegmentI64ToF32(MemorySegment input, MemorySegment output, long[] geometry) { for (int i = 0; i < COUNT; i++) output.setAtIndex(floatLayout(), geometry[3] + i, (float) input.getAtIndex(longLayout(), geometry[2] + i)); sink += Float.floatToRawIntBits(output.getAtIndex(floatLayout(), 0)); }
    private static void directGeneralArrayF32ToBool(float[] input, byte[] output, long[] geometry) { directGeneralF32ToBool(input, output, geometry); sink += output[0]; }
    private static void directDenseMixedBoolToF64(byte[] input, MemorySegment output, long[] geometry) { for (int i = 0; i < COUNT; i++) output.setAtIndex(doubleLayout(), geometry[3] + i, input[(int) geometry[2] + i] == 0 ? 0d : 1d); sink += Double.doubleToRawLongBits(output.getAtIndex(doubleLayout(), 0)); }
    private static void directSameTypeRawIdentity(float[] input, float[] output, long[] geometry) { int inputAddress = (int) geometry[2], outputAddress = (int) geometry[3], end = COUNT, index = 0; do { output[outputAddress] = input[inputAddress]; inputAddress++; outputAddress++; index++; } while (index < end); sink += Float.floatToRawIntBits(output[0]); }
    private static void directRoundingSensitiveTwoCastChain(double[] input, short[] output, long[] geometry) { int inputAddress = (int) geometry[2], outputAddress = (int) geometry[3], end = COUNT, index = 0; do { float intermediate = narrowFloat(input[inputAddress]); output[outputAddress] = BFloat16Bits.fromFloat(intermediate); inputAddress++; outputAddress++; index++; } while (index < end); sink += output[0]; }

    private static CpuKernelIr ir(Row row, DataType source, DataType target) {
        CpuAccessPlan read = plan(CpuAccessPlan.AccessKind.READ, row==Row.BLOCK_OUTER_BF16_TO_F64?CpuAccessPlan.Regime.BLOCK_OUTER:row.name().startsWith("GENERAL")?CpuAccessPlan.Regime.DENSE_LINEAR:CpuAccessPlan.Regime.DENSE_LINEAR, row==Row.BLOCK_OUTER_BF16_TO_F64?2:row.name().startsWith("GENERAL")?2:1);
        CpuAccessPlan write = plan(CpuAccessPlan.AccessKind.WRITE, row.name().startsWith("GENERAL")?CpuAccessPlan.Regime.GENERAL_ODOMETER:CpuAccessPlan.Regime.DENSE_LINEAR, read.iterationRank());
        if(row==Row.ROUNDING_SENSITIVE_TWO_CAST_CHAIN) return new CpuKernelIr(List.of(new CpuKernelIr.Value(0,source,CpuKernelIr.Value.Kind.INPUT,read),new CpuKernelIr.Value(1,DataType.FLOAT32,CpuKernelIr.Value.Kind.VIRTUAL,read),new CpuKernelIr.Value(2,target,CpuKernelIr.Value.Kind.OUTPUT,write)),List.of(new CpuKernelIr.Instruction(CpuPointwiseOpcode.CAST,List.of(0),1),new CpuKernelIr.Instruction(CpuPointwiseOpcode.CAST,List.of(1),2)),new CpuKernelIr.Loop("start","end"),List.of(new CpuKernelIr.Store(2,0)));
        return new CpuKernelIr(List.of(new CpuKernelIr.Value(0,source,CpuKernelIr.Value.Kind.INPUT,read),new CpuKernelIr.Value(1,target,CpuKernelIr.Value.Kind.OUTPUT,write)),List.of(new CpuKernelIr.Instruction(CpuPointwiseOpcode.CAST,List.of(0),1)),new CpuKernelIr.Loop("start","end"),List.of(new CpuKernelIr.Store(1,0)));
    }

    private static CpuAccessPlan plan(CpuAccessPlan.AccessKind kind,CpuAccessPlan.Regime regime,int rank){ List<CpuAccessPlan.AxisRole> roles=rank==1?List.of(CpuAccessPlan.AxisRole.CONTIGUOUS):regime==CpuAccessPlan.Regime.BLOCK_OUTER?List.of(CpuAccessPlan.AxisRole.STRIDED,CpuAccessPlan.AxisRole.CONTIGUOUS):regime==CpuAccessPlan.Regime.GENERAL_ODOMETER?List.of(CpuAccessPlan.AxisRole.STRIDED,CpuAccessPlan.AxisRole.STRIDED):List.of(CpuAccessPlan.AxisRole.CONTIGUOUS,CpuAccessPlan.AxisRole.CONTIGUOUS); return new CpuAccessPlan(kind,regime,rank,roles,regime==CpuAccessPlan.Regime.BLOCK_OUTER?1:regime==CpuAccessPlan.Regime.DENSE_LINEAR?rank:0); }
    private static List<CpuKernelSpecialization.CarrierAccess> carriers(Row r){return switch(r){case DENSE_ARRAY_F64_TO_BF16,ROUNDING_SENSITIVE_TWO_CAST_CHAIN->List.of(CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY,CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY);case GENERAL_MIXED_I64_TO_BF16->List.of(CpuKernelSpecialization.CarrierAccess.LONG_ARRAY,CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT);case OFFSET_DENSE_MIXED_F64_TO_I64->List.of(CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY,CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT);case BLOCK_OUTER_BF16_TO_F64->List.of(CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY,CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY);case DENSE_SEGMENT_I64_TO_F32->List.of(CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT,CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT);case GENERAL_ARRAY_F32_TO_BOOL->List.of(CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY,CpuKernelSpecialization.CarrierAccess.BYTE_ARRAY);case DENSE_MIXED_BOOL_TO_F64->List.of(CpuKernelSpecialization.CarrierAccess.BYTE_ARRAY,CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT);case SAME_TYPE_RAW_IDENTITY->List.of(CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY,CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY);};}
    private static DataType source(Row r){return switch(r){case DENSE_ARRAY_F64_TO_BF16,OFFSET_DENSE_MIXED_F64_TO_I64,ROUNDING_SENSITIVE_TWO_CAST_CHAIN->DataType.FLOAT64;case GENERAL_MIXED_I64_TO_BF16,DENSE_SEGMENT_I64_TO_F32->DataType.INT64;case BLOCK_OUTER_BF16_TO_F64->DataType.BFLOAT16;case GENERAL_ARRAY_F32_TO_BOOL,SAME_TYPE_RAW_IDENTITY->DataType.FLOAT32;case DENSE_MIXED_BOOL_TO_F64->DataType.BOOL;};}
    private static DataType target(Row r){return switch(r){case DENSE_ARRAY_F64_TO_BF16,GENERAL_MIXED_I64_TO_BF16,ROUNDING_SENSITIVE_TWO_CAST_CHAIN->DataType.BFLOAT16;case OFFSET_DENSE_MIXED_F64_TO_I64->DataType.INT64;case BLOCK_OUTER_BF16_TO_F64,DENSE_MIXED_BOOL_TO_F64->DataType.FLOAT64;case DENSE_SEGMENT_I64_TO_F32,SAME_TYPE_RAW_IDENTITY->DataType.FLOAT32;case GENERAL_ARRAY_F32_TO_BOOL->DataType.BOOL;};}
    private static Object input(Row r){Random z=new Random(0x5eedL+r.ordinal());return switch(source(r)){case FLOAT64->{double[] a=new double[COUNT+32];for(int i=0;i<a.length;i++)a[i]=Double.longBitsToDouble(z.nextLong());if(r==Row.ROUNDING_SENSITIVE_TWO_CAST_CHAIN){a[0]=1.0039062501d;a[1]=-1.0039062501d;a[2]=1.00390625d;}yield a;}case INT64->{long[] a=new long[COUNT+32];for(int i=0;i<a.length;i++)a[i]=r==Row.GENERAL_MIXED_I64_TO_BF16?(i%65_535)-32_767:z.nextLong();yield r==Row.DENSE_SEGMENT_I64_TO_F32?MemorySegment.ofArray(a):a;}case BFLOAT16->{short[]a=new short[COUNT*2+32];for(int i=0;i<a.length;i++)a[i]=(short)z.nextInt();yield a;}case FLOAT32->{float[]a=new float[COUNT+32];for(int i=0;i<a.length;i++)a[i]=Float.intBitsToFloat(z.nextInt());yield a;}case BOOL->{byte[]a=new byte[COUNT+32];for(int i=0;i<a.length;i++)a[i]=(byte)(i&1);yield a;}default->throw new AssertionError();};}
    private static Object output(Row r){return switch(target(r)){case BFLOAT16->r==Row.GENERAL_MIXED_I64_TO_BF16?MemorySegment.ofArray(new short[COUNT*4+32]):new short[COUNT+32];case INT64->MemorySegment.ofArray(new long[COUNT+32]);case FLOAT64->r==Row.DENSE_MIXED_BOOL_TO_F64?MemorySegment.ofArray(new double[COUNT+32]):new double[COUNT+32];case FLOAT32->r==Row.DENSE_SEGMENT_I64_TO_F32?MemorySegment.ofArray(new float[COUNT+32]):new float[COUNT+32];case BOOL->new byte[COUNT*4+32];default->throw new AssertionError();};}
    private static long[] geometry(Row r){if(r.name().startsWith("GENERAL")){return new long[]{256,256,0,0,0,0,256,1,1024,2,COUNT+32L,COUNT*4L+32};}if(r==Row.BLOCK_OUTER_BF16_TO_F64)return new long[]{256,256,0,0,0,0,512,1,256,1,0,0,256,256};long in=r==Row.OFFSET_DENSE_MIXED_F64_TO_I64?7:0,out=r==Row.OFFSET_DENSE_MIXED_F64_TO_I64?11:0;return new long[]{COUNT,0,in,out,1,1,COUNT+32L,COUNT+32L};}
    private static short bf16(double value){
        long bits=Double.doubleToRawLongBits(value), fraction=bits&0x000fffffffffffffL;
        int sign=(int)(bits>>>48)&0x8000, rawExponent=(int)(bits>>>52)&0x7ff;
        if(rawExponent==0x7ff)return(short)(fraction==0?sign|0x7f80:0x7fc0);
        if(rawExponent>1150)return(short)(sign|0x7f80);
        long significand; int exponent; int shift;
        if(rawExponent==0){significand=fraction;exponent=-127;shift=941;}
        else {significand=(1L<<52)|fraction;exponent=rawExponent-1023;shift=45;
            if(exponent< -126){shift+=-126-exponent;exponent=-127;}}
        long rounded=round(significand,shift);
        if(rounded==256){rounded=128;exponent++;}
        if(exponent>127)return(short)(sign|0x7f80);
        return(short)(sign|((exponent+127)<<7)|((int)rounded&127));
    }
    private static short bf16(long value){if(value==0)return 0;int sign=value<0?0x8000:0;long magnitude=value==Long.MIN_VALUE?Long.MIN_VALUE:value<0?-value:value;int exponent=63-Long.numberOfLeadingZeros(magnitude),shift=Math.max(0,exponent-7);long significand=shift==0?magnitude<<(7-exponent):round(magnitude,shift);if(significand==256){significand=128;exponent++;}return(short)(sign|((exponent+127)<<7)|((int)significand&127));}
    private static long round(long value,int shift){if(shift>=63)return 0;long result=value>>>shift,discard=value&((1L<<shift)-1),half=1L<<(shift-1);return discard>half||discard==half&&(result&1)!=0?result+1:result;}
    private static void directGeneralI64ToBfloat(long[] input, MemorySegment output, long[] g) {
        long inputAddress=g[4], outputAddress=g[5], row=0, column=0;
        for(int logical=0;logical<COUNT;logical++) {
            output.setAtIndex(shortLayout(),outputAddress,bf16(input[(int)inputAddress]));
            inputAddress++;
            if(++column==g[1]) {column=0;row++;outputAddress+=g[9]+g[8]-g[9]*g[1];}
            else outputAddress+=g[9];
        }
    }
    private static void directGeneralF32ToBool(float[] input, byte[] output, long[] g) {
        long inputAddress=g[4], outputAddress=g[5], row=0, column=0;
        for(int logical=0;logical<COUNT;logical++) {
            output[(int)outputAddress]=(byte)(input[(int)inputAddress]==0f?0:1);
            inputAddress++;
            if(++column==g[1]) {column=0;row++;outputAddress+=g[9]+g[8]-g[9]*g[1];}
            else outputAddress+=g[9];
        }
    }
    private static void directBlockOuterBfloatToF64Kernel(short[] input, double[] output, long[] g) {
        long inputAddress=g[4], outputAddress=g[5], inputInner=g[10], outputInner=g[11];
        long inputRow=0, outputRow=0;
        for(int logical=0;logical<COUNT;logical++) {
            output[(int)outputAddress]=widen(input[(int)inputAddress]);
            inputAddress++; outputAddress++; inputInner++; outputInner++;
            if(inputInner==g[12]) {inputInner=0;inputAddress-=g[12];if(++inputRow==g[0]) {inputRow=0;inputAddress-=g[6]*g[0];}else inputAddress+=g[6];}
            if(outputInner==g[13]) {outputInner=0;outputAddress-=g[13];if(++outputRow==g[0]) {outputRow=0;outputAddress-=g[8]*g[0];}else outputAddress+=g[8];}
        }
    }
    private static long toLong(double v){if(Double.isNaN(v))return 0;if(v>=Long.MAX_VALUE)return Long.MAX_VALUE;if(v<=Long.MIN_VALUE)return Long.MIN_VALUE;return(long)v;}
    private static double widen(short bits){int b=Short.toUnsignedInt(bits);if((b&0x7f80)==0x7f80&&(b&127)!=0)return Double.longBitsToDouble(((long)(b&0x8000)<<48)|0x7ff0000000000000L|((long)(b&127)<<45));return(float)Float.intBitsToFloat(b<<16);}
    private static ValueLayout.OfShort shortLayout(){return ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.nativeOrder());} private static ValueLayout.OfLong longLayout(){return ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.nativeOrder());} private static ValueLayout.OfFloat floatLayout(){return ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.nativeOrder());} private static ValueLayout.OfDouble doubleLayout(){return ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.nativeOrder());}
    private static long observe(Prepared p){return switch(targetFor(p.output)){case SHORT->((p.output instanceof short[] a)?a[0]:((MemorySegment)p.output).getAtIndex(shortLayout(),0));case LONG->((MemorySegment)p.output).getAtIndex(longLayout(),0);case DOUBLE->p.output instanceof double[] a?Double.doubleToRawLongBits(a[0]):Double.doubleToRawLongBits(((MemorySegment)p.output).getAtIndex(doubleLayout(),0));case FLOAT->p.output instanceof float[] a?Float.floatToRawIntBits(a[0]):Float.floatToRawIntBits(((MemorySegment)p.output).getAtIndex(floatLayout(),0));case BYTE->((byte[])p.output)[0];};}
    private enum Carrier {SHORT,LONG,DOUBLE,FLOAT,BYTE} private static Carrier targetFor(Object o){if(o instanceof short[]||o instanceof MemorySegment s&&s.byteSize()%2==0&&s.byteSize()%4!=0)return Carrier.SHORT;if(o instanceof long[]||o instanceof MemorySegment s&&s.byteSize()%8==0)return Carrier.LONG;if(o instanceof double[])return Carrier.DOUBLE;if(o instanceof float[])return Carrier.FLOAT;return Carrier.BYTE;}
    private static void verify(Row row,Prepared p)throws Throwable{Object before=cloneOutput(p.output);generated(row,p);Object generated=cloneOutput(p.output);restore(p.output,before);direct(row,p);assertTrue(equal(generated,p.output),row+" direct raw mismatch "+firstDifference(generated,p.output));}
    private static int firstDifference(Object expected,Object actual){
        if(expected instanceof short[] a&&actual instanceof short[] b){for(int i=0;i<a.length;i++)if(a[i]!=b[i])return i;}
        else if(expected instanceof long[] a&&actual instanceof long[] b){for(int i=0;i<a.length;i++)if(a[i]!=b[i])return i;}
        else if(expected instanceof double[] a&&actual instanceof double[] b){for(int i=0;i<a.length;i++)if(Double.doubleToRawLongBits(a[i])!=Double.doubleToRawLongBits(b[i]))return i;}
        else if(expected instanceof float[] a&&actual instanceof float[] b){for(int i=0;i<a.length;i++)if(Float.floatToRawIntBits(a[i])!=Float.floatToRawIntBits(b[i]))return i;}
        else {byte[]a=expected instanceof byte[] b?b:segmentBytes((MemorySegment)expected),b=actual instanceof byte[] x?x:segmentBytes((MemorySegment)actual);for(int i=0;i<a.length;i++)if(a[i]!=b[i])return i;}return -1;}
    private static Object cloneOutput(Object o){if(o instanceof MemorySegment s)return segmentBytes(s);if(o instanceof short[] a)return a.clone();if(o instanceof long[] a)return a.clone();if(o instanceof double[] a)return a.clone();if(o instanceof float[] a)return a.clone();return ((byte[])o).clone();} private static byte[] segmentBytes(MemorySegment s){byte[]copy=new byte[Math.toIntExact(s.byteSize())];for(int i=0;i<copy.length;i++)copy[i]=s.getAtIndex(ValueLayout.JAVA_BYTE,i);return copy;} private static void restore(Object o,Object saved){if(o instanceof MemorySegment s){byte[]copy=(byte[])saved;for(int i=0;i<copy.length;i++)s.setAtIndex(ValueLayout.JAVA_BYTE,i,copy[i]);}else if(o instanceof short[]a)System.arraycopy(saved,0,a,0,a.length);else if(o instanceof long[]a)System.arraycopy(saved,0,a,0,a.length);else if(o instanceof double[]a)System.arraycopy(saved,0,a,0,a.length);else if(o instanceof float[]a)System.arraycopy(saved,0,a,0,a.length);else System.arraycopy(saved,0,o,0,((byte[])o).length);} private static boolean equal(Object a,Object b){return a instanceof byte[]x&&b instanceof MemorySegment s?Arrays.equals(x,segmentBytes(s)):a instanceof short[]x&&b instanceof short[]y?Arrays.equals(x,y):a instanceof long[]x&&b instanceof long[]y?Arrays.equals(x,y):a instanceof double[]x&&b instanceof double[]y?Arrays.equals(x,y):a instanceof float[]x&&b instanceof float[]y?Arrays.equals(x,y):Arrays.equals((byte[])a,(byte[])b);}
    private static void validateFork(Path file,int fork)throws Exception{List<String>lines=Files.readAllLines(file);assertEquals(10,lines.size());for(int i=1;i<lines.size();i++){String[]v=lines.get(i).split(",");assertEquals(Row.values()[i-1].name(),v[0]);assertTrue(Double.parseDouble(v[2])<=LIMIT,"fork "+fork+" "+v[0]);}}
    private static void aggregate(Path root)throws Exception{StringBuilder report=new StringBuilder("row,fork0,fork1,fork2,fork3,fork4,median,accepted\n");for(int row=0;row<Row.values().length;row++){double[]v=new double[FORKS];for(int fork=0;fork<FORKS;fork++)v[fork]=Double.parseDouble(Files.readAllLines(root.resolve("raw-fork-"+fork+".csv")).get(row+1).split(",")[2]);double[]sorted=v.clone();Arrays.sort(sorted);assertTrue(sorted[2]<=LIMIT,Row.values()[row]+" median");report.append(Row.values()[row]).append(',').append(v[0]).append(',').append(v[1]).append(',').append(v[2]).append(',').append(v[3]).append(',').append(v[4]).append(',').append(sorted[2]).append(",true\n");}Files.writeString(root.resolve("aggregate.csv"),report);}
    private static void manifest(Path root)throws Exception{MessageDigest d=MessageDigest.getInstance("SHA-256");StringBuilder out=new StringBuilder();try(var paths=Files.walk(root)){for(Path p:paths.filter(Files::isRegularFile).sorted().toList())if(!p.getFileName().toString().startsWith("manifest"))out.append(HexFormat.of().formatHex(d.digest(Files.readAllBytes(p)))).append("  ").append(root.relativize(p)).append('\n');}Files.writeString(root.resolve("manifest.sha256"),out);Files.writeString(root.resolve("manifest.digest"),HexFormat.of().formatHex(d.digest(out.toString().getBytes(StandardCharsets.UTF_8))));}
    private static Path sourceFile(){
        Path cwd=Path.of(System.getProperty("user.dir")).toAbsolutePath();
        Path moduleSource=cwd.resolve("src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuCastPerformanceTest.java");
        if(Files.isRegularFile(moduleSource))return moduleSource;
        Path repositorySource=cwd.resolve("backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuCastPerformanceTest.java");
        if(Files.isRegularFile(repositorySource))return repositorySource;
        throw new IllegalStateException("CpuCastPerformanceTest.java not found from user.dir: "+cwd);
    }
    @FunctionalInterface
    private interface Invocation { long elapsed(int count) throws Throwable; }

    private record Measurement(long generatedBeforeNanos, long directAfterNanos,
                               long generatedAfterNanos, long directBeforeNanos) {
        long generatedNanos() { return generatedBeforeNanos + generatedAfterNanos; }
        long directNanos() { return directAfterNanos + directBeforeNanos; }
        double ratio() {
            return symmetricRatio(generatedBeforeNanos, directAfterNanos,
                    generatedAfterNanos, directBeforeNanos);
        }
    }

    private record Prepared(MethodHandle entry,Object input,Object output,long[] geometry,
                            Invocation generatedAction, Invocation directAction) {}
}
