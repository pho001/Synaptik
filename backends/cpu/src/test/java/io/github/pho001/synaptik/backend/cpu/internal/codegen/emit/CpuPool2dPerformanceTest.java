package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.backend.cpu.internal.cache.*;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.ir.*;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPool2dLowering;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.model.datatype.*;
import java.lang.classfile.ClassFile;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.*;
import java.security.MessageDigest;
import java.time.Instant;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Opt-in five-fork generated-versus-optimal-clean-Java Pool2d evidence owner. */
public final class CpuPool2dPerformanceTest {
    private static final long MIN_BATCH_NANOS = 25_000_000L;
    private static volatile long sink;

    @FunctionalInterface
    private interface Action {
        long run() throws Throwable;
    }

    private record Case(
            String name,
            Action generated,
            Action direct,
            Runnable verify,
            byte[] bytes,
            CpuKernelSpecialization specialization) {}

    private CpuPool2dPerformanceTest() {}

    /** Launches exactly five fixed-heap forks and retains their aggregate when explicitly enabled. */
    @Test
    void retainedFiveForkEvidence() throws Throwable {
        assumeTrue("true".equals(System.getenv("SYNAPTIK_CPU_POOL2D_PERFORMANCE")));
        Path root = root();
        assertFalse(Files.exists(root), "evidence root must be fresh");
        Files.createDirectories(root);
        String java = Path.of(System.getProperty("java.home"), "bin", "java").toString();
        String cp = System.getProperty("java.class.path");
        for (int fork = 0; fork < 5; fork++) {
            Process process =
                    new ProcessBuilder(
                                    java,
                                    "-Xms1g",
                                    "-Xmx1g",
                                    "--add-modules",
                                    "jdk.incubator.vector",
                                    "-cp",
                                    cp,
                                    getClass().getName(),
                                    Integer.toString(fork))
                            .inheritIO()
                            .start();
            assertEquals(0, process.waitFor(), "fork " + fork);
        }
        aggregate(root);
        retainArtifacts(root);
        writeManifest(root);
    }

    /**
     * Executes one isolated measured fork.
     *
     * @param args one exact fork index
     * @throws Throwable if semantic, timing, or evidence retention fails
     */
    public static void main(String[] args) throws Throwable {
        int fork = Integer.parseInt(args[0]);
        Path root = root();
        StringBuilder report = new StringBuilder();
        report
                .append("ENV,java=")
                .append(System.getProperty("java.version"))
                .append(",vm=")
                .append(System.getProperty("java.vm.name"))
                .append(",processors=")
                .append(Runtime.getRuntime().availableProcessors())
                .append(",maxMemory=")
                .append(Runtime.getRuntime().maxMemory())
                .append('\n');
        try (Arena arena = Arena.ofConfined()) {
            List<Case> cases = cases(arena);
            int[] reps = new int[cases.size() * 2];
            for (int i = 0; i < cases.size(); i++) {
                gate(cases.get(i));
                reps[2 * i] = repetitions(cases.get(i).generated);
                reps[2 * i + 1] = repetitions(cases.get(i).direct);
            }
            long[][] g = new long[cases.size()][9], d = new long[cases.size()][9];
            Random random = new Random(0x0008_600D_2026L ^ fork * 0x9e3779b97f4a7c15L);
            for (int round = -5; round < 9; round++) {
                List<Integer> order = new ArrayList<>();
                for (int i = 0; i < cases.size(); i++) order.add(i);
                Collections.shuffle(order, random);
                for (int i : order) {
                    long gt, dt;
                    if (random.nextBoolean()) {
                        gt = time(cases.get(i).generated, reps[2 * i]);
                        dt = time(cases.get(i).direct, reps[2 * i + 1]);
                    } else {
                        dt = time(cases.get(i).direct, reps[2 * i + 1]);
                        gt = time(cases.get(i).generated, reps[2 * i]);
                    }
                    if (round >= 0) {
                        g[i][round] = gt / reps[2 * i];
                        d[i][round] = dt / reps[2 * i + 1];
                        gate(cases.get(i));
                    }
                }
            }
            int failures = 0;
            for (int i = 0; i < cases.size(); i++) {
                long gm = median(g[i]), dm = median(d[i]);
                double ratio = (double) gm / dm;
                if (ratio > 1.15) failures++;
                report.append(
                        String.format(
                                Locale.ROOT,
                                "RESULT,%s,%d,%d,%.9f,%s,%s%n",
                                cases.get(i).name,
                                gm,
                                dm,
                                ratio,
                                Arrays.toString(g[i]),
                                Arrays.toString(d[i])));
            }
            report.append("SINK,").append(sink).append('\n');
            if (failures > 0) throw new AssertionError("ratio failures " + failures);
        } catch (Throwable failure) {
            report.append("FAILED,").append(failure).append('\n');
            retain(root, false, fork, report.toString());
            throw failure;
        }
        retain(root, true, fork, report.toString());
    }

    private static List<Case> cases(Arena arena) throws Exception {
        return List.of(
                arrayCase(
                        "DENSE-F64-MAX-3X3",
                        DataType.FLOAT64,
                        CpuPool2dIr.Kind.MAX,
                        2,
                        8,
                        64,
                        64,
                        3,
                        3,
                        1,
                        0,
                        1),
                arrayCase(
                        "PAD-DILATE-CEIL-F32-MAX",
                        DataType.FLOAT32,
                        CpuPool2dIr.Kind.MAX,
                        2,
                        8,
                        64,
                        64,
                        3,
                        3,
                        2,
                        2,
                        2),
                bf16Case("MIXED-BF16-MAX", CpuPool2dIr.Kind.MAX, arena),
                arrayCase(
                        "DENSE-F64-AVERAGE-3X3",
                        DataType.FLOAT64,
                        CpuPool2dIr.Kind.AVERAGE,
                        2,
                        8,
                        64,
                        64,
                        3,
                        3,
                        1,
                        0,
                        1),
                arrayCase(
                        "PAD-DILATE-CEIL-F32-AVERAGE",
                        DataType.FLOAT32,
                        CpuPool2dIr.Kind.AVERAGE,
                        2,
                        8,
                        64,
                        64,
                        3,
                        3,
                        2,
                        2,
                        2),
                bf16Case("MIXED-BF16-AVERAGE", CpuPool2dIr.Kind.AVERAGE, arena));
    }

    private static Case arrayCase(
            String name,
            DataType type,
            CpuPool2dIr.Kind kind,
            long n,
            long c,
            long h,
            long w,
            long kh,
            long kw,
            long stride,
            long pad,
            long dilation)
            throws Exception {
        long oh = extent(h, kh, stride, pad, dilation, true),
                ow = extent(w, kw, stride, pad, dilation, true);
        long count = n * c * oh * ow;
        var geometry = geometry(type, kind, n, c, h, w, oh, ow, kh, kw, stride, pad, dilation);
        CarrierAccess carrier =
                type == DataType.FLOAT64 ? CarrierAccess.DOUBLE_ARRAY : CarrierAccess.FLOAT_ARRAY;
        var generated = generated(geometry, carrier, carrier);
        Object input =
                type == DataType.FLOAT64
                        ? new double[(int) (n * c * h * w)]
                        : new float[(int) (n * c * h * w)];
        Object go = type == DataType.FLOAT64 ? new double[(int) count] : new float[(int) count];
        Object direct = type == DataType.FLOAT64 ? new double[(int) count] : new float[(int) count];
        fill(input);
        Action ga =
                () -> {
                    generated.handle.invokeWithArguments(input, go, geometry.pack(0, 0), 0L, count);
                    return checksum(go);
                };
        Action da =
                () -> {
                    if (type == DataType.FLOAT64) direct64(geometry, (double[]) input, (double[]) direct);
                    else direct32(geometry, (float[]) input, (float[]) direct);
                    return checksum(direct);
                };
        return new Case(
                name,
                ga,
                da,
                () -> {
                    if (!equal(go, direct)) throw new AssertionError(name);
                },
                generated.bytes,
                generated.specialization);
    }

    private static Case bf16Case(String name, CpuPool2dIr.Kind kind, Arena arena) throws Exception {
        long n = 2, c = 8, h = 48, w = 48, oh = 46, ow = 46, count = n * c * oh * ow;
        var geometry = geometry(DataType.BFLOAT16, kind, n, c, h, w, oh, ow, 3, 3, 1, 0, 1);
        var generated = generated(geometry, CarrierAccess.MEMORY_SEGMENT, CarrierAccess.SHORT_ARRAY);
        MemorySegment input = arena.allocate(n * c * h * w * 2, 2);
        short[] go = new short[(int) count], direct = new short[(int) count];
        for (long i = 0; i < n * c * h * w; i++)
            input.set(
                    ValueLayout.JAVA_SHORT_UNALIGNED, i * 2, BFloat16Bits.fromFloat((i % 31 - 15) * .03125f));
        Action ga =
                () -> {
                    generated.handle.invokeExact(input, go, geometry.pack(0, 0), 0L, count);
                    return checksum(go);
                };
        Action da =
                () -> {
                    direct16(geometry, input, direct);
                    return checksum(direct);
                };
        return new Case(
                name,
                ga,
                da,
                () -> {
                    if (!Arrays.equals(go, direct)) throw new AssertionError(name);
                },
                generated.bytes,
                generated.specialization);
    }

    private record Generated(
            MethodHandle handle, byte[] bytes, CpuKernelSpecialization specialization) {}

    private static Generated generated(
            CpuPool2dLowering.Geometry g, CarrierAccess in, CarrierAccess out) {
        var roles = Collections.nCopies(4, CpuAccessPlan.AxisRole.CONTIGUOUS);
        var pool =
                new CpuPool2dIr(
                        g.kind(),
                        g.dataType(),
                        CpuPool2dIr.Realization.DIRECT_SCALAR,
                        new CpuAccessPlan(
                                CpuAccessPlan.AccessKind.READ, CpuAccessPlan.Regime.DENSE_LINEAR, 4, roles, 4),
                        new CpuAccessPlan(
                                CpuAccessPlan.AccessKind.WRITE, CpuAccessPlan.Regime.DENSE_LINEAR, 4, roles, 4));
        var ir = pool.encodedKernelIr();
        var s =
                new CpuKernelSpecialization(
                        CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                        CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                        CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,
                        List.of(g.dataType(), g.dataType()),
                        List.of(in, out),
                        0,
                        -1,
                        List.of(),
                        false,
                        55);
        var generator = new CpuClassFileKernelGenerator();
        byte[] bytes = generator.generateClassBytes(s, ir);
        return new Generated(generator.defineClassBytes(s, bytes).entryPoint(), bytes, s);
    }

    private static CpuPool2dLowering.Geometry geometry(
            DataType t,
            CpuPool2dIr.Kind k,
            long n,
            long c,
            long h,
            long w,
            long oh,
            long ow,
            long kh,
            long kw,
            long stride,
            long pad,
            long dilation) {
        return new CpuPool2dLowering.Geometry(
                k,
                t,
                new CpuPool2dLowering.Layout(
                        new long[] {n, c, h, w}, 0, new long[] {c * h * w, h * w, w, 1}),
                new CpuPool2dLowering.Layout(
                        new long[] {n, c, oh, ow}, 0, new long[] {c * oh * ow, oh * ow, ow, 1}),
                kh,
                kw,
                stride,
                stride,
                pad,
                pad,
                dilation,
                dilation,
                kh * kw,
                n * c * oh * ow);
    }

    private static void direct64(CpuPool2dLowering.Geometry g, double[] x, double[] y) {
        long[] e = g.input().extents(), o = g.output().extents();
        for (long cell = 0; cell < g.outputCount(); cell++) {
            long q = cell, ow = q % o[3];
            q /= o[3];
            long oh = q % o[2];
            q /= o[2];
            long c = q % o[1],
                    n = q / o[1],
                    ih0 = oh * g.strideHeight() - g.paddingHeight(),
                    iw0 = ow * g.strideWidth() - g.paddingWidth();
            if (g.kind() == CpuPool2dIr.Kind.MAX) {
                boolean found = false;
                double best = Double.NEGATIVE_INFINITY;
                outer:
                for (long kh = 0; kh < g.kernelHeight(); kh++)
                    for (long kw = 0; kw < g.kernelWidth(); kw++) {
                        long ih = ih0 + kh * g.dilationHeight(), iw = iw0 + kw * g.dilationWidth();
                        if (ih < 0 || iw < 0 || ih >= e[2] || iw >= e[3]) continue;
                        double v = x[(int) (((n * e[1] + c) * e[2] + ih) * e[3] + iw)];
                        if (Double.isNaN(v)) {
                            best = v;
                            found = true;
                            break outer;
                        }
                        if (!found
                                || v > best
                                || v == 0
                                        && best == 0
                                        && Double.doubleToRawLongBits(v) == 0
                                        && Double.doubleToRawLongBits(best) != 0) {
                            best = v;
                            found = true;
                        }
                    }
                y[(int) cell] = found ? best : Double.NEGATIVE_INFINITY;
            } else {
                double sum = 0;
                boolean neg = true;
                for (long kh = 0; kh < g.kernelHeight(); kh++)
                    for (long kw = 0; kw < g.kernelWidth(); kw++) {
                        long ih = ih0 + kh * g.dilationHeight(), iw = iw0 + kw * g.dilationWidth();
                        if (ih < 0 || iw < 0 || ih >= e[2] || iw >= e[3]) {
                            neg = false;
                            continue;
                        }
                        double v = x[(int) (((n * e[1] + c) * e[2] + ih) * e[3] + iw)];
                        neg &= Double.doubleToRawLongBits(v) == Long.MIN_VALUE;
                        sum += v;
                    }
                double r = sum / g.divisor();
                y[(int) cell] = r == 0 ? (neg ? -0.0 : +0.0) : r;
            }
        }
    }

    private static void direct32(CpuPool2dLowering.Geometry g, float[] x, float[] y) {
        long[] e = g.input().extents(), o = g.output().extents();
        for (long cell = 0; cell < g.outputCount(); cell++) {
            long q = cell, ow = q % o[3];
            q /= o[3];
            long oh = q % o[2];
            q /= o[2];
            long c = q % o[1],
                    n = q / o[1],
                    ih0 = oh * g.strideHeight() - g.paddingHeight(),
                    iw0 = ow * g.strideWidth() - g.paddingWidth();
            if (g.kind() == CpuPool2dIr.Kind.MAX) {
                boolean found = false;
                float best = Float.NEGATIVE_INFINITY;
                outer:
                for (long kh = 0; kh < g.kernelHeight(); kh++)
                    for (long kw = 0; kw < g.kernelWidth(); kw++) {
                        long ih = ih0 + kh * g.dilationHeight(), iw = iw0 + kw * g.dilationWidth();
                        if (ih < 0 || iw < 0 || ih >= e[2] || iw >= e[3]) continue;
                        float v = x[(int) (((n * e[1] + c) * e[2] + ih) * e[3] + iw)];
                        if (Float.isNaN(v)) {
                            best = v;
                            found = true;
                            break outer;
                        }
                        if (!found
                                || v > best
                                || v == 0
                                        && best == 0
                                        && Float.floatToRawIntBits(v) == 0
                                        && Float.floatToRawIntBits(best) != 0) {
                            best = v;
                            found = true;
                        }
                    }
                y[(int) cell] = found ? best : Float.NEGATIVE_INFINITY;
            } else {
                float sum = 0;
                boolean neg = true;
                for (long kh = 0; kh < g.kernelHeight(); kh++)
                    for (long kw = 0; kw < g.kernelWidth(); kw++) {
                        long ih = ih0 + kh * g.dilationHeight(), iw = iw0 + kw * g.dilationWidth();
                        if (ih < 0 || iw < 0 || ih >= e[2] || iw >= e[3]) {
                            neg = false;
                            continue;
                        }
                        float v = x[(int) (((n * e[1] + c) * e[2] + ih) * e[3] + iw)];
                        neg &= Float.floatToRawIntBits(v) == Integer.MIN_VALUE;
                        sum += v;
                    }
                float r = sum / (float) g.divisor();
                y[(int) cell] = r == 0 ? (neg ? -0.0f : +0.0f) : r;
            }
        }
    }

    private static void direct16(CpuPool2dLowering.Geometry g, MemorySegment x, short[] y) {
        long[] e = g.input().extents(), o = g.output().extents();
        for (long cell = 0; cell < g.outputCount(); cell++) {
            long q = cell, ow = q % o[3];
            q /= o[3];
            long oh = q % o[2];
            q /= o[2];
            long c = q % o[1],
                    n = q / o[1],
                    ih0 = oh * g.strideHeight() - g.paddingHeight(),
                    iw0 = ow * g.strideWidth() - g.paddingWidth();
            if (g.kind() == CpuPool2dIr.Kind.MAX) {
                boolean found = false;
                float best = Float.NEGATIVE_INFINITY;
                short winner = (short) 0xff80;
                outer:
                for (long kh = 0; kh < g.kernelHeight(); kh++)
                    for (long kw = 0; kw < g.kernelWidth(); kw++) {
                        long ih = ih0 + kh * g.dilationHeight(), iw = iw0 + kw * g.dilationWidth();
                        if (ih < 0 || iw < 0 || ih >= e[2] || iw >= e[3]) continue;
                        short bits =
                                x.get(
                                        ValueLayout.JAVA_SHORT_UNALIGNED,
                                        2 * (((n * e[1] + c) * e[2] + ih) * e[3] + iw));
                        float v = BFloat16Bits.toFloat(bits);
                        if (Float.isNaN(v)) {
                            winner = bits;
                            found = true;
                            break outer;
                        }
                        if (!found || v > best || v == 0 && best == 0 && bits == 0 && winner != 0) {
                            best = v;
                            winner = bits;
                            found = true;
                        }
                    }
                y[(int) cell] = winner;
            } else {
                float sum = 0;
                boolean neg = true;
                for (long kh = 0; kh < g.kernelHeight(); kh++)
                    for (long kw = 0; kw < g.kernelWidth(); kw++) {
                        long ih = ih0 + kh * g.dilationHeight(), iw = iw0 + kw * g.dilationWidth();
                        if (ih < 0 || iw < 0 || ih >= e[2] || iw >= e[3]) {
                            neg = false;
                            continue;
                        }
                        short bits =
                                x.get(
                                        ValueLayout.JAVA_SHORT_UNALIGNED,
                                        2 * (((n * e[1] + c) * e[2] + ih) * e[3] + iw));
                        neg &= bits == (short) 0x8000;
                        sum += BFloat16Bits.toFloat(bits);
                    }
                float r = sum / (float) g.divisor();
                y[(int) cell] = BFloat16Bits.fromFloat(r == 0 ? (neg ? -0.0f : +0.0f) : r);
            }
        }
    }

    private static long extent(long d, long k, long s, long p, long dilation, boolean ceil) {
        long n = d + 2 * p - (dilation * (k - 1) + 1);
        return n / s + (ceil && n % s != 0 ? 1 : 0) + 1;
    }

    private static void fill(Object x) {
        if (x instanceof double[] a) for (int i = 0; i < a.length; i++) a[i] = (i % 31 - 15) * .03125;
        else {
            float[] a = (float[]) x;
            for (int i = 0; i < a.length; i++) a[i] = (i % 31 - 15) * .03125f;
        }
    }

    private static boolean equal(Object a, Object b) {
        return a instanceof double[] x
                ? Arrays.equals(x, (double[]) b)
                : Arrays.equals((float[]) a, (float[]) b);
    }

    private static long checksum(Object x) {
        long h = 0;
        if (x instanceof double[] a)
            for (double v : a) h = Long.rotateLeft(h, 1) ^ Double.doubleToRawLongBits(v);
        else if (x instanceof float[] a)
            for (float v : a)
                h = Long.rotateLeft(h, 1) ^ Integer.toUnsignedLong(Float.floatToRawIntBits(v));
        else for (short v : (short[]) x) h = Long.rotateLeft(h, 1) ^ Short.toUnsignedLong(v);
        return h;
    }

    private static void gate(Case c) throws Throwable {
        long g = c.generated.run(), d = c.direct.run();
        if (g != d) throw new AssertionError(c.name + " checksum");
        c.verify.run();
    }

    private static long time(Action a, int n) throws Throwable {
        long start = System.nanoTime(), v = 0;
        for (int i = 0; i < n; i++) v ^= a.run();
        sink ^= v;
        return System.nanoTime() - start;
    }

    private static int repetitions(Action a) throws Throwable {
        int n = 1;
        while (time(a, n) < MIN_BATCH_NANOS) n = Math.multiplyExact(n, 2);
        return n;
    }

    private static long median(long[] v) {
        long[] copy = v.clone();
        Arrays.sort(copy);
        return copy[4];
    }

    private static Path root() {
        String value = System.getenv("SYNAPTIK_CPU_POOL2D_EVIDENCE_ROOT");
        if (value == null || value.isBlank())
            throw new IllegalStateException("SYNAPTIK_CPU_POOL2D_EVIDENCE_ROOT is required");
        return Path.of(value).toAbsolutePath();
    }

    private static void retain(Path root, boolean ok, int fork, String report) throws Exception {
        Path dir = root.resolve(ok ? "forks" : "failed-forks");
        Files.createDirectories(dir);
        Files.writeString(
                dir.resolve("fork-" + fork + "-" + Instant.now().toEpochMilli() + ".csv"), report);
    }

    private static void aggregate(Path root) throws Exception {
        List<Path> forks;
        try (var stream = Files.list(root.resolve("forks"))) {
            forks = stream.sorted().toList();
        }
        assertEquals(5, forks.size());
        StringBuilder out = new StringBuilder();
        List<String> names = null;
        double[][] ratios = new double[5][6];
        for (int f = 0; f < 5; f++) {
            List<String> found = new ArrayList<>();
            int row = 0;
            for (String line : Files.readAllLines(forks.get(f)))
                if (line.startsWith("RESULT,")) {
                    String[] v = line.split(",", 6);
                    found.add(v[1]);
                    ratios[f][row++] = Double.parseDouble(v[4]);
                }
            assertEquals(6, row);
            if (names == null) names = found;
            else assertEquals(names, found);
        }
        for (int row = 0; row < 6; row++) {
            double[] v = new double[5];
            for (int f = 0; f < 5; f++) v[f] = ratios[f][row];
            Arrays.sort(v);
            assertTrue(v[2] <= 1.15, names.get(row));
            out.append(
                    String.format(
                            Locale.ROOT, "AGGREGATE,%s,%.9f,%s%n", names.get(row), v[2], Arrays.toString(v)));
        }
        Files.writeString(root.resolve("aggregate.csv"), out);
    }

    private static void retainArtifacts(Path root) throws Exception {
        Path generated = root.resolve("generated");
        Files.createDirectories(generated);
        try (Arena arena = Arena.ofConfined()) {
            for (Case c : cases(arena)) {
                Path file = generated.resolve(c.name + ".class");
                Files.write(file, c.bytes);
                var model = ClassFile.of().parse(c.bytes);
                String refs =
                        java.util.stream.StreamSupport.stream(model.constantPool().spliterator(), false)
                                .filter(MemberRefEntry.class::isInstance)
                                .map(MemberRefEntry.class::cast)
                                .map(
                                        e ->
                                                e.owner().asInternalName()
                                                        + "."
                                                        + e.name().stringValue()
                                                        + e.type().stringValue())
                                .reduce("", (left, right) -> left.isEmpty() ? right : left + '\n' + right);
                assertFalse(refs.contains("synaptik"));
                Files.writeString(generated.resolve(c.name + ".members"), refs);
                Files.writeString(
                        generated.resolve(c.name + ".specialization"), c.specialization.toString());
                Process p =
                        new ProcessBuilder(
                                        Path.of(System.getProperty("java.home"), "bin", "javap").toString(),
                                        "-c",
                                        "-v",
                                        "-p",
                                        file.toString())
                                .redirectOutput(generated.resolve(c.name + ".javap").toFile())
                                .start();
                assertEquals(0, p.waitFor());
            }
        }
        Path source =
                Path.of(
                        "src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPool2dPerformanceTest.java");
        Files.copy(source, root.resolve("direct-oracle-source.java"));
        Files.writeString(root.resolve("environment.txt"), System.getProperties().toString());
    }

    private static void writeManifest(Path root) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        StringBuilder manifest = new StringBuilder();
        try (var files = Files.walk(root)) {
            for (Path file : files.filter(Files::isRegularFile).sorted().toList()) {
                if (file.getFileName().toString().equals("manifest.sha256")) continue;
                manifest
                        .append(HexFormat.of().formatHex(digest.digest(Files.readAllBytes(file))))
                        .append("  ")
                        .append(root.relativize(file))
                        .append('\n');
            }
        }
        Files.writeString(root.resolve("manifest.sha256"), manifest);
        Files.writeString(
                root.resolve("manifest.digest"),
                HexFormat.of()
                        .formatHex(
                                digest.digest(
                                        manifest.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8))));
    }
}
