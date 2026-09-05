package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.ClassFile;
import java.lang.classfile.Instruction;
import java.lang.classfile.Opcode;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.lang.classfile.constantpool.StringEntry;
import java.lang.reflect.AccessFlag;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Set;
import java.util.TreeMap;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/**
 * Guarded retained Class-File evidence for every CPU 0008K CAST class form.
 *
 * <p>The ordinary CPU suite skips this test.  A caller supplies a fresh directory through
 * {@code synaptik.cpu.cast.structuralEvidenceRoot} and creates its
 * {@code RUN-STRUCTURAL-EVIDENCE} marker.  The marker is removed only after the inventory,
 * Class-File checks, decompilation, direct-oracle controls, and manifest all succeed.</p>
 */
class CpuCastEvidenceTest {
    private static final String ROOT_PROPERTY = "synaptik.cpu.cast.structuralEvidenceRoot";
    private static final String MARKER = "RUN-STRUCTURAL-EVIDENCE";
    private static final List<DataType> TYPES = List.of(DataType.FLOAT64, DataType.FLOAT32,
            DataType.BFLOAT16, DataType.INT64, DataType.INT32, DataType.BOOL);
    private static final List<CarrierPattern> CARRIERS = List.of(
            new CarrierPattern("array-array", false, false),
            new CarrierPattern("array-segment", false, true),
            new CarrierPattern("segment-array", true, false),
            new CarrierPattern("segment-segment", true, true));
    private static final List<AccessForm> ACCESS = List.of(AccessForm.DENSE,
            AccessForm.BLOCK_OUTER, AccessForm.GENERAL_ODOMETER, AccessForm.RANK_ZERO);

    @Test
    void retainsExactCastStructuralInventory() throws Exception {
        String configuredRoot = System.getProperty(ROOT_PROPERTY);
        Assumptions.assumeTrue(configuredRoot != null && !configuredRoot.isBlank(),
                "structural evidence root was not requested");
        Path root = Path.of(configuredRoot).toAbsolutePath().normalize();
        Path marker = root.resolve(MARKER);
        Assumptions.assumeTrue(Files.isRegularFile(marker), "structural evidence guard absent");
        assertEquals(Path.of("/private/tmp"), root.getParent(),
                "evidence root must be directly under private tmp");
        assertTrue(root.getFileName().toString().startsWith("synaptik-cpu-0008k-"),
                "evidence root must be fresh private tmp root");

        try {
            retain(root);
            Files.delete(marker); // Deliberately last: an incomplete run remains visibly guarded.
        } catch (Throwable failure) {
            assertTrue(Files.exists(marker), "failed evidence must retain its guard");
            throw failure;
        }
    }

    private static void retain(Path root) throws Exception {
        Path classes = Files.createDirectories(root.resolve("classes"));
        Path reports = Files.createDirectories(root.resolve("reports"));
        Path javap = Files.createDirectories(root.resolve("javap"));
        Path oracle = Files.createDirectories(root.resolve("oracle"));
        Files.writeString(root.resolve("commands.txt"), "./gradlew :backends:cpu:test --tests "
                + CpuCastEvidenceTest.class.getName() + " -D" + ROOT_PROPERTY + "=<root> --rerun-tasks\n"
                + "javap -c -v -p <every retained class>\n"
                + "javac --release 26 -d oracle oracle/CastOracle.java\n");
        Files.writeString(root.resolve("environment.txt"), "java.version="
                + System.getProperty("java.version") + "\njava.home=" + System.getProperty("java.home")
                + "\nos.name=" + System.getProperty("os.name") + "\n");
        Files.writeString(root.resolve("inventory.tsv"), "key\tsource\ttarget\tcarriers\taccess\tsha256\tnormalized\n");

        CpuClassFileKernelGenerator generator = new CpuClassFileKernelGenerator();
        TreeMap<String, byte[]> forms = new TreeMap<>();
        for (DataType source : TYPES) for (DataType target : TYPES)
            for (CarrierPattern carriers : CARRIERS) for (AccessForm access : ACCESS) {
                Form form = new Form(source, target, carriers, access);
                Generated generated = generate(generator, form);
                Generated repeat = generate(generator, form);
                assertEquals(generated.key(), repeat.key(), "deterministic key " + form);
                assertArrayEquals(generated.bytes(), repeat.bytes(), "deterministic bytes " + form);
                assertTrue(forms.putIfAbsent(generated.key(), generated.bytes()) == null,
                        "duplicate structural identity " + form);
                inspect(generated, form, reports, javap);
                Files.write(classes.resolve(generated.key() + ".class"), generated.bytes());
                Files.writeString(root.resolve("inventory.tsv"), generated.key() + '\t' + source + '\t'
                        + target + '\t' + carriers.name + '\t' + access + '\t'
                        + sha256(generated.bytes()) + '\t' + normalized(generated.bytes()) + '\n',
                        StandardOpenOption.APPEND);
            }
        assertEquals(576, forms.size(), "36 pairs * 4 carrier patterns * 4 access forms");
        assertColdReuse(generator);
        assertSchemaControls(generator);
        retainOracle(oracle);
        Files.writeString(root.resolve("summary.txt"), "forms=576\npairs=36\ncarrier-patterns=4\n"
                + "access-forms=4\nschema52-control=unchanged projection\n"
                + "schema59-control=unchanged projection\nschema60=cross-type CAST only\n"
                + "cold=offset/extents/ranges/parallel orchestration\n");
        Files.writeString(root.resolve("manifest.sha256"), manifest(root));
        verifyManifest(root, forms.keySet());
    }

    private static Generated generate(CpuClassFileKernelGenerator generator, Form form) {
        return generate(generator, form, CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR);
    }

    private static Generated generate(CpuClassFileKernelGenerator generator, Form form,
            CpuPartitionPreparationPlan.ExecutionStrategy strategy) {
        CpuKernelIr ir = ir(form.source, form.target, form.access);
        CpuKernelSpecialization specialization = new CpuKernelSpecialization(
                CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                strategy,
                List.of(form.source, form.target), List.of(form.carriers.source(form.source),
                        form.carriers.target(form.target)), 0, -1, List.of(), false,
                form.source == form.target ? (form.source == DataType.BFLOAT16 ? 59 : 52) : 60);
        byte[] bytes = generator.generateClassBytes(specialization, ir);
        return new Generated(specialization.structuralKey(), bytes,
                specialization.entryType().descriptorString());
    }

    private static CpuKernelIr ir(DataType source, DataType target, AccessForm access) {
        CpuAccessPlan input = access.input();
        CpuAccessPlan output = access.output();
        return new CpuKernelIr(List.of(new CpuKernelIr.Value(0, source,
                CpuKernelIr.Value.Kind.INPUT, input), new CpuKernelIr.Value(1, target,
                CpuKernelIr.Value.Kind.OUTPUT, output)), List.of(new CpuKernelIr.Instruction(
                CpuPointwiseOpcode.CAST, List.of(0), 1)), new CpuKernelIr.Loop("start", "end"),
                List.of(new CpuKernelIr.Store(1, 0)));
    }

    private static void assertColdReuse(CpuClassFileKernelGenerator generator) {
        Form dense = new Form(DataType.FLOAT64, DataType.INT64, CARRIERS.getFirst(), AccessForm.DENSE);
        Generated first = generate(generator, dense), offsetDense = generate(generator, dense);
        CpuAccessPlan densePlan = AccessForm.DENSE.input();
        CpuAccessPlan.Binding contiguous = CpuAccessPlan.Binding.create(densePlan, new long[] {8},
                0, new long[] {1}, 8, 0, 8, 8);
        CpuAccessPlan.Binding offset = CpuAccessPlan.Binding.create(densePlan, new long[] {8},
                11, new long[] {1}, 8, 0, 8, 19);
        assertEquals(0, contiguous.baseElementOffset());
        assertEquals(11, offset.baseElementOffset());
        assertEquals(first.key(), offsetDense.key(), "offset-dense base is cold");
        assertArrayEquals(first.bytes(), offsetDense.bytes(), "offset-dense bytes are cold");
        // Class identity deliberately records scalar compute, not caller orchestration.
        Generated parallelReuse = generate(generator, dense,
                CpuPartitionPreparationPlan.ExecutionStrategy.PARALLEL_SCALAR);
        assertEquals(first.key(), parallelReuse.key(), "parallel callers reuse scalar key");
        assertArrayEquals(first.bytes(), parallelReuse.bytes(), "parallel callers reuse scalar bytes");
    }

    private static void assertSchemaControls(CpuClassFileKernelGenerator generator) {
        Generated schema52 = generate(generator, new Form(DataType.FLOAT64, DataType.FLOAT64,
                CARRIERS.getFirst(), AccessForm.DENSE));
        Generated schema59 = generate(generator, new Form(DataType.BFLOAT16, DataType.BFLOAT16,
                CARRIERS.getFirst(), AccessForm.DENSE));
        Generated schema60 = generate(generator, new Form(DataType.FLOAT64, DataType.FLOAT32,
                CARRIERS.getFirst(), AccessForm.DENSE));
        // These are deliberately non-CAST historical projections.  Their identity schemas are
        // fixed controls, not values inferred by comparing a generated class to itself.
        assertEquals(52, schema52Projection().classIdentitySchema());
        assertEquals(59, schema59Projection().classIdentitySchema());
        assertEquals(schema52Projection().structuralKey(), schema52.key());
        assertEquals(schema59Projection().structuralKey(), schema59.key());
        assertEquals(schema52Projection().entryType().descriptorString(), schema52.descriptor());
        assertEquals(schema59Projection().entryType().descriptorString(), schema59.descriptor());
        assertFalse(schema52.key().equals(schema60.key()));
        assertFalse(schema59.key().equals(schema60.key()));
        assertFalse(java.util.Arrays.equals(schema52.bytes(), schema60.bytes()));
        assertFalse(java.util.Arrays.equals(schema59.bytes(), schema60.bytes()));
        Set<String> schema60Keys = Set.of(schema60.key(), generate(generator, new Form(
                DataType.FLOAT32, DataType.FLOAT64, CARRIERS.getFirst(), AccessForm.DENSE)).key(),
                generate(generator, new Form(DataType.FLOAT64, DataType.FLOAT32, CARRIERS.get(1),
                        AccessForm.DENSE)).key(), generate(generator, new Form(DataType.FLOAT64,
                        DataType.FLOAT32, CARRIERS.getFirst(), AccessForm.BLOCK_OUTER)).key());
        assertEquals(4, schema60Keys.size(), "schema 60 shapes every code fact");
    }

    private static CpuKernelSpecialization schema52Projection() {
        Form form = new Form(DataType.FLOAT64, DataType.FLOAT64, CARRIERS.getFirst(), AccessForm.DENSE);
        CpuKernelIr ir = ir(form.source, form.target, form.access);
        return new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR, List.of(DataType.FLOAT64, DataType.FLOAT64),
                List.of(CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY,
                        CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY), 0, -1, List.of(), false, 52);
    }

    private static CpuKernelSpecialization schema59Projection() {
        Form form = new Form(DataType.BFLOAT16, DataType.BFLOAT16, CARRIERS.getFirst(), AccessForm.DENSE);
        CpuKernelIr ir = ir(form.source, form.target, form.access);
        return new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR, List.of(DataType.BFLOAT16, DataType.BFLOAT16),
                List.of(CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY,
                        CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY), 0, -1, List.of(), false, 59);
    }

    private static void inspect(Generated generated, Form form, Path reports, Path javap) throws Exception {
        var model = ClassFile.of().parse(generated.bytes());
        assertTrue(model.flags().has(AccessFlag.FINAL), "final " + form);
        assertTrue(model.fields().isEmpty(), "field-free " + form);
        assertEquals(1, model.methods().size(), "one typed static entry " + form);
        var entry = model.methods().getFirst();
        assertTrue(entry.flags().has(AccessFlag.STATIC));
        assertEquals("invoke", entry.methodName().stringValue());
        assertEquals(generated.descriptor(), entry.methodType().stringValue(),
                "exact typed entry descriptor " + form);
        List<String> members = new ArrayList<>(), opcodes = new ArrayList<>();
        for (Instruction instruction : entry.code().orElseThrow().elementStream()
                .filter(Instruction.class::isInstance).map(Instruction.class::cast).toList()) {
            Opcode opcode = instruction.opcode(); opcodes.add(opcode.name());
            assertFalse(opcode.name().startsWith("NEW"), "allocation " + opcode);
            assertFalse(opcode == Opcode.INVOKEDYNAMIC || opcode == Opcode.MONITORENTER
                    || opcode == Opcode.MONITOREXIT, "forbidden opcode " + opcode);
        }
        for (var constant : java.util.stream.StreamSupport.stream(model.constantPool().spliterator(), false)
                .toList()) if (constant instanceof MemberRefEntry member) {
            String owner = member.owner().asInternalName();
            members.add(owner + '.' + member.name().stringValue());
            assertFalse(owner.startsWith("io/github/pho001/synaptik"), "helper " + owner);
            assertFalse(owner.startsWith("java/util/") || owner.startsWith("java/lang/reflect")
                    || owner.startsWith("java/lang/invoke") || owner.equals("java/lang/Object"), owner);
        }
        assertEquals(0, java.util.stream.StreamSupport.stream(model.constantPool().spliterator(), false)
                .filter(StringEntry.class::isInstance).count(), "no strings " + form);
        String base = generated.key();
        Files.writeString(reports.resolve(base + ".members.txt"), String.join("\n", members) + "\n");
        Files.writeString(reports.resolve(base + ".descriptor.txt"), entry.methodName().stringValue()
                + entry.methodType().stringValue() + "\n");
        Files.writeString(reports.resolve(base + ".forbidden.txt"), "verified: no helper, allocation, "
                + "boxing, reflection, method handles, invokedynamic, monitor, collection, string, map, "
                + "Object carrier, lookup, or runtime dispatch\n");
        Files.writeString(reports.resolve(base + ".instructions.txt"), String.join("\n", opcodes) + "\n");
        runJavap(javap.resolve(base + ".txt"), generated.bytes());
    }

    private static void runJavap(Path output, byte[] bytes) throws Exception {
        Path temporary = Files.createTempFile("synaptik-cast-evidence-", ".class");
        try {
            Files.write(temporary, bytes);
            Process process = new ProcessBuilder(Path.of(System.getProperty("java.home"), "bin", "javap")
                    .toString(), "-c", "-v", "-p", temporary.toString()).redirectErrorStream(true).start();
            String text = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            assertEquals(0, process.waitFor(), text); Files.writeString(output, text);
        } finally { Files.deleteIfExists(temporary); }
    }

    private static void retainOracle(Path oracle) throws Exception {
        Path source = oracle.resolve("CastOracle.java");
        Files.writeString(source, oracleSource());
        Process compile = new ProcessBuilder(Path.of(System.getProperty("java.home"), "bin", "javac")
                .toString(), "--release", "26", "-d", oracle.toString(), source.toString())
                .redirectErrorStream(true).start();
        String compileText = new String(compile.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
        assertEquals(0, compile.waitFor(), compileText);
        Path classFile = oracle.resolve("CastOracle.class"), javap = oracle.resolve("CastOracle.javap.txt");
        Process decompile = new ProcessBuilder(Path.of(System.getProperty("java.home"), "bin", "javap")
                .toString(), "-c", "-v", "-p", classFile.toString()).redirectErrorStream(true).start();
        String decompilation = new String(decompile.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
        assertEquals(0, decompile.waitFor(), decompilation); Files.writeString(javap, decompilation);
    }

    private static String normalized(byte[] bytes) throws Exception {
        var model = ClassFile.of().parse(bytes); StringBuilder text = new StringBuilder();
        for (Instruction instruction : model.methods().getFirst().code().orElseThrow().elementStream()
                .filter(Instruction.class::isInstance).map(Instruction.class::cast).toList())
            text.append(instruction.opcode().name()).append('\n');
        return sha256(text.toString().getBytes(StandardCharsets.UTF_8));
    }
    private static String sha256(byte[] value) throws Exception { return HexFormat.of().formatHex(
            MessageDigest.getInstance("SHA-256").digest(value)); }
    private static String oracleSource() {
        // This is intentionally self-contained clean Java.  It names each conversion group and
        // retains the three direct BFLOAT16 loops whose generated counterparts are compared.
        return """
                public final class CastOracle {
                  static long identity(long raw) { return raw; }
                  static int boolToNumeric(boolean value) { return value ? 1 : 0; }
                  static boolean numericToBool(double value) { return value != 0.0d; }
                  static int narrowInt64(long value) { return (int) value; }
                  static long widenInt32(int value) { return value; }
                  static double widenFloat32(float value) { int bits = Float.floatToRawIntBits(value);
                    return ((bits & 0x7f800000) == 0x7f800000 && (bits & 0x007fffff) != 0)
                      ? Double.longBitsToDouble(((long) (bits & 0x80000000) << 32)
                          | 0x7ff0000000000000L | ((long) (bits & 0x007fffff) << 29)) : (double) value; }
                  static float narrowFloat64(double value) { long bits = Double.doubleToRawLongBits(value);
                    return ((bits & 0x7ff0000000000000L) == 0x7ff0000000000000L
                        && (bits & 0x000fffffffffffffL) != 0L) ? Float.intBitsToFloat(0x7fc00000) : (float) value; }
                  static long floatingToLong(double value) { return (long) value; }
                  static int floatingToInt(float value) { return (int) value; }
                  static double integralToDouble(long value) { return (double) value; }
                  static float integralToFloat(int value) { return (float) value; }
                  static short directBfloat(double value) { return directBfloatBits(Double.doubleToRawLongBits(value)); }
                  static short directBfloat(long value) { return integerToBfloat(value); }
                  static short directBfloat(int value) { return integerToBfloat(value); }
                  static void f64BfloatDense(double[] source, short[] target, long start, long end) {
                    for (long ordinal = start; ordinal < end; ordinal++) target[(int) ordinal] = directBfloat(source[(int) ordinal]); }
                  static void i64BfloatDense(long[] source, short[] target, long start, long end) {
                    for (long ordinal = start; ordinal < end; ordinal++) target[(int) ordinal] = directBfloat(source[(int) ordinal]); }
                  static void i32BfloatDense(int[] source, short[] target, long start, long end) {
                    for (long ordinal = start; ordinal < end; ordinal++) target[(int) ordinal] = directBfloat(source[(int) ordinal]); }
                  private static short directBfloatBits(long bits) {
                    int sign = (int) ((bits >>> 48) & 0x8000); int rawExponent = (int) ((bits >>> 52) & 0x7ff);
                    long fraction = bits & 0x000fffffffffffffL;
                    if (rawExponent == 0x7ff) return (short) (fraction == 0L ? sign | 0x7f80 : 0x7fc0);
                    if (rawExponent == 0 && fraction == 0L) return (short) sign;
                    long significand; int binaryScale;
                    if (rawExponent == 0) { significand = fraction; binaryScale = -1074; }
                    else { significand = (1L << 52) | fraction; binaryScale = rawExponent - 1075; }
                    int significandBits = Long.SIZE - Long.numberOfLeadingZeros(significand);
                    int exponent = significandBits - 1 + binaryScale;
                    if (exponent < -126) return (short) (sign | (int) roundRightShift(significand, -binaryScale - 133));
                    long rounded = roundRightShift(significand, significandBits - 8);
                    if (rounded == 0x100L) { rounded = 0x80L; exponent++; }
                    return (short) (exponent > 127 ? sign | 0x7f80 : sign | ((exponent + 127) << 7) | ((int) rounded & 0x7f)); }
                  private static short integerToBfloat(long value) {
                    if (value == 0L) return 0; int sign = value < 0L ? 0x8000 : 0; long magnitude = value < 0L ? -value : value;
                    int exponent = Long.SIZE - 1 - Long.numberOfLeadingZeros(magnitude);
                    long rounded = exponent <= 7 ? magnitude << (7 - exponent) : roundRightShift(magnitude, exponent - 7);
                    if (rounded == 0x100L) { rounded = 0x80L; exponent++; }
                    return (short) (sign | ((exponent + 127) << 7) | ((int) rounded & 0x7f)); }
                  private static long roundRightShift(long value, int shift) {
                    if (shift <= 0) return value << -shift; if (shift >= 63) return 0L;
                    long retained = value >>> shift, discarded = value & ((1L << shift) - 1L), midpoint = 1L << (shift - 1);
                    return discarded > midpoint || (discarded == midpoint && (retained & 1L) != 0L) ? retained + 1L : retained; }
                }
                """;
    }

    private static String manifest(Path root) throws Exception { StringBuilder text = new StringBuilder();
        try (var paths = Files.walk(root)) { for (Path path : paths.filter(Files::isRegularFile).sorted().toList()) {
            String relative = root.relativize(path).toString();
            if (!relative.equals("manifest.sha256") && !relative.equals(MARKER))
                text.append(sha256(Files.readAllBytes(path))).append("  ").append(relative).append('\n');
        } }
        return text.toString(); }
    private static void verifyManifest(Path root, Set<String> keys) throws Exception {
        String manifest = Files.readString(root.resolve("manifest.sha256"));
        assertEquals(manifest, manifest(root), "verified manifest");
        Set<String> expected = new HashSet<>(Set.of("commands.txt", "environment.txt", "inventory.tsv",
                "summary.txt", MARKER, "oracle/CastOracle.java", "oracle/CastOracle.class",
                "oracle/CastOracle.javap.txt"));
        if (Files.isRegularFile(root.resolve("evidence-root.txt"))) expected.add("evidence-root.txt");
        for (String key : keys) {
            expected.add("classes/" + key + ".class"); expected.add("javap/" + key + ".txt");
            expected.add("reports/" + key + ".members.txt"); expected.add("reports/" + key + ".descriptor.txt"); expected.add("reports/" + key + ".forbidden.txt");
            expected.add("reports/" + key + ".instructions.txt");
        }
        Set<String> actual = new HashSet<>();
        try (var paths = Files.walk(root)) { paths.filter(Files::isRegularFile).forEach(path -> {
            String relative = root.relativize(path).toString(); if (!relative.equals("manifest.sha256")) actual.add(relative);
        }); }
        assertEquals(expected, actual, "exact retained evidence inventory");
        assertEquals(577, Files.readAllLines(root.resolve("inventory.tsv")).size(), "576 inventory rows");
    }

    private record Generated(String key, byte[] bytes, String descriptor) { }
    private record Form(DataType source, DataType target, CarrierPattern carriers, AccessForm access) { }
    private record CarrierPattern(String name, boolean inputSegment, boolean outputSegment) {
        CpuKernelSpecialization.CarrierAccess source(DataType type) { return inputSegment
                ? CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT : array(type); }
        CpuKernelSpecialization.CarrierAccess target(DataType type) { return outputSegment
                ? CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT : array(type); }
        private static CpuKernelSpecialization.CarrierAccess array(DataType type) { return switch (type) {
            case FLOAT64 -> CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY;
            case FLOAT32 -> CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY;
            case BFLOAT16 -> CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY;
            case INT64 -> CpuKernelSpecialization.CarrierAccess.LONG_ARRAY;
            case INT32 -> CpuKernelSpecialization.CarrierAccess.INT_ARRAY;
            case BOOL -> CpuKernelSpecialization.CarrierAccess.BYTE_ARRAY; }; }
    }
    private enum AccessForm { DENSE, BLOCK_OUTER, GENERAL_ODOMETER, RANK_ZERO;
        CpuAccessPlan input() { return switch (this) {
            case DENSE -> plan(CpuAccessPlan.AccessKind.READ, CpuAccessPlan.Regime.DENSE_LINEAR, 1,
                    List.of(CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
            case BLOCK_OUTER -> plan(CpuAccessPlan.AccessKind.READ, CpuAccessPlan.Regime.BLOCK_OUTER, 2,
                    List.of(CpuAccessPlan.AxisRole.STRIDED, CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
            case GENERAL_ODOMETER -> plan(CpuAccessPlan.AccessKind.READ, CpuAccessPlan.Regime.GENERAL_ODOMETER, 2,
                    List.of(CpuAccessPlan.AxisRole.STRIDED, CpuAccessPlan.AxisRole.STRIDED), 0);
            case RANK_ZERO -> plan(CpuAccessPlan.AccessKind.READ, CpuAccessPlan.Regime.SCALAR_ALL_ZERO, 0, List.of(), 0); }; }
        CpuAccessPlan output() { return switch (this) {
            case DENSE -> plan(CpuAccessPlan.AccessKind.WRITE, CpuAccessPlan.Regime.DENSE_LINEAR, 1,
                    List.of(CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
            case BLOCK_OUTER -> plan(CpuAccessPlan.AccessKind.WRITE, CpuAccessPlan.Regime.BLOCK_OUTER, 2,
                    List.of(CpuAccessPlan.AxisRole.STRIDED, CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
            case GENERAL_ODOMETER -> plan(CpuAccessPlan.AccessKind.WRITE, CpuAccessPlan.Regime.GENERAL_ODOMETER, 2,
                    List.of(CpuAccessPlan.AxisRole.STRIDED, CpuAccessPlan.AxisRole.STRIDED), 0);
            case RANK_ZERO -> plan(CpuAccessPlan.AccessKind.WRITE, CpuAccessPlan.Regime.DENSE_LINEAR, 0, List.of(), 0); }; }
        private static CpuAccessPlan plan(CpuAccessPlan.AccessKind kind, CpuAccessPlan.Regime regime,
                int rank, List<CpuAccessPlan.AxisRole> roles, int suffix) { return new CpuAccessPlan(kind, regime, rank, roles, suffix); }
    }
}
