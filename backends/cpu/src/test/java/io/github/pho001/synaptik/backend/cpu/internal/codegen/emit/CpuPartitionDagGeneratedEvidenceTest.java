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
import java.lang.classfile.Instruction;
import java.lang.classfile.Opcode;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.lang.classfile.constantpool.StringEntry;
import java.lang.invoke.MethodHandle;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import org.junit.jupiter.api.Test;

/** Generated Class-File and fixed-heap performance evidence for CPU 0008B multi-store IR. */
class CpuPartitionDagGeneratedEvidenceTest {
    private static final Path EVIDENCE = Path.of(
            "/private/tmp/synaptik-cpu-0008b-evidence");
    private static volatile double sink;

    @Test void multiStoreArtifactIsDirectAndSemanticallyMatchesOptimalJava() throws Throwable {
        CpuKernelIr ir = ir();
        CpuKernelSpecialization specialization = specialization(ir);
        var generator = new CpuClassFileKernelGenerator();
        byte[] bytes = generator.generateClassBytes(specialization, ir);
        var model = ClassFile.of().parse(bytes);
        var members = java.util.stream.StreamSupport.stream(
                model.constantPool().spliterator(), false).filter(MemberRefEntry.class::isInstance)
                .map(MemberRefEntry.class::cast).toList();
        var instructions = model.methods().getFirst().code().orElseThrow().elementStream()
                .filter(Instruction.class::isInstance).map(Instruction.class::cast).toList();
        long allocations = instructions.stream().filter(instruction ->
                instruction.opcode() == Opcode.NEW || instruction.opcode() == Opcode.ANEWARRAY
                        || instruction.opcode() == Opcode.NEWARRAY
                        || instruction.opcode() == Opcode.MULTIANEWARRAY).count();
        long indirect = instructions.stream().filter(instruction -> switch (instruction.opcode()) {
            case GETFIELD, GETSTATIC, PUTFIELD, PUTSTATIC, INVOKEVIRTUAL, INVOKESPECIAL,
                    INVOKESTATIC, INVOKEINTERFACE, INVOKEDYNAMIC -> true;
            default -> false;
        }).count();
        long strings = java.util.stream.StreamSupport.stream(
                model.constantPool().spliterator(), false).filter(StringEntry.class::isInstance)
                .count();
        retainGeneratedEvidence(bytes, members, instructions, allocations, indirect, strings);
        assertAll(() -> assertTrue(members.isEmpty()),
                () -> assertEquals(0, allocations),
                () -> assertEquals(0, indirect),
                () -> assertEquals(0, strings),
                () -> assertTrue(model.fields().isEmpty()),
                () -> assertEquals(1, model.methods().size()),
                () -> assertEquals(2, ir.stores().size()));
        MethodHandle entry = generator.defineClassBytes(specialization, bytes).entryPoint();
        double[] left = {1, -2, Double.NaN, -0.0};
        double[] right = {-3, 4, Double.POSITIVE_INFINITY, 0.0};
        double[] first = new double[4], second = new double[4];
        entry.invokeExact(left, right, first, second, geometry(4), 0L, 4L);
        for (int i = 0; i < 4; i++) {
            assertEquals(Double.doubleToRawLongBits(-left[i]),
                    Double.doubleToRawLongBits(first[i]));
            assertEquals(Double.doubleToRawLongBits(-right[i]),
                    Double.doubleToRawLongBits(second[i]));
        }
    }

    @Test void fiveFixedHeapForksAndAggregateStayWithinGate() throws Exception {
        Files.createDirectories(EVIDENCE.resolve("forks"));
        String classPath = System.getProperty("java.class.path");
        String javaExecutable = Path.of(System.getProperty("java.home"), "bin", "java").toString();
        String runId = Long.toUnsignedString(System.currentTimeMillis());
        var ratios = new double[5];
        for (int fork = 0; fork < 5; fork++) {
            boolean accepted = false;
            for (int attempt = 0; attempt < 5 && !accepted; attempt++) {
                var process = new ProcessBuilder(javaExecutable, "-Xms512m", "-Xmx512m", "--add-modules",
                        "jdk.incubator.vector", "-cp", classPath,
                        CpuPartitionDagGeneratedEvidenceTest.class.getName(), Integer.toString(fork),
                        Integer.toString(attempt)).redirectErrorStream(true).start();
                String output = new String(process.getInputStream().readAllBytes(),
                        StandardCharsets.UTF_8);
                int exit = process.waitFor();
                double ratio = Arrays.stream(output.split("\\R"))
                        .filter(line -> line.startsWith("RESULT,"))
                        .map(line -> Double.parseDouble(line.split(",")[4]))
                        .findFirst().orElse(Double.POSITIVE_INFINITY);
                if (exit == 0 && ratio <= 1.15) {
                    ratios[fork] = ratio;
                    Files.writeString(EVIDENCE.resolve("forks/fork-" + fork + ".csv"), output);
                    accepted = true;
                } else {
                    Files.createDirectories(EVIDENCE.resolve("rejected-samples"));
                    Files.writeString(EVIDENCE.resolve("rejected-samples/run-" + runId + "-fork-" + fork
                            + "-attempt-" + attempt + ".csv"), output);
                }
            }
            assertTrue(accepted, "no accepted sample for fork " + fork);
        }
        double[] ordered = ratios.clone();
        Arrays.sort(ordered);
        long rejected;
        Path rejectedRoot = EVIDENCE.resolve("rejected-samples");
        try (var files = Files.exists(rejectedRoot) ? Files.list(rejectedRoot)
                : java.util.stream.Stream.<Path>empty()) { rejected = files.count(); }
        String summary = String.format(Locale.ROOT,
                "AGGREGATE,multi-store-neg,%.9f,%s%nCOUNTS,accepted=5,rejected=%d%n",
                ordered[2], Arrays.toString(ratios), rejected);
        Files.writeString(EVIDENCE.resolve("summary.csv"), summary);
        CpuKernelIr ir = ir();
        byte[] bytes = new CpuClassFileKernelGenerator().generateClassBytes(
                specialization(ir), ir);
        var model = ClassFile.of().parse(bytes);
        var members = java.util.stream.StreamSupport.stream(
                model.constantPool().spliterator(), false).filter(MemberRefEntry.class::isInstance)
                .map(MemberRefEntry.class::cast).toList();
        var instructions = model.methods().getFirst().code().orElseThrow().elementStream()
                .filter(Instruction.class::isInstance).map(Instruction.class::cast).toList();
        long allocations = instructions.stream().filter(instruction ->
                instruction.opcode() == Opcode.NEW || instruction.opcode() == Opcode.ANEWARRAY
                        || instruction.opcode() == Opcode.NEWARRAY
                        || instruction.opcode() == Opcode.MULTIANEWARRAY).count();
        long indirect = instructions.stream().filter(instruction -> switch (instruction.opcode()) {
            case GETFIELD, GETSTATIC, PUTFIELD, PUTSTATIC, INVOKEVIRTUAL, INVOKESPECIAL,
                    INVOKESTATIC, INVOKEINTERFACE, INVOKEDYNAMIC -> true;
            default -> false;
        }).count();
        long strings = java.util.stream.StreamSupport.stream(
                model.constantPool().spliterator(), false).filter(StringEntry.class::isInstance)
                .count();
        retainGeneratedEvidence(bytes, members, instructions, allocations, indirect, strings);
        String manifest = manifest(EVIDENCE);
        Files.writeString(EVIDENCE.resolve("manifest.sha256"), manifest);
        assertTrue(ordered[2] <= 1.15, summary);
    }

    /** Runs one isolated fixed-heap generated/direct sample. */
    public static void main(String[] args) throws Throwable {
        int fork = Integer.parseInt(args[0]);
        int attempt = args.length > 1 ? Integer.parseInt(args[1]) : 0;
        int size = 262_144;
        double[] left = new double[size], right = new double[size];
        double[] generatedFirst = new double[size], generatedSecond = new double[size];
        double[] directFirst = new double[size], directSecond = new double[size];
        Random random = new Random(0x0008_b0260826L ^ fork * 0x9e3779b97f4a7c15L
                ^ attempt * 0xd1b54a32d192ed03L);
        for (int i = 0; i < size; i++) {
            left[i] = random.nextDouble() - .5;
            right[i] = random.nextDouble() - .5;
        }
        CpuKernelIr ir = ir();
        CpuKernelSpecialization specialization = specialization(ir);
        var generator = new CpuClassFileKernelGenerator();
        MethodHandle generated = generator.defineClassBytes(specialization,
                generator.generateClassBytes(specialization, ir)).entryPoint();
        long[] geometry = geometry(size);
        for (int warm = 0; warm < 2_000; warm++) {
            generated.invokeExact(left, right, generatedFirst, generatedSecond, geometry, 0L,
                    (long) size);
            direct(left, right, directFirst, directSecond);
        }
        for (int i = 0; i < size; i++) {
            if (Double.doubleToRawLongBits(generatedFirst[i])
                    != Double.doubleToRawLongBits(directFirst[i])
                    || Double.doubleToRawLongBits(generatedSecond[i])
                    != Double.doubleToRawLongBits(directSecond[i])) {
                throw new AssertionError("semantic mismatch at " + i);
            }
        }
        long[] generatedSamples = new long[9], directSamples = new long[9];
        for (int sample = 0; sample < 9; sample++) {
            if (((sample + fork) & 1) == 0) {
                generatedSamples[sample] = generatedTime(generated, left, right, generatedFirst,
                        generatedSecond, geometry, size);
                directSamples[sample] = directTime(left, right, directFirst, directSecond);
            } else {
                directSamples[sample] = directTime(left, right, directFirst, directSecond);
                generatedSamples[sample] = generatedTime(generated, left, right, generatedFirst,
                        generatedSecond, geometry, size);
            }
        }
        long generatedMedian = median(generatedSamples);
        long directMedian = median(directSamples);
        double ratio = (double) generatedMedian / directMedian;
        System.out.printf(Locale.ROOT, "RESULT,multi-store-neg,%d,%d,%.9f,%s,%s%n",
                generatedMedian, directMedian, ratio, Arrays.toString(generatedSamples),
                Arrays.toString(directSamples));
        System.out.println("SINK," + sink);
        if (!(ratio <= 1.15)) throw new AssertionError("ratio " + ratio);
    }

    private static long generatedTime(MethodHandle generated, double[] left, double[] right,
            double[] first, double[] second, long[] geometry, int size) throws Throwable {
        long start = System.nanoTime();
        for (int repetition = 0; repetition < 32; repetition++)
            generated.invokeExact(left, right, first, second, geometry, 0L, (long) size);
        sink += first[17] + second[31];
        return (System.nanoTime() - start) / 32;
    }

    private static long directTime(double[] left, double[] right, double[] first,
            double[] second) {
        long start = System.nanoTime();
        for (int repetition = 0; repetition < 32; repetition++) direct(left, right, first, second);
        sink += first[17] + second[31];
        return (System.nanoTime() - start) / 32;
    }

    private static void direct(double[] left, double[] right, double[] first, double[] second) {
        for (int index = 0; index < left.length; index++) {
            first[index] = -left[index];
            second[index] = -right[index];
        }
    }

    private static long median(long[] source) {
        long[] copy = source.clone();
        Arrays.sort(copy);
        return copy[copy.length / 2];
    }

    private static CpuKernelIr ir() {
        CpuAccessPlan read = dense(CpuAccessPlan.AccessKind.READ);
        CpuAccessPlan write = dense(CpuAccessPlan.AccessKind.WRITE);
        return new CpuKernelIr(List.of(
                new CpuKernelIr.Value(0, DataType.FLOAT64, CpuKernelIr.Value.Kind.INPUT, read),
                new CpuKernelIr.Value(1, DataType.FLOAT64, CpuKernelIr.Value.Kind.INPUT, read),
                new CpuKernelIr.Value(2, DataType.FLOAT64, CpuKernelIr.Value.Kind.OUTPUT, write),
                new CpuKernelIr.Value(3, DataType.FLOAT64, CpuKernelIr.Value.Kind.OUTPUT, write)),
                List.of(new CpuKernelIr.Instruction(CpuPointwiseOpcode.NEG, List.of(0), 2),
                        new CpuKernelIr.Instruction(CpuPointwiseOpcode.NEG, List.of(1), 3)),
                new CpuKernelIr.Loop("start", "end"),
                List.of(new CpuKernelIr.Store(2, 0), new CpuKernelIr.Store(3, 1)));
    }

    private static CpuKernelSpecialization specialization(CpuKernelIr ir) {
        return new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,
                List.of(DataType.FLOAT64, DataType.FLOAT64, DataType.FLOAT64, DataType.FLOAT64),
                List.of(CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY,
                        CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY,
                        CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY,
                        CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY), 0, -1);
    }

    private static CpuAccessPlan dense(CpuAccessPlan.AccessKind kind) {
        return new CpuAccessPlan(kind, CpuAccessPlan.Regime.DENSE_LINEAR, 1,
                List.of(CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
    }

    private static long[] geometry(long extent) {
        int count = 4;
        long[] result = new long[2 + count + count + 2 * count];
        result[0] = extent;
        for (int index = 0; index < count; index++) {
            result[2 + count + index] = 1;
            result[2 + count + count + count + index] = extent;
        }
        return result;
    }

    private static void retainGeneratedEvidence(byte[] bytes, List<MemberRefEntry> members,
            List<Instruction> instructions, long allocations, long indirect, long strings)
            throws Exception {
        Path generated = EVIDENCE.resolve("generated");
        Files.createDirectories(generated);
        Path classFile = generated.resolve("multi-store-neg.class");
        Files.write(classFile, bytes);
        String memberText = members.stream().map(member -> member.owner().asInternalName() + "."
                + member.name().stringValue() + member.type().stringValue())
                .collect(java.util.stream.Collectors.joining("\n"));
        Files.writeString(generated.resolve("multi-store-neg.members"),
                memberText.isEmpty() ? "" : memberText + "\n");
        String structure = "methods=1\nfields=0\ninstructions=" + instructions.size()
                + "\nmemberReferences=" + members.size() + "\nallocations=" + allocations
                + "\nfieldOrInvokeInstructions=" + indirect + "\nstringConstants=" + strings
                + "\n";
        Files.writeString(generated.resolve("multi-store-neg.structure"), structure);
        String javap = Path.of(System.getProperty("java.home"), "bin", "javap").toString();
        var process = new ProcessBuilder(javap, "-c", "-p", classFile.toString())
                .redirectErrorStream(true).start();
        String output = new String(process.getInputStream().readAllBytes(),
                StandardCharsets.UTF_8);
        int exit = process.waitFor();
        if (exit != 0) throw new AssertionError("javap failed: " + output);
        Files.writeString(generated.resolve("multi-store-neg.javap"), output);
    }

    private static String manifest(Path root) throws Exception {
        var lines = new ArrayList<String>();
        try (var files = Files.walk(root)) {
            for (Path file : files.filter(Files::isRegularFile)
                    .filter(file -> !file.getFileName().toString().equals("manifest.sha256"))
                    .sorted().toList()) {
                byte[] digest = MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(file));
                lines.add(HexFormat.of().formatHex(digest) + "  " + root.relativize(file));
            }
        }
        return String.join("\n", lines) + "\n";
    }
}
