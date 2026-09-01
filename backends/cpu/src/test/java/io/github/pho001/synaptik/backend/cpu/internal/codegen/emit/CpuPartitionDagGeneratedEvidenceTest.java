package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFusionDecision;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionDagDecomposer;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionLowering;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratorSchema;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
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
import java.util.Map;
import java.util.Optional;
import java.util.Random;
import org.junit.jupiter.api.Test;

/** Generated Class-File and fixed-heap performance evidence for CPU 0008B and CPU 0008D. */
class CpuPartitionDagGeneratedEvidenceTest {
    private static final Path EVIDENCE = Path.of(
            "/private/tmp/synaptik-cpu-0008b-evidence");
    private static final Path FUSION_EVIDENCE = Path.of(
            "/private/tmp/synaptik-cpu-0008d-evidence");
    private static final double GENERATED_DIRECT_MEDIAN_GATE = 1.25;
    private static final double GENERATED_DIRECT_FORK_GATE = 1.40;
    private static final double FUSED_SPLIT_MEDIAN_GATE = 0.90;
    private static final double FUSED_SPLIT_FORK_GATE = 1.10;
    private static volatile double sink;

    @Test void decisionFactsStayOutsideStructuralAndClassFileIdentity() {
        var plan = CpuPartitionPreparerTest.analyze(Shape.of(8)).plan();
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        byte[] first = generator.generateClassBytes(route.specialization(), route.kernelIr());
        byte[] second = generator.generateClassBytes(route.specialization(), route.kernelIr());
        String constants = new String(first, StandardCharsets.ISO_8859_1);
        assertAll(
                () -> assertFalse(plan.fusionDecisions().isEmpty()),
                () -> assertEquals(57, CpuGeneratorSchema.CURRENT_VERSION),
                () -> assertArrayEquals(first, second),
                () -> assertFalse(constants.contains("CpuFusionDecision")),
                () -> assertFalse(constants.contains("CpuFusionProfitabilitySelector")),
                () -> assertFalse(constants.contains("CpuSpecializedSubgraph")),
                () -> assertFalse(constants.contains("CpuPartitionPreparationPlan")));
    }

    @Test void freshFusionEvidenceRetainsEverySampleAndBothRequiredComparisons()
            throws Exception {
        var plan = new CpuPartitionPreparer().analyze(fusionContext(8)).plan();
        var selection = plan.fusionDecisions().stream()
                .filter(CpuFusionDecision.Selection.class::isInstance)
                .map(CpuFusionDecision.Selection.class::cast).findFirst().orElseThrow();
        CpuKernelIr fusedIr = plan.units().getFirst().portablePlan().kernelIr();
        assertAll(
                () -> assertEquals(CpuFusionDecision.SelectionReason.PROFITABLE_FUSION,
                        selection.reason()),
                () -> assertEquals(1, plan.units().size()),
                () -> assertEquals(2, selection.canonicalSplit().units().size()),
                () -> assertEquals(2, fusedIr.instructions().size()));

        String runId = Long.toUnsignedString(System.currentTimeMillis());
        Path runRoot = FUSION_EVIDENCE.resolve("run-" + runId);
        Files.createDirectories(runRoot.resolve("forks"));
        var enumeration = new CpuPartitionDagDecomposer().enumerate(fusionContext(8),
                new CpuPartitionLowering(), List.of());
        CpuKernelIr firstIr = enumeration.canonicalSplit().get(0).lowering().kernelIr();
        CpuKernelIr secondIr = enumeration.canonicalSplit().get(1).lowering().kernelIr();
        retainFusionClassEvidence(runRoot, "selected-fused-add", fusedIr);
        retainFusionClassEvidence(runRoot, "canonical-split-add-1", firstIr);
        retainFusionClassEvidence(runRoot, "canonical-split-add-2", secondIr);

        String classPath = System.getProperty("java.class.path");
        String javaExecutable = Path.of(System.getProperty("java.home"), "bin", "java")
                .toString();
        var generatedDirectRatios = new ArrayList<Double>();
        var fusedSplitRatios = new ArrayList<Double>();
        var generatedDirectForkMedians = new ArrayList<Double>();
        var fusedSplitForkMedians = new ArrayList<Double>();
        for (int fork = 0; fork < 5; fork++) {
            var process = new ProcessBuilder(javaExecutable, "-Xms512m", "-Xmx512m",
                    "--add-modules", "jdk.incubator.vector", "-cp", classPath,
                    CpuPartitionDagGeneratedEvidenceTest.class.getName(), "0008d",
                    Integer.toString(fork)).redirectErrorStream(true).start();
            String output = new String(process.getInputStream().readAllBytes(),
                    StandardCharsets.UTF_8);
            int exit = process.waitFor();
            Files.writeString(runRoot.resolve("forks/fork-" + fork + ".csv"), output);
            assertEquals(0, exit, "0008D performance fork failed:\n" + output);
            for (String line : output.split("\\R")) {
                String[] fields = line.split(",");
                if (fields.length == 7 && fields[0].equals("SAMPLE")) {
                    (fields[1].equals("generated-direct") ? generatedDirectRatios
                            : fusedSplitRatios).add(Double.parseDouble(fields[6]));
                } else if (fields.length == 6 && fields[0].equals("FORK")) {
                    (fields[1].equals("generated-direct") ? generatedDirectForkMedians
                            : fusedSplitForkMedians).add(Double.parseDouble(fields[3]));
                }
            }
        }
        assertAll(() -> assertEquals(45, generatedDirectRatios.size()),
                () -> assertEquals(45, fusedSplitRatios.size()),
                () -> assertEquals(5, generatedDirectForkMedians.size()),
                () -> assertEquals(5, fusedSplitForkMedians.size()));
        double generatedDirectMedian = median(generatedDirectRatios);
        double fusedSplitMedian = median(fusedSplitRatios);
        double generatedDirectMaximumFork = generatedDirectForkMedians.stream()
                .mapToDouble(Double::doubleValue).max().orElseThrow();
        double fusedSplitMaximumFork = fusedSplitForkMedians.stream()
                .mapToDouble(Double::doubleValue).max().orElseThrow();
        String summary = String.format(Locale.ROOT,
                "DISTRIBUTION,generated-direct,count=%d,min=%.9f,median=%.9f,max=%.9f,"
                + "forkMedians=%s,samples=%s%n"
                + "DISTRIBUTION,fused-split,count=%d,min=%.9f,median=%.9f,max=%.9f,"
                + "forkMedians=%s,samples=%s%n"
                + "GATES,generatedMedian<=%.2f,generatedFork<=%.2f,fusedMedian<=%.2f,"
                + "fusedFork<=%.2f%n"
                + "RATIONALE,aggregate medians absorb scheduler noise; per-fork ceilings detect "
                + "a consistently regressed isolated process; topology comparison requires a "
                + "clear aggregate win while allowing one noisy fork%n",
                generatedDirectRatios.size(), minimum(generatedDirectRatios),
                generatedDirectMedian, maximum(generatedDirectRatios),
                generatedDirectForkMedians, generatedDirectRatios,
                fusedSplitRatios.size(), minimum(fusedSplitRatios), fusedSplitMedian,
                maximum(fusedSplitRatios), fusedSplitForkMedians, fusedSplitRatios,
                GENERATED_DIRECT_MEDIAN_GATE, GENERATED_DIRECT_FORK_GATE,
                FUSED_SPLIT_MEDIAN_GATE, FUSED_SPLIT_FORK_GATE);
        Files.writeString(runRoot.resolve("summary.csv"), summary);
        Files.writeString(runRoot.resolve("manifest.sha256"), manifest(runRoot));
        Files.writeString(FUSION_EVIDENCE.resolve("latest-run.txt"), runRoot + "\n");
        assertAll(
                () -> assertTrue(generatedDirectMedian <= GENERATED_DIRECT_MEDIAN_GATE, summary),
                () -> assertTrue(generatedDirectMaximumFork <= GENERATED_DIRECT_FORK_GATE, summary),
                () -> assertTrue(fusedSplitMedian <= FUSED_SPLIT_MEDIAN_GATE, summary),
                () -> assertTrue(fusedSplitMaximumFork <= FUSED_SPLIT_FORK_GATE, summary));
    }

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
        if (args.length > 0 && args[0].equals("0008d")) {
            runFusionFork(Integer.parseInt(args[1]));
            return;
        }
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

    private static void runFusionFork(int fork) throws Throwable {
        int size = 65_536;
        int[] input = new int[size], generatedOutput = new int[size], directOutput = new int[size];
        int[] splitIntermediate = new int[size], splitOutput = new int[size];
        Random random = new Random(0x0008_d0260827L ^ fork * 0x9e3779b97f4a7c15L);
        for (int index = 0; index < size; index++) input[index] = random.nextInt();
        var generator = new CpuClassFileKernelGenerator();
        var context = fusionContext(size);
        CpuKernelIr fusedIr = new CpuPartitionPreparer().analyze(context).plan()
                .units().getFirst().portablePlan().kernelIr();
        var enumeration = new CpuPartitionDagDecomposer().enumerate(context,
                new CpuPartitionLowering(), List.of());
        CpuKernelIr firstIr = enumeration.canonicalSplit().get(0).lowering().kernelIr();
        CpuKernelIr secondIr = enumeration.canonicalSplit().get(1).lowering().kernelIr();
        MethodHandle fused = generatedHandle(generator, fusedIr);
        MethodHandle first = generatedHandle(generator, firstIr);
        MethodHandle second = generatedHandle(generator, secondIr);
        long[] geometry = geometry(2, size);
        for (int warm = 0; warm < 1_000; warm++) {
            fused.invokeExact(input, generatedOutput, geometry, 0L, (long) size);
            directFused(input, directOutput);
            first.invokeExact(input, splitIntermediate, geometry, 0L, (long) size);
            second.invokeExact(splitIntermediate, splitOutput, geometry, 0L, (long) size);
        }
        assertIntArraysEqual(generatedOutput, directOutput, "generated/direct");
        assertIntArraysEqual(generatedOutput, splitOutput, "fused/split");
        long[] generated = new long[9], direct = new long[9];
        long[] fusedTimes = new long[9], splitTimes = new long[9];
        double[] generatedDirectRatios = new double[9], fusedSplitRatios = new double[9];
        for (int sample = 0; sample < 9; sample++) {
            if (((sample + fork) & 1) == 0) {
                generated[sample] = fusedTime(fused, input, generatedOutput, geometry, size);
                direct[sample] = directFusedTime(input, directOutput);
                fusedTimes[sample] = fusedTime(fused, input, generatedOutput, geometry, size);
                splitTimes[sample] = splitTime(first, second, input, splitIntermediate,
                        splitOutput, geometry, size);
            } else {
                direct[sample] = directFusedTime(input, directOutput);
                generated[sample] = fusedTime(fused, input, generatedOutput, geometry, size);
                splitTimes[sample] = splitTime(first, second, input, splitIntermediate,
                        splitOutput, geometry, size);
                fusedTimes[sample] = fusedTime(fused, input, generatedOutput, geometry, size);
            }
            generatedDirectRatios[sample] = (double) generated[sample] / direct[sample];
            fusedSplitRatios[sample] = (double) fusedTimes[sample] / splitTimes[sample];
            System.out.printf(Locale.ROOT, "SAMPLE,generated-direct,%d,%d,%d,%d,%.9f%n",
                    fork, sample, generated[sample], direct[sample],
                    generatedDirectRatios[sample]);
            System.out.printf(Locale.ROOT, "SAMPLE,fused-split,%d,%d,%d,%d,%.9f%n",
                    fork, sample, fusedTimes[sample], splitTimes[sample],
                    fusedSplitRatios[sample]);
        }
        System.out.printf(Locale.ROOT, "FORK,generated-direct,%d,%.9f,%.9f,%.9f%n", fork,
                median(generatedDirectRatios), minimum(generatedDirectRatios),
                maximum(generatedDirectRatios));
        System.out.printf(Locale.ROOT, "FORK,fused-split,%d,%.9f,%.9f,%.9f%n", fork,
                median(fusedSplitRatios), minimum(fusedSplitRatios),
                maximum(fusedSplitRatios));
        System.out.println("SINK," + sink);
    }

    private static MethodHandle generatedHandle(CpuClassFileKernelGenerator generator,
            CpuKernelIr ir) {
        CpuKernelSpecialization specialization = intSpecialization(ir);
        return generator.defineClassBytes(specialization,
                generator.generateClassBytes(specialization, ir)).entryPoint();
    }

    private static long fusedTime(MethodHandle fused, int[] input, int[] output,
            long[] geometry, int size) throws Throwable {
        long start = System.nanoTime();
        for (int repetition = 0; repetition < 64; repetition++)
            fused.invokeExact(input, output, geometry, 0L, (long) size);
        sink += output[17];
        return (System.nanoTime() - start) / 64;
    }

    private static long splitTime(MethodHandle first, MethodHandle second, int[] input,
            int[] intermediate, int[] output, long[] geometry, int size) throws Throwable {
        long start = System.nanoTime();
        for (int repetition = 0; repetition < 64; repetition++) {
            first.invokeExact(input, intermediate, geometry, 0L, (long) size);
            second.invokeExact(intermediate, output, geometry, 0L, (long) size);
        }
        sink += output[31];
        return (System.nanoTime() - start) / 64;
    }

    private static long directFusedTime(int[] input, int[] output) {
        long start = System.nanoTime();
        for (int repetition = 0; repetition < 64; repetition++) directFused(input, output);
        sink += output[47];
        return (System.nanoTime() - start) / 64;
    }

    private static void directFused(int[] input, int[] output) {
        for (int index = 0; index < input.length; index++) output[index] = input[index] + 1 + 2;
    }

    private static void assertIntArraysEqual(int[] actual, int[] expected, String comparison) {
        for (int index = 0; index < actual.length; index++) {
            if (actual[index] != expected[index]) throw new AssertionError(
                    comparison + " semantic mismatch at " + index);
        }
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

    private static double median(double[] source) {
        double[] copy = source.clone();
        Arrays.sort(copy);
        return copy[copy.length / 2];
    }

    private static double median(List<Double> source) {
        return median(source.stream().mapToDouble(Double::doubleValue).toArray());
    }

    private static double minimum(double[] source) {
        return Arrays.stream(source).min().orElseThrow();
    }

    private static double maximum(double[] source) {
        return Arrays.stream(source).max().orElseThrow();
    }

    private static double minimum(List<Double> source) {
        return source.stream().mapToDouble(Double::doubleValue).min().orElseThrow();
    }

    private static double maximum(List<Double> source) {
        return source.stream().mapToDouble(Double::doubleValue).max().orElseThrow();
    }

    private static CpuKernelSpecialization intSpecialization(CpuKernelIr ir) {
        return new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,
                List.of(DataType.INT32, DataType.INT32),
                List.of(CpuKernelSpecialization.CarrierAccess.INT_ARRAY,
                        CpuKernelSpecialization.CarrierAccess.INT_ARRAY), 0, -1);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> fusionContext(long elementCount) {
        Shape shape = Shape.of(elementCount);
        var descriptor = new TensorDescriptor(DataType.INT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        var nodes = List.of(
                new CompiledNode(new NodeId(0), new Operation(ScalarElementwiseKind.ADD,
                        new ScalarValueAttrs(ScalarValue.int32(1))), List.of(new ValueId(0)),
                        List.of(new ValueId(1))),
                new CompiledNode(new NodeId(1), new Operation(ScalarElementwiseKind.ADD,
                        new ScalarValueAttrs(ScalarValue.int32(2))), List.of(new ValueId(1)),
                        List.of(new ValueId(2))));
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                nodes.stream().map(CompiledNode::id).toList());
        var values = java.util.stream.LongStream.rangeClosed(0, 2)
                .mapToObj(index -> new GraphValue(new ValueId(index), descriptor)).toList();
        var memory = List.of(
                new LogicalMemoryRequirement(new ValueId(0), descriptor, Optional.empty(),
                        List.of(partition), false),
                new LogicalMemoryRequirement(new ValueId(1), descriptor, Optional.of(partition),
                        List.of(partition), false),
                new LogicalMemoryRequirement(new ValueId(2), descriptor, Optional.of(partition),
                        List.of(), true));
        return new PrepareContext<>(partition, nodes, values, memory, Map.of(),
                CpuPartitionAnalysisInputs.DEFAULT);
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
        return geometry(4, extent);
    }

    private static long[] geometry(int count, long extent) {
        long[] result = new long[2 + count + count + 2 * count];
        result[0] = extent;
        for (int index = 0; index < count; index++) {
            result[2 + count + index] = 1;
            result[2 + count + count + count + index] = extent;
        }
        return result;
    }

    private static void retainFusionClassEvidence(Path runRoot, String name, CpuKernelIr ir)
            throws Exception {
        var generator = new CpuClassFileKernelGenerator();
        byte[] bytes = generator.generateClassBytes(intSpecialization(ir), ir);
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
        long dispatch = instructions.stream().filter(instruction -> switch (instruction.opcode()) {
            case GETFIELD, GETSTATIC, PUTFIELD, PUTSTATIC, INVOKEVIRTUAL, INVOKESPECIAL,
                    INVOKESTATIC, INVOKEINTERFACE, INVOKEDYNAMIC -> true;
            default -> false;
        }).count();
        long strings = java.util.stream.StreamSupport.stream(
                model.constantPool().spliterator(), false).filter(StringEntry.class::isInstance)
                .count();
        Path generated = runRoot.resolve("generated");
        Files.createDirectories(generated);
        Path classFile = generated.resolve(name + ".class");
        Files.write(classFile, bytes);
        Files.writeString(generated.resolve(name + ".members"), members.stream()
                .map(member -> member.owner().asInternalName() + "." + member.name().stringValue()
                        + member.type().stringValue())
                .collect(java.util.stream.Collectors.joining("\n")));
        Files.writeString(generated.resolve(name + ".structure"),
                "methods=" + model.methods().size() + "\nfields=" + model.fields().size()
                + "\ninstructions=" + instructions.size() + "\nmemberReferences="
                + members.size() + "\nallocations=" + allocations + "\ndispatchInstructions="
                + dispatch + "\nstringConstants=" + strings + "\nstructuralKey="
                + ir.structuralKey() + "\n");
        String javap = Path.of(System.getProperty("java.home"), "bin", "javap").toString();
        var process = new ProcessBuilder(javap, "-c", "-v", "-p", classFile.toString())
                .redirectErrorStream(true).start();
        String output = new String(process.getInputStream().readAllBytes(),
                StandardCharsets.UTF_8);
        int exit = process.waitFor();
        Files.writeString(generated.resolve(name + ".javap"), output);
        assertAll(() -> assertEquals(0, exit, output), () -> assertTrue(members.isEmpty()),
                () -> assertEquals(0, allocations), () -> assertEquals(0, dispatch),
                () -> assertEquals(0, strings), () -> assertTrue(model.fields().isEmpty()),
                () -> assertEquals(1, model.methods().size()));
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
