package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.io.BufferedReader;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/**
 * Opt-in retained five-fork generated-versus-direct performance gate for CPU 0008J.
 *
 * <p>Each direct side is a fixed primitive Java loop for the exact generated shape: it uses raw
 * {@code short} BFLOAT16 carriers, expands numerical inputs, performs the same FLOAT32/binary64
 * operation, and writes one rounded raw result at each logical node.  Forks retain every result;
 * no sample is retried, discarded, or averaged away.  This owner is deliberately excluded from
 * ordinary CPU test runs unless its explicit environment flag and evidence root are supplied.</p>
 */
class CpuBFloat16PointwisePerformanceTest {
    private static final String ENABLE = "SYNAPTIK_CPU_BFLOAT16_POINTWISE_PERFORMANCE";
    private static final String ROOT = "SYNAPTIK_CPU_BFLOAT16_POINTWISE_EVIDENCE_ROOT";
    private static final int FORKS = 5;
    private static final int WARMUPS = 5;
    private static final long MINIMUM_NANOS = 25_000_000L;
    private static final double LIMIT = 1.15d;
    private static final int COUNT = 65_536;
    private static volatile long checksum;

    enum Row { GELU_ARRAY, POW_SEGMENT, CLAMP_MIXED_GENERAL, COMPARISON_BROADCAST, FUSED_WHERE }

    public static void main(String[] arguments) throws Exception {
        if (arguments.length == 2 && arguments[0].equals("--fork"))
            runFork(Path.of(requireRoot()), Integer.parseInt(arguments[1]));
        else runParent(Path.of(requireRoot()));
    }

    @Test void retainedFiveForkProtocol() throws Exception {
        Assumptions.assumeTrue("true".equals(System.getenv(ENABLE)));
        runParent(Path.of(requireRoot()));
    }

    @Test void exactInventoryIsFiveDistinctGeneratedShapes() {
        assertTrue(Row.values().length == 5);
    }

    @Test void directOraclesMatchGeneratedRawResults() {
        for (Row row : Row.values()) verifyEquivalent(row, prepare(row));
    }

    @Test void retainsExactBenchmarkGeneratedClassFilesWhenExplicitlyRequested() throws Exception {
        String requested = System.getProperty("synaptik.cpu.0008j.evidenceRoot");
        Assumptions.assumeTrue(requested != null && !requested.isBlank());
        Path directory = Path.of(requested).resolve("generated-classes");
        Files.createDirectories(directory);
        var generator = new CpuClassFileKernelGenerator();
        for (Row row : Row.values()) {
            CpuKernelIr kernelIr = ir(row);
            Files.write(directory.resolve(classFileName(row)), generator.generateClassBytes(
                    specialization(row, kernelIr), kernelIr));
        }
    }

    private static String requireRoot() {
        String root = System.getenv(ROOT);
        if (root == null || root.isBlank()) throw new IllegalArgumentException(ROOT + " is required");
        return root;
    }

    private static void runParent(Path root) throws Exception {
        Files.createDirectories(root);
        Files.writeString(root.resolve("bfloat16-pointwise-performance-protocol.txt"),
                "forks=5\nwarmups=5\nminimum_side_ns=25000000\norder=seeded-randomized\n"
                + "retry=false\ndiscard=false\nfixed_heap=-Xms1g,-Xmx1g\nthreshold=1.15\n"
                + "rows=GELU_ARRAY,POW_SEGMENT,CLAMP_MIXED_GENERAL,COMPARISON_BROADCAST,FUSED_WHERE\n");
        Path retainedSource = root.resolve("CpuBFloat16PointwisePerformanceTest.java");
        Files.copy(sourceFile(), retainedSource, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        try (var resource = CpuBFloat16PointwisePerformanceTest.class.getResourceAsStream(
                "CpuBFloat16PointwisePerformanceTest.class")) {
            if (resource == null) throw new IllegalStateException("performance owner class missing");
            Files.write(root.resolve("CpuBFloat16PointwisePerformanceTest.class"),
                    resource.readAllBytes());
        }
        retainOracleDecompilation(root);
        for (int fork = 0; fork < FORKS; fork++) {
            List<String> command = List.of(Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                    "-Xms1g", "-Xmx1g", "-XX:-TieredCompilation", "-Xbatch", "--add-modules",
                    "jdk.incubator.vector", "-cp", System.getProperty("java.class.path"),
                    CpuBFloat16PointwisePerformanceTest.class.getName(), "--fork", Integer.toString(fork));
            Process process = new ProcessBuilder(command).redirectOutput(root.resolve("bfloat16-fork-"
                    + fork + ".stdout").toFile()).redirectError(root.resolve("bfloat16-fork-" + fork
                    + ".stderr").toFile()).start();
            if (process.waitFor() != 0) throw new AssertionError("fork " + fork + " failed");
            validate(root.resolve("bfloat16-raw-fork-" + fork + ".csv"));
        }
        StringBuilder report = new StringBuilder("row,fork0,fork1,fork2,fork3,fork4,median,accepted\n");
        for (Row row : Row.values()) {
            double[] ratios = new double[FORKS];
            for (int fork = 0; fork < FORKS; fork++) ratios[fork] = ratio(root.resolve(
                    "bfloat16-raw-fork-" + fork + ".csv"), row);
            double[] sorted = ratios.clone(); Arrays.sort(sorted);
            assertTrue(sorted[2] <= LIMIT, row + " median=" + sorted[2]);
            report.append(row).append(',').append(ratios[0]).append(',').append(ratios[1]).append(',')
                    .append(ratios[2]).append(',').append(ratios[3]).append(',').append(ratios[4])
                    .append(',').append(sorted[2]).append(",true\n");
        }
        Files.writeString(root.resolve("bfloat16-pointwise-performance-report.csv"), report.toString());
    }

    private static void runFork(Path root, int fork) throws Exception {
        Random random = new Random(0x8_0000_0000L + fork);
        StringBuilder csv = new StringBuilder("row,iterations,statistic,ratio,checksum\n");
        for (Row row : Row.values()) {
            Prepared prepared = prepare(row);
            verifyEquivalent(row, prepared);
            int iterations = calibrate(row, prepared);
            for (int warmup = 0; warmup < WARMUPS; warmup++) {
                if (random.nextBoolean()) { elapsed(row, prepared, true, iterations); elapsed(row, prepared, false, iterations); }
                else { elapsed(row, prepared, false, iterations); elapsed(row, prepared, true, iterations); }
            }
            var ratios = new ArrayList<Double>(9);
            long shortestGenerated = Long.MAX_VALUE;
            long shortestDirect = Long.MAX_VALUE;
            StringBuilder measurements = new StringBuilder("row,sample,iterations,generated_ns,direct_ns,ratio\n");
            for (int sample = 0; sample < 9; sample++) {
                boolean generatedFirst = random.nextBoolean();
                long first = elapsed(row, prepared, generatedFirst, iterations);
                long second = elapsed(row, prepared, !generatedFirst, iterations);
                long generated = generatedFirst ? first : second;
                long direct = generatedFirst ? second : first;
                shortestGenerated = Math.min(shortestGenerated, generated);
                shortestDirect = Math.min(shortestDirect, direct);
                double sampleRatio = (double) generated / direct;
                ratios.add(sampleRatio);
                measurements.append(row).append(',').append(sample).append(',').append(iterations)
                        .append(',').append(generated).append(',').append(direct).append(',')
                        .append(sampleRatio).append('\n');
            }
            ratios.sort(Double::compare);
            double ratio = ratios.get(4);
            Files.writeString(root.resolve("bfloat16-measurements-fork-" + fork + "-" + row
                    + ".csv"), measurements.toString());
            assertTrue(shortestGenerated >= MINIMUM_NANOS,
                    row + " generated sample below 25 ms: " + shortestGenerated);
            assertTrue(shortestDirect >= MINIMUM_NANOS,
                    row + " direct sample below 25 ms: " + shortestDirect);
            assertTrue(ratio <= LIMIT, row + " fork " + fork + " ratio=" + ratio);
            csv.append(row).append(',').append(iterations).append(",median-of-nine,")
                    .append(ratio).append(',').append(checksum).append('\n');
        }
        Files.writeString(root.resolve("bfloat16-raw-fork-" + fork + ".csv"), csv.toString());
    }

    private static int calibrate(Row row, Prepared prepared) {
        int iterations = 1;
        while (Math.min(elapsed(row, prepared, true, iterations),
                elapsed(row, prepared, false, iterations)) < MINIMUM_NANOS) iterations <<= 1;
        return iterations;
    }

    private static long elapsed(Row row, Prepared prepared, boolean generated, int iterations) {
        long start = System.nanoTime();
        for (int iteration = 0; iteration < iterations; iteration++) {
            if (generated) invokeGenerated(row, prepared); else invokeDirect(row, prepared);
        }
        return System.nanoTime() - start;
    }

    private static void invokeGenerated(Row row, Prepared p) {
        try { switch (row) {
            case GELU_ARRAY -> p.entry.invokeExact(p.left, p.output, p.geometry, 0L, (long) COUNT);
            case POW_SEGMENT -> p.entry.invokeExact(p.leftSegment, p.rightSegment, p.outputSegment,
                    p.geometry, 0L, (long) COUNT);
            case CLAMP_MIXED_GENERAL -> p.entry.invokeExact(p.left, p.outputSegment, p.geometry,
                    0L, (long) COUNT);
            case COMPARISON_BROADCAST -> p.entry.invokeExact(p.left, p.right, p.boolOutput,
                    p.geometry, 0L, (long) COUNT);
            case FUSED_WHERE -> p.entry.invokeExact(p.left, p.right, p.otherwise, p.output,
                    p.geometry, 0L, (long) COUNT);
        }} catch (Throwable failure) { throw new AssertionError("generated " + row, failure); }
        checksum += observe(row, p);
    }

    private static void invokeDirect(Row row, Prepared p) {
        switch (row) {
            case GELU_ARRAY -> directGelu(p.left, p.output, p.geometry, 0, COUNT);
            case POW_SEGMENT -> directPow(p.leftSegment, p.rightSegment, p.outputSegment, p.geometry, 0, COUNT);
            case CLAMP_MIXED_GENERAL -> directClamp(p.left, p.outputSegment, p.geometry, 0, COUNT);
            case COMPARISON_BROADCAST -> directComparison(p.left, p.right, p.boolOutput, p.geometry, 0, COUNT);
            case FUSED_WHERE -> directFused(p.left, p.right, p.otherwise, p.output, p.geometry, 0, COUNT);
        }
        checksum += observe(row, p);
    }

    private static long observe(Row row, Prepared p) { return switch (row) {
        case POW_SEGMENT -> p.outputSegment.getAtIndex(
                java.lang.foreign.ValueLayout.JAVA_SHORT_UNALIGNED, p.geometry[4]);
        case CLAMP_MIXED_GENERAL -> p.outputSegment.getAtIndex(
                java.lang.foreign.ValueLayout.JAVA_SHORT_UNALIGNED, p.geometry[5]);
        case COMPARISON_BROADCAST -> p.boolOutput[(int) p.geometry[4]];
        case GELU_ARRAY -> p.output[(int) p.geometry[3]];
        case FUSED_WHERE -> p.output[(int) p.geometry[5]];
    }; }

    private static Prepared prepare(Row row) {
        CpuKernelIr ir = ir(row);
        CpuKernelSpecialization specialization = specialization(row, ir);
        var generator = new CpuClassFileKernelGenerator();
        MethodHandle entry = generator.defineClassBytes(specialization,
                generator.generateClassBytes(specialization, ir)).entryPoint();
        short[] left = new short[COUNT + 8], right = new short[COUNT + 8];
        short[] otherwise = new short[COUNT + 8];
        short[] output = new short[row == Row.CLAMP_MIXED_GENERAL ? 4 * COUNT + 8 : COUNT + 8];
        byte[] boolOutput = new byte[COUNT + 8];
        for (int i = 0; i < left.length; i++) {
            left[i] = BFloat16Bits.fromFloat((i % 257 - 128) * .03125f);
            right[i] = BFloat16Bits.fromFloat((i % 17 - 8) * .125f);
            otherwise[i] = (short) (0x7f80 | (i & 0x7f));
        }
        return new Prepared(entry, left, right, otherwise, output, boolOutput, MemorySegment.ofArray(left),
                MemorySegment.ofArray(right), MemorySegment.ofArray(output),
                row == Row.CLAMP_MIXED_GENERAL ? generalClampGeometry()
                        : denseGeometry(COUNT, specialization.boundaryDataTypes().size()));
    }

    private static CpuKernelSpecialization specialization(Row row, CpuKernelIr ir) {
        List<CpuKernelIr.Value> values = ir.values().stream().filter(value -> value.kind()
                != CpuKernelIr.Value.Kind.VIRTUAL).toList();
        List<DataType> types = values.stream().map(CpuKernelIr.Value::dataType).toList();
        List<CpuKernelSpecialization.CarrierAccess> carriers = carriers(row);
        return new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR, types, carriers, 0, -1,
                List.of(), false, 59);
    }

    private static String classFileName(Row row) { return switch (row) {
        case GELU_ARRAY -> "gelu.class";
        case POW_SEGMENT -> "pow.class";
        case CLAMP_MIXED_GENERAL -> "clamp.class";
        case COMPARISON_BROADCAST -> "comparison.class";
        case FUSED_WHERE -> "fused-access.class";
    }; }

    private static List<CpuKernelSpecialization.CarrierAccess> carriers(Row row) { return switch (row) {
        case GELU_ARRAY -> List.of(CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY, CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY);
        case POW_SEGMENT -> List.of(CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT, CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT, CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT);
        case CLAMP_MIXED_GENERAL -> List.of(CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY, CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT);
        case COMPARISON_BROADCAST -> List.of(CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY, CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY, CpuKernelSpecialization.CarrierAccess.BYTE_ARRAY);
        case FUSED_WHERE -> List.of(CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY, CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY, CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY, CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY);
    }; }

    private static CpuKernelIr ir(Row row) { return switch (row) {
        case GELU_ARRAY -> unary(CpuPointwiseOpcode.GELU_EXACT);
        case POW_SEGMENT -> binary(CpuPointwiseOpcode.POW, DataType.BFLOAT16);
        case CLAMP_MIXED_GENERAL -> clamp();
        case COMPARISON_BROADCAST -> comparison();
        case FUSED_WHERE -> fused();
    }; }

    private static CpuKernelIr unary(CpuPointwiseOpcode opcode) { return kernel(List.of(value(0, DataType.BFLOAT16, CpuKernelIr.Value.Kind.INPUT, denseRead()), value(1, DataType.BFLOAT16, CpuKernelIr.Value.Kind.OUTPUT, denseWrite())), List.of(new CpuKernelIr.Instruction(opcode, List.of(0), 1))); }
    private static CpuKernelIr binary(CpuPointwiseOpcode opcode, DataType type) { return kernel(List.of(value(0,type,CpuKernelIr.Value.Kind.INPUT,denseRead()),value(1,type,CpuKernelIr.Value.Kind.INPUT,denseRead()),value(2,type,CpuKernelIr.Value.Kind.OUTPUT,denseWrite())),List.of(new CpuKernelIr.Instruction(opcode,List.of(0,1),2))); }
    private static CpuKernelIr clamp() {
        var input = new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,
                CpuAccessPlan.Regime.DENSE_LINEAR, 2,
                List.of(CpuAccessPlan.AxisRole.CONTIGUOUS,
                        CpuAccessPlan.AxisRole.CONTIGUOUS), 2);
        var output = new CpuAccessPlan(CpuAccessPlan.AccessKind.WRITE,
                CpuAccessPlan.Regime.GENERAL_ODOMETER, 2,
                List.of(CpuAccessPlan.AxisRole.STRIDED,
                        CpuAccessPlan.AxisRole.STRIDED), 0);
        return kernel(List.of(value(0, DataType.BFLOAT16, CpuKernelIr.Value.Kind.INPUT, input),
                value(1, DataType.BFLOAT16, CpuKernelIr.Value.Kind.OUTPUT, output)),
                List.of(new CpuKernelIr.Instruction(CpuPointwiseOpcode.SCALAR_CLAMP,
                        List.of(0), 1, new CpuKernelIr.ClampImmediate(
                                new CpuKernelIr.ScalarImmediate(DataType.BFLOAT16, 0xbf80),
                                new CpuKernelIr.ScalarImmediate(DataType.BFLOAT16, 0x3f80)))));
    }
    private static CpuKernelIr comparison() { var scalar=new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,CpuAccessPlan.Regime.SCALAR_ALL_ZERO,1,List.of(CpuAccessPlan.AxisRole.BROADCAST),0); return kernel(List.of(value(0,DataType.BFLOAT16,CpuKernelIr.Value.Kind.INPUT,denseRead()),value(1,DataType.BFLOAT16,CpuKernelIr.Value.Kind.INPUT,scalar),value(2,DataType.BOOL,CpuKernelIr.Value.Kind.OUTPUT,denseWrite())),List.of(new CpuKernelIr.Instruction(CpuPointwiseOpcode.GREATER_THAN,List.of(0,1),2))); }
    private static CpuKernelIr fused() { return kernel(List.of(value(0,DataType.BFLOAT16,CpuKernelIr.Value.Kind.INPUT,denseRead()),value(1,DataType.BFLOAT16,CpuKernelIr.Value.Kind.INPUT,denseRead()),value(2,DataType.BFLOAT16,CpuKernelIr.Value.Kind.INPUT,denseRead()),value(3,DataType.BFLOAT16,CpuKernelIr.Value.Kind.VIRTUAL,denseRead()),value(4,DataType.BOOL,CpuKernelIr.Value.Kind.VIRTUAL,denseRead()),value(5,DataType.BFLOAT16,CpuKernelIr.Value.Kind.VIRTUAL,denseRead()),value(6,DataType.BFLOAT16,CpuKernelIr.Value.Kind.OUTPUT,denseWrite())),List.of(new CpuKernelIr.Instruction(CpuPointwiseOpcode.ADD,List.of(0,1),3),new CpuKernelIr.Instruction(CpuPointwiseOpcode.IS_FINITE,List.of(3),4),new CpuKernelIr.Instruction(CpuPointwiseOpcode.SIGMOID,List.of(3),5),new CpuKernelIr.Instruction(CpuPointwiseOpcode.WHERE,List.of(4,5,2),6))); }
    private static CpuKernelIr kernel(List<CpuKernelIr.Value> values,List<CpuKernelIr.Instruction> instructions){return new CpuKernelIr(values,instructions,new CpuKernelIr.Loop("start","end"),List.of(new CpuKernelIr.Store(values.getLast().ordinal(),0)));}
    private static CpuKernelIr.Value value(int id,DataType type,CpuKernelIr.Value.Kind kind,CpuAccessPlan plan){return new CpuKernelIr.Value(id,type,kind,plan);}
    private static CpuAccessPlan denseRead(){return new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,CpuAccessPlan.Regime.DENSE_LINEAR,1,List.of(CpuAccessPlan.AxisRole.CONTIGUOUS),1);}
    private static CpuAccessPlan denseWrite(){return new CpuAccessPlan(CpuAccessPlan.AccessKind.WRITE,CpuAccessPlan.Regime.DENSE_LINEAR,1,List.of(CpuAccessPlan.AxisRole.CONTIGUOUS),1);}
    private static long[] denseGeometry(int count,int boundaries){long[] g=new long[2+4*boundaries];g[0]=count;for(int i=0;i<boundaries;i++){g[2+i]=i+1;g[2+boundaries+i]=1;g[2+2*boundaries+i]=count+8;}return g;}
    private static long[] generalClampGeometry() {
        return new long[] {256, 256, 0, 0, 1, 3, 256, 1, 1_024, 2,
                COUNT + 8L, 4L * COUNT + 8L};
    }

    private static void directGelu(short[] input, short[] output, long[] geometry,
            long start, long end) {
        if (start >= end) return;
        int inputAddress = (int) geometry[2];
        int outputAddress = (int) geometry[3];
        int index = (int) start;
        int exclusiveEnd = (int) end;
        while (index < exclusiveEnd) {
            float value = Float.intBitsToFloat(input[inputAddress] << 16);
            double result;
            if (value == Double.NEGATIVE_INFINITY) result = -0.0d;
            else {
                double argument = value / Math.sqrt(2.0d);
                double magnitude;
                if (Double.isNaN(argument)) magnitude = Double.NaN;
                else if (argument == 0.0d) magnitude = argument;
                else if (argument == Double.POSITIVE_INFINITY) magnitude = 1.0d;
                else if (argument == Double.NEGATIVE_INFINITY) magnitude = -1.0d;
                else {
                    double x = Math.abs(argument);
                    if (x <= 1.0d) {
                        double z = x * x;
                        double numerator = 9.60497373987051638749E0;
                        numerator = numerator * z + 9.00260197203842689217E1;
                        numerator = numerator * z + 2.23200534594684319226E3;
                        numerator = numerator * z + 7.00332514112805075473E3;
                        numerator = numerator * z + 5.55923013010394962768E4;
                        double denominator = z + 3.35617141647503099647E1;
                        denominator = denominator * z + 5.21357949780152679795E2;
                        denominator = denominator * z + 4.59432382970980127987E3;
                        denominator = denominator * z + 2.26290000613890934246E4;
                        denominator = denominator * z + 4.92673942608635921086E4;
                        magnitude = x * numerator / denominator;
                    } else {
                        double numerator;
                        double denominator;
                        if (x < 8.0d) {
                            numerator = 2.46196981473530512524E-10;
                            numerator = numerator * x + 5.64189564831068821977E-1;
                            numerator = numerator * x + 7.46321056442269912687E0;
                            numerator = numerator * x + 4.86371970985681366614E1;
                            numerator = numerator * x + 1.96520832956077098242E2;
                            numerator = numerator * x + 5.26445194995477358631E2;
                            numerator = numerator * x + 9.34528527171957607540E2;
                            numerator = numerator * x + 1.02755188689515710272E3;
                            numerator = numerator * x + 5.57535335369399327526E2;
                            denominator = x + 1.32281951154744992508E1;
                            denominator = denominator * x + 8.67072140885989742329E1;
                            denominator = denominator * x + 3.54937778887819891062E2;
                            denominator = denominator * x + 9.75708501743205489753E2;
                            denominator = denominator * x + 1.82390916687909736289E3;
                            denominator = denominator * x + 2.24633760818710981792E3;
                            denominator = denominator * x + 1.65666309194161350182E3;
                            denominator = denominator * x + 5.57535340817727675546E2;
                        } else {
                            numerator = 5.64189583547755073984E-1;
                            numerator = numerator * x + 1.27536670759978104416E0;
                            numerator = numerator * x + 5.01905042251180477414E0;
                            numerator = numerator * x + 6.16021097993053585195E0;
                            numerator = numerator * x + 7.40974269950448939160E0;
                            numerator = numerator * x + 2.97886665372100240670E0;
                            denominator = x + 2.26052863220117276590E0;
                            denominator = denominator * x + 9.39603524938001434673E0;
                            denominator = denominator * x + 1.20489539808096656605E1;
                            denominator = denominator * x + 1.70814450747565897222E1;
                            denominator = denominator * x + 9.60896809063285878198E0;
                            denominator = denominator * x + 3.36907645100081516050E0;
                        }
                        magnitude = 1.0d - Math.exp(-x * x) * numerator / denominator;
                    }
                    magnitude = Math.copySign(magnitude, argument);
                }
                result = 0.5d * value * (1.0d + magnitude);
            }
            float narrowed = (float) result;
            int bits = Float.floatToRawIntBits(narrowed);
            int upper = (bits & 0x7fff_ffff) > 0x7f80_0000 ? 0x7fc0
                    : (bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16;
            output[outputAddress] = (short) upper;
            inputAddress++;
            outputAddress++;
            index++;
        }
    }

    private static void directPow(MemorySegment left, MemorySegment right, MemorySegment output,
            long[] geometry, long start, long end) {
        if (start >= end) return;
        var layout = java.lang.foreign.ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(
                java.nio.ByteOrder.nativeOrder());
        long leftAddress = geometry[2];
        long rightAddress = geometry[3];
        long outputAddress = geometry[4];
        long index = start;
        while (index < end) {
            short leftBits = left.get(layout, leftAddress * Short.BYTES);
            short rightBits = right.get(layout, rightAddress * Short.BYTES);
            float value = (float) StrictMath.pow(
                    Float.intBitsToFloat(leftBits << 16),
                    Float.intBitsToFloat(rightBits << 16));
            int bits = Float.floatToRawIntBits(value);
            int upper = (bits & 0x7fff_ffff) > 0x7f80_0000 ? 0x7fc0
                    : (bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16;
            output.set(layout, outputAddress * Short.BYTES, (short) upper);
            if (++index >= end) break;
            leftAddress++;
            rightAddress++;
            outputAddress++;
        }
    }

    private static void directClamp(short[] input, MemorySegment output, long[] geometry,
            long start, long end) {
        if (start >= end) return;
        long index = start;
        long inputAddress = geometry[4];
        long outputAddress = geometry[5];
        long row = geometry[2];
        long column = geometry[3];
        var layout = java.lang.foreign.ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(
                java.nio.ByteOrder.nativeOrder());
        while (index < end) {
            float value = Float.intBitsToFloat(input[(int) inputAddress] << 16);
            float result = Math.min(Math.max(value, -1.0f), 1.0f);
            int bits = Float.floatToRawIntBits(result);
            int upper = (bits & 0x7fff_ffff) > 0x7f80_0000 ? 0x7fc0
                    : (bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16;
            output.set(layout, outputAddress * Short.BYTES, (short) upper);
            if (++index >= end) break;
            inputAddress++;
            column++;
            outputAddress += geometry[9];
            if (column >= geometry[1]) {
                column = 0;
                outputAddress -= geometry[9] * geometry[1];
                row++;
                outputAddress += geometry[8];
                if (row >= geometry[0]) {
                    row = 0;
                    outputAddress -= geometry[8] * geometry[0];
                }
            }
        }
    }

    private static void directComparison(short[] left, short[] right, byte[] output,
            long[] geometry, long start, long end) {
        if (start >= end) return;
        int leftAddress = (int) geometry[2];
        int scalarAddress = (int) geometry[3];
        int outputAddress = (int) geometry[4];
        int index = (int) start;
        int exclusiveEnd = (int) end;
        while (index < exclusiveEnd) {
            float value = Float.intBitsToFloat(left[leftAddress] << 16);
            float scalar = Float.intBitsToFloat(right[scalarAddress] << 16);
            output[outputAddress] = (byte) (value > scalar ? 1 : 0);
            leftAddress++;
            outputAddress++;
            index++;
        }
    }

    private static void directFused(short[] left, short[] right, short[] otherwise,
            short[] output, long[] geometry, long start, long end) {
        if (start >= end) return;
        int leftAddress = (int) geometry[2];
        int rightAddress = (int) geometry[3];
        int otherwiseAddress = (int) geometry[4];
        int outputAddress = (int) geometry[5];
        int index = (int) start;
        int exclusiveEnd = (int) end;
        while (index < exclusiveEnd) {
            float sumValue = Float.intBitsToFloat(left[leftAddress] << 16)
                    + Float.intBitsToFloat(right[rightAddress] << 16);
            int sumBits = Float.floatToRawIntBits(sumValue);
            int sumUpper = (sumBits & 0x7fff_ffff) > 0x7f80_0000 ? 0x7fc0
                    : (sumBits + 0x7fff + ((sumBits >>> 16) & 1)) >>> 16;
            float roundedSum = Float.intBitsToFloat(sumUpper << 16);
            double sigmoid;
            if (roundedSum >= 0.0f) sigmoid = 1.0d / (1.0d + StrictMath.exp(-roundedSum));
            else {
                double exponential = StrictMath.exp(roundedSum);
                sigmoid = exponential / (1.0d + exponential);
            }
            float sigmoidValue = (float) sigmoid;
            int sigmoidBits = Float.floatToRawIntBits(sigmoidValue);
            int sigmoidUpper = (sigmoidBits & 0x7fff_ffff) > 0x7f80_0000 ? 0x7fc0
                    : (sigmoidBits + 0x7fff + ((sigmoidBits >>> 16) & 1)) >>> 16;
            output[outputAddress] = Float.isFinite(roundedSum)
                    ? (short) sigmoidUpper : otherwise[otherwiseAddress];
            leftAddress++;
            rightAddress++;
            otherwiseAddress++;
            outputAddress++;
            index++;
        }
    }
    private static double ratio(Path file,Row wanted)throws Exception{try(BufferedReader r=Files.newBufferedReader(file)){r.readLine();for(String line;(line=r.readLine())!=null;){String[] p=line.split(",");if(p[0].equals(wanted.name()))return Double.parseDouble(p[3]);}}throw new IllegalStateException(wanted.name());}
    private static void validate(Path file)throws Exception{for(Row row:Row.values())assertTrue(ratio(file,row)<=LIMIT,row.name());}
    private static void verifyEquivalent(Row row, Prepared prepared) {
        Arrays.fill(prepared.output, (short) 0x5a5a);
        Arrays.fill(prepared.boolOutput, (byte) 0x5a);
        invokeGenerated(row, prepared);
        short[] generatedShorts = prepared.output.clone();
        byte[] generatedBools = prepared.boolOutput.clone();
        Arrays.fill(prepared.output, (short) 0x5a5a);
        Arrays.fill(prepared.boolOutput, (byte) 0x5a);
        invokeDirect(row, prepared);
        assertTrue(Arrays.equals(generatedShorts, prepared.output), row + " raw short mismatch");
        assertTrue(Arrays.equals(generatedBools, prepared.boolOutput), row + " raw BOOL mismatch");
    }

    private static Path sourceFile() {
        List<Path> candidates = List.of(
                Path.of("backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/"
                        + "codegen/emit/CpuBFloat16PointwisePerformanceTest.java"),
                Path.of("src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/"
                        + "CpuBFloat16PointwisePerformanceTest.java"));
        return candidates.stream().filter(Files::isRegularFile).findFirst().orElseThrow(
                () -> new IllegalStateException("performance owner source file missing"));
    }

    private static void retainOracleDecompilation(Path root) throws Exception {
        Path classFile = root.resolve("CpuBFloat16PointwisePerformanceTest.class");
        Process process = new ProcessBuilder(
                Path.of(System.getProperty("java.home"), "bin", "javap").toString(),
                "-c", "-v", "-p", classFile.toString())
                .redirectOutput(root.resolve("CpuBFloat16PointwisePerformanceTest.javap.txt").toFile())
                .redirectError(root.resolve("CpuBFloat16PointwisePerformanceTest.javap.stderr").toFile())
                .start();
        if (process.waitFor() != 0) throw new AssertionError("direct oracle javap failed");
    }
    private record Prepared(MethodHandle entry,short[] left,short[] right,short[] otherwise,short[] output,byte[] boolOutput,MemorySegment leftSegment,MemorySegment rightSegment,MemorySegment outputSegment,long[] geometry) { }
}
