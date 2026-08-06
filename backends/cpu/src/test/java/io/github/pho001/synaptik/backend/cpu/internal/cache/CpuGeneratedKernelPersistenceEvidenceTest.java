package io.github.pho001.synaptik.backend.cpu.internal.cache;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.internal.ir.*;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionStrategy;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.security.*;
import java.util.*;
import java.util.concurrent.TimeUnit;
import jdk.incubator.vector.DoubleVector;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/** Opt-in, report-only development evidence for verified generated-class-byte persistence. */
class CpuGeneratedKernelPersistenceEvidenceTest {
    private static final int FORKS = 7, WARMUPS = 20, SAMPLES = 50;
    private static final Path CPU_PROJECT = cpuProject();

    @Test void recordsFixedForkedPersistenceEvidence() throws Exception {
        Assumptions.assumeTrue("1".equals(System.getenv("SYNAPTIK_CPU_PERSISTENCE_EVIDENCE")),
                "explicit persistence evidence is disabled");
        Path scratch = Files.createTempDirectory(CPU_PROJECT.resolve("build"),
                "persistence-evidence-");
        var results = new ArrayList<FixtureResult>();
        String failure = "";
        try {
            for (String compute : List.of("scalar", "vector")) for (String pattern
                    : List.of("all-segment", "all-heap", "mixed")) {
                var noRoot = new ArrayList<Long>(); var hit = new ArrayList<Long>();
                long classBytes = 0, envelopeBytes = 0; int hits = 0, fallbacks = 0;
                for (int fork = 0; fork < FORKS; fork++) {
                    boolean hitFirst = (fork & 1) != 0;
                    WorkerResult first = runWorker(hitFirst ? "hit" : "no-root", compute, pattern,
                            scratch.resolve(compute + "-" + pattern + "-" + fork));
                    WorkerResult second = runWorker(hitFirst ? "no-root" : "hit", compute, pattern,
                            scratch.resolve(compute + "-" + pattern + "-" + fork));
                    WorkerResult noRootResult = hitFirst ? second : first;
                    WorkerResult hitResult = hitFirst ? first : second;
                    noRoot.addAll(noRootResult.samples); hit.addAll(hitResult.samples);
                    classBytes = hitResult.classBytes; envelopeBytes = hitResult.envelopeBytes;
                    hits += hitResult.hits; fallbacks += hitResult.fallbacks;
                }
                results.add(new FixtureResult(compute + "-" + pattern, noRoot, hit, classBytes,
                        envelopeBytes, hits, fallbacks));
            }
        } catch (Exception environmentFailure) {
            failure = environmentFailure.getClass().getSimpleName() + ": "
                    + String.valueOf(environmentFailure.getMessage());
        }
        boolean sufficient = results.size() == 6 && results.stream().allMatch(
                result -> result.noRoot.size() == 350 && result.hit.size() == 350);
        String verdict = !failure.isEmpty() || !sufficient ? "INCONCLUSIVE"
                : results.stream().allMatch(FixtureResult::eligible)
                        ? "ENABLE_ELIGIBLE" : "KEEP_DISABLED";
        String payload = payload(results, verdict, failure);
        String hash = hex(sha256(payload.getBytes(StandardCharsets.UTF_8)));
        Path report = CPU_PROJECT.resolve("build/reports/evidence/cpu-0005d-persistence.json");
        Files.createDirectories(report.getParent());
        Files.writeString(report, payload.substring(0, payload.length() - 2)
                + ",\n  \"reportHash\": \"" + hash + "\"\n}\n", StandardCharsets.UTF_8);
        assertAll(() -> assertTrue(Set.of("ENABLE_ELIGIBLE", "KEEP_DISABLED", "INCONCLUSIVE")
                        .contains(verdict)),
                () -> assertEquals(64, hash.length()), () -> assertTrue(Files.isRegularFile(report)));
    }

    public static void main(String[] args) throws Exception {
        String mode = args[0], compute = args[1], pattern = args[2];
        Path root = Path.of(args[3]);
        var fixture = fixture(compute, pattern);
        CpuGeneratedKernelArtifactStore store = mode.equals("hit")
                ? new CpuGeneratedKernelArtifactStore(Optional.of(root))
                : new CpuGeneratedKernelArtifactStore();
        long envelopeSize = 0;
        if (mode.equals("hit")) {
            Files.createDirectories(root);
            CpuGeneratedKernelArtifactStore.clearLoadedForTests();
            var seed = store.loadOrGenerateObserved(fixture.specialization, fixture.ir);
            if (seed.source() != CpuGeneratedKernelArtifactStore.RealizationSource.GENERATED) {
                throw new IllegalStateException("trusted-root pre-seed did not generate");
            }
            envelopeSize = Files.size(root.resolve(fixture.specialization.structuralKey() + ".artifact"));
            CpuGeneratedKernelArtifactStore.clearLoadedForTests();
            if (store.loadOrGenerateObserved(fixture.specialization, fixture.ir).source()
                    != CpuGeneratedKernelArtifactStore.RealizationSource.PERSISTED_HIT) {
                throw new IllegalStateException("trusted-root pre-seed did not verify as a hit");
            }
        }
        for (int i = 0; i < WARMUPS; i++) {
            CpuGeneratedKernelArtifactStore.clearLoadedForTests();
            store.loadOrGenerate(fixture.specialization, fixture.ir);
        }
        var samples = new StringJoiner(","); long classSize = 0; int hits = 0, fallbacks = 0;
        for (int i = 0; i < SAMPLES; i++) {
            CpuGeneratedKernelArtifactStore.clearLoadedForTests();
            long start = System.nanoTime();
            var realization = store.loadOrGenerateObserved(fixture.specialization, fixture.ir);
            samples.add(Long.toString(System.nanoTime() - start));
            classSize = realization.artifact().classBytes().length;
            if (mode.equals("hit")) {
                if (realization.source()
                        == CpuGeneratedKernelArtifactStore.RealizationSource.PERSISTED_HIT) hits++;
                else fallbacks++;
            } else if (realization.source()
                    != CpuGeneratedKernelArtifactStore.RealizationSource.GENERATED) {
                throw new IllegalStateException("no-root realization did not generate");
            }
        }
        System.out.println("RESULT|" + classSize + "|" + envelopeSize + "|" + hits + "|"
                + fallbacks + "|" + samples);
    }

    private static WorkerResult runWorker(String mode, String compute, String pattern, Path root)
            throws Exception {
        String java = Path.of(System.getProperty("java.home"), "bin", "java").toString();
        String separator = System.getProperty("path.separator");
        var classpath = new LinkedHashSet<String>();
        classpath.add(System.getProperty("java.class.path"));
        classpath.add(CPU_PROJECT.resolve("build/classes/java/test").toString());
        classpath.add(CPU_PROJECT.resolve("build/classes/java/main").toString());
        Path repository = CPU_PROJECT.getParent().getParent();
        for (String area : List.of("modules", "backends/openblas-provider")) {
            Path path = repository.resolve(area);
            if (Files.exists(path)) try (var files = Files.walk(path)) {
                files.filter(file -> file.toString().endsWith(".jar")
                        && file.toString().contains("build/libs")).forEach(file ->
                        classpath.add(file.toAbsolutePath().toString()));
            }
        }
        var process = new ProcessBuilder(java, "--add-modules", "jdk.incubator.vector", "-cp",
                String.join(separator, classpath), CpuGeneratedKernelPersistenceEvidenceTest.class.getName(),
                mode, compute, pattern, root.toAbsolutePath().toString()).redirectErrorStream(true).start();
        String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
        if (!process.waitFor(60, TimeUnit.SECONDS)) { process.destroyForcibly();
            throw new IllegalStateException("evidence fork timed out"); }
        if (process.exitValue() != 0) throw new IllegalStateException("evidence fork failed: " + output);
        String line = output.lines().filter(value -> value.startsWith("RESULT|")).findFirst()
                .orElseThrow(() -> new IllegalStateException("missing fork result: " + output));
        String[] fields = line.split("\\|", 6);
        var samples = Arrays.stream(fields[5].split(",")).map(Long::parseLong).toList();
        return new WorkerResult(samples, Long.parseLong(fields[1]), Long.parseLong(fields[2]),
                Integer.parseInt(fields[3]), Integer.parseInt(fields[4]));
    }

    private static Fixture fixture(String compute, String pattern) {
        var access = new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,
                CpuAccessPlan.Regime.DENSE_LINEAR, 1, List.of(CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
        var output = new CpuAccessPlan(CpuAccessPlan.AccessKind.WRITE,
                CpuAccessPlan.Regime.DENSE_LINEAR, 1, List.of(CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
        var ir = new CpuKernelIr(List.of(
                new CpuKernelIr.Value(0, DataType.FLOAT64, CpuKernelIr.Value.Kind.INPUT, access),
                new CpuKernelIr.Value(1, DataType.FLOAT64, CpuKernelIr.Value.Kind.INPUT, access),
                new CpuKernelIr.Value(2, DataType.FLOAT64, CpuKernelIr.Value.Kind.INPUT, access),
                new CpuKernelIr.Value(3, DataType.FLOAT64, CpuKernelIr.Value.Kind.VIRTUAL, access),
                new CpuKernelIr.Value(4, DataType.FLOAT64, CpuKernelIr.Value.Kind.VIRTUAL, access),
                new CpuKernelIr.Value(5, DataType.FLOAT64, CpuKernelIr.Value.Kind.OUTPUT, output)),
                List.of(new CpuKernelIr.Instruction(CpuKernelIr.Instruction.Semantic.ADD, List.of(0,1),3),
                        new CpuKernelIr.Instruction(CpuKernelIr.Instruction.Semantic.GELU_EXACT,List.of(3),4),
                        new CpuKernelIr.Instruction(CpuKernelIr.Instruction.Semantic.MUL,List.of(4,2),5)),
                new CpuKernelIr.Loop("start", "end"), List.of(new CpuKernelIr.Store(5, 0)));
        var segment = CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT;
        var heap = CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY;
        List<CpuKernelSpecialization.CarrierAccess> carriers = switch (pattern) {
            case "all-segment" -> List.of(segment,segment,segment,segment);
            case "all-heap" -> List.of(heap,heap,heap,heap);
            default -> List.of(heap,segment,heap,segment);
        };
        boolean vector = compute.equals("vector");
        var specialization = new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                vector ? ExecutionStrategy.VECTOR : ExecutionStrategy.SCALAR, carriers,
                vector ? DoubleVector.SPECIES_PREFERRED.vectorBitSize() : 0);
        return new Fixture(ir, specialization);
    }

    private static String payload(List<FixtureResult> results, String verdict, String failure) {
        var text = new StringBuilder("{\n  \"schema\": ").append(CpuGeneratorSchema.CURRENT_VERSION)
                .append(",\n  \"command\": \"SYNAPTIK_CPU_PERSISTENCE_EVIDENCE=1 ./gradlew :backends:cpu:test --rerun-tasks --tests '*CpuGeneratedKernelPersistenceEvidenceTest'\"")
                .append(",\n  \"javaVendor\": \"").append(escape(System.getProperty("java.vendor")))
                .append("\",\n  \"javaVersion\": \"").append(escape(System.getProperty("java.version")))
                .append("\",\n  \"os\": \"").append(escape(System.getProperty("os.name"))).append(' ')
                .append(escape(System.getProperty("os.version"))).append("\",\n  \"architecture\": \"")
                .append(escape(System.getProperty("os.arch"))).append("\",\n  \"availableProcessors\": ")
                .append(Runtime.getRuntime().availableProcessors()).append(",\n  \"methodology\": {\"fixtures\": 6, \"forksPerMode\": 7, \"warmupsPerFork\": 20, \"samplesPerFork\": 50},\n")
                .append("  \"thresholds\": {\"medianRatioMaximum\": 0.8, \"minimumMedianSavingNs\": 200000, \"p95RatioMaximum\": 0.9, \"requiredFallbacks\": 0},\n")
                .append("  \"fixtures\": [\n");
        for (int i = 0; i < results.size(); i++) { var r = results.get(i);
            text.append("    {\"name\": \"").append(r.name).append("\", \"sampleCount\": ")
                    .append(r.hit.size()).append(", \"noRootMedianNs\": ").append(r.noRootMedian())
                    .append(", \"hitMedianNs\": ").append(r.hitMedian())
                    .append(", \"noRootP95Ns\": ").append(r.noRootP95())
                    .append(", \"hitP95Ns\": ").append(r.hitP95())
                    .append(", \"medianSavingNs\": ").append(r.noRootMedian()-r.hitMedian())
                    .append(", \"medianRatio\": ").append(String.format(Locale.ROOT,"%.6f",r.ratio()))
                    .append(", \"hitCount\": ").append(r.hits).append(", \"fallbackCount\": ")
                    .append(r.fallbacks).append(", \"classBytes\": ").append(r.classBytes)
                    .append(", \"envelopeBytes\": ").append(r.envelopeBytes).append('}');
            text.append(i + 1 == results.size() ? "\n" : ",\n");
        }
        text.append("  ],\n  \"verdict\": \"").append(verdict).append('"');
        if (!failure.isEmpty()) text.append(",\n  \"failure\": \"").append(escape(failure)).append('"');
        return text.append("\n}\n").toString();
    }
    private static long percentile(List<Long> values, double fraction) { var sorted = values.stream().sorted().toList();
        return sorted.get(Math.max(0, (int)Math.ceil(sorted.size()*fraction)-1)); }
    private static byte[] sha256(byte[] value) throws Exception { return MessageDigest.getInstance("SHA-256").digest(value); }
    private static String hex(byte[] value) { return HexFormat.of().formatHex(value); }
    private static String escape(String value) {
        return value.replace("\\", "\\\\").replace("\"", "\\\"")
                .replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t");
    }
    private static Path cpuProject() {
        Path working = Path.of(System.getProperty("user.dir")).toAbsolutePath().normalize();
        if (Files.isDirectory(working.resolve("src/main/java"))
                && working.endsWith(Path.of("backends/cpu"))) return working;
        Path nested = working.resolve("backends/cpu");
        if (Files.isDirectory(nested.resolve("src/main/java"))) return nested;
        throw new IllegalStateException("cannot locate backends/cpu from " + working);
    }
    private record Fixture(CpuKernelIr ir, CpuKernelSpecialization specialization) { }
    private record WorkerResult(List<Long> samples,long classBytes,long envelopeBytes,int hits,int fallbacks) { }
    private record FixtureResult(String name,List<Long> noRoot,List<Long> hit,long classBytes,long envelopeBytes,int hits,int fallbacks) {
        long noRootMedian(){return percentile(noRoot,.5);} long hitMedian(){return percentile(hit,.5);}
        long noRootP95(){return percentile(noRoot,.95);} long hitP95(){return percentile(hit,.95);}
        double ratio(){return (double)hitMedian()/noRootMedian();}
        boolean eligible(){return hits==350 && fallbacks==0 && ratio()<=.8
                && noRootMedian()-hitMedian()>=200_000
                && (double)hitP95()/noRootP95()<=.9;}
    }
}
