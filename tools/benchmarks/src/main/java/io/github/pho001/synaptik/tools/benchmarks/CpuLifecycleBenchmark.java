package io.github.pho001.synaptik.tools.benchmarks;

import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.engine.InferenceSession;
import io.github.pho001.synaptik.engine.RunResult;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.management.ManagementFactory;
import java.nio.ByteBuffer;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HexFormat;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.LongVector;

/**
 * Runnable, report-only CPU Engine lifecycle benchmark using fixed public Tensor, Engine, and
 * InferenceSession APIs.
 *
 * <p>The evidence profile records immutable run/source/configuration identities and makes every
 * retained repeated-run batch span its declared minimum duration. Public Engine does not expose
 * backend-private selected-plan facts, so reports state that limitation and are ineligible by
 * themselves to authorize a production route/default change.</p>
 */
public final class CpuLifecycleBenchmark {
    private static final Arena ARENA = Arena.ofShared();
    private static final long EVIDENCE_SAMPLE_NANOS = 25_000_000L;
    private static final int MAX_SAMPLE_ITERATIONS = 1_000_000;
    private static final String PROPERTY_PREFIX = "synaptik.benchmark.";

    private CpuLifecycleBenchmark() {}

    /**
     * Runs the selected fixed profile and writes one JSON report to standard output.
     *
     * @param args zero arguments for {@code smoke}, or one of {@code smoke}, {@code baseline}, and
     *     {@code evidence}; spelling is normalized with {@link Locale#ROOT}
     * @throws IllegalArgumentException if the requested profile is unknown or evidence metadata is
     *     absent
     * @throws IllegalStateException if the evidence JVM flags or retained sample floor disagree
     */
    public static void main(String[] args) {
        if (args.length > 1) {
            throw new IllegalArgumentException("expected at most one benchmark profile");
        }
        String requested = args.length == 0 ? "smoke" : args[0].toLowerCase(Locale.ROOT);
        Profile profile = switch (requested) {
            case "smoke" -> new Profile("smoke", 1, 2, 2, 0, smokeWorkloads());
            case "baseline" -> new Profile("baseline", 3, 10, 5, 0, smokeWorkloads());
            case "evidence" -> new Profile("evidence", 5, 7, 3, EVIDENCE_SAMPLE_NANOS,
                    evidenceWorkloads());
            default -> throw new IllegalArgumentException(
                    "profile must be smoke, baseline, or evidence");
        };
        EvidenceMetadata metadata = profile.name().equals("evidence")
                ? EvidenceMetadata.required() : null;
        long started = System.nanoTime();
        List<String> reports = new ArrayList<>();
        boolean mandatoryFailure = false;
        for (Workload workload : profile.workloads()) {
            try {
                reports.add(run(workload, profile));
            } catch (RuntimeException failure) {
                mandatoryFailure = true;
                reports.add("{\"name\":\"" + esc(workload.name())
                        + "\",\"status\":\"unavailable\",\"error\":\""
                        + esc(failure.getMessage() == null
                                ? failure.getClass().getSimpleName() : failure.getMessage())
                        + "\"}");
            }
        }
        long elapsed = System.nanoTime() - started;
        var runtime = ManagementFactory.getRuntimeMXBean();
        System.out.println("{\"schema\":3,\"profile\":\"" + profile.name()
                + "\",\"warmupIterations\":" + profile.warmups()
                + ",\"measurementIterations\":" + profile.measurements()
                + ",\"oneShotIterations\":" + profile.oneShotMeasurements()
                + ",\"minimumRetainedBatchNanos\":" + profile.minimumSampleNanos()
                + ",\"retainedBatchFloorEnforced\":true"
                + ",\"eligibleForProductionDecision\":false"
                + ",\"planFacts\":{\"status\":\"unavailable\",\"reason\":\""
                + "public Engine lifecycle intentionally hides backend-private selected plans\"}"
                + ",\"evidenceMetadata\":" + (metadata == null ? "null" : metadata.json())
                + ",\"harnessClassSha256\":\"" + classSha256() + "\""
                + ",\"java\":\"" + esc(System.getProperty("java.version"))
                + "\",\"javaVm\":\"" + esc(System.getProperty("java.vm.name") + " "
                        + System.getProperty("java.vm.version"))
                + "\",\"jvmArguments\":" + strings(runtime.getInputArguments())
                + ",\"os\":\"" + esc(System.getProperty("os.name") + " "
                        + System.getProperty("os.version") + " "
                        + System.getProperty("os.arch"))
                + "\",\"availableProcessors\":" + Runtime.getRuntime().availableProcessors()
                + ",\"initialMemoryBytesObserved\":" + Runtime.getRuntime().totalMemory()
                + ",\"maxMemoryBytes\":" + Runtime.getRuntime().maxMemory()
                + ",\"preferredSpecies\":" + preferredSpeciesJson()
                + ",\"elapsedNanos\":" + elapsed + ",\"workloads\":["
                + String.join(",", reports) + "]}");
        if (mandatoryFailure) System.exit(2);
    }

    private static String run(Workload workload, Profile profile) {
        long compileNanos;
        long prepareNanos;
        long checksum;
        List<Long> repeated = new ArrayList<>();
        List<Long> repeatedBatch = new ArrayList<>();
        List<Integer> sampleIterations = new ArrayList<>();
        List<Long> oneShot = new ArrayList<>();
        try (Engine engine = Engine.standard()) {
            long begin = System.nanoTime();
            var graph = engine.compile(List.of(workload.output()));
            compileNanos = System.nanoTime() - begin;
            begin = System.nanoTime();
            try (InferenceSession session = engine.session(graph)) {
                prepareNanos = System.nanoTime() - begin;
                checksum = checksumRun(session, workload);
                for (int i = 0; i < profile.warmups(); i++) {
                    runBatch(session, workload.inputs(), 1);
                    requireChecksum(workload, checksum, checksumRun(session, workload));
                }
                int suggestedIterations = 1;
                for (int i = 0; i < profile.measurements(); i++) {
                    TimedSample sample = timedSample(session, workload.inputs(),
                            suggestedIterations, profile.minimumSampleNanos());
                    suggestedIterations = sample.iterations();
                    sampleIterations.add(sample.iterations());
                    repeatedBatch.add(sample.elapsedNanos());
                    repeated.add(sample.elapsedNanos() / sample.iterations());
                    requireChecksum(workload, checksum, checksumRun(session, workload));
                }
            }
            for (int i = 0; i < profile.oneShotMeasurements(); i++) {
                begin = System.nanoTime();
                HostTensorValue value = engine.compute(workload.output());
                oneShot.add(System.nanoTime() - begin);
                requireChecksum(workload, checksum, checksum(value));
            }
        }
        if (profile.minimumSampleNanos() > 0 && repeatedBatch.stream()
                .anyMatch(value -> value < profile.minimumSampleNanos())) {
            throw new IllegalStateException("retained sample fell below declared floor");
        }
        return "{\"name\":\"" + workload.name() + "\",\"inputs\":"
                + workload.inputs().size() + ",\"inputShapes\":"
                + strings(workload.inputs().stream()
                        .map(input -> input.descriptor().shape().toString()).toList())
                + ",\"outputShape\":\"" + esc(workload.output().descriptor().shape().toString())
                + "\",\"compileNanos\":" + compileNanos + ",\"prepareNanos\":"
                + prepareNanos + ",\"sampleIterations\":" + integers(sampleIterations)
                + ",\"repeatedRunNanos\":" + longs(repeated)
                + ",\"repeatedBatchNanos\":" + longs(repeatedBatch)
                + ",\"repeatedRunDistribution\":" + distribution(repeated)
                + ",\"oneShotComputeNanos\":" + longs(oneShot)
                + ",\"checksum\":" + checksum + "}";
    }

    private static TimedSample timedSample(InferenceSession session, List<Tensor> inputs,
            int suggestedIterations, long minimumSampleNanos) {
        int iterations = 0;
        int chunk = Math.max(1, suggestedIterations);
        long begin = System.nanoTime();
        long elapsed;
        do {
            if (chunk > MAX_SAMPLE_ITERATIONS - iterations) {
                throw new IllegalStateException("retained sample iteration ceiling exceeded");
            }
            runBatch(session, inputs, chunk);
            iterations += chunk;
            elapsed = System.nanoTime() - begin;
            if (elapsed < minimumSampleNanos) {
                long capacity = MAX_SAMPLE_ITERATIONS - (long) iterations;
                if (capacity == 0) {
                    throw new IllegalStateException("retained sample did not reach declared floor");
                }
                long average = Math.max(1L, elapsed / iterations);
                long remaining = minimumSampleNanos - elapsed;
                long required = Math.max(1L, Math.ceilDiv(remaining, average));
                chunk = Math.toIntExact(Math.min(capacity,
                        Math.max(required, Math.ceilDiv(iterations, 4))));
            }
        } while (elapsed < minimumSampleNanos);
        return new TimedSample(iterations, elapsed);
    }

    private static void runBatch(InferenceSession session, List<Tensor> inputs, int batchSize) {
        for (int i = 0; i < batchSize; i++) {
            try (RunResult ignored = session.run(inputs)) {
                // Closing the result keeps the timed boundary on repeated execution, not export.
            }
        }
    }

    private static long checksumRun(InferenceSession session, Workload workload) {
        try (RunResult result = session.run(workload.inputs())) {
            HostTensorValue value = result.materialize(
                    result.publications().getFirst(), Long.MAX_VALUE);
            return checksum(value);
        }
    }

    private static void requireChecksum(Workload workload, long expected, long actual) {
        if (actual != expected) {
            throw new IllegalStateException("checksum changed for " + workload.name()
                    + ": expected " + expected + " but observed " + actual);
        }
    }

    private static long checksum(HostTensorValue value) {
        ByteBuffer bytes = value.bytes();
        long hash = 0xcbf29ce484222325L;
        while (bytes.hasRemaining()) {
            hash = (hash ^ (bytes.get() & 0xffL)) * 0x100000001b3L;
        }
        return hash;
    }

    private static String distribution(List<Long> values) {
        List<Long> sorted = new ArrayList<>(values);
        Collections.sort(sorted);
        return "{\"minimum\":" + sorted.getFirst() + ",\"median\":"
                + sorted.get(sorted.size() / 2) + ",\"maximum\":" + sorted.getLast() + "}";
    }

    private static String longs(List<Long> values) {
        StringBuilder out = new StringBuilder("[");
        for (int i = 0; i < values.size(); i++) {
            if (i > 0) out.append(',');
            out.append(values.get(i));
        }
        return out.append(']').toString();
    }

    private static String integers(List<Integer> values) {
        StringBuilder out = new StringBuilder("[");
        for (int i = 0; i < values.size(); i++) {
            if (i > 0) out.append(',');
            out.append(values.get(i));
        }
        return out.append(']').toString();
    }

    private static String strings(List<String> values) {
        StringBuilder out = new StringBuilder("[");
        for (int i = 0; i < values.size(); i++) {
            if (i > 0) out.append(',');
            out.append('"').append(esc(values.get(i))).append('"');
        }
        return out.append(']').toString();
    }

    private static String esc(String value) {
        StringBuilder escaped = new StringBuilder(value.length());
        for (int index = 0; index < value.length(); index++) {
            char character = value.charAt(index);
            switch (character) {
                case '"' -> escaped.append("\\\"");
                case '\\' -> escaped.append("\\\\");
                case '\b' -> escaped.append("\\b");
                case '\f' -> escaped.append("\\f");
                case '\n' -> escaped.append("\\n");
                case '\r' -> escaped.append("\\r");
                case '\t' -> escaped.append("\\t");
                default -> {
                    if (character < 0x20) escaped.append(String.format("\\u%04x", (int) character));
                    else escaped.append(character);
                }
            }
        }
        return escaped.toString();
    }

    private static String classSha256() {
        try (InputStream stream = CpuLifecycleBenchmark.class.getResourceAsStream(
                "CpuLifecycleBenchmark.class")) {
            if (stream == null) throw new IllegalStateException("benchmark class bytes unavailable");
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256")
                    .digest(stream.readAllBytes()));
        } catch (IOException failure) {
            throw new IllegalStateException("cannot hash benchmark class", failure);
        } catch (NoSuchAlgorithmException impossible) {
            throw new ExceptionInInitializerError(impossible);
        }
    }

    private static String preferredSpeciesJson() {
        return "{\"float32\":{\"bits\":" + FloatVector.SPECIES_PREFERRED.vectorBitSize()
                + ",\"lanes\":" + FloatVector.SPECIES_PREFERRED.length()
                + "},\"float64\":{\"bits\":" + DoubleVector.SPECIES_PREFERRED.vectorBitSize()
                + ",\"lanes\":" + DoubleVector.SPECIES_PREFERRED.length()
                + "},\"int32\":{\"bits\":" + IntVector.SPECIES_PREFERRED.vectorBitSize()
                + ",\"lanes\":" + IntVector.SPECIES_PREFERRED.length()
                + "},\"int64\":{\"bits\":" + LongVector.SPECIES_PREFERRED.vectorBitSize()
                + ",\"lanes\":" + LongVector.SPECIES_PREFERRED.length()
                + "},\"bool\":{\"bits\":" + ByteVector.SPECIES_PREFERRED.vectorBitSize()
                + ",\"lanes\":" + ByteVector.SPECIES_PREFERRED.length() + "}}";
    }

    private static List<Workload> smokeWorkloads() {
        return List.of(
                binaryAdd("pointwise", tensor(Shape.of(2, 3)), tensor(Shape.of(2, 3))),
                matmul("matmul", tensor(Shape.of(2, 3)), tensor(Shape.of(3, 4))),
                conv2d("conv2d", tensor(Shape.of(1, 2, 8, 8)),
                        tensor(Shape.of(3, 2, 3, 3))),
                conv3d("conv3d", tensor(Shape.of(1, 2, 5, 5, 5)),
                        tensor(Shape.of(3, 2, 3, 3, 3))),
                reduction("reduction", tensor(Shape.of(4, 16)), true),
                normalization("normalization", tensor(Shape.of(4, 16)), Shape.of(16)));
    }

    private static List<Workload> evidenceWorkloads() {
        return List.of(
                binaryAdd("pointwise", tensor(Shape.of(1_024, 1_024)),
                        tensor(Shape.of(1_024, 1_024))),
                matmul("matmul", tensor(Shape.of(256, 512)), tensor(Shape.of(512, 1))),
                conv2d("conv2d", tensor(Shape.of(1, 16, 64, 64)),
                        tensor(Shape.of(32, 16, 3, 3))),
                conv3d("conv3d", tensor(Shape.of(1, 8, 24, 24, 24)),
                        tensor(Shape.of(12, 8, 3, 3, 3))),
                reduction("reduction", tensor(Shape.of(1_024, 2_048)), false),
                normalization("normalization", tensor(Shape.of(1_024, 2_048)),
                        Shape.of(2_048)));
    }

    private static Workload matmul(String name, Tensor input, Tensor second) {
        return finish(name, input.matmul(second).contiguous(), input, second);
    }

    private static Workload conv2d(String name, Tensor input, Tensor weights) {
        return finish(name, input.conv2d(weights, Conv2dAttrs.defaults()), input, weights);
    }

    private static Workload conv3d(String name, Tensor input, Tensor weights) {
        return finish(name, input.conv3d(weights, Conv3dAttrs.defaults()), input, weights);
    }

    private static Workload reduction(String name, Tensor input, boolean full) {
        return finish(name, (full ? input.sum() : input.sum(1)).contiguous(), input);
    }

    private static Workload normalization(String name, Tensor input, Shape normalizedShape) {
        return finish(name, input.layerNorm(normalizedShape,
                ScalarValue.float32(1e-5f)).contiguous(), input);
    }

    private static Workload binaryAdd(String name, Tensor input, Tensor second) {
        return finish(name, input.add(second).contiguous(), input, second);
    }

    private static Workload finish(String name, Tensor output, Tensor... inputs) {
        return new Workload(name, output, List.of(inputs));
    }

    private static Tensor tensor(Shape shape) {
        int count = Math.toIntExact(shape.knownElementCount().orElseThrow());
        float[] data = new float[count];
        for (int i = 0; i < count; i++) {
            data[i] = ((i * 17) % 101 - 50) / 101.0f;
        }
        MemorySegment segment = ARENA.allocate((long) count * Float.BYTES, Float.BYTES);
        MemorySegment.copy(MemorySegment.ofArray(data), 0, segment, 0, segment.byteSize());
        TensorDescriptor descriptor = new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        return TensorFactory.create(descriptor, Optional.empty(),
                Optional.of(new MemorySegmentStorage(DataType.FLOAT32, count, segment)));
    }

    private record Profile(String name, int warmups, int measurements, int oneShotMeasurements,
            long minimumSampleNanos, List<Workload> workloads) {
        private Profile {
            workloads = List.copyOf(workloads);
        }
    }

    private record Workload(String name, Tensor output, List<Tensor> inputs) {
        private Workload {
            inputs = List.copyOf(inputs);
        }
    }

    private record TimedSample(int iterations, long elapsedNanos) {}

    private record EvidenceMetadata(String baseRevision, String sourceIdentity,
            String gitTreeObjectSha1, String harnessSourceSha256,
            String productionSourceSha256, String configuration, int fork, long orderSeed,
            int orderIndex, String plannedOrder, String cpuIdentity, String cpuFeatures,
            String computePreference, int configuredMaximumParallelism, int availableParallelism,
            long minimumElementsPerWorker, int workerCount, String materializationPolicy,
            String openBlasPolicy) {
        private static EvidenceMetadata required() {
            List<String> arguments = ManagementFactory.getRuntimeMXBean().getInputArguments();
            for (String required : List.of("-Xms1g", "-Xmx1g", "-XX:-TieredCompilation",
                    "-Xbatch")) {
                if (!arguments.contains(required)) {
                    throw new IllegalStateException("evidence JVM is missing " + required);
                }
            }
            return new EvidenceMetadata(property("baseRevision"), property("sourceIdentity"),
                    property("gitTreeObjectSha1"), property("harnessSourceSha256"),
                    property("productionSourceSha256"), property("configuration"),
                    positiveInt("fork"), longValue("orderSeed"), positiveInt("orderIndex"),
                    property("plannedOrder"), property("cpuIdentity"), property("cpuFeatures"),
                    property("computePreference"), positiveInt("configuredMaximumParallelism"),
                    positiveInt("availableParallelism"),
                    positiveLong("minimumElementsPerWorker"), nonNegativeInt("workerCount"),
                    property("materializationPolicy"), property("openBlasPolicy"));
        }

        private String json() {
            return "{\"baseRevision\":\"" + esc(baseRevision)
                    + "\",\"sourceIdentity\":\"" + esc(sourceIdentity)
                    + "\",\"gitTreeObjectSha1\":\"" + esc(gitTreeObjectSha1)
                    + "\",\"harnessSourceSha256\":\"" + esc(harnessSourceSha256)
                    + "\",\"productionSourceSha256\":\"" + esc(productionSourceSha256)
                    + "\",\"configuration\":\"" + esc(configuration)
                    + "\",\"fork\":" + fork + ",\"orderSeed\":" + orderSeed
                    + ",\"orderIndex\":" + orderIndex + ",\"plannedOrder\":\""
                    + esc(plannedOrder) + "\",\"noRetryOrDiscard\":true,\"cpuIdentity\":\""
                    + esc(cpuIdentity) + "\",\"cpuFeatures\":\"" + esc(cpuFeatures)
                    + "\",\"portableExecution\":{\"computePreference\":\""
                    + esc(computePreference) + "\",\"configuredMaximumParallelism\":"
                    + configuredMaximumParallelism + ",\"availableParallelism\":"
                    + availableParallelism + ",\"minimumElementsPerWorker\":"
                    + minimumElementsPerWorker + ",\"workerCount\":" + workerCount
                    + "},\"materializationPolicy\":\"" + esc(materializationPolicy)
                    + "\",\"openBlasPolicy\":\"" + esc(openBlasPolicy) + "\"}";
        }

        private static String property(String name) {
            String value = System.getProperty(PROPERTY_PREFIX + name);
            if (value == null || value.isBlank()) {
                throw new IllegalArgumentException("missing evidence metadata property " + name);
            }
            return value;
        }

        private static int positiveInt(String name) {
            int value = Integer.parseInt(property(name));
            if (value <= 0) throw new IllegalArgumentException(name + " must be positive");
            return value;
        }

        private static int nonNegativeInt(String name) {
            int value = Integer.parseInt(property(name));
            if (value < 0) throw new IllegalArgumentException(name + " must be non-negative");
            return value;
        }

        private static long positiveLong(String name) {
            long value = Long.parseLong(property(name));
            if (value <= 0) throw new IllegalArgumentException(name + " must be positive");
            return value;
        }

        private static long longValue(String name) {
            return Long.parseLong(property(name));
        }
    }
}
