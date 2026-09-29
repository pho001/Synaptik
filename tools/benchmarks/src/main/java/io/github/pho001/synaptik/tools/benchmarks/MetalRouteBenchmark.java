package io.github.pho001.synaptik.tools.benchmarks;

import io.github.pho001.synaptik.backend.metal.MetalBackendConfiguration;
import io.github.pho001.synaptik.backend.metal.MetalBackendIntegration;
import io.github.pho001.synaptik.backend.metal.MetalCapabilityProvider;
import io.github.pho001.synaptik.backend.metal.MetalLocalWorkloadTuning;
import io.github.pho001.synaptik.backend.metal.MetalInvocationPlan;
import io.github.pho001.synaptik.backend.metal.MetalPreparationStructure;
import io.github.pho001.synaptik.compiler.CompileArtifacts;
import io.github.pho001.synaptik.compiler.GraphCompilationPort;
import io.github.pho001.synaptik.config.compile.BackendIntent;
import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.prepare.GraphPreparation;
import io.github.pho001.synaptik.runtime.execution.PreparedExecution;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.run.PreparedExecutionRunner;
import io.github.pho001.synaptik.runtime.run.RunResult;
import io.github.pho001.synaptik.trace.TraceEvent;
import io.github.pho001.synaptik.trace.TraceLevel;
import io.github.pho001.synaptik.trace.TracePayload;
import io.github.pho001.synaptik.trace.TracePhase;
import io.github.pho001.synaptik.trace.payload.BackendInvocationOutcome;
import io.github.pho001.synaptik.trace.payload.BackendPreparationOutcome;
import io.github.pho001.synaptik.trace.payload.TraceCacheStatus;
import io.github.pho001.synaptik.trace.payload.TraceNativeStatusKind;
import io.github.pho001.synaptik.trace.payload.TraceNumericalProfile;
import io.github.pho001.synaptik.trace.payload.TraceOutcomeStatus;
import io.github.pho001.synaptik.trace.payload.TraceRouteKind;
import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.management.ManagementFactory;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HexFormat;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Optional;

/**
 * Fixed report-only Metal singleton-NEG benchmark over both current local route candidates.
 *
 * <p>Route attestation uses a separate traced integration and contributes no timings. Timed work
 * uses the ordinary no-trace integration, enumerates both opaque candidates, and records both
 * without selecting a winner or interacting with tuning caches. Reports are machine-local
 * observations and are never eligible to authorize a production decision.</p>
 */
public final class MetalRouteBenchmark {
    private static final int ELEMENT_COUNT = 1_048_576;
    private static final Shape WORKLOAD_SHAPE = Shape.of(ELEMENT_COUNT);
    private static final long BASELINE_SAMPLE_NANOS = 25_000_000L;
    private static final int MAX_SAMPLE_ITERATIONS = 1_000_000;
    private static final String LIBRARY_ENVIRONMENT = "SYNAPTIK_METAL_TEST_LIBRARY";
    private static final String PROPERTY_PREFIX = "synaptik.benchmark.metal.";
    private static final PreparedExecutionRunner RUNNER = new PreparedExecutionRunner();
    private static final int GENERATOR_SCHEMA_VERSION = 2;
    private static final long GENERATOR_MULTIPLIER = 17L;
    private static final long GENERATOR_MODULUS = 101L;
    private static final long GENERATOR_OFFSET = 50L;
    private static final float GENERATOR_DENOMINATOR = 101.0f;
    private static final int[] SENTINEL_BITS = {
        0x00000000,
        0x80000000,
        0x00000001,
        0x80000001,
        0x007fffff,
        0x807fffff,
        0x00800000,
        0x80800000,
        0x3f800000,
        0xbf800000,
        0x7f800000,
        0xff800000,
        0x7fc12345,
        0xffc54321,
        0x7fa12345,
        0xffa54321
    };

    private MetalRouteBenchmark() {}

    /**
     * Runs exactly one fixed profile and writes one schema-1 JSON document to standard output.
     *
     * @param args zero arguments for {@code smoke}, or exactly one {@code smoke} or
     *     {@code baseline} argument
     */
    public static void main(String[] args) {
        Profile profile = profile(args);
        BaselineMetadata metadata = profile.name().equals("baseline")
                ? BaselineMetadata.required()
                : null;
        requireSupportedHost();
        Path library = configuredLibrary();
        long started = System.nanoTime();
        try (Arena arena = Arena.ofShared()) {
            Workload workload = workload(arena);
            List<Attestation> attestations = attest(library, workload);
            TimedReport timed = measure(library, workload, attestations, profile);
            long elapsed = System.nanoTime() - started;
            System.out.println(report(
                    profile, metadata, library, workload, attestations, timed, elapsed));
        }
    }

    private static Profile profile(String[] args) {
        if (args.length > 1) {
            throw new IllegalArgumentException("expected at most one Metal benchmark profile");
        }
        String requested = args.length == 0 ? "smoke" : args[0].toLowerCase(Locale.ROOT);
        return switch (requested) {
            case "smoke" -> new Profile("smoke", 2, 2, 0L);
            case "baseline" -> new Profile("baseline", 4, 8, BASELINE_SAMPLE_NANOS);
            default -> throw new IllegalArgumentException("profile must be smoke or baseline");
        };
    }

    private static void requireSupportedHost() {
        String os = System.getProperty("os.name");
        String architecture = System.getProperty("os.arch");
        if (!"Mac OS X".equals(os)
                || !("aarch64".equals(architecture) || "arm64".equals(architecture))) {
            throw new IllegalStateException(
                    "Metal benchmark requires Apple-silicon macOS: " + os + " " + architecture);
        }
    }

    private static Path configuredLibrary() {
        String configured = System.getenv(LIBRARY_ENVIRONMENT);
        if (configured == null || configured.isBlank()) {
            throw new IllegalArgumentException(LIBRARY_ENVIRONMENT + " must name the Metal dylib");
        }
        Path path = Path.of(configured);
        if (!path.isAbsolute()) {
            throw new IllegalArgumentException(LIBRARY_ENVIRONMENT + " must be an absolute path");
        }
        if (!Files.isRegularFile(path)) {
            throw new IllegalArgumentException("configured Metal dylib is not a regular file: "
                    + path.normalize());
        }
        try {
            return path.toRealPath();
        } catch (IOException failure) {
            throw new IllegalStateException("cannot resolve configured Metal dylib", failure);
        }
    }

    private static Workload workload(Arena arena) {
        byte[] canonicalInput = new byte[Math.multiplyExact(ELEMENT_COUNT, Float.BYTES)];
        byte[] expectedOutput = new byte[canonicalInput.length];
        ByteBuffer inputBytes = ByteBuffer.wrap(canonicalInput);
        ByteBuffer outputBytes = ByteBuffer.wrap(expectedOutput);
        MemorySegment segment = arena.allocate(canonicalInput.length, Float.BYTES);
        for (int index = 0; index < ELEMENT_COUNT; index++) {
            int bits = inputBits(index);
            inputBytes.putInt(bits);
            outputBytes.putInt(bits ^ 0x80000000);
            segment.setAtIndex(ValueLayout.JAVA_INT, index, bits);
        }
        var storage = new MemorySegmentStorage(DataType.FLOAT32, ELEMENT_COUNT, segment);
        var descriptor = new TensorDescriptor(
                DataType.FLOAT32,
                WORKLOAD_SHAPE,
                Optional.of(LayoutDescriptor.contiguous(WORKLOAD_SHAPE)),
                false);
        Tensor input = TensorFactory.create(
                descriptor, Optional.empty(), Optional.of(storage));
        return new Workload(
                input.neg(),
                storage,
                expectedOutput,
                sha256(canonicalInput),
                sha256(expectedOutput));
    }

    private static int inputBits(int index) {
        if (index < SENTINEL_BITS.length) {
            return SENTINEL_BITS[index];
        }
        long numerator = ((index * GENERATOR_MULTIPLIER) % GENERATOR_MODULUS)
                - GENERATOR_OFFSET;
        return Float.floatToRawIntBits((float) numerator / GENERATOR_DENOMINATOR);
    }

    private static List<Attestation> attest(Path library, Workload workload) {
        TraceCollector collector = new TraceCollector();
        List<TraceEvent<? extends TracePayload>> observedEvents = List.of();
        List<Attestation> attestations = new ArrayList<>(2);
        try (MetalBackendIntegration integration = MetalBackendIntegration.open(
                        new MetalBackendConfiguration(library), collector::record);
                BufferRepresentation input = integration.borrow(workload.storage())) {
            List<BufferRepresentation> callerInputs = List.of(input);
            CandidateSuite suite = candidates(integration, workload);
            requireCompleteCandidatePair(suite);
            for (int index = 0; index < suite.candidates().size(); index++) {
                int eventStart = observedEvents.size();
                var candidate = suite.candidates().get(index);
                var preparation = suite.tuning().trialPreparation(suite.batch(), candidate);
                try (PreparedExecution execution = GraphPreparation.prepare(
                        suite.artifacts(), List.of(preparation), integration.scheduleAssembler())) {
                    String checksum = materializedChecksum(
                            integration, execution, callerInputs, suite.outputDescriptor(), workload,
                            "attestation candidate " + index);
                    if (!checksum.equals(workload.expectedSha256())) {
                        throw new IllegalStateException(
                                "attestation checksum disagrees with fixed expected output");
                    }
                }
                List<TraceEvent<? extends TracePayload>> snapshot = collector.snapshot();
                if (snapshot.size() != eventStart + 4
                        || !snapshot.subList(0, eventStart).equals(observedEvents)) {
                    throw new IllegalStateException(
                            "attestation candidate must add PREPARE structure/outcome "
                                    + "and RUN plan/outcome events");
                }
                if (snapshot.get(eventStart).phase() != TracePhase.PREPARE
                        || snapshot.get(eventStart).level() != TraceLevel.INFO
                        || !(snapshot.get(eventStart).payload()
                                instanceof MetalPreparationStructure)
                        || snapshot.get(eventStart + 2).phase() != TracePhase.RUN
                        || snapshot.get(eventStart + 2).level() != TraceLevel.INFO
                        || !(snapshot.get(eventStart + 2).payload()
                                instanceof MetalInvocationPlan)) {
                    throw new IllegalStateException(
                            "attestation structural PREPARE/RUN events have wrong shape");
                }
                TraceRouteKind expectedRoute = index == 0
                        ? TraceRouteKind.CUSTOM_KERNEL
                        : TraceRouteKind.GRAPH_EXECUTABLE;
                attestations.add(attestation(
                        index,
                        identityHex(suite, index),
                        expectedRoute,
                        snapshot.get(eventStart + 1),
                        snapshot.get(eventStart + 3)));
                observedEvents = snapshot;
            }
        }
        return List.copyOf(attestations);
    }

    private static Attestation attestation(
            int candidateIndex,
            String candidateIdentityHex,
            TraceRouteKind expectedRoute,
            TraceEvent<? extends TracePayload> preparationEvent,
            TraceEvent<? extends TracePayload> invocationEvent) {
        if (preparationEvent.phase() != TracePhase.PREPARE
                || preparationEvent.level() != TraceLevel.INFO
                || !(preparationEvent.payload() instanceof BackendPreparationOutcome preparation)) {
            throw new IllegalStateException("attestation preparation event has wrong shape");
        }
        if (invocationEvent.phase() != TracePhase.RUN
                || invocationEvent.level() != TraceLevel.INFO
                || !(invocationEvent.payload() instanceof BackendInvocationOutcome invocation)) {
            throw new IllegalStateException("attestation invocation event has wrong shape");
        }
        requireSuccessfulTrace(
                preparation.status(), preparation.profile(), preparation.route(),
                preparation.nativeStatus().orElseThrow().kind(),
                preparation.nativeStatus().orElseThrow().code(), expectedRoute, "PREPARE");
        requireSuccessfulTrace(
                invocation.status(), invocation.profile(), invocation.route(),
                invocation.nativeStatus().orElseThrow().kind(),
                invocation.nativeStatus().orElseThrow().code(), expectedRoute, "RUN");
        if (preparation.cacheStatus() != TraceCacheStatus.NOT_QUERIED) {
            throw new IllegalStateException("Metal preparation trace must report NOT_QUERIED");
        }
        if (!preparation.preparedUnitId().equals(invocation.preparedUnitId())) {
            throw new IllegalStateException("PREPARE and RUN trace units do not match");
        }
        return new Attestation(candidateIndex, candidateIdentityHex, expectedRoute);
    }

    private static void requireSuccessfulTrace(
            TraceOutcomeStatus status,
            TraceNumericalProfile profile,
            TraceRouteKind route,
            TraceNativeStatusKind nativeStatus,
            int nativeCode,
            TraceRouteKind expectedRoute,
            String phase) {
        if (status != TraceOutcomeStatus.SUCCEEDED
                || profile != TraceNumericalProfile.STRICT_IEEE
                || route != expectedRoute
                || nativeStatus != TraceNativeStatusKind.SUCCESS
                || nativeCode != 0) {
            throw new IllegalStateException(phase + " trace does not attest expected success");
        }
    }

    private static TimedReport measure(
            Path library,
            Workload workload,
            List<Attestation> attestations,
            Profile profile) {
        try (MetalBackendIntegration integration = MetalBackendIntegration.open(
                new MetalBackendConfiguration(library))) {
            CandidateSuite suite = candidates(integration, workload);
            requireCompleteCandidatePair(suite);
            for (int index = 0; index < attestations.size(); index++) {
                if (!identityHex(suite, index).equals(attestations.get(index).candidateIdentityHex())) {
                    throw new IllegalStateException(
                            "timed candidate identity does not match traced attestation");
                }
            }

            PreparedRoute first = prepareRoute(integration, suite, 0);
            try (PreparedExecution ignoredFirst = first.execution()) {
                PreparedRoute second = prepareRoute(integration, suite, 1);
                try (PreparedExecution ignoredSecond = second.execution();
                        BufferRepresentation input = integration.borrow(workload.storage())) {
                    List<BufferRepresentation> callerInputs = List.of(input);
                    List<PreparedRoute> prepared = List.of(first, second);
                    List<RouteMeasurements> routes = List.of(
                            new RouteMeasurements(0, first.prepareNanos()),
                            new RouteMeasurements(1, second.prepareNanos()));
                    for (RouteMeasurements route : routes) {
                        route.precheckSha256 = materializedChecksum(
                                integration,
                                prepared.get(route.candidateIndex).execution(),
                                callerInputs,
                                suite.outputDescriptor(),
                                workload,
                                "precheck candidate " + route.candidateIndex);
                    }

                    for (int round = 0; round < profile.warmupRounds(); round++) {
                        for (int candidateIndex : order(round)) {
                            runBatch(
                                    prepared.get(candidateIndex).execution(), callerInputs, 1);
                        }
                    }

                    List<List<Integer>> roundOrders = new ArrayList<>(profile.measurementRounds());
                    for (int round = 0; round < profile.measurementRounds(); round++) {
                        int[] order = order(round);
                        roundOrders.add(List.of(order[0], order[1]));
                        for (int candidateIndex : order) {
                            RouteMeasurements route = routes.get(candidateIndex);
                            TimedSample sample = timedSample(
                                    prepared.get(candidateIndex).execution(),
                                    callerInputs,
                                    route.suggestedIterations,
                                    profile.minimumSampleNanos());
                            route.suggestedIterations = sample.iterations();
                            route.sampleIterations.add(sample.iterations());
                            route.repeatedBatchNanos.add(sample.elapsedNanos());
                            route.repeatedRunNanos.add(
                                    sample.elapsedNanos() / sample.iterations());
                            route.postBatchSha256.add(materializedChecksum(
                                    integration,
                                    prepared.get(candidateIndex).execution(),
                                    callerInputs,
                                    suite.outputDescriptor(),
                                    workload,
                                    "post-batch candidate " + candidateIndex));
                        }
                    }
                    for (RouteMeasurements route : routes) {
                        if (!route.precheckSha256.equals(workload.expectedSha256())
                                || route.postBatchSha256.stream()
                                        .anyMatch(value -> !value.equals(workload.expectedSha256()))) {
                            throw new IllegalStateException(
                                    "retained correctness checksum changed for candidate "
                                            + route.candidateIndex);
                        }
                        if (route.repeatedBatchNanos.stream()
                                .anyMatch(value -> value < profile.minimumSampleNanos())) {
                            throw new IllegalStateException(
                                    "retained sample fell below declared floor");
                        }
                    }
                    return new TimedReport(
                            suite.compileNanos(),
                            suite.enumerationNanos(),
                            List.copyOf(routes),
                            List.copyOf(roundOrders),
                            suite.compatibilitySchema(),
                            suite.reuseScope().name());
                }
            }
        }
    }

    private static PreparedRoute prepareRoute(
            MetalBackendIntegration integration, CandidateSuite suite, int candidateIndex) {
        long begin = System.nanoTime();
        var preparation = suite.tuning().trialPreparation(
                suite.batch(), suite.candidates().get(candidateIndex));
        PreparedExecution execution = GraphPreparation.prepare(
                suite.artifacts(), List.of(preparation), integration.scheduleAssembler());
        return new PreparedRoute(execution, System.nanoTime() - begin);
    }

    private static TimedSample timedSample(
            PreparedExecution execution,
            List<BufferRepresentation> callerInputs,
            int suggestedIterations,
            long minimumSampleNanos) {
        int iterations = 0;
        int chunk = Math.max(1, suggestedIterations);
        long begin = System.nanoTime();
        long elapsed;
        do {
            if (chunk > MAX_SAMPLE_ITERATIONS - iterations) {
                throw new IllegalStateException("retained sample iteration ceiling exceeded");
            }
            runBatch(execution, callerInputs, chunk);
            iterations += chunk;
            elapsed = System.nanoTime() - begin;
            if (elapsed < minimumSampleNanos) {
                long capacity = MAX_SAMPLE_ITERATIONS - (long) iterations;
                if (capacity == 0L) {
                    throw new IllegalStateException(
                            "retained sample did not reach declared floor");
                }
                long average = Math.max(1L, elapsed / iterations);
                long remaining = minimumSampleNanos - elapsed;
                long required = Math.max(1L, Math.ceilDiv(remaining, average));
                chunk = Math.toIntExact(Math.min(
                        capacity, Math.max(required, Math.ceilDiv(iterations, 4))));
            }
        } while (elapsed < minimumSampleNanos);
        return new TimedSample(iterations, elapsed);
    }

    private static void runBatch(
            PreparedExecution execution,
            List<BufferRepresentation> callerInputs,
            int iterations) {
        for (int index = 0; index < iterations; index++) {
            try (RunResult ignored = RUNNER.run(execution, callerInputs)) {
                // The retained boundary includes result closure but excludes host materialization.
            }
        }
    }

    private static String materializedChecksum(
            MetalBackendIntegration integration,
            PreparedExecution execution,
            List<BufferRepresentation> callerInputs,
            TensorDescriptor outputDescriptor,
            Workload workload,
            String phase) {
        byte[] actual;
        try (RunResult result = RUNNER.run(execution, callerInputs)) {
            if (result.resultCount() != 1) {
                throw new IllegalStateException(phase + " must publish exactly one result");
            }
            actual = integration.copyToCanonicalHostBytes(
                    result.publicationRepresentation(0),
                    outputDescriptor,
                    workload.expectedOutput().length);
        }
        if (!Arrays.equals(workload.expectedOutput(), actual)) {
            int word = firstDifferentWord(workload.expectedOutput(), actual);
            int expected = ByteBuffer.wrap(workload.expectedOutput()).getInt(word * Integer.BYTES);
            int observed = actual.length >= (word + 1) * Integer.BYTES
                    ? ByteBuffer.wrap(actual).getInt(word * Integer.BYTES)
                    : 0;
            throw new IllegalStateException(phase + " raw-bit mismatch at word " + word
                    + ": expected 0x" + String.format("%08x", expected)
                    + ", observed 0x" + String.format("%08x", observed));
        }
        return sha256(actual);
    }

    private static int firstDifferentWord(byte[] expected, byte[] actual) {
        int common = Math.min(expected.length, actual.length);
        for (int index = 0; index < common; index++) {
            if (expected[index] != actual[index]) {
                return index / Integer.BYTES;
            }
        }
        return common / Integer.BYTES;
    }

    private static CandidateSuite candidates(
            MetalBackendIntegration integration, Workload workload) {
        long begin = System.nanoTime();
        CompileArtifacts artifacts = GraphCompilationPort.compile(
                CompileMode.FORWARD_ONLY,
                NumericalProfile.STRICT_IEEE,
                List.of(workload.output()),
                Optional.empty(),
                GraphOptimizationConfig.disabled(),
                BackendIntent.unconstrained(),
                PartitionScoringConfig.neutral(),
                List.of(new MetalCapabilityProvider()),
                List.of(integration.availabilitySnapshot()));
        long compileNanos = System.nanoTime() - begin;
        if (artifacts.partitions().size() != 1
                || !artifacts.partitions().getFirst().owner()
                        .equals(MetalCapabilityProvider.METAL_BACKEND_ID)) {
            throw new IllegalStateException(
                    "fixed benchmark must compile to exactly one Metal partition");
        }
        var projected = GraphPreparation.project(
                artifacts,
                artifacts.partitions().getFirst(),
                integration.partitionPreparation().backendInputs());
        MetalLocalWorkloadTuning tuning = integration.localWorkloadTuning();
        begin = System.nanoTime();
        var handoff = tuning.candidateHandoff(projected).orElseThrow(() ->
                new IllegalStateException("fixed singleton NEG must expose both Metal routes"));
        var batch = handoff.candidateBatch();
        List<MetalLocalWorkloadTuning.Candidate> candidates = tuning.candidates(batch);
        long enumerationNanos = System.nanoTime() - begin;
        if (handoff.partition() != artifacts.partitions().getFirst()
                || handoff.selectedDecision().isPresent()) {
            throw new IllegalStateException("fixed candidate handoff association is invalid");
        }
        var compatibility = tuning.compatibility(batch);
        var publicationValueId = artifacts.publication().forwardBindings().getFirst().valueId();
        TensorDescriptor outputDescriptor = artifacts.graph().values().stream()
                .filter(value -> value.id().equals(publicationValueId))
                .findFirst()
                .orElseThrow(() -> new IllegalStateException(
                        "fixed benchmark publication descriptor is unavailable"))
                .descriptor();
        return new CandidateSuite(
                artifacts,
                tuning,
                batch,
                List.copyOf(candidates),
                compileNanos,
                enumerationNanos,
                compatibility.schemaVersion(),
                outputDescriptor,
                compatibility.reuseScope());
    }

    private static void requireCompleteCandidatePair(CandidateSuite suite) {
        if (suite.candidates().size() != 2) {
            throw new IllegalStateException(
                    "fixed Metal benchmark requires exactly two candidates");
        }
        String first = identityHex(suite, 0);
        String second = identityHex(suite, 1);
        if (first.equals(second)) {
            throw new IllegalStateException("fixed Metal candidates must have distinct identities");
        }
    }

    private static String identityHex(CandidateSuite suite, int candidateIndex) {
        return HexFormat.of().formatHex(suite.tuning().candidateIdentity(
                suite.candidates().get(candidateIndex)).bytes());
    }

    private static int[] order(int round) {
        return round % 2 == 0 ? new int[] {0, 1} : new int[] {1, 0};
    }

    private static String report(
            Profile profile,
            BaselineMetadata metadata,
            Path library,
            Workload workload,
            List<Attestation> attestations,
            TimedReport timed,
            long elapsedNanos) {
        var runtime = ManagementFactory.getRuntimeMXBean();
        StringBuilder out = new StringBuilder(16_384);
        out.append("{\"schema\":1")
                .append(",\"benchmark\":\"metal-singleton-neg-routes\"")
                .append(",\"profile\":\"").append(profile.name()).append('"')
                .append(",\"eligibleForProductionDecision\":false")
                .append(",\"autotuningEvidence\":false")
                .append(",\"selectionMode\":\"fixed-candidate-enumeration-no-winner\"")
                .append(",\"source\":{")
                .append("\"benchmarkClassSha256\":\"").append(classSha256()).append('"')
                .append(",\"baselineMetadata\":")
                .append(metadata == null ? "null" : metadata.json())
                .append('}')
                .append(",\"environment\":{")
                .append("\"java\":\"").append(esc(System.getProperty("java.version"))).append('"')
                .append(",\"javaVm\":\"")
                .append(esc(System.getProperty("java.vm.name") + " "
                        + System.getProperty("java.vm.version"))).append('"')
                .append(",\"jvmArguments\":").append(strings(runtime.getInputArguments()))
                .append(",\"osName\":\"").append(esc(System.getProperty("os.name"))).append('"')
                .append(",\"osVersion\":\"").append(esc(System.getProperty("os.version"))).append('"')
                .append(",\"osArch\":\"").append(esc(System.getProperty("os.arch"))).append('"')
                .append(",\"availableProcessors\":")
                .append(Runtime.getRuntime().availableProcessors())
                .append(",\"initialMemoryBytesObserved\":")
                .append(Runtime.getRuntime().totalMemory())
                .append(",\"maxMemoryBytes\":").append(Runtime.getRuntime().maxMemory())
                .append('}')
                .append(",\"nativeLibrary\":{")
                .append("\"realPath\":\"").append(esc(library.toString())).append('"')
                .append(",\"sizeBytes\":").append(fileSize(library))
                .append(",\"sha256\":\"").append(fileSha256(library)).append("\"}")
                .append(",\"protocol\":{")
                .append("\"traceAttestation\":\"separate-untimed-integration\"")
                .append(",\"timedIntegration\":\"ordinary-no-trace-open\"")
                .append(",\"timedBoundary\":\"PreparedExecutionRunner.run+RunResult.close\"")
                .append(",\"outputMaterializationInsideTiming\":false")
                .append(",\"warmupRounds\":").append(profile.warmupRounds())
                .append(",\"measurementRounds\":").append(profile.measurementRounds())
                .append(",\"minimumRetainedBatchNanos\":")
                .append(profile.minimumSampleNanos())
                .append(",\"maximumSampleIterations\":").append(MAX_SAMPLE_ITERATIONS)
                .append(",\"noRetryOrDiscard\":true")
                .append(",\"independentUnit\":\"fresh-jvm-invocation\"")
                .append(",\"withinProcessSamplesCorrelated\":true")
                .append(",\"roundOrders\":").append(nestedIntegers(timed.roundOrders()))
                .append('}')
                .append(",\"workload\":{")
                .append("\"generatorSchema\":").append(GENERATOR_SCHEMA_VERSION)
                .append(",\"generator\":\"sentinel-prefix-plus-indexed-binary32-mod101\"")
                .append(",\"indexOrigin\":0")
                .append(",\"sentinelCount\":").append(SENTINEL_BITS.length)
                .append(",\"sentinelRawWordsHex\":").append(sentinelRawWordsHex())
                .append(",\"tail\":{")
                .append("\"firstIndex\":").append(SENTINEL_BITS.length)
                .append(",\"lastIndex\":").append(ELEMENT_COUNT - 1)
                .append(",\"formula\":\"rawBits(roundTiesToEvenBinary32(exactBinary32((((int64) i * 17) % 101) - 50) / exactBinary32(101)))\"")
                .append(",\"integerArithmetic\":\"exact signed 64-bit over declared index range\"")
                .append(",\"remainder\":\"Java % on nonnegative dividend; result 0..100\"")
                .append(",\"numeratorRange\":[-50,50]")
                .append(",\"numeratorConversion\":\"exact integer-to-binary32\"")
                .append(",\"denominatorBinary32RawWordHex\":\"0x42ca0000\"")
                .append(",\"division\":\"strict IEEE 754 binary32 roundTiesToEven\"")
                .append(",\"encoding\":\"Float.floatToRawIntBits\"}")
                .append(",\"expectedRawWordFormula\":\"inputRawWord XOR 0x80000000\"")
                .append(",\"operation\":\"NEG\"")
                .append(",\"numericalProfile\":\"STRICT_IEEE\"")
                .append(",\"graphOptimizations\":\"disabled\"")
                .append(",\"dataType\":\"FLOAT32\"")
                .append(",\"layout\":\"CONTIGUOUS\"")
                .append(",\"requiresGrad\":false")
                .append(",\"shape\":[").append(ELEMENT_COUNT).append(']')
                .append(",\"elementCount\":").append(ELEMENT_COUNT)
                .append(",\"inputSha256\":\"").append(workload.inputSha256()).append('"')
                .append(",\"expectedOutputSha256\":\"")
                .append(workload.expectedSha256()).append("\"}")
                .append(",\"compileNanos\":").append(timed.compileNanos())
                .append(",\"candidateEnumerationNanos\":")
                .append(timed.enumerationNanos())
                .append(",\"elapsedNanos\":").append(elapsedNanos)
                .append(",\"routes\":[");
        for (int index = 0; index < timed.routes().size(); index++) {
            if (index > 0) out.append(',');
            RouteMeasurements route = timed.routes().get(index);
            Attestation attestation = attestations.get(index);
            out.append("{\"candidateIndex\":").append(index)
                    .append(",\"candidateIdentityHex\":\"")
                    .append(attestation.candidateIdentityHex()).append('"')
                    .append(",\"compatibilitySchema\":")
                    .append(timed.compatibilitySchema())
                    .append(",\"reuseScope\":\"").append(timed.reuseScope()).append('"')
                    .append(",\"attestation\":{")
                    .append("\"route\":\"").append(attestation.route()).append('"')
                    .append(",\"prepareStatus\":\"SUCCEEDED\"")
                    .append(",\"runStatus\":\"SUCCEEDED\"")
                    .append(",\"cacheStatus\":\"NOT_QUERIED\"")
                    .append(",\"nativeStatus\":\"SUCCESS\"}")
                    .append(",\"prepareNanos\":").append(route.prepareNanos)
                    .append(",\"correctness\":{")
                    .append("\"expectedSha256\":\"").append(workload.expectedSha256()).append('"')
                    .append(",\"precheckSha256\":\"").append(route.precheckSha256).append('"')
                    .append(",\"postBatchSha256\":").append(strings(route.postBatchSha256))
                    .append(",\"exactRawBits\":true}")
                    .append(",\"sampleIterations\":").append(integers(route.sampleIterations))
                    .append(",\"repeatedBatchNanos\":").append(longs(route.repeatedBatchNanos))
                    .append(",\"repeatedRunNanos\":").append(longs(route.repeatedRunNanos))
                    .append(",\"distribution\":")
                    .append(distribution(route.repeatedRunNanos))
                    .append('}');
        }
        return out.append("]}").toString();
    }

    private static String distribution(List<Long> values) {
        List<Long> sorted = new ArrayList<>(values);
        Collections.sort(sorted);
        return "{\"minimum\":" + sorted.getFirst()
                + ",\"median\":" + sorted.get(sorted.size() / 2)
                + ",\"maximum\":" + sorted.getLast() + "}";
    }

    private static String strings(List<String> values) {
        StringBuilder out = new StringBuilder("[");
        for (int index = 0; index < values.size(); index++) {
            if (index > 0) out.append(',');
            out.append('"').append(esc(values.get(index))).append('"');
        }
        return out.append(']').toString();
    }

    private static String integers(List<Integer> values) {
        StringBuilder out = new StringBuilder("[");
        for (int index = 0; index < values.size(); index++) {
            if (index > 0) out.append(',');
            out.append(values.get(index));
        }
        return out.append(']').toString();
    }

    private static String nestedIntegers(List<List<Integer>> values) {
        StringBuilder out = new StringBuilder("[");
        for (int index = 0; index < values.size(); index++) {
            if (index > 0) out.append(',');
            out.append(integers(values.get(index)));
        }
        return out.append(']').toString();
    }

    private static String longs(List<Long> values) {
        StringBuilder out = new StringBuilder("[");
        for (int index = 0; index < values.size(); index++) {
            if (index > 0) out.append(',');
            out.append(values.get(index));
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
                    if (character < 0x20) {
                        escaped.append(String.format("\\u%04x", (int) character));
                    } else {
                        escaped.append(character);
                    }
                }
            }
        }
        return escaped.toString();
    }

    private static String classSha256() {
        try (InputStream stream = MetalRouteBenchmark.class.getResourceAsStream(
                "MetalRouteBenchmark.class")) {
            if (stream == null) {
                throw new IllegalStateException("Metal benchmark class bytes unavailable");
            }
            return sha256(stream.readAllBytes());
        } catch (IOException failure) {
            throw new IllegalStateException("cannot hash Metal benchmark class", failure);
        }
    }

    private static long fileSize(Path path) {
        try {
            return Files.size(path);
        } catch (IOException failure) {
            throw new IllegalStateException("cannot size Metal dylib", failure);
        }
    }

    private static String fileSha256(Path path) {
        try (InputStream stream = Files.newInputStream(path)) {
            MessageDigest digest = sha256Digest();
            byte[] buffer = new byte[16 * 1024];
            int count;
            while ((count = stream.read(buffer)) >= 0) {
                if (count > 0) digest.update(buffer, 0, count);
            }
            return HexFormat.of().formatHex(digest.digest());
        } catch (IOException failure) {
            throw new IllegalStateException("cannot hash Metal dylib", failure);
        }
    }

    private static String sha256(byte[] bytes) {
        return HexFormat.of().formatHex(sha256Digest().digest(bytes));
    }

    private static MessageDigest sha256Digest() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException impossible) {
            throw new ExceptionInInitializerError(impossible);
        }
    }

    private static String sentinelRawWordsHex() {
        StringBuilder out = new StringBuilder("[");
        for (int index = 0; index < SENTINEL_BITS.length; index++) {
            if (index > 0) out.append(',');
            out.append("\"0x")
                    .append(HexFormat.of().toHexDigits(SENTINEL_BITS[index]))
                    .append('"');
        }
        return out.append(']').toString();
    }

    private record Profile(
            String name, int warmupRounds, int measurementRounds, long minimumSampleNanos) {}

    private record Workload(
            Tensor output,
            MemorySegmentStorage storage,
            byte[] expectedOutput,
            String inputSha256,
            String expectedSha256) {}

    private record CandidateSuite(
            CompileArtifacts artifacts,
            MetalLocalWorkloadTuning tuning,
            MetalLocalWorkloadTuning.CandidateBatch batch,
            List<MetalLocalWorkloadTuning.Candidate> candidates,
            long compileNanos,
            long enumerationNanos,
            int compatibilitySchema,
            TensorDescriptor outputDescriptor,
            MetalLocalWorkloadTuning.ReuseScope reuseScope) {}

    private record Attestation(
            int candidateIndex, String candidateIdentityHex, TraceRouteKind route) {}

    private record PreparedRoute(PreparedExecution execution, long prepareNanos) {}

    private record TimedSample(int iterations, long elapsedNanos) {}

    private record TimedReport(
            long compileNanos,
            long enumerationNanos,
            List<RouteMeasurements> routes,
            List<List<Integer>> roundOrders,
            int compatibilitySchema,
            String reuseScope) {}

    private static final class TraceCollector {
        private final List<TraceEvent<? extends TracePayload>> events = new ArrayList<>();

        private synchronized void record(TraceEvent<? extends TracePayload> event) {
            events.add(Objects.requireNonNull(event, "event"));
        }

        private synchronized List<TraceEvent<? extends TracePayload>> snapshot() {
            return List.copyOf(events);
        }
    }

    private static final class RouteMeasurements {
        private final int candidateIndex;
        private final long prepareNanos;
        private final List<Integer> sampleIterations = new ArrayList<>();
        private final List<Long> repeatedBatchNanos = new ArrayList<>();
        private final List<Long> repeatedRunNanos = new ArrayList<>();
        private final List<String> postBatchSha256 = new ArrayList<>();
        private int suggestedIterations = 1;
        private String precheckSha256;

        private RouteMeasurements(int candidateIndex, long prepareNanos) {
            this.candidateIndex = candidateIndex;
            this.prepareNanos = prepareNanos;
        }
    }

    private record BaselineMetadata(
            String baseRevision,
            String sourceIdentity,
            String gitTreeObjectSha1,
            String harnessSourceSha256,
            String hostIdentity,
            String metalDeviceIdentity,
            String osBuild,
            String nativeBuildIdentity,
            String powerState,
            String thermalState,
            int fork) {
        private static BaselineMetadata required() {
            List<String> arguments = ManagementFactory.getRuntimeMXBean().getInputArguments();
            for (String required : List.of(
                    "-Xms1g",
                    "-Xmx1g",
                    "-XX:-TieredCompilation",
                    "-Xbatch",
                    "--enable-native-access=ALL-UNNAMED")) {
                if (!arguments.contains(required)) {
                    throw new IllegalStateException("baseline JVM is missing " + required);
                }
            }
            return new BaselineMetadata(
                    property("baseRevision"),
                    property("sourceIdentity"),
                    property("gitTreeObjectSha1"),
                    property("harnessSourceSha256"),
                    property("hostIdentity"),
                    property("metalDeviceIdentity"),
                    property("osBuild"),
                    property("nativeBuildIdentity"),
                    property("powerState"),
                    property("thermalState"),
                    positiveInt("fork"));
        }

        private String json() {
            return "{\"baseRevision\":\"" + esc(baseRevision)
                    + "\",\"sourceIdentity\":\"" + esc(sourceIdentity)
                    + "\",\"gitTreeObjectSha1\":\"" + esc(gitTreeObjectSha1)
                    + "\",\"harnessSourceSha256\":\"" + esc(harnessSourceSha256)
                    + "\",\"hostIdentity\":{\"value\":\"" + esc(hostIdentity)
                    + "\",\"source\":\"caller-supplied\"}"
                    + ",\"metalDeviceIdentity\":{\"value\":\"" + esc(metalDeviceIdentity)
                    + "\",\"source\":\"caller-supplied\"}"
                    + ",\"osBuild\":{\"value\":\"" + esc(osBuild)
                    + "\",\"source\":\"caller-supplied\"}"
                    + ",\"nativeBuildIdentity\":{\"value\":\"" + esc(nativeBuildIdentity)
                    + "\",\"source\":\"caller-supplied\"}"
                    + ",\"powerState\":{\"value\":\"" + esc(powerState)
                    + "\",\"source\":\"caller-supplied\"}"
                    + ",\"thermalState\":{\"value\":\"" + esc(thermalState)
                    + "\",\"source\":\"caller-supplied\"}"
                    + ",\"fork\":" + fork + "}";
        }

        private static String property(String name) {
            String value = System.getProperty(PROPERTY_PREFIX + name);
            if (value == null || value.isBlank()) {
                throw new IllegalArgumentException(
                        "missing baseline metadata property " + name);
            }
            return value;
        }

        private static int positiveInt(String name) {
            int value = Integer.parseInt(property(name));
            if (value <= 0) {
                throw new IllegalArgumentException(name + " must be positive");
            }
            return value;
        }
    }
}
