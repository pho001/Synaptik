package io.github.pho001.synaptik.tools.benchmarks;

import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration.OneDnnApi;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.engine.PreparedExecution;
import io.github.pho001.synaptik.engine.RunResult;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.MemorySegment;
import java.lang.management.ManagementFactory;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.concurrent.TimeUnit;

/**
 * Report-only comparison of the same public forward/automatic-differentiation graphs under the
 * default CPU composition and three explicitly configured oneDNN API policies.
 *
 * <p>Only repeated {@link Engine#run} and {@link RunResult#close} are timed. That whole-plan boundary
 * includes caller heap ingress copies, run allocation, binding, native stream/memory/tensor handles,
 * scratch, synchronization, generated remainder, publication, and cleanup whenever selected by the
 * prepared plan. It excludes graph construction, Engine opening, compile, prepare, detached host
 * materialization, and independent numerical checks. Opening, compile, and prepare are reported
 * separately. Every publication, including actual Compiler-generated dX/dW, is checked outside the
 * timed batches against deterministic scalar or closed-form equations, never another Engine.</p>
 *
 * <p>Requested API policy is not evidence that native execution won selection. Public Engine does
 * not expose private selected-plan facts. Use a separate functional ONEDNN_VERBOSE execution to
 * establish native execution; tracing is prohibited here because it invalidates timing. Neither
 * profile authorizes optimization, a production default change, or a numerical-policy change.
 * CPU_DEFAULT means {@code Engine.standard()}, not a forced-portable configuration.</p>
 */
public final class CpuOneDnnBenchmark {
    private static final long BASELINE_BATCH_NANOS = 25_000_000L;
    private static final int MAX_BATCH_ITERATIONS = 100_000;
    private static final int LINEAR_BATCH = 64;
    private static final int LINEAR_INPUTS = 256;
    private static final int LINEAR_OUTPUTS = 128;
    private static final int CONV_BATCH = 4;
    private static final int CONV_CHANNELS = 8;
    private static final int CONV_GROUPS = 2;
    private static final int CONV_SIDE = 6;
    private static final int CONV_KERNEL = 3;
    private static final int CONV_OUTPUT_SIDE = CONV_SIDE - CONV_KERNEL + 1;
    private static final double ABSOLUTE_TOLERANCE = 1e-7;
    private static final double RELATIVE_TOLERANCE = 2e-5;

    private CpuOneDnnBenchmark() {}

    /**
     * Writes a JSON report after every requested configuration passes all numerical checks.
     *
     * @param args exactly two arguments: {@code smoke|baseline} and an absolute path to the required
     *     oneDNN 3.14.0 LP64 little-endian CPU SEQ library; no library is bundled or substituted
     * @throws IllegalArgumentException if arguments, path, or verbose environment are invalid
     * @throws IllegalStateException if a numerical check or retained sample floor fails
     * @throws IOException if the exact native file cannot be read for its SHA-256 identity
     */
    public static void main(String[] args) throws IOException {
        if (args.length != 2) {
            throw new IllegalArgumentException("expected smoke|baseline and absolute oneDNN library path");
        }
        String name = args[0].toLowerCase(Locale.ROOT);
        Profile profile = switch (name) {
            case "smoke" -> new Profile(name, 2, 3, 0);
            case "baseline" -> new Profile(name, 3, 7, BASELINE_BATCH_NANOS);
            default -> throw new IllegalArgumentException("profile must be smoke or baseline");
        };
        rejectVerbose("ONEDNN_VERBOSE");
        rejectVerbose("DNNL_VERBOSE");
        Path library = Path.of(args[1]);
        if (!library.isAbsolute() || !Files.isRegularFile(library)) {
            throw new IllegalArgumentException("oneDNN library must name an existing absolute file");
        }
        String librarySha256;
        try (InputStream bytes = Files.newInputStream(library)) {
            librarySha256 = sha256(bytes);
        }
        List<String> workloads = new ArrayList<>();
        for (Workload workload : List.of(linearMse(), groupedConv3d())) {
            List<String> configurations = new ArrayList<>();
            for (Configuration configuration : Configuration.values()) {
                configurations.add(measure(workload, configuration, library, profile));
            }
            workloads.add("{\"name\":" + quote(workload.name())
                    + ",\"description\":" + quote(workload.description())
                    + ",\"inputShapes\":" + strings(workload.inputs().stream()
                            .map(input -> input.descriptor().shape().toString()).toList())
                    + ",\"publicationOrder\":" + strings(List.of("loss", "forward", "dX", "dW"))
                    + ",\"configurations\":[" + String.join(",", configurations) + "]}");
        }
        var runtime = ManagementFactory.getRuntimeMXBean();
        System.out.println("{\"schema\":1,\"benchmark\":\"CpuOneDnnBenchmark\",\"status\":\"passed\""
                + ",\"profile\":" + quote(profile.name())
                + ",\"reportOnly\":true,\"eligibleForProductionDecision\":false"
                + ",\"timingEligible\":true,\"verboseDisabled\":true"
                + ",\"warmupBatches\":" + profile.warmups()
                + ",\"measurementBatches\":" + profile.samples()
                + ",\"minimumBatchNanos\":" + profile.minimumBatchNanos()
                + ",\"maximumBatchIterations\":" + MAX_BATCH_ITERATIONS
                + ",\"timedBoundary\":" + quote("repeated Engine.run(prepared, callerHeapInputs) + RunResult.close; "
                        + "includes whole-plan ingress copies, runtime allocations, binding, opaque native handles, "
                        + "scratch, native execute/wait, generated remainder, publication and cleanup")
                + ",\"excludedBoundary\":" + quote("fixture/graph construction, Engine opening, compile, prepare, "
                        + "host materialization and independent forward/gradient checks; no optimizer update timed")
                + ",\"planFacts\":{\"status\":\"unavailable\",\"nativeExecutionProven\":false,\"reason\":"
                        + quote("public Engine hides backend-private selection; requested oneDNN configuration "
                                + "does not prove selection; use separate functional ONEDNN_VERBOSE proof") + "}"
                + ",\"openBlasComparison\":{\"status\":\"ineligible\",\"reason\":"
                        + quote("current OpenBLAS route requires whole-partition bare 2D MATMUL, optionally with "
                                + "a private PERMUTE prefix; these complete forward/loss/AD graphs do not qualify; "
                                + "no forced OpenBLAS row or OpenBLAS speedup is claimed") + "}"
                + ",\"cpuDefaultMeaning\":" + quote("Engine.standard(), including existing optional OpenBLAS discovery; "
                        + "not asserted to be forced portable")
                + ",\"numericalCheck\":{\"oracle\":\"independent deterministic scalar/closed-form equations\""
                        + ",\"allPublicationsChecked\":true,\"absoluteTolerance\":" + ABSOLUTE_TOLERANCE
                        + ",\"relativeTolerance\":" + RELATIVE_TOLERANCE
                        + ",\"toleranceMeaning\":\"benchmark check only; not a production numerical-policy grant\"}"
                + ",\"nativeLibrary\":{\"path\":" + quote(library.toString())
                        + ",\"sha256\":" + quote(librarySha256)
                        + ",\"requiredAbi\":\"oneDNN 3.14.0 LP64 little-endian CPU SEQ\"}"
                + ",\"harnessClassSha256\":" + quote(classSha256())
                + ",\"os\":" + quote(System.getProperty("os.name") + " " + System.getProperty("os.version"))
                + ",\"architecture\":" + quote(System.getProperty("os.arch"))
                + ",\"cpuIdentity\":" + quote(cpuIdentity())
                + ",\"availableProcessors\":" + Runtime.getRuntime().availableProcessors()
                + ",\"jdk\":" + quote(System.getProperty("java.runtime.version"))
                + ",\"vm\":" + quote(System.getProperty("java.vm.name") + " " + System.getProperty("java.vm.version"))
                + ",\"jvmArguments\":" + strings(runtime.getInputArguments())
                + ",\"maxHeapBytes\":" + Runtime.getRuntime().maxMemory()
                + ",\"configurationOrder\":" + strings(Arrays.stream(Configuration.values()).map(Enum::name).toList())
                + ",\"orderPolicy\":\"fixed order in one JVM; no randomized-fork or statistical winner claim\""
                + ",\"workloads\":[" + String.join(",", workloads) + "]}");
    }

    /**
     * Prevents functional native tracing from being mistaken for performance evidence.
     *
     * @param variable oneDNN's current or legacy verbose environment-variable name
     * @throws IllegalArgumentException if its value is anything except absent, blank or zero
     */
    private static void rejectVerbose(String variable) {
        String value = System.getenv(variable);
        if (value != null && !value.isBlank() && !value.trim().equals("0")) {
            throw new IllegalArgumentException(variable + " must be unset or 0 for timing; "
                    + "run functional native-execution tracing separately");
        }
    }

    /**
     * Prepares one exact graph and retains all per-sample raw whole-plan measurements.
     *
     * @param workload immutable graph/heap fixtures shared unchanged across configurations
     * @param configuration requested CPU composition, not a selected-plan assertion
     * @param library required exact native file used only by explicit oneDNN configurations
     * @param profile bounded warmup and measurement policy
     * @return JSON containing separate open/compile/prepare durations and validated batch samples
     */
    private static String measure(Workload workload, Configuration configuration, Path library, Profile profile) {
        List<Sample> samples = new ArrayList<>();
        List<Sample> warmups = new ArrayList<>();
        long begin = System.nanoTime();
        try (Engine engine = configuration.open(library)) {
            long openNanos = System.nanoTime() - begin;
            begin = System.nanoTime();
            var graph = engine.compile(workload.outputs(), workload.seeds(), workload.gradientInputs());
            long compileNanos = System.nanoTime() - begin;
            begin = System.nanoTime();
            try (PreparedExecution prepared = engine.prepare(graph)) {
                long prepareNanos = System.nanoTime() - begin;
                check(engine, prepared, workload);
                int iterations = 1;
                for (int i = 0; i < profile.warmups(); i++) {
                    Sample sample = sample(engine, prepared, workload.inputs(), iterations, profile.minimumBatchNanos());
                    warmups.add(sample);
                    iterations = sample.iterations();
                }
                check(engine, prepared, workload);
                for (int i = 0; i < profile.samples(); i++) {
                    Sample sample = sample(engine, prepared, workload.inputs(), iterations, profile.minimumBatchNanos());
                    samples.add(sample);
                    iterations = sample.iterations();
                    check(engine, prepared, workload);
                }
                double[] perRun = samples.stream().mapToDouble(Sample::nanosPerRun).sorted().toArray();
                return "{\"requestedConfiguration\":" + configuration.json()
                        + ",\"numerics\":\"passed-all-forward-and-gradient-lanes\""
                        + ",\"engineOpenNanos\":" + openNanos + ",\"compileNanos\":" + compileNanos
                        + ",\"prepareNanos\":" + prepareNanos
                        + ",\"warmupSamples\":" + sampleJson(warmups)
                        + ",\"rawSamples\":" + sampleJson(samples)
                        + ",\"runNanosDistribution\":{\"minimum\":" + perRun[0]
                        + ",\"median\":" + perRun[perRun.length / 2]
                        + ",\"maximum\":" + perRun[perRun.length - 1] + "}}";
            }
        }
    }

    /**
     * Measures repeated public runs with the existing lifecycle benchmark's elapsed batch-floor
     * approach. Batch growth changes only measurement length, never execution policy.
     *
     * @param engine open Engine borrowing the fixture arrays
     * @param prepared open preparation owned by that Engine
     * @param inputs exact compiled Tensor identities, backed by caller-owned heap arrays
     * @param suggestedIterations positive initial batch size from the preceding batch
     * @param minimumNanos required retained elapsed duration, zero for smoke
     * @return elapsed nanoseconds and the exact number of completed-and-closed runs
     * @throws IllegalStateException if the iteration ceiling prevents satisfying the duration floor
     */
    private static Sample sample(Engine engine, PreparedExecution prepared, List<Tensor> inputs,
            int suggestedIterations, long minimumNanos) {
        int iterations = 0;
        int chunk = suggestedIterations;
        long begin = System.nanoTime();
        long elapsed;
        do {
            if (chunk > MAX_BATCH_ITERATIONS - iterations) {
                throw new IllegalStateException("batch iteration ceiling exceeded");
            }
            for (int i = 0; i < chunk; i++) {
                try (RunResult ignored = engine.run(prepared, inputs)) {
                    // Materialization is intentionally outside this whole-plan execution boundary.
                }
            }
            iterations += chunk;
            elapsed = System.nanoTime() - begin;
            if (elapsed < minimumNanos) {
                long capacity = MAX_BATCH_ITERATIONS - (long) iterations;
                if (capacity == 0) throw new IllegalStateException("batch did not reach required duration");
                long average = Math.max(1L, elapsed / iterations);
                chunk = Math.toIntExact(Math.min(capacity, Math.max(1L,
                        Math.ceilDiv(minimumNanos - elapsed, average))));
            }
        } while (elapsed < minimumNanos);
        return new Sample(iterations, elapsed);
    }

    /**
     * Materializes every publication in a separate, untimed execution and checks every lane.
     *
     * @param engine open Engine associated with the preparation
     * @param prepared reusable preparation containing forward and actual generated adjoints
     * @param workload expected loss, forward, dX and dW in public publication order
     * @throws IllegalStateException if publication metadata or any finite expected value disagrees
     */
    private static void check(Engine engine, PreparedExecution prepared, Workload workload) {
        try (RunResult result = engine.run(prepared, workload.inputs())) {
            if (result.resultCount() != workload.expected().length) {
                throw new IllegalStateException("unexpected publication count for " + workload.name());
            }
            for (int port = 0; port < workload.expected().length; port++) {
                double[] expected = workload.expected()[port];
                HostTensorValue value = result.materialize(result.publications().get(port),
                        (long) expected.length * Float.BYTES);
                if (value.dataType() != DataType.FLOAT32 || value.byteSize() != (long) expected.length * Float.BYTES) {
                    throw new IllegalStateException("unexpected publication metadata for " + workload.name());
                }
                ByteBuffer bytes = value.bytes();
                for (int lane = 0; lane < expected.length; lane++) {
                    float actual = bytes.getFloat();
                    double tolerance = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * Math.abs(expected[lane]);
                    if (!Float.isFinite(actual) || Math.abs(actual - expected[lane]) > tolerance) {
                        throw new IllegalStateException(workload.name() + " port " + port + " lane " + lane
                                + ": expected " + expected[lane] + " but observed " + actual);
                    }
                }
            }
        }
    }

    /**
     * Constructs a dense F32 linear/MSE graph with three large forward/dX/dW contractions.
     *
     * @return one shared graph and independently computed scalar equations for all publications
     */
    private static Workload linearMse() {
        int batches = LINEAR_BATCH, inputs = LINEAR_INPUTS, outputs = LINEAR_OUTPUTS;
        float[] x = new float[batches * inputs], w = new float[outputs * inputs];
        float[] target = new float[batches * outputs];
        for (int b = 0; b < batches; b++) for (int i = 0; i < inputs; i++) {
            x[b * inputs + i] = (b % 7 - 3) / 16f + (i % 5 - 2) / 32f;
        }
        for (int o = 0; o < outputs; o++) for (int i = 0; i < inputs; i++) {
            w[o * inputs + i] = (o % 5 - 2) / 32f + (i % 7 - 3) / 64f;
        }
        for (int b = 0; b < batches; b++) for (int o = 0; o < outputs; o++) {
            target[b * outputs + o] = ((b + 3 * o) % 9 - 4) / 16f;
        }
        double[] y = new double[target.length], dx = new double[x.length], dw = new double[w.length];
        double loss = 0;
        for (int b = 0; b < batches; b++) for (int o = 0; o < outputs; o++) {
            int cell = b * outputs + o;
            for (int i = 0; i < inputs; i++) y[cell] += (double) x[b * inputs + i] * w[o * inputs + i];
            double error = y[cell] - target[cell];
            loss += error * error / y.length;
            double dy = 2 * error / y.length;
            for (int i = 0; i < inputs; i++) {
                dx[b * inputs + i] += dy * w[o * inputs + i];
                dw[o * inputs + i] += dy * x[b * inputs + i];
            }
        }
        Tensor tx = tensor(Shape.of(batches, inputs), true, x);
        Tensor tw = tensor(Shape.of(outputs, inputs), true, w);
        Tensor tt = tensor(Shape.of(batches, outputs), false, target);
        Tensor forward = tx.linear(tw);
        Tensor error = forward.sub(tt);
        return workload("linear-mse-training", "F32 B64 I256 O128; mean((X.linear(W)-target)^2), forward, actual dX/dW",
                error.mul(error).mean(), forward, List.of(tx, tw, tt), List.of(tx, tw),
                new double[][] {{loss}, y, dx, dw});
    }

    /**
     * Builds grouped NCDHW Conv3d training with a spatially constant but batch/channel-varying
     * fixture. Its oracle uses channel sums and exact boundary multiplicities, not Tensor/AD or a
     * second convolution implementation. Forward Conv remains the existing CPU unit; only qualified
     * ordinary contractions in the generated reverse graph can become oneDNN units.
     *
     * @return shared grouped graph with exact binary-fraction forward/gradient expectations
     */
    private static Workload groupedConv3d() {
        int channelsPerGroup = CONV_CHANNELS / CONV_GROUPS;
        int volume = CONV_SIDE * CONV_SIDE * CONV_SIDE;
        int kernelVolume = CONV_KERNEL * CONV_KERNEL * CONV_KERNEL;
        int outputVolume = CONV_OUTPUT_SIDE * CONV_OUTPUT_SIDE * CONV_OUTPUT_SIDE;
        float[] x = new float[CONV_BATCH * CONV_CHANNELS * volume];
        float[] w = new float[CONV_CHANNELS * channelsPerGroup * kernelVolume];
        for (int b = 0; b < CONV_BATCH; b++) for (int c = 0; c < CONV_CHANNELS; c++) {
            Arrays.fill(x, (b * CONV_CHANNELS + c) * volume,
                    (b * CONV_CHANNELS + c + 1) * volume, (1 + (b * 3 + c) % 7) / 16f);
        }
        for (int o = 0; o < CONV_CHANNELS; o++) for (int c = 0; c < channelsPerGroup; c++) {
            Arrays.fill(w, (o * channelsPerGroup + c) * kernelVolume,
                    (o * channelsPerGroup + c + 1) * kernelVolume, (1 + (o * 3 + c) % 5) / 32f);
        }
        double[] y = new double[CONV_BATCH * CONV_CHANNELS * outputVolume];
        double[] dx = new double[x.length], dw = new double[w.length];
        double loss = 0;
        for (int b = 0; b < CONV_BATCH; b++) for (int o = 0; o < CONV_CHANNELS; o++) {
            int groupStart = (o / channelsPerGroup) * channelsPerGroup;
            double channelSum = 0;
            for (int c = 0; c < channelsPerGroup; c++) {
                channelSum += (double) x[(b * CONV_CHANNELS + groupStart + c) * volume]
                        * w[(o * channelsPerGroup + c) * kernelVolume];
            }
            double forward = channelSum * kernelVolume;
            Arrays.fill(y, (b * CONV_CHANNELS + o) * outputVolume,
                    (b * CONV_CHANNELS + o + 1) * outputVolume, forward);
            loss += outputVolume * forward * forward;
            double dy = 2 * forward;
            for (int c = 0; c < channelsPerGroup; c++) {
                double weightGradient = outputVolume * dy * x[(b * CONV_CHANNELS + groupStart + c) * volume];
                int weightBase = (o * channelsPerGroup + c) * kernelVolume;
                for (int k = 0; k < kernelVolume; k++) dw[weightBase + k] += weightGradient;
                double inputContribution = dy * w[weightBase];
                for (int d = 0; d < CONV_SIDE; d++) for (int h = 0; h < CONV_SIDE; h++) {
                    for (int v = 0; v < CONV_SIDE; v++) {
                        int cell = (b * CONV_CHANNELS + groupStart + c) * volume
                                + (d * CONV_SIDE + h) * CONV_SIDE + v;
                        dx[cell] += inputContribution * coverage(d) * coverage(h) * coverage(v);
                    }
                }
            }
        }
        Tensor tx = tensor(Shape.of(CONV_BATCH, CONV_CHANNELS, CONV_SIDE, CONV_SIDE, CONV_SIDE), true, x);
        Tensor tw = tensor(Shape.of(CONV_CHANNELS, channelsPerGroup, CONV_KERNEL, CONV_KERNEL, CONV_KERNEL), true, w);
        Tensor forward = tx.conv3d(tw, new Conv3dAttrs(1, 1, 1, 0, 0, 0, 1, 1, 1, CONV_GROUPS));
        return workload("grouped-conv3d-training", "F32 B4 Cin8 Cout8 groups2 input6^3 kernel3^3; sum(Y^2), forward, actual dX/dW; "
                        + "forward Conv is existing CPU, native eligibility is limited to reverse bare contractions",
                forward.mul(forward).sum(), forward, List.of(tx, tw), List.of(tx, tw),
                new double[][] {{loss}, y, dx, dw});
    }

    /**
     * Counts how many valid unit-stride unpadded kernel windows contain a coordinate on one axis.
     *
     * @param coordinate input-axis coordinate in the fixed Conv3d fixture
     * @return positive number of contributing output/kernel pairs on that axis
     */
    private static int coverage(int coordinate) {
        return Math.min(CONV_KERNEL - 1, coordinate) - Math.max(0, coordinate - CONV_OUTPUT_SIDE + 1) + 1;
    }

    /**
     * Adds explicit loss and zero-forward seeds without introducing a reference Engine.
     *
     * @param name report identity
     * @param description fixed graph dimensions and formula
     * @param loss scalar differentiated root
     * @param forward separately published forward output with a zero cotangent
     * @param leaves caller-owned data input identities
     * @param gradientInputs exact ordered leaves for dX and dW publications
     * @param expected independent loss/forward/dX/dW arrays in public order
     * @return complete immutable fixture references, all backed by automatically managed heap storage
     */
    private static Workload workload(String name, String description, Tensor loss, Tensor forward,
            List<Tensor> leaves, List<Tensor> gradientInputs, double[][] expected) {
        Tensor one = tensor(Shape.scalar(), false, new float[] {1});
        Tensor zero = tensor(forward.descriptor().shape(), false, new float[expected[1].length]);
        List<Tensor> inputs = new ArrayList<>(leaves);
        inputs.add(one);
        inputs.add(zero);
        return new Workload(name, description, List.of(loss, forward), List.of(one, zero),
                gradientInputs, List.copyOf(inputs), expected);
    }

    /**
     * Attaches a borrowed heap array; no Arena or native owner is hidden in fixture creation.
     *
     * @param shape static contiguous shape with element count equal to the array length
     * @param gradient whether Compiler should admit this leaf for differentiation
     * @param values live caller-owned F32 storage, retained unchanged across all configurations
     * @return tensor borrowing the supplied heap array without a copy
     * @throws IllegalArgumentException if the shape and storage disagree
     */
    private static Tensor tensor(Shape shape, boolean gradient, float[] values) {
        if (shape.knownElementCount().orElseThrow() != values.length) {
            throw new IllegalArgumentException("fixture shape/storage mismatch");
        }
        return TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), gradient), Optional.empty(),
                Optional.of(new MemorySegmentStorage(DataType.FLOAT32, values.length, MemorySegment.ofArray(values))));
    }

    /** Requested compositions only; these names never describe observed private execution plans. */
    private enum Configuration {
        CPU_DEFAULT(null), ONEDNN_PRIMITIVES(OneDnnApi.PRIMITIVES), ONEDNN_GRAPH(OneDnnApi.GRAPH), ONEDNN_AUTO(OneDnnApi.AUTO);

        private final OneDnnApi api;

        /**
         * Retains the fixed requested policy.
         *
         * @param api oneDNN preference, or null for the unchanged default Engine
         */
        Configuration(OneDnnApi api) {
            this.api = api;
        }

        /**
         * Transfers an explicit CPU owner to its Engine through the supported builder protocol.
         *
         * @param library exact required oneDNN file; unused by CPU_DEFAULT
         * @return new Engine that owns its CPU integration and closes it on failure or normal cleanup
         */
        Engine open(Path library) {
            if (api == null) return Engine.standard();
            try (Engine.Builder builder = Engine.builder()) {
                return builder.takeOwnership(CpuBackendIntegration.openOneDnn(library, api)).build();
            }
        }

        /**
         * Describes configuration intent without inferring a selected route.
         *
         * @return JSON with requested factory/API and unchanged default shared CPU policy
         */
        String json() {
            return "{\"name\":" + quote(name()) + ",\"factory\":"
                    + quote(api == null ? "Engine.standard" : "Engine.builder.takeOwnership(CpuBackendIntegration.openOneDnn).build")
                    + ",\"oneDnnApi\":" + (api == null ? "null" : quote(api.name()))
                    + ",\"sharedCpuPolicy\":\"defaults, no execution-policy overrides\"}";
        }
    }

    /** Fixed report profile; it never changes CPU or native execution settings. */
    private record Profile(String name, int warmups, int samples, long minimumBatchNanos) {}

    /** Public expression identities and independent expected arrays, shared across compositions. */
    private record Workload(String name, String description, List<Tensor> outputs, List<Tensor> seeds,
            List<Tensor> gradientInputs, List<Tensor> inputs, double[][] expected) {}

    /** One retained elapsed-time batch including every close. */
    private record Sample(int iterations, long elapsedNanos) {
        /**
         * Computes the per-run report statistic without discarding raw batch data.
         *
         * @return elapsed batch nanoseconds divided by its positive completed-run count
         */
        double nanosPerRun() {
            return (double) elapsedNanos / iterations;
        }
    }

    /**
     * Serializes raw measurement evidence outside every timed section.
     *
     * @param samples retained batches in execution order
     * @return JSON array of exact durations/counts and derived per-run nanoseconds
     */
    private static String sampleJson(List<Sample> samples) {
        return "[" + String.join(",", samples.stream().map(sample -> "{\"iterations\":" + sample.iterations()
                + ",\"elapsedNanos\":" + sample.elapsedNanos() + ",\"nanosPerRun\":" + sample.nanosPerRun() + "}").toList()) + "]";
    }

    /**
     * Serializes a list without a new report-library dependency.
     *
     * @param values non-null strings in desired report order
     * @return JSON array of escaped strings
     */
    private static String strings(List<String> values) {
        return "[" + String.join(",", values.stream().map(CpuOneDnnBenchmark::quote).toList()) + "]";
    }

    /**
     * Quotes one JSON string, including control characters that can occur in host metadata.
     *
     * @param value non-null report text
     * @return valid quoted JSON string
     */
    private static String quote(String value) {
        StringBuilder out = new StringBuilder("\"");
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            switch (c) {
                case '"' -> out.append("\\\"");
                case '\\' -> out.append("\\\\");
                case '\n' -> out.append("\\n");
                case '\r' -> out.append("\\r");
                case '\t' -> out.append("\\t");
                default -> {
                    if (c < 0x20) out.append(String.format(Locale.ROOT, "\\u%04x", (int) c));
                    else out.append(c);
                }
            }
        }
        return out.append('"').toString();
    }

    /**
     * Hashes bytes incrementally so library size does not create a matching heap allocation.
     *
     * @param stream open borrowed input, consumed to EOF but not closed here
     * @return lowercase SHA-256 file/content identity
     * @throws IOException if content cannot be read
     */
    private static String sha256(InputStream stream) throws IOException {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] buffer = new byte[64 * 1024];
            int count;
            while ((count = stream.read(buffer)) != -1) digest.update(buffer, 0, count);
            return HexFormat.of().formatHex(digest.digest());
        } catch (NoSuchAlgorithmException impossible) {
            throw new ExceptionInInitializerError(impossible);
        }
    }

    /**
     * Identifies the executable benchmark class in addition to the exact native library.
     *
     * @return SHA-256 of this compiled class resource
     * @throws IOException if the resource cannot be read
     * @throws IllegalStateException if the class loader does not expose the resource
     */
    private static String classSha256() throws IOException {
        try (InputStream stream = CpuOneDnnBenchmark.class.getResourceAsStream("CpuOneDnnBenchmark.class")) {
            if (stream == null) throw new IllegalStateException("benchmark class bytes unavailable");
            return sha256(stream);
        }
    }

    /**
     * Reads CPU identity outside timed work on supported macOS/Linux hosts. Unavailable metadata is
     * reported honestly rather than inferred from architecture or processor count.
     *
     * @return observed CPU brand/model, or an explicit unavailable reason
     */
    private static String cpuIdentity() {
        try {
            if (System.getProperty("os.name").equals("Linux")) {
                try (var lines = Files.lines(Path.of("/proc/cpuinfo"))) {
                    return lines.filter(line -> line.startsWith("model name") || line.startsWith("Hardware"))
                            .map(line -> line.substring(line.indexOf(':') + 1).trim()).findFirst()
                            .orElse("unavailable: /proc/cpuinfo has no model name or Hardware field");
                }
            }
            if (System.getProperty("os.name").equals("Mac OS X")) {
                Process process = new ProcessBuilder("/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string")
                        .redirectErrorStream(true).start();
                try {
                    if (!process.waitFor(2, TimeUnit.SECONDS)) return "unavailable: sysctl timed out";
                    if (process.exitValue() != 0) return "unavailable: sysctl exited " + process.exitValue();
                    try (InputStream output = process.getInputStream()) {
                        return new String(output.readAllBytes(), StandardCharsets.UTF_8).trim();
                    }
                } finally {
                    process.destroyForcibly();
                }
            }
            return "unavailable: unsupported OS CPU identity query";
        } catch (IOException failure) {
            return "unavailable: " + failure.getMessage();
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            return "unavailable: CPU identity query interrupted";
        }
    }
}
