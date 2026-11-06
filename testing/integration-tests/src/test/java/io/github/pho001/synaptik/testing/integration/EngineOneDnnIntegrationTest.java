package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration.OneDnnApi;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.engine.RunResult;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/**
 * Public CPU composition regressions with an explicitly supplied oneDNN library. Scalar equations
 * independently check publications and Compiler-generated gradients; no assertion selects a route.
 * Heap fixtures have automatic lifetime, and every prepared execution and run is explicitly closed.
 */
final class EngineOneDnnIntegrationTest {
    @ParameterizedTest
    @EnumSource(OneDnnApi.class)
    void lowPrecisionComputationDoesNotDependOnNativeTypeAvailability(OneDnnApi api) {
        // A valid required library need not implement every native precision. Ordinary CPU
        // computation must remain correct whether cold selection chooses native or portable.
        Shape shape = Shape.of(2, 2);
        var descriptor = new TensorDescriptor(DataType.BFLOAT16, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        short[] leftBits = {0x3f80, 0x4000, 0x4040, 0x4080}; // 1, 2, 3, 4
        short[] rightBits = {0x40a0, 0x40c0, 0x40e0, 0x4100}; // 5, 6, 7, 8
        Tensor left = TensorFactory.create(descriptor, Optional.empty(), Optional.of(
                new MemorySegmentStorage(DataType.BFLOAT16, 4, MemorySegment.ofArray(leftBits))));
        Tensor right = TensorFactory.create(descriptor, Optional.empty(), Optional.of(
                new MemorySegmentStorage(DataType.BFLOAT16, 4, MemorySegment.ofArray(rightBits))));
        try (Engine engine = configured(api)) {
            var graph = engine.compile(List.of(left.matmul(right)));
            try (var prepared = engine.prepare(graph);
                    var result = engine.run(prepared, List.of(left, right))) {
                var bytes = result.materialize(result.publications().getFirst(), 8).bytes();
                for (float expected : new float[] {19, 22, 43, 50}) {
                    float actual = Float.intBitsToFloat(Short.toUnsignedInt(bytes.getShort()) << 16);
                    assertEquals(expected, actual);
                }
            }
        }
    }

    @ParameterizedTest
    @EnumSource(value = OneDnnApi.class, names = {"PRIMITIVES", "GRAPH"})
    void savedPreactivationAndAllLinearGradientsRefreshSharedInputsAtTheSameAddresses(OneDnnApi api) {
        // Each forward/reverse contraction has 32^3 MACs. The same external x is read by the
        // forward contraction and dW, so changing it also exercises shared ingress refresh.
        float[] x = pattern(32 * 32, 13, .03125f);
        float[] w = pattern(32 * 32, 17, .015625f);
        float[] b = pattern(32, 7, .0625f);
        Tensor tx = tensor(Shape.of(32, 32), true, x);
        Tensor tw = tensor(Shape.of(32, 32), true, w);
        Tensor tb = tensor(Shape.of(32), true, b);
        Tensor raw = tx.linear(tw);
        Tensor z = raw.add(tb);
        Tensor activation = z.tanh();
        Tensor loss = activation.mul(activation).sum();
        Tensor one = tensor(Shape.scalar(), false, new float[] {1});
        Tensor zero = tensor(Shape.of(32, 32), false, new float[32 * 32]);
        try (Engine engine = configured(api)) {
            var compiled = engine.compile(List.of(loss, raw, z), List.of(one, zero, zero),
                    List.of(tx, tw, tb));
            try (var prepared = engine.prepare(compiled)) {
                double[][] initial = linearOracle(x, w, b);
                HostTensorValue retained;
                try (RunResult result = engine.run(prepared, List.of(tx, tw, tb, one, zero))) {
                    assertPublications(result, initial);
                    retained = result.materialize(result.publications().get(2), 32L * 32 * Float.BYTES);
                }
                // Mutate the existing arrays, not Tensor identities or storage attachments.
                change(w, -.5f, .0078125f);
                change(b, .5f, .03125f);
                change(x, -.75f, .015625f);
                double[][] changed = linearOracle(x, w, b);
                assertFalse(Arrays.equals(initial[2], changed[2]), "weight/input update must be observable");
                try (RunResult result = engine.run(prepared, List.of(tb, one, tw, zero, tx))) {
                    assertPublications(result, changed);
                }
                assertValue(initial[2], retained);
            }
        }
    }

    @ParameterizedTest
    @EnumSource(value = OneDnnApi.class, names = {"PRIMITIVES", "GRAPH"})
    void groupedConv3dLossAndAllGradientsReuseMutableWeights(OneDnnApi api) {
        // B=4, Cin=Cout=8, groups=2, 4^3 input and 2^3 kernel: 27,648 MACs.
        // Existing Conv AD produces the grouped/broadcast contractions; the oracle below is
        // direct NCDHW window arithmetic, not a second Tensor/AD implementation.
        float[] x = pattern(4 * 8 * 64, 7, .25f);
        float[] w = pattern(8 * 4 * 8, 5, .125f);
        float[] b = pattern(8, 5, .0625f);
        Tensor tx = tensor(Shape.of(4, 8, 4, 4, 4), true, x);
        Tensor tw = tensor(Shape.of(8, 4, 2, 2, 2), true, w);
        Tensor tb = tensor(Shape.of(8), true, b);
        Tensor output = tx.conv3d(tw, tb, new Conv3dAttrs(1, 1, 1, 0, 0, 0, 1, 1, 1, 2));
        Tensor loss = output.mul(output).sum();
        Tensor one = tensor(Shape.scalar(), false, new float[] {1});
        Tensor zero = tensor(Shape.of(4, 8, 3, 3, 3), false, new float[4 * 8 * 27]);
        try (Engine engine = configured(api)) {
            var compiled = engine.compile(List.of(loss, output), List.of(one, zero), List.of(tx, tw, tb));
            try (var prepared = engine.prepare(compiled)) {
                double[][] initial = convolutionOracle(x, w, b);
                try (RunResult result = engine.run(prepared, List.of(tx, tw, tb, one, zero))) {
                    assertPublications(result, initial);
                }
                change(w, -.5f, .03125f);
                change(b, -.5f, .125f);
                double[][] changed = convolutionOracle(x, w, b);
                assertFalse(Arrays.equals(initial[1], changed[1]), "updated convolution must change");
                try (RunResult result = engine.run(prepared, List.of(zero, tw, one, tx, tb))) {
                    assertPublications(result, changed);
                }
            }
        }
    }

    @ParameterizedTest
    @EnumSource(value = OneDnnApi.class, names = {"PRIMITIVES", "GRAPH"})
    void modelConv2dAndConv3dPreserveNegativeZeroIncludingExternalBias(OneDnnApi api) {
        Tensor bias = tensor(Shape.of(1), false, new float[] {-0.0f});
        Tensor x2 = tensor(Shape.of(1, 1, 1, 3), false, new float[] {-0.0f, -0.0f, -0.0f});
        Tensor w2 = tensor(Shape.of(1, 1, 1, 3), false, new float[] {1, 1, 1});
        Tensor x3 = tensor(Shape.of(1, 1, 1, 1, 3), false, new float[] {-0.0f, -0.0f, -0.0f});
        Tensor w3 = tensor(Shape.of(1, 1, 1, 1, 3), false, new float[] {1, 1, 1});
        List<Tensor> outputs = List.of(
                x2.conv2d(w2, bias, new Conv2dAttrs(1, 1, 0, 0, 1, 1, 1)),
                x3.conv3d(w3, bias, new Conv3dAttrs(1, 1, 1, 0, 0, 0, 1, 1, 1, 1)));
        try (Engine engine = configured(api);
                var prepared = engine.prepare(engine.compile(outputs));
                RunResult result = engine.run(prepared, List.of(x2, w2, x3, w3, bias))) {
            assertEquals(2, result.resultCount());
            for (int port = 0; port < 2; port++) {
                HostTensorValue value = result.materialize(result.publications().get(port), Float.BYTES);
                assertEquals(Float.BYTES, value.byteSize());
                assertEquals(0x80000000, Float.floatToRawIntBits(value.bytes().getFloat()),
                        "Model Conv" + (port == 0 ? "2d" : "3d") + " must retain -0");
            }
        }
    }

    @ParameterizedTest
    @EnumSource(value = OneDnnApi.class, names = {"PRIMITIVES", "GRAPH"})
    void pointContractionsAndSeparateEpiloguesPreserveExactSpecialValues(OneDnnApi api) {
        try (Engine engine = configured(api)) {
            Tensor point = tensor(Shape.of(1, 1), false, new float[] {-0.0f});
            Tensor one = tensor(Shape.of(1, 1), false, new float[] {1});
            assertEquals(0x80000000, Float.floatToRawIntBits(
                    forward(engine, point.matmul(one), point, one).bytes().getFloat()));
            Shape scalarMatrix = Shape.of(1, 1);
            var descriptor = new TensorDescriptor(DataType.FLOAT64, scalarMatrix,
                    Optional.of(LayoutDescriptor.contiguous(scalarMatrix)), false);
            Tensor doublePoint = TensorFactory.fromFlatArray(descriptor, Optional.empty(), new double[] {-0.0});
            Tensor doubleOne = TensorFactory.fromFlatArray(descriptor, Optional.empty(), new double[] {1});
            assertEquals(Long.MIN_VALUE, Double.doubleToRawLongBits(forward(
                    engine, doublePoint.matmul(doubleOne), doublePoint, doubleOne).bytes().getDouble()));
            Tensor emptyLeft = tensor(Shape.of(1, 0), false, new float[0]);
            Tensor emptyRight = tensor(Shape.of(0, 1), false, new float[0]);
            assertEquals(0, Float.floatToRawIntBits(forward(
                    engine, emptyLeft.matmul(emptyRight), emptyLeft, emptyRight).bytes().getFloat()));

            int size = 64;
            float[] values = new float[size * 2], weights = new float[2 * size];
            Arrays.fill(weights, 1);
            for (int row = 0; row < size; row++) {
                values[row * 2] = row == 0 ? Float.NaN : Float.NEGATIVE_INFINITY;
            }
            Tensor x = tensor(Shape.of(size, 2), false, values);
            Tensor w = tensor(Shape.of(2, size), false, weights);
            Tensor contraction = x.matmul(w);
            for (Tensor activation : List.of(contraction.relu(), contraction.sigmoid())) {
                var bytes = forward(engine, activation, x, w).bytes();
                for (int row = 0; row < size; row++) for (int col = 0; col < size; col++) {
                    float value = bytes.getFloat();
                    if (row == 0) assertTrue(Float.isNaN(value));
                    else assertEquals(0, Float.floatToRawIntBits(value));
                }
            }
            float[] cancellation = new float[size * 3], right = new float[3 * size], bias = new float[size];
            Arrays.fill(right, 1);
            Arrays.fill(bias, -1);
            for (int row = 0; row < size; row++) {
                cancellation[row * 3] = 33554432;
                cancellation[row * 3 + 1] = -33554432;
                cancellation[row * 3 + 2] = 1;
            }
            Tensor a = tensor(Shape.of(size, 3), false, cancellation);
            Tensor b = tensor(Shape.of(3, size), false, right);
            Tensor addend = tensor(Shape.of(size), false, bias);
            var result = forward(engine, a.matmul(b).add(addend), a, b, addend).bytes();
            while (result.hasRemaining()) {
                float value = result.getFloat();
                assertTrue(value == 0 || value == -1,
                        "external ADD must follow the rounded contraction, not seed its accumulator");
            }
        }
    }

    @ParameterizedTest
    @EnumSource(value = OneDnnApi.class, names = {"PRIMITIVES", "GRAPH"})
    void concurrentRunsKeepFreshBindingsScratchAndRetainedPublicationsIsolated(OneDnnApi api)
            throws Exception {
        int size = 64;
        Tensor tx = tensor(Shape.of(size, size), false, new float[size * size]);
        Tensor tw = tensor(Shape.of(size, size), false, new float[size * size]);
        try (Engine engine = configured(api)) {
            var compiled = engine.compile(List.of(tx.matmul(tw)));
            try (var prepared = engine.prepare(compiled);
                    var workers = Executors.newFixedThreadPool(2)) {
                var barrier = new CyclicBarrier(2);
                var jobs = new ArrayList<Future<Void>>();
                for (int worker = 0; worker < 2; worker++) {
                    int sign = worker == 0 ? 1 : -1;
                    jobs.add(workers.submit(() -> {
                        float[] weights = new float[size * size];
                        for (int row = 0; row < size; row++) for (int col = 0; col < size; col++) {
                            weights[row * size + col] = col % 3 + 1;
                        }
                        Tensor actualW = tensor(tw.descriptor().shape(), false, weights);
                        for (int step = 1; step <= 4; step++) {
                            float factor = sign * step;
                            float[] values = new float[size * size];
                            Arrays.fill(values, factor);
                            Tensor actualX = tensor(tx.descriptor().shape(), false, values);
                            var inputs = compiled.inputs().stream().map(input ->
                                    input.tensorId().equals(tx.id()) ? actualX : actualW).toList();
                            barrier.await(30, TimeUnit.SECONDS);
                            try (RunResult result = engine.runValues(prepared, inputs)) {
                                // Both distinct results must remain valid after both runs complete.
                                barrier.await(30, TimeUnit.SECONDS);
                                var bytes = result.materialize(
                                        result.publications().getFirst(), (long) size * size * Float.BYTES).bytes();
                                for (int row = 0; row < size; row++) for (int col = 0; col < size; col++) {
                                    assertEquals(factor * size * (col % 3 + 1), bytes.getFloat());
                                }
                            }
                        }
                        return null;
                    }));
                }
                for (var job : jobs) job.get(60, TimeUnit.SECONDS);
            }
        }
    }

    @ParameterizedTest
    @EnumSource(value = OneDnnApi.class, names = {"PRIMITIVES", "GRAPH"})
    void offsetGappedInputFeedsDenseConsumerWithoutReadingOrWritingHoles(OneDnnApi api) {
        Shape shape = Shape.of(32, 32);
        LayoutDescriptor layout = LayoutDescriptor.of(shape, new long[] {67, 2}, 3, true);
        float[] physical = new float[Math.toIntExact(layout.referencedElementSpan())];
        Arrays.fill(physical, Float.NaN);
        float[] logical = pattern(32 * 32, 11, .0625f);
        float[] w = pattern(32 * 32, 13, .03125f);
        float[] v = pattern(32 * 32, 17, .015625f);
        Tensor input = TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(layout), false), Optional.empty(), Optional.of(new MemorySegmentStorage(
                        DataType.FLOAT32, physical.length, MemorySegment.ofArray(physical))));
        Tensor weight = tensor(shape, false, w);
        Tensor secondWeight = tensor(shape, false, v);
        Tensor first = input.matmul(weight);
        Tensor second = first.matmul(secondWeight);
        try (Engine engine = configured(api);
                var prepared = engine.prepare(engine.compile(List.of(first, second)))) {
            for (int run = 0; run < 2; run++) {
                for (int row = 0; row < 32; row++) for (int col = 0; col < 32; col++) {
                    physical[3 + row * 67 + col * 2] = logical[row * 32 + col];
                }
                float[] before = physical.clone();
                double[] expectedFirst = product(logical, w);
                double[] expectedSecond = product(expectedFirst, v);
                try (RunResult result = engine.run(prepared, List.of(secondWeight, input, weight))) {
                    assertPublications(result, new double[][] {expectedFirst, expectedSecond});
                }
                assertArrayEquals(before, physical, "input and NaN guard cells remain borrowed/read-only");
                change(logical, -.5f, .03125f);
                change(v, -.5f, .0078125f);
            }
        }
    }

    @ParameterizedTest
    @EnumSource(OneDnnApi.class)
    void publicPredicatesPublishCanonicalBytesAndPreserveUnorderedRules(OneDnnApi api) {
        int count = 65535;
        float[] pattern = {-0f, 0f, Float.NaN, Float.NEGATIVE_INFINITY,
                Float.POSITIVE_INFINITY, -1f, 1f};
        float[] left = new float[count], right = new float[count];
        for (int index = 0; index < count; index++) {
            left[index] = pattern[index % pattern.length];
            right[index] = pattern[(index + 3) % pattern.length];
        }
        Tensor a = tensor(Shape.of(count), false, left);
        Tensor b = tensor(Shape.of(count), false, right);
        List<Tensor> expressions = List.of(a.lessOrEqual(b), a.greaterThan(b), a.notEqualTo(b));
        try (Engine engine = configured(api)) {
            for (int operation = 0; operation < expressions.size(); operation++) {
                var graph = engine.compile(List.of(expressions.get(operation)));
                List<Tensor> inputs = graph.inputs().stream().map(metadata ->
                        metadata.tensorId().equals(a.id()) ? a : b).toList();
                try (var prepared = engine.prepare(graph);
                        var result = engine.run(prepared, inputs)) {
                    var value = result.materialize(result.publications().getFirst(), count);
                    assertEquals(DataType.BOOL, value.dataType());
                    var bytes = value.bytes();
                    assertEquals(count, bytes.remaining());
                    for (int index = 0; index < count; index++) {
                        boolean expected = switch (operation) {
                            case 0 -> left[index] <= right[index];
                            case 1 -> left[index] > right[index];
                            default -> left[index] != right[index];
                        };
                        assertEquals((byte) (expected ? 1 : 0), bytes.get(index),
                                "canonical predicate at logical lane " + index);
                    }
                }
            }
        }
    }

    @ParameterizedTest
    @EnumSource(OneDnnApi.class)
    void scalarMeanRemainsPreparableAcrossApiQueriesAndUsesActualDivision(OneDnnApi api) {
        int count = 65535;
        float[] values = new float[count];
        values[0] = 257f;
        Tensor input = tensor(Shape.of(count), false, values);
        try (Engine engine = configured(api)) {
            var compiled = engine.compile(List.of(input.mean()));
            for (int preparation = 0; preparation < 2; preparation++) {
                try (var prepared = engine.prepare(compiled)) {
                    for (int invocation = 0; invocation < 2; invocation++) {
                        values[0] = invocation == 0 ? 257f : -257f;
                        try (var result = engine.run(prepared, List.of(input))) {
                            float actual = result.materialize(result.publications().getFirst(),
                                    Float.BYTES).bytes().getFloat();
                            assertEquals(Float.floatToRawIntBits(values[0] / count),
                                    Float.floatToRawIntBits(actual));
                        }
                    }
                }
            }
        }
    }

    @ParameterizedTest
    @EnumSource(OneDnnApi.class)
    void meanDoesNotRoundTheExactContributorCountToAnF32Divisor(OneDnnApi api) {
        int count = (1 << 24) + 1;
        float[] values = new float[count];
        values[0] = 1 << 24;
        Tensor input = tensor(Shape.of(count), false, values);
        try (Engine engine = configured(api);
                var prepared = engine.prepare(engine.compile(List.of(input.mean())));
                var result = engine.run(prepared, List.of(input))) {
            float expected = (float) ((double) values[0] / count);
            float actual = result.materialize(result.publications().getFirst(),
                    Float.BYTES).bytes().getFloat();
            assertEquals(Float.floatToRawIntBits(expected), Float.floatToRawIntBits(actual));
        }
    }

    @ParameterizedTest
    @EnumSource(OneDnnApi.class)
    void singletonAxisReductionsPreserveValuesWithoutUnsupportedNativeDescriptors(OneDnnApi api) {
        int count = 65535;
        float[] values = new float[count];
        for (int index = 0; index < count; index++) values[index] = (index % 11 + 1) * .125f;
        Tensor input = tensor(Shape.of(count, 1), false, values);
        try (Engine engine = configured(api)) {
            for (Tensor output : List.of(input.sum(1, false), input.mean(1, false), input.prod(1, false))) {
                try (var prepared = engine.prepare(engine.compile(List.of(output)));
                        var result = engine.run(prepared, List.of(input))) {
                    var bytes = result.materialize(result.publications().getFirst(),
                            (long) count * Float.BYTES).bytes();
                    for (float expected : values) {
                        assertEquals(Float.floatToRawIntBits(expected),
                                Float.floatToRawIntBits(bytes.getFloat()));
                    }
                    assertEquals(0, bytes.remaining());
                }
            }
        }
    }

    /**
     * Runs one public forward recipe and detaches its publication before closing all owners.
     * @param engine borrowed open Engine
     * @param output exact requested publication
     * @param inputs caller-owned tensor identities and live heap storage
     * @return independent canonical bytes that remain valid after the run and preparation close
     */
    private static HostTensorValue forward(Engine engine, Tensor output, Tensor... inputs) {
        try (var prepared = engine.prepare(engine.compile(List.of(output)));
                RunResult result = engine.run(prepared, List.of(inputs))) {
            return result.materialize(result.publications().getFirst(), Long.MAX_VALUE);
        }
    }

    /**
     * Opens the ordinary configured CPU composition without changing process environment.
     *
     * @param api non-null oneDNN API preference supplied by the parameterized test
     * @return a newly owned Engine; only an absent/blank library setting aborts the test
     */
    private static Engine configured(OneDnnApi api) {
        String path = System.getenv("SYNAPTIK_ONEDNN_LIBRARY");
        assumeTrue(path != null && !path.isBlank(), "SYNAPTIK_ONEDNN_LIBRARY is not configured");
        try (Engine.Builder builder = Engine.builder()) {
            return builder.takeOwnership(CpuBackendIntegration.openOneDnn(java.nio.file.Path.of(path), api)).build();
        }
    }

    /**
     * Attaches a caller-owned heap array without copying, so mutations preserve backing identity.
     *
     * @param shape fully static canonical shape matching the array length
     * @param gradient whether the leaf is a requested differentiation role
     * @param values mutable FLOAT32 storage with automatic lifetime and no closeable owner
     * @return a tensor borrowing exactly the supplied array
     */
    private static Tensor tensor(Shape shape, boolean gradient, float[] values) {
        assertEquals(shape.knownElementCount().orElseThrow(), values.length);
        return TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), gradient), Optional.empty(),
                Optional.of(new MemorySegmentStorage(DataType.FLOAT32, values.length, MemorySegment.ofArray(values))));
    }

    /**
     * Generates bounded exactly represented, nonuniform fixture values.
     *
     * @param size number of lanes
     * @param period odd repeat length
     * @param scale binary fraction multiplying centered integers
     * @return a fresh mutable array
     */
    private static float[] pattern(int size, int period, float scale) {
        float[] result = new float[size];
        for (int i = 0; i < size; i++) result[i] = ((i * 19 + 3) % period - period / 2) * scale;
        return result;
    }

    /**
     * Changes values in place without replacing their storage.
     *
     * @param values caller-owned mutable fixture
     * @param factor scale applied to each old value
     * @param increment lane-dependent additive change
     */
    private static void change(float[] values, float factor, float increment) {
        for (int i = 0; i < values.length; i++) values[i] = values[i] * factor + (i % 3 - 1) * increment;
    }

    /**
     * Checks ordered detached forward/gradient publications while the run remains open.
     *
     * @param result live run whose entire publication boundary is asserted
     * @param expected independent logical-order values for each publication
     */
    private static void assertPublications(RunResult result, double[][] expected) {
        assertEquals(expected.length, result.resultCount());
        for (int i = 0; i < expected.length; i++) {
            assertValue(expected[i], result.materialize(result.publications().get(i),
                    (long) expected[i].length * Float.BYTES));
        }
    }

    /**
     * Compares FLOAT32 arithmetic against a double scalar reference using test-only finite tolerance.
     *
     * @param expected independent logical-order values
     * @param actual detached host publication
     */
    private static void assertValue(double[] expected, HostTensorValue actual) {
        assertEquals(DataType.FLOAT32, actual.dataType());
        assertEquals((long) expected.length * Float.BYTES, actual.byteSize());
        var bytes = actual.bytes();
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], bytes.getFloat(), 8e-5 * Math.max(1, Math.abs(expected[i])), "lane " + i);
        }
    }

    /**
     * Evaluates one biased 32-by-32 linear layer, tanh-square loss and all three analytical roles.
     *
     * @param x row-major input with 32 batches and 32 features
     * @param w output-major linear weights
     * @param b per-output bias
     * @return loss, raw contraction, saved preactivation, dX, dW and db, in publication order
     */
    private static double[][] linearOracle(float[] x, float[] w, float[] b) {
        double[] raw = new double[32 * 32], z = new double[raw.length];
        double[] dx = new double[x.length], dw = new double[w.length], db = new double[b.length];
        double loss = 0;
        for (int n = 0; n < 32; n++) for (int o = 0; o < 32; o++) {
            int p = n * 32 + o;
            for (int k = 0; k < 32; k++) raw[p] += (double) x[n * 32 + k] * w[o * 32 + k];
            z[p] = raw[p] + b[o];
            double h = StrictMath.tanh(z[p]);
            loss += h * h;
            double dz = 2 * h * (1 - h * h);
            db[o] += dz;
            for (int k = 0; k < 32; k++) {
                dx[n * 32 + k] += dz * w[o * 32 + k];
                dw[o * 32 + k] += dz * x[n * 32 + k];
            }
        }
        return new double[][] {{loss}, raw, z, dx, dw, db};
    }

    /**
     * Evaluates grouped valid Conv3d and exact window derivatives of the squared-output objective.
     *
     * @param x NCDHW input for [4,8,4,4,4]
     * @param w weights for [8,4,2,2,2], split into two channel groups
     * @param b eight output-channel biases
     * @return loss, output, dX, dW and db, in publication order
     */
    private static double[][] convolutionOracle(float[] x, float[] w, float[] b) {
        double[] output = new double[4 * 8 * 27];
        double[] dx = new double[x.length], dw = new double[w.length], db = new double[b.length];
        double loss = 0;
        for (int n = 0; n < 4; n++) for (int oc = 0; oc < 8; oc++) {
            for (int d = 0; d < 3; d++) for (int h = 0; h < 3; h++) for (int c = 0; c < 3; c++) {
                int p = (n * 8 + oc) * 27 + (d * 3 + h) * 3 + c;
                double value = 0;
                for (int ic = 0; ic < 4; ic++) {
                    for (int kd = 0; kd < 2; kd++) for (int kh = 0; kh < 2; kh++) for (int kw = 0; kw < 2; kw++) {
                        int xi = ((n * 8 + oc / 4 * 4 + ic) * 4 + d + kd) * 16 + (h + kh) * 4 + c + kw;
                        int wi = (oc * 4 + ic) * 8 + (kd * 2 + kh) * 2 + kw;
                        value += (double) x[xi] * w[wi];
                    }
                }
                value += b[oc];
                output[p] = value;
                loss += value * value;
                double seed = 2 * value;
                db[oc] += seed;
                for (int ic = 0; ic < 4; ic++) {
                    for (int kd = 0; kd < 2; kd++) for (int kh = 0; kh < 2; kh++) for (int kw = 0; kw < 2; kw++) {
                        int xi = ((n * 8 + oc / 4 * 4 + ic) * 4 + d + kd) * 16 + (h + kh) * 4 + c + kw;
                        int wi = (oc * 4 + ic) * 8 + (kd * 2 + kh) * 2 + kw;
                        dx[xi] += seed * w[wi];
                        dw[wi] += seed * x[xi];
                    }
                }
            }
        }
        return new double[][] {{loss}, output, dx, dw, db};
    }

    /**
     * Multiplies two canonical 32-by-32 matrices with independent double accumulation.
     *
     * @param left row-major input values
     * @param right row-major right-hand matrix
     * @return row-major product
     */
    private static double[] product(float[] left, float[] right) {
        double[] widened = new double[left.length];
        for (int i = 0; i < left.length; i++) widened[i] = left[i];
        return product(widened, right);
    }

    /**
     * Multiplies the unrounded first scalar reference by a FLOAT32 right-hand matrix.
     *
     * @param left row-major double reference for the first product
     * @param right row-major right-hand matrix
     * @return row-major product
     */
    private static double[] product(double[] left, float[] right) {
        double[] result = new double[32 * 32];
        for (int row = 0; row < 32; row++) for (int col = 0; col < 32; col++) {
            for (int k = 0; k < 32; k++) result[row * 32 + col] += left[row * 32 + k] * right[k * 32 + col];
        }
        return result;
    }
}
