package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.backend.contract.BackendDeviceId;
import io.github.pho001.synaptik.backend.contract.DeviceClass;
import io.github.pho001.synaptik.compiler.FunctionalGradientRequest;
import io.github.pho001.synaptik.compiler.GraphCompilationPort;
import io.github.pho001.synaptik.config.compile.BackendIntent;
import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorId;
import io.github.pho001.synaptik.prepare.GraphPreparation;
import io.github.pho001.synaptik.prepare.PartitionPreparation;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.run.PreparedExecutionRunner;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

/** Direct native regressions for rank bounds, exact low words, and creating-context identity. */
class MetalNativeBoundaryRepairTest {
    @Test
    void publicOrdinaryEmptyNegSumMatmulAndTopKPublishTheirCompleteOutputs() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
            Shape leftShape = Shape.of(2, 0);
            Shape rightShape = Shape.of(0, 3);
            Tensor left = TensorFactory.create(new TensorDescriptor(type, leftShape,
                    Optional.of(LayoutDescriptor.contiguous(leftShape)), false));
            Tensor right = TensorFactory.create(new TensorDescriptor(type, rightShape,
                    Optional.of(LayoutDescriptor.contiguous(rightShape)), false));
            var selected = left.topK(0, 1);
            var actual = prepared(List.of(left.neg(), left.sum(1), left.matmul(right),
                            selected.values(), selected.indices()),
                    Map.of(left.id(), new byte[0], right.id(), new byte[0]), Optional.empty());
            assertEquals(5, actual.size());
            assertArrayEquals(new byte[0], actual.get(0), type + " NEG");
            assertArrayEquals(new byte[2 * type.byteWidth()], actual.get(1), type + " SUM");
            assertArrayEquals(new byte[6 * type.byteWidth()], actual.get(2), type + " MATMUL");
            assertArrayEquals(new byte[0], actual.get(3), type + " TOP_K values");
            assertArrayEquals(new byte[0], actual.get(4), type + " TOP_K indices");
        }
    }

    @Test
    void emptyMatmulWithAnchorEpiloguePublishesAnEmptyTerminal() {
        Shape leftShape = Shape.of(0, 2), rightShape = Shape.of(2, 3);
        Tensor left = TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, leftShape,
                Optional.of(LayoutDescriptor.contiguous(leftShape)), false));
        Tensor right = TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, rightShape,
                Optional.of(LayoutDescriptor.contiguous(rightShape)), false));
        assertArrayEquals(new byte[0], prepared(List.of(left.matmul(right).relu()),
                Map.of(left.id(), new byte[0], right.id(), numeric(DataType.FLOAT32, 1, 2, 3, 4, 5, 6)),
                Optional.empty()).getFirst());
    }

    @Test
    void publicSourceOnlyFloat64GeneralScanPublishesExactIdentityHistory() {
        Shape stepShape = Shape.of(1), sequenceShape = Shape.of(2, 1);
        Tensor formal = TensorFactory.create(new TensorDescriptor(DataType.FLOAT64, stepShape,
                Optional.of(LayoutDescriptor.contiguous(stepShape)), false));
        Tensor sequence = TensorFactory.create(new TensorDescriptor(DataType.FLOAT64, sequenceShape,
                Optional.of(LayoutDescriptor.contiguous(sequenceShape)), false));
        var body = new io.github.pho001.synaptik.model.tensor.ScanBody(
                List.of(), List.of(formal), List.of(), List.of(), List.of(formal));
        Tensor output = io.github.pho001.synaptik.model.tensor.Scan.apply(
                List.of(), List.of(sequence), List.of(), body, false).outputs().getFirst();
        for (long[] words : List.of(
                new long[] {Double.doubleToRawLongBits(3.0), Double.doubleToRawLongBits(4.0)},
                new long[] {0x8000000000000000L, 0xfff0000000004321L})) {
            ByteBuffer storage = ByteBuffer.allocate(16).order(ByteOrder.nativeOrder());
            for (long word : words) storage.putLong(word);
            assertArrayEquals(storage.array(), prepared(List.of(output),
                    Map.of(sequence.id(), storage.array()), Optional.empty()).getFirst());
        }
    }

    @Test
    void noncanonicalFloat64ScanRejectsBeforeNativePreparation() {
        Shape stepShape = Shape.of(1);
        Tensor formal = TensorFactory.create(new TensorDescriptor(DataType.FLOAT64, stepShape,
                Optional.of(LayoutDescriptor.contiguous(stepShape)), false));
        Tensor sequence = external(DataType.FLOAT64, Shape.of(2, 1), new long[] {2, 1}, 1, false);
        var body = new io.github.pho001.synaptik.model.tensor.ScanBody(
                List.of(), List.of(formal), List.of(), List.of(), List.of(formal));
        Tensor output = io.github.pho001.synaptik.model.tensor.Scan.apply(
                List.of(), List.of(sequence), List.of(), body, false).outputs().getFirst();
        var availability = new BackendAvailabilitySnapshot(MetalCapabilityProvider.METAL_BACKEND_ID,
                Map.of(new BackendDeviceId(MetalCapabilityProvider.METAL_BACKEND_ID, "default"),
                        DeviceClass.ACCELERATOR));
        assertThrows(IllegalStateException.class, () -> GraphCompilationPort.compile(
                CompileMode.FORWARD_ONLY, List.of(output), Optional.empty(),
                GraphOptimizationConfig.disabled(), BackendIntent.unconstrained(),
                PartitionScoringConfig.neutral(), List.of(new MetalCapabilityProvider()), List.of(availability)));
    }

    @Test
    void publicExternalTransposeAndOffsetStridedMatmulUseLogicalContraction() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
            for (boolean transpose : List.of(false, true)) {
                Tensor left = external(type, Shape.of(2, 3),
                        transpose ? new long[] {1, 2} : new long[] {8, 2},
                        transpose ? 0 : 1, false);
                Tensor right = external(type, Shape.of(3, 2), new long[] {5, 2}, 1, false);
                byte[] leftStorage = transpose ? numeric(type, 1, 4, 2, 5, 3, 6)
                        : numeric(type, 90, 1, 91, 2, 92, 3, 93, 94, 95, 4, 96, 5, 97, 6);
                byte[] rightStorage = numeric(type, 80, 7, 81, 8, 82, 83, 9, 84, 10, 85, 86, 11, 87, 12);
                byte[] actual = prepared(List.of(left.matmul(right)),
                        Map.of(left.id(), leftStorage, right.id(), rightStorage),
                        Optional.empty()).getFirst();
                assertArrayEquals(numeric(type, 58, 64, 139, 154), actual, type + " transpose=" + transpose);
            }
        }
    }

    @Test
    void affineExpReciprocalAndSigmoidReadOriginalCarriersAndPermutedCoordinates() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
            Tensor input = external(type, Shape.of(2, 2), new long[] {5, 2}, 1, false);
            for (boolean permuted : List.of(false, true)) {
                Tensor value = permuted ? input.permute(1, 0) : input;
                var actual = prepared(List.of(value.exp(), value.reciprocal(), value.sigmoid()),
                        Map.of(input.id(), numeric(type, 99, Float.NEGATIVE_INFINITY, 98, 0,
                                97, 96, Float.POSITIVE_INFINITY, 95, Float.NEGATIVE_INFINITY)),
                        Optional.empty());
                assertArrayEquals(permuted ? numeric(type, 0, Float.POSITIVE_INFINITY, 1, 0)
                                : numeric(type, 0, 1, Float.POSITIVE_INFINITY, 0),
                        actual.get(0), type + " EXP permuted=" + permuted);
                assertArrayEquals(permuted ? numeric(type, -0.0f, 0.0f, Float.POSITIVE_INFINITY, -0.0f)
                                : numeric(type, -0.0f, Float.POSITIVE_INFINITY, 0.0f, -0.0f),
                        actual.get(1), type + " RECIPROCAL permuted=" + permuted);
                assertArrayEquals(permuted ? numeric(type, 0, 1, .5f, 0) : numeric(type, 0, .5f, 1, 0),
                        actual.get(2), type + " SIGMOID permuted=" + permuted);
            }
        }
    }

    @Test
    void publicExternalAffineComparisonsReadLogicalElementsForEveryNumericCarrier() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
            for (boolean gradient : List.of(false, true)) {
                Tensor left = external(type, Shape.of(2), new long[] {2}, 1, gradient);
                Tensor right = external(type, Shape.of(2), new long[] {1}, 0, false);
                var actual = prepared(List.of(left.greaterThan(right), left.equalTo(right),
                                left.greaterOrEqual(right), left.lessThan(right),
                                left.lessOrEqual(right), left.notEqualTo(right)),
                        Map.of(left.id(), numeric(type, 99, -2, 98, 3),
                                right.id(), numeric(type, -1, 3)), Optional.empty());
                assertArrayEquals(new byte[] {0, 0}, actual.get(0), type + " greater");
                assertArrayEquals(new byte[] {0, 1}, actual.get(1), type + " equal");
                assertArrayEquals(new byte[] {0, 1}, actual.get(2), type + " greater-or-equal");
                assertArrayEquals(new byte[] {1, 0}, actual.get(3), type + " less");
                assertArrayEquals(new byte[] {1, 1}, actual.get(4), type + " less-or-equal");
                assertArrayEquals(new byte[] {1, 0}, actual.get(5), type + " not-equal");
            }
        }
    }

    @Test
    void publicExternalBroadcastFloat64FloorRetainsExactSubnormalAndNegativeZero() {
        for (long word : new long[] {0x8000000000000000L, 0x8000000000000001L,
                0x3ff0000000000001L}) {
            Tensor input = external(DataType.FLOAT64, Shape.of(2, 3), new long[] {0, 0}, 1, false);
            ByteBuffer storage = ByteBuffer.allocate(16).order(ByteOrder.nativeOrder());
            storage.putLong(0x7ff8000000004321L).putLong(word);
            byte[] actual = prepared(List.of(input.floor()),
                    Map.of(input.id(), storage.array()), Optional.empty()).getFirst();
            ByteBuffer expected = ByteBuffer.allocate(48).order(ByteOrder.nativeOrder());
            long result = Double.doubleToRawLongBits(Math.floor(Double.longBitsToDouble(word)));
            for (int i = 0; i < 6; i++) expected.putLong(result);
            assertArrayEquals(expected.array(), actual);
        }
    }

    @Test
    void publicPermutedAffineNegPublishesLogicalValuesAndActualGradient() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
            Tensor input = external(type, Shape.of(2, 2), new long[] {5, 2}, 1, true);
            Tensor root = input.permute(1, 0).neg();
            Tensor seed = external(type, Shape.of(2, 2), new long[] {2, 1}, 0, false);
            var request = new FunctionalGradientRequest(List.of(new FunctionalGradientRequest.Stage(
                    List.of(new FunctionalGradientRequest.ForwardTensorReference(root)),
                    List.of(Optional.of(seed)), List.of(input), false,
                    FunctionalGradientRequest.DisconnectedPolicy.ERROR)));
            var actual = prepared(List.of(root), Map.of(
                    input.id(), numeric(type, 90, -0.0f, 91, 2, 92, 93, -3, 94, 4),
                    seed.id(), numeric(type, 1, 2, 3, 4)), Optional.of(request));
            assertArrayEquals(numeric(type, 0.0f, 3, -2, -4), actual.get(0), type + " primal");
            assertArrayEquals(numeric(type, -1, -3, -2, -4), actual.get(1), type + " adjoint");
        }
    }

    private static Tensor external(DataType type, Shape shape, long[] strides, long offset,
            boolean gradient) {
        return TensorFactory.create(new TensorDescriptor(type, shape,
                Optional.of(LayoutDescriptor.of(shape, strides, offset, true)), gradient));
    }

    private static byte[] numeric(DataType type, float... values) {
        ByteBuffer bytes = ByteBuffer.allocate(values.length * type.byteWidth()).order(ByteOrder.nativeOrder());
        for (float value : values) {
            if (type == DataType.FLOAT32) bytes.putInt(Float.floatToRawIntBits(value));
            else if (type == DataType.BFLOAT16) bytes.putShort(
                    io.github.pho001.synaptik.model.datatype.ScalarValue.bfloat16(value).bfloat16Bits());
            else bytes.putShort(Float.floatToFloat16(value));
        }
        return bytes.array();
    }

    private static List<byte[]> prepared(List<Tensor> roots, Map<TensorId, byte[]> inputs,
            Optional<FunctionalGradientRequest> gradients) {
        String library = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(library != null && !library.isBlank(), "native device library not configured");
        var availability = new BackendAvailabilitySnapshot(MetalCapabilityProvider.METAL_BACKEND_ID,
                Map.of(new BackendDeviceId(MetalCapabilityProvider.METAL_BACKEND_ID, "default"),
                        DeviceClass.ACCELERATOR));
        var artifacts = GraphCompilationPort.compile(gradients.isPresent()
                        ? CompileMode.FORWARD_AND_BACKWARD : CompileMode.FORWARD_ONLY,
                roots, gradients, GraphOptimizationConfig.disabled(), BackendIntent.unconstrained(),
                PartitionScoringConfig.neutral(), List.of(new MetalCapabilityProvider()), List.of(availability));
        assertEquals(1, artifacts.partitions().size(), "public graph must remain Metal-only");
        List<ValueId> publications = new ArrayList<>();
        artifacts.publication().forwardBindings().forEach(binding -> publications.add(binding.valueId()));
        artifacts.publication().gradientBindings().forEach(binding -> publications.add(binding.valueId()));
        try (MetalDeviceContext context = MetalDeviceContext.open(Path.of(library));
                Arena arena = Arena.ofConfined()) {
            var analyzed = new AtomicReference<MetalNegPreparationPlan>();
            var preparer = new MetalNegPartitionPreparer();
            var preparation = new PartitionPreparation<>(new MetalNegAnalysisInputs(context), c -> {
                var result = preparer.analyze(c);
                analyzed.set(result.plan());
                return result;
            }, new MetalNegPartitionFinalizer(context));
            Map<TensorId, MetalBufferRepresentation> buffers = new HashMap<>();
            try (var execution = GraphPreparation.prepare(artifacts, List.of(preparation), c ->
                    new MetalNegPreparedScheduleAssembler(context, analyzed.get(), publications).assemble(c))) {
                for (var binding : artifacts.constants().bindableInputBindings()) {
                    byte[] bytes = java.util.Objects.requireNonNull(inputs.get(binding.tensorId()));
                    var buffer = context.createBuffer(bytes.length);
                    buffers.put(binding.tensorId(), buffer);
                    if (bytes.length != 0) {
                        var host = arena.allocateFrom(JAVA_BYTE, bytes);
                        buffer.upload(0, host, 0, bytes.length);
                    }
                }
                var bound = artifacts.constants().bindableInputBindings().stream()
                        .<BufferRepresentation>map(binding -> buffers.get(binding.tensorId())).toList();
                try (var result = new PreparedExecutionRunner().run(execution, bound)) {
                    var actual = new ArrayList<byte[]>();
                    for (int i = 0; i < publications.size(); i++) {
                        var buffer = (MetalBufferRepresentation) result.publicationRepresentation(i);
                        var host = arena.allocate(buffer.byteSize(), 8);
                        if (buffer.byteSize() != 0) buffer.download(0, host, 0, host.byteSize());
                        actual.add(host.toArray(JAVA_BYTE));
                    }
                    for (var entry : buffers.entrySet()) {
                        byte[] original = inputs.get(entry.getKey());
                        var host = arena.allocate(original.length, 8);
                        if (original.length != 0) entry.getValue().download(0, host, 0, original.length);
                        assertArrayEquals(original, host.toArray(JAVA_BYTE), "external storage remains unchanged");
                    }
                    return List.copyOf(actual);
                }
            } finally {
                for (var buffer : buffers.values()) buffer.close();
            }
        }
    }

    @Test
    void rankSixteenFalseWhereBranchRetainsLastAxisAndBroadcastsExactly() {
        try (NativeScope scope = new NativeScope()) {
            for (DataType type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
                for (boolean broadcast : List.of(false, true)) {
                    long[] output = new long[16];
                    Arrays.fill(output, 1);
                    output[14] = 2;
                    output[15] = 3;
                    long[] falseShape = output.clone();
                    if (broadcast) falseShape[15] = 1;
                    var values = List.of(value(DataType.BOOL, 3), value(type, 2, broadcast ? 3 : 1),
                            value(type, falseShape), value(type, output));
                    var node = MetalMpsGraphProgram.Node.generic(
                            MetalMpsGraphProgram.NodeKind.WHERE, new int[] {0, 1, 2},
                            new int[] {3}, MetalMpsGraphProgram.AttributeKind.NONE, new long[0]);
                    int nan = type == DataType.FLOAT32 ? 0xff812345
                            : type == DataType.BFLOAT16 ? 0xff81 : 0xfc01;
                    int negativeZero = type == DataType.FLOAT32 ? 0x80000000 : 0x8000;
                    byte[] whenFalse = words(type, broadcast
                            ? new int[] {nan, 1} : new int[] {nan, 1, negativeZero, 2, 3, 4});
                    byte[] expected = words(type, broadcast
                            ? new int[] {nan, negativeZero, nan, 1, 0, 1}
                            : new int[] {nan, negativeZero, negativeZero, 2, 0, 4});
                    assertArrayEquals(expected, scope.execute(values, node,
                            List.of(new byte[] {0, 1, 0}, words(type, broadcast
                                    ? new int[] {negativeZero, negativeZero, negativeZero, 0, 0, 0}
                                    : new int[] {negativeZero, 0}), whenFalse)),
                            type + " broadcast=" + broadcast);
                }
            }
        }
    }

    @Test
    void lowScatterCopiesEveryUntouchedEncodingForBothIndexWidths() {
        try (NativeScope scope = new NativeScope()) {
            for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
                int one = type == DataType.BFLOAT16 ? 0x3f80 : 0x3c00;
                int three = type == DataType.BFLOAT16 ? 0x4040 : 0x4200;
                int[] base = type == DataType.BFLOAT16
                        ? new int[] {one, 0x7f81, 0xff81, 0x7fc5, 0xffc7, 0, 0x8000, 1, 0x8001}
                        : new int[] {one, 0x7c01, 0xfc01, 0x7e05, 0xfe07, 0, 0x8000, 1, 0x8001};
                int[] expected = base.clone();
                expected[0] = three;
                for (DataType indexType : List.of(DataType.INT32, DataType.INT64)) {
                    var values = List.of(value(type, base.length), value(indexType, 2),
                            value(type, 2), value(type, base.length));
                    byte[] indices = new byte[Math.toIntExact(2 * indexType.byteWidth())];
                    assertArrayEquals(words(type, expected), scope.execute(values,
                            MetalMpsGraphProgram.Node.scatterAdd(0, 1, 2, 3, 0),
                            List.of(words(type, base), indices, words(type, one, one))),
                            type + "/" + indexType);
                }
            }
        }
    }

    @Test
    void lowScansSeedFirstContributorAndKeepExclusiveIdentitiesInBothDirections() {
        try (NativeScope scope = new NativeScope()) {
            for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
                int one = type == DataType.BFLOAT16 ? 0x3f80 : 0x3c00;
                int two = 0x4000;
                int three = type == DataType.BFLOAT16 ? 0x4040 : 0x4200;
                for (boolean reverse : List.of(false, true)) {
                    for (boolean exclusive : List.of(false, true)) {
                        for (var kind : List.of(MetalMpsGraphProgram.NodeKind.CUM_SUM,
                                MetalMpsGraphProgram.NodeKind.CUM_PROD)) {
                            boolean product = kind == MetalMpsGraphProgram.NodeKind.CUM_PROD;
                            // Two independent lines: all-negative zero and ordinary finite terms.
                            int[] input = {0x8000, 0x8000, 0x8000, 0x8000, one, two};
                            int[] expected = product
                                    ? new int[] {0x8000, 0, 0x8000, 0x8000, 0x8000, 0x8000}
                                    : new int[] {0x8000, 0x8000, 0x8000, 0x8000, one, three};
                            if (exclusive) {
                                expected = product
                                        ? new int[] {one, 0x8000, 0, one, 0x8000, 0x8000}
                                        : new int[] {0, 0x8000, 0x8000, 0, 0x8000, one};
                            }
                            if (reverse) {
                                reverseLines(input);
                                reverseLines(expected);
                            }
                            assertArrayEquals(words(type, expected), scope.execute(
                                    List.of(value(type, 2, 3), value(type, 2, 3)),
                                    MetalMpsGraphProgram.Node.scan(kind, 0, 1, 1, exclusive, reverse),
                                    List.of(words(type, input))),
                                    type + "/" + kind + "/exclusive=" + exclusive + "/reverse=" + reverse);
                        }
                    }
                }
            }
        }
    }

    @Test
    void allThreeNativeRoutesRejectForeignBuffersBeforeMutationAndSurviveOwnerRelease() {
        try (NativeScope scope = new NativeScope()) {
            // Both contexts use MTLCreateSystemDefaultDevice; device equality is not ownership.
            var foreignContext = scope.context();
            var localInput = scope.buffer(scope.owner, words(DataType.FLOAT32, 0x3f800000));
            var localOutput = scope.buffer(scope.owner, words(DataType.FLOAT32, 0x5a5a5a5a));
            var foreignInput = scope.buffer(foreignContext, words(DataType.FLOAT32, 0x40000000));
            var foreignOutput = scope.buffer(foreignContext, words(DataType.FLOAT32, 0x6a6a6a6a));
            for (MetalPreparedRoute route : MetalPreparedRoute.values()) {
                var executable = scope.executable(route, List.of(value(DataType.FLOAT32, 1),
                        value(DataType.FLOAT32, 1)), route == MetalPreparedRoute.CUSTOM_PROGRAM
                                ? MetalMpsGraphProgram.Node.generic(MetalMpsGraphProgram.NodeKind.FLOOR,
                                        new int[] {0}, new int[] {1},
                                        MetalMpsGraphProgram.AttributeKind.NONE, new long[0])
                                : MetalMpsGraphProgram.Node.neg(0, 1));
                for (int foreignRole = 1; foreignRole <= 3; foreignRole++) {
                    var input = (foreignRole & 1) != 0 ? foreignInput : localInput;
                    var output = (foreignRole & 2) != 0 ? foreignOutput : localOutput;
                    var failure = assertThrows(MetalNativeApi.NativeFailure.class,
                            () -> scope.run(route, executable, List.of(input, output)));
                    assertEquals(MetalNativeApi.Status.INCOMPATIBLE_RESOURCE, failure.status());
                    assertArrayEquals(words(DataType.FLOAT32, 0x5a5a5a5a), scope.download(localOutput, 4));
                    assertArrayEquals(words(DataType.FLOAT32, 0x6a6a6a6a), scope.download(foreignOutput, 4));
                    assertArrayEquals(words(DataType.FLOAT32, 0x3f800000), scope.download(localInput, 4));
                    assertArrayEquals(words(DataType.FLOAT32, 0x40000000), scope.download(foreignInput, 4));
                }
                scope.run(route, executable, List.of(localInput, localOutput));
                assertArrayEquals(words(DataType.FLOAT32,
                        route == MetalPreparedRoute.CUSTOM_PROGRAM ? 0x3f800000 : 0xbf800000),
                        scope.download(localOutput, 4));
                scope.upload(localOutput, words(DataType.FLOAT32, 0x5a5a5a5a));
            }
            // Buffers keep their identity alive after their context handle is consumed, without
            // owning its queue. New contexts must never accept those surviving foreign buffers.
            scope.releaseContext(foreignContext);
            for (int iteration = 0; iteration < 16; iteration++) {
                var replacement = scope.context();
                var output = scope.buffer(replacement, words(DataType.FLOAT32, 0x5a5a5a5a));
                var pipeline = scope.api.createNegKernelPipeline(replacement, 1);
                try {
                    var failure = assertThrows(MetalNativeApi.NativeFailure.class,
                            () -> scope.api.runNegKernelPipeline(pipeline, foreignInput, output));
                    assertEquals(MetalNativeApi.Status.INCOMPATIBLE_RESOURCE, failure.status());
                    assertArrayEquals(words(DataType.FLOAT32, 0x5a5a5a5a), scope.download(output, 4));
                } finally {
                    scope.api.releaseNegKernelPipeline(pipeline);
                    scope.releaseContext(replacement);
                }
            }
            scope.releaseContext(scope.owner);
            // Existing executables already retain their context. Consuming its external handle
            // must not invalidate those resources, and release order remains caller-owned.
            for (int index = 0; index < scope.executables.size(); index++) {
                scope.run(scope.routes.get(index), scope.executables.get(index), List.of(localInput, localOutput));
                assertArrayEquals(words(DataType.FLOAT32,
                        scope.routes.get(index) == MetalPreparedRoute.CUSTOM_PROGRAM ? 0x3f800000 : 0xbf800000),
                        scope.download(localOutput, 4));
            }
        }
    }

    private static void reverseLines(int[] words) {
        for (int start = 0; start < words.length; start += 3) {
            int first = words[start];
            words[start] = words[start + 2];
            words[start + 2] = first;
        }
    }

    private static MetalMpsGraphProgram.ValueDescriptor value(DataType type, long... shape) {
        return new MetalMpsGraphProgram.ValueDescriptor(type, shape, false);
    }

    private static byte[] words(DataType type, int... words) {
        ByteBuffer bytes = ByteBuffer.allocate(Math.toIntExact(words.length * type.byteWidth()))
                .order(ByteOrder.nativeOrder());
        for (int word : words) {
            if (type == DataType.FLOAT32) bytes.putInt(word);
            else bytes.putShort((short) word);
        }
        return bytes.array();
    }

    /** Owns raw handles only; intentionally bypasses Java context-bound representation checks. */
    private static final class NativeScope implements AutoCloseable {
        private final MetalNativeApi api;
        private final List<MetalNativeApi.Handle> contexts = new ArrayList<>();
        private final List<MetalNativeApi.Handle> buffers = new ArrayList<>();
        private final List<MetalNativeApi.Handle> executables = new ArrayList<>();
        private final List<MetalPreparedRoute> routes = new ArrayList<>();
        private final MetalNativeApi.Handle owner;

        private NativeScope() {
            String library = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
            assumeTrue(library != null && !library.isBlank(), "native device library not configured");
            api = MetalNativeApi.open(Path.of(library));
            try {
                owner = context();
            } catch (RuntimeException | Error failure) {
                api.close();
                throw failure;
            }
        }

        private MetalNativeApi.Handle context() {
            var context = api.createContext();
            contexts.add(context);
            return context;
        }

        private void releaseContext(MetalNativeApi.Handle context) {
            api.releaseContext(context);
            contexts.remove(context);
        }

        private MetalNativeApi.Handle buffer(MetalNativeApi.Handle context, byte[] bytes) {
            var buffer = api.createBuffer(context, bytes.length);
            buffers.add(buffer);
            upload(buffer, bytes);
            return buffer;
        }

        private void upload(MetalNativeApi.Handle buffer, byte[] bytes) {
            try (Arena arena = Arena.ofConfined()) {
                var source = arena.allocate(bytes.length, 1);
                source.copyFrom(MemorySegment.ofArray(bytes));
                api.upload(buffer, 0, source, bytes.length);
            }
        }

        private byte[] download(MetalNativeApi.Handle buffer, int count) {
            try (Arena arena = Arena.ofConfined()) {
                var target = arena.allocate(count, 1);
                api.download(buffer, 0, target, count);
                return target.toArray(JAVA_BYTE);
            }
        }

        private MetalNativeApi.Handle executable(MetalPreparedRoute route,
                List<MetalMpsGraphProgram.ValueDescriptor> values, MetalMpsGraphProgram.Node node) {
            int[] feeds = new int[values.size() - 1];
            for (int index = 0; index < feeds.length; index++) feeds[index] = index;
            var executable = route == MetalPreparedRoute.CUSTOM_SINGLE_NEG
                    ? api.createNegKernelPipeline(owner, 1)
                    : api.createProgramExecutable(owner, values, new MetalMpsGraphProgram(List.of(node)),
                            feeds, new int[] {values.size() - 1}, route);
            executables.add(executable);
            routes.add(route);
            return executable;
        }

        private byte[] execute(List<MetalMpsGraphProgram.ValueDescriptor> values,
                MetalMpsGraphProgram.Node node, List<byte[]> inputs) {
            var route = MetalPreparedRoute.CUSTOM_PROGRAM;
            var executable = executable(route, values, node);
            var table = new ArrayList<MetalNativeApi.Handle>();
            for (int index = 0; index < values.size(); index++) {
                table.add(buffer(owner, index < inputs.size() ? inputs.get(index)
                        : new byte[Math.toIntExact(values.get(index).byteCount())]));
            }
            run(route, executable, table);
            for (int index = 0; index < inputs.size(); index++) {
                assertArrayEquals(inputs.get(index), download(table.get(index), inputs.get(index).length));
            }
            return download(table.getLast(), Math.toIntExact(values.getLast().byteCount()));
        }

        private void run(MetalPreparedRoute route, MetalNativeApi.Handle executable,
                List<MetalNativeApi.Handle> table) {
            if (route == MetalPreparedRoute.CUSTOM_SINGLE_NEG) {
                api.runNegKernelPipeline(executable, table.getFirst(), table.getLast());
                return;
            }
            try (Arena arena = Arena.ofConfined()) {
                int count = route == MetalPreparedRoute.CUSTOM_PROGRAM ? table.size() : table.size() - 1;
                var inputs = arena.allocate(ADDRESS, count);
                for (int index = 0; index < count; index++) {
                    inputs.setAtIndex(ADDRESS, index, table.get(index).carrier());
                }
                var outputs = arena.allocate(ADDRESS);
                outputs.set(ADDRESS, 0, table.getLast().carrier());
                api.runExecutable(executable, count, inputs, 1, outputs);
            }
        }

        @Override
        public void close() {
            for (int index = executables.size(); index-- > 0;) {
                if (routes.get(index) == MetalPreparedRoute.CUSTOM_SINGLE_NEG) {
                    api.releaseNegKernelPipeline(executables.get(index));
                } else api.releaseExecutable(executables.get(index));
            }
            for (var buffer : buffers.reversed()) api.releaseBuffer(buffer);
            for (var context : contexts.reversed()) api.releaseContext(context);
            api.close();
        }
    }
}
