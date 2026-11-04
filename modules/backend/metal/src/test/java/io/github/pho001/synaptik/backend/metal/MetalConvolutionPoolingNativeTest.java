package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.backend.contract.BackendDeviceId;
import io.github.pho001.synaptik.backend.contract.DeviceClass;
import io.github.pho001.synaptik.compiler.GraphCompilationPort;
import io.github.pho001.synaptik.config.compile.BackendIntent;
import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.tensor.Scan;
import io.github.pho001.synaptik.model.tensor.ScanBody;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorId;
import io.github.pho001.synaptik.prepare.GraphPreparation;
import io.github.pho001.synaptik.prepare.PartitionPreparation;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.run.PreparedExecutionRunner;
import java.util.HashMap;
import java.util.Map;
import java.util.OptionalInt;
import java.util.concurrent.atomic.AtomicReference;
import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.Float16Bits;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class MetalConvolutionPoolingNativeTest {
    /**
     * Executes the shared convolution body under length-aware Scan control in both directions.
     * Inactive sequence cells contain NaN and must neither contribute nor publish a bias value.
     */
    @Test
    void scanConvolutionAddsBiasAfterContractionAndSkipsInactiveCells() {
        String library = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(library != null && !library.isBlank(), "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        for (boolean threeDimensional : List.of(false, true)) {
            for (boolean reverse : List.of(false, true)) {
                long[] cellShape = threeDimensional
                        ? new long[] {2, 1, 1, 1, 2} : new long[] {2, 1, 1, 2};
                long[] weightShape = threeDimensional
                        ? new long[] {1, 1, 1, 1, 2} : new long[] {1, 1, 1, 2};
                long[] sequenceShape = threeDimensional
                        ? new long[] {2, 2, 1, 1, 1, 2} : new long[] {2, 2, 1, 1, 2};
                Tensor cell = scanTensor(DataType.FLOAT32, cellShape);
                Tensor weight = scanTensor(DataType.FLOAT32, weightShape);
                Tensor bias = scanTensor(DataType.FLOAT32, 1);
                Tensor convolution = threeDimensional
                        ? cell.conv3d(weight, bias,
                                new io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs(
                                        1, 1, 1, 0, 0, 0, 1, 1, 1, 1))
                        : cell.conv2d(weight, bias,
                                new io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs(
                                        1, 1, 0, 0, 1, 1, 1));
                Tensor sequence = scanTensor(DataType.FLOAT32, sequenceShape);
                Tensor actualWeight = scanTensor(DataType.FLOAT32, weightShape);
                Tensor actualBias = scanTensor(DataType.FLOAT32, 1);
                Tensor lengths = scanTensor(DataType.INT64, 2);
                var body = new ScanBody(List.of(), List.of(cell), List.of(weight, bias),
                        List.of(), List.of(convolution));
                var scan = Scan.apply(List.of(), List.of(sequence), List.of(actualWeight, actualBias),
                        lengths, 0, List.of(OptionalInt.of(0)), body, reverse);
                var availability = new BackendAvailabilitySnapshot(MetalCapabilityProvider.METAL_BACKEND_ID,
                        Map.of(new BackendDeviceId(MetalCapabilityProvider.METAL_BACKEND_ID, "default"),
                                DeviceClass.ACCELERATOR));
                var artifacts = GraphCompilationPort.compile(CompileMode.FORWARD_ONLY, scan.outputs(),
                        Optional.empty(), GraphOptimizationConfig.disabled(), BackendIntent.unconstrained(),
                        PartitionScoringConfig.neutral(), List.of(new MetalCapabilityProvider()),
                        List.of(availability));
                assertEquals(1, artifacts.partitions().size());
                List<ValueId> publications = new ArrayList<>();
                artifacts.publication().forwardBindings().forEach(binding -> publications.add(binding.valueId()));
                try (MetalDeviceContext context = MetalDeviceContext.open(Path.of(library));
                        Arena arena = Arena.ofConfined()) {
                    var analyzed = new AtomicReference<MetalNegPreparationPlan>();
                    var preparer = new MetalNegPartitionPreparer();
                    var preparation = new PartitionPreparation<>(new MetalNegAnalysisInputs(context),
                            prepareContext -> {
                                var result = preparer.analyze(prepareContext);
                                analyzed.set(result.plan());
                                return result;
                            }, new MetalNegPartitionFinalizer(context));
                    try (var execution = GraphPreparation.prepare(artifacts, List.of(preparation),
                            scheduleContext -> new MetalNegPreparedScheduleAssembler(
                                    context, analyzed.get(), publications).assemble(scheduleContext))) {
                        Map<TensorId, MetalBufferRepresentation> buffers = new HashMap<>();
                        try {
                            putScanInput(buffers, sequence, floatWords(1, 1, 1, 1, 1, 1,
                                    Float.NaN, Float.NaN), context, arena);
                            putScanInput(buffers, actualWeight, floatWords(0x1.0p24f, -0x1.0p24f),
                                    context, arena);
                            putScanInput(buffers, actualBias, floatWords(1), context, arena);
                            putScanInput(buffers, lengths, ByteBuffer.allocate(16).order(ByteOrder.nativeOrder())
                                    .putLong(2).putLong(1).array(), context, arena);
                            var inputs = artifacts.constants().bindableInputBindings().stream()
                                    .<BufferRepresentation>map(binding -> buffers.get(binding.tensorId())).toList();
                            try (var result = new PreparedExecutionRunner().run(execution, inputs)) {
                                var output = (MetalBufferRepresentation) result.publicationRepresentation(0);
                                var host = arena.allocate(output.byteSize(), Float.BYTES);
                                output.download(0, host, 0, host.byteSize());
                                assertArrayEquals(floatWords(1, 1, 1, 0), host.toArray(
                                        java.lang.foreign.ValueLayout.JAVA_BYTE));
                            }
                        } finally {
                            for (var buffer : buffers.values()) buffer.close();
                        }
                    }
                }
            }
        }
    }

    /**
     * Creates a canonical non-gradient Scan input.
     *
     * @param type original storage carrier
     * @param dimensions concrete logical extents
     * @return a fresh symbolic input with contiguous layout
     */
    private static Tensor scanTensor(DataType type, long... dimensions) {
        Shape shape = Shape.of(dimensions);
        return TensorFactory.create(new TensorDescriptor(type, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false));
    }

    /**
     * Uploads one invocation input and records its owned buffer for cleanup.
     *
     * @param buffers invocation-owned buffers indexed by input identity
     * @param tensor symbolic input receiving the buffer
     * @param bytes exact represented input bytes
     * @param context owning Metal context
     * @param arena lifetime owner of the temporary host upload segment
     */
    private static void putScanInput(Map<TensorId, MetalBufferRepresentation> buffers,
            Tensor tensor, byte[] bytes, MetalDeviceContext context, Arena arena) {
        var buffer = context.createBuffer(bytes.length);
        buffers.put(tensor.id(), buffer);
        var host = arena.allocate(bytes.length, 8);
        host.copyFrom(MemorySegment.ofArray(bytes));
        buffer.upload(0, host, 0, bytes.length);
    }

    /**
     * Cancellation must finish in FLOAT32 before bias addition and the sole carrier narrowing.
     * Both ranks also exercise the absent-bias path, where no third operand may be read.
     */
    @Test
    void convolutionAddsBiasAfterCancellationForBothRanksAndFloatingCarriers() {
        for (boolean threeDimensional : List.of(false, true)) {
            for (DataType type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
                for (boolean biased : List.of(false, true)) {
                    long[] inputShape = threeDimensional
                            ? new long[] {1, 1, 1, 1, 2} : new long[] {1, 1, 1, 2};
                    long[] outputShape = threeDimensional
                            ? new long[] {1, 1, 1, 1, 1} : new long[] {1, 1, 1, 1};
                    int output = biased ? 3 : 2;
                    var convolution = node(
                            threeDimensional ? MetalMpsGraphProgram.NodeKind.CONV3D
                                    : MetalMpsGraphProgram.NodeKind.CONV2D,
                            biased ? new int[] {0, 1, 2} : new int[] {0, 1},
                            new int[] {output},
                            threeDimensional ? MetalMpsGraphProgram.AttributeKind.CONV_3D
                                    : MetalMpsGraphProgram.AttributeKind.CONV_2D,
                            threeDimensional ? new long[] {1, 1, 1, 0, 0, 0, 1, 1, 1, 1}
                                    : new long[] {1, 1, 0, 0, 1, 1, 1});
                    var values = new ArrayList<MetalMpsGraphProgram.ValueDescriptor>();
                    values.add(gradient(type, inputShape));
                    values.add(gradient(type, inputShape));
                    if (biased) values.add(gradient(type, 1));
                    values.add(gradient(type, outputShape));
                    var inputs = new ArrayList<byte[]>();
                    inputs.add(type == DataType.FLOAT32 ? floatWords(1, 1) : lowWords(type, 4096, 4096));
                    inputs.add(type == DataType.FLOAT32
                            ? floatWords(0x1.0p24f, -0x1.0p24f) : lowWords(type, 4096, -4096));
                    if (biased) inputs.add(type == DataType.FLOAT32 ? floatWords(1) : lowWords(type, 1));
                    List<byte[]> actual = execute(new MetalMpsGraphProgram(List.of(convolution)), values,
                            biased ? new int[] {0, 1, 2} : new int[] {0, 1}, new int[] {output}, inputs);
                    assertArrayEquals(type == DataType.FLOAT32 ? floatWords(biased ? 1 : 0)
                                    : lowWords(type, biased ? 1 : 0),
                            actual.getFirst(), type + " rank=" + inputShape.length + " bias=" + biased);
                }
            }
        }
    }

    /** The shared Conv2d anchor must supply the post-contraction bias value to fused ADD. */
    @Test
    void fusedConvolutionEpilogueReceivesPostContractionBias() {
        var convolution = node(MetalMpsGraphProgram.NodeKind.CONV2D,
                new int[] {0, 1, 2}, new int[] {3},
                MetalMpsGraphProgram.AttributeKind.CONV_2D, 1, 1, 0, 0, 1, 1, 1);
        var program = new MetalMpsGraphProgram(List.of(convolution,
                MetalMpsGraphProgram.Node.binary(MetalMpsGraphProgram.NodeKind.ADD, 3, 4, 5)));
        var values = List.of(typed(DataType.FLOAT32, 1, 1, 1, 2),
                typed(DataType.FLOAT32, 1, 1, 1, 2), typed(DataType.FLOAT32, 1),
                typed(DataType.FLOAT32, 1, 1, 1, 1), typed(DataType.FLOAT32, 1),
                typed(DataType.FLOAT32, 1, 1, 1, 1));
        int[] feeds = {0, 1, 2, 4};
        int[] targets = {5};
        assertEquals(MetalPointwiseFusionPlan.StepKind.ANCHOR_EPILOGUE,
                MetalPointwiseFusionPlanner.plan(program, values, feeds, targets,
                        MetalPreparedRoute.CUSTOM_PROGRAM).steps().getFirst().kind());
        assertArrayEquals(floatWords(3), execute(program, values, feeds, targets,
                List.of(floatWords(1, 1), floatWords(0x1.0p24f, -0x1.0p24f),
                        floatWords(1), floatWords(2))).getFirst());
    }

    @Test
    void groupedMixedCarrierConvolutionAndPaddingMultiplicationAreLiteral() {
        var grouped = node(
                MetalMpsGraphProgram.NodeKind.CONV2D,
                new int[] {0, 1, 2}, new int[] {3},
                MetalMpsGraphProgram.AttributeKind.CONV_2D,
                1, 1, 0, 0, 1, 1, 2);
        List<MetalMpsGraphProgram.ValueDescriptor> groupedValues = List.of(
                typed(DataType.FLOAT32, 1, 2, 2, 2),
                typed(DataType.BFLOAT16, 2, 1, 1, 1),
                typed(DataType.FLOAT32, 2),
                typed(DataType.FLOAT32, 1, 2, 2, 2));
        List<byte[]> actual = execute(
                new MetalMpsGraphProgram(List.of(grouped)), groupedValues,
                new int[] {0, 1, 2}, new int[] {3},
                List.of(
                        floatWords(1, 2, 3, 4, 5, 6, 7, 8),
                        shortWords(0x4000, 0xbf80),
                        floatWords(0.5f, 1.0f)));
        assertArrayEquals(floatWords(2.5f, 4.5f, 6.5f, 8.5f, -4, -5, -6, -7),
                actual.getFirst());
        var padded = node(
                MetalMpsGraphProgram.NodeKind.CONV2D,
                new int[] {0, 1}, new int[] {2},
                MetalMpsGraphProgram.AttributeKind.CONV_2D,
                1, 1, 1, 1, 1, 1, 1);
        int[] weights = new int[9];
        weights[0] = 0x7f800000;
        List<byte[]> paddedActual = execute(
                new MetalMpsGraphProgram(List.of(padded)),
                List.of(
                        typed(DataType.FLOAT32, 1, 1, 1, 1),
                        typed(DataType.FLOAT32, 1, 1, 3, 3),
                        typed(DataType.FLOAT32, 1, 1, 1, 1)),
                new int[] {0, 1}, new int[] {2},
                List.of(floatWords(1), intWords(weights)));
        assertTrue(Float.isNaN(scalarFloat(paddedActual.getFirst())),
                "conceptual +0 times infinity must participate in convolution arithmetic");

        var conv3d = node(
                MetalMpsGraphProgram.NodeKind.CONV3D,
                new int[] {0, 1}, new int[] {2},
                MetalMpsGraphProgram.AttributeKind.CONV_3D,
                1, 1, 1, 0, 0, 0, 1, 1, 1, 1);
        List<byte[]> conv3dActual = execute(
                new MetalMpsGraphProgram(List.of(conv3d)),
                List.of(
                        typed(DataType.FLOAT32, 1, 1, 2, 2, 2),
                        typed(DataType.FLOAT32, 1, 1, 2, 2, 2),
                        typed(DataType.FLOAT32, 1, 1, 1, 1, 1)),
                new int[] {0, 1}, new int[] {2},
                List.of(floatWords(1, 2, 3, 4, 5, 6, 7, 8),
                        floatWords(1, 1, 1, 1, 1, 1, 1, 1)));
        assertArrayEquals(floatWords(36), conv3dActual.getFirst());
    }

    @Test
    void maximumPoolingPreservesWinningBitsAndAllPaddingIdentity() {
        for (DataType type : List.of(
                DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
            long firstNaN = switch (type) {
                case FLOAT64 -> 0x7ff0_0000_0000_1234L;
                case FLOAT32 -> 0x7fa1_2345L;
                case BFLOAT16 -> 0x7f81L;
                case FLOAT16 -> 0x7c01L;
                default -> throw new AssertionError(type);
            };
            long secondNaN = switch (type) {
                case FLOAT64 -> 0xfff8_0000_0000_5678L;
                case FLOAT32 -> 0xffc5_4321L;
                case BFLOAT16 -> 0xffc2L;
                case FLOAT16 -> 0xfe02L;
                default -> throw new AssertionError(type);
            };
            var max2d = node(
                    MetalMpsGraphProgram.NodeKind.MAX_POOL2D,
                    new int[] {0}, new int[] {1},
                    MetalMpsGraphProgram.AttributeKind.WINDOW_2D,
                    2, 2, 1, 1, 0, 0, 1, 1, 0);
            List<byte[]> result = execute(
                    new MetalMpsGraphProgram(List.of(max2d)),
                    List.of(typed(type, 1, 1, 2, 2), typed(type, 1, 1, 1, 1)),
                    new int[] {0}, new int[] {1},
                    List.of(rawWords(type, firstNaN, secondNaN, 0, negativeZero(type))));
            assertArrayEquals(rawWords(type, firstNaN), result.getFirst(), type.toString());
        }

        for (DataType type : List.of(
                DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
            var numericMax = node(
                    MetalMpsGraphProgram.NodeKind.MAX_POOL2D,
                    new int[] {0}, new int[] {1},
                    MetalMpsGraphProgram.AttributeKind.WINDOW_2D,
                    1, 2, 1, 2, 0, 0, 1, 1, 0);
            List<byte[]> result = execute(
                    new MetalMpsGraphProgram(List.of(numericMax)),
                    List.of(typed(type, 1, 1, 1, 10), typed(type, 1, 1, 1, 5)),
                    new int[] {0}, new int[] {1},
                    List.of(rawWords(
                            type,
                            negativeInfinity(type), negativeOne(type),
                            positiveInfinity(type), positiveOne(type),
                            negativeZero(type), 0,
                            0, negativeZero(type),
                            negativeOne(type), positiveOne(type))));
            assertArrayEquals(
                    rawWords(
                            type,
                            negativeOne(type),
                            positiveInfinity(type),
                            0,
                            0,
                            positiveOne(type)),
                    result.getFirst(),
                    type + " production numeric comparator");
        }

        var max3d = node(
                MetalMpsGraphProgram.NodeKind.MAX_POOL3D,
                new int[] {0}, new int[] {1},
                MetalMpsGraphProgram.AttributeKind.WINDOW_3D,
                1, 1, 2, 1, 1, 2, 0, 0, 2, 1, 1, 1, 1);
        List<byte[]> allPadding = execute(
                new MetalMpsGraphProgram(List.of(max3d)),
                List.of(
                        typed(DataType.BFLOAT16, 1, 1, 1, 1, 1),
                        typed(DataType.BFLOAT16, 1, 1, 1, 1, 3)),
                new int[] {0}, new int[] {1}, List.of(shortWords(0x3f80)));
        assertArrayEquals(shortWords(0xff80, 0x3f80, 0xff80), allPadding.getFirst());
    }

    @Test
    void averagePoolingUsesFixedDivisorAndPreservesAllNegativeZeroOnly() {
        var average2d = node(
                MetalMpsGraphProgram.NodeKind.AVERAGE_POOL2D,
                new int[] {0}, new int[] {1},
                MetalMpsGraphProgram.AttributeKind.WINDOW_2D,
                1, 2, 1, 1, 0, 1, 1, 1, 0);
        List<byte[]> padded = execute(
                new MetalMpsGraphProgram(List.of(average2d)),
                List.of(
                        typed(DataType.FLOAT32, 1, 1, 1, 2),
                        typed(DataType.FLOAT32, 1, 1, 1, 3)),
                new int[] {0}, new int[] {1}, List.of(floatWords(2, 4)));
        assertArrayEquals(floatWords(1, 3, 2), padded.getFirst());

        var negativeZero = node(
                MetalMpsGraphProgram.NodeKind.AVERAGE_POOL2D,
                new int[] {0}, new int[] {1},
                MetalMpsGraphProgram.AttributeKind.WINDOW_2D,
                1, 2, 1, 1, 0, 0, 1, 1, 0);
        List<byte[]> zero = execute(
                new MetalMpsGraphProgram(List.of(negativeZero)),
                List.of(
                        typed(DataType.FLOAT32, 1, 1, 1, 2),
                        typed(DataType.FLOAT32, 1, 1, 1, 1)),
                new int[] {0}, new int[] {1}, List.of(intWords(0x80000000, 0x80000000)));
        assertArrayEquals(intWords(0x80000000), zero.getFirst());

        var average3d = node(
                MetalMpsGraphProgram.NodeKind.AVERAGE_POOL3D,
                new int[] {0}, new int[] {1},
                MetalMpsGraphProgram.AttributeKind.WINDOW_3D,
                2, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0);
        List<byte[]> infinity = execute(
                new MetalMpsGraphProgram(List.of(average3d)),
                List.of(
                        typed(DataType.FLOAT32, 1, 1, 2, 1, 1),
                        typed(DataType.FLOAT32, 1, 1, 1, 1, 1)),
                new int[] {0}, new int[] {1},
                List.of(intWords(0x7f800000, 0xff800000)));
        assertTrue(Float.isNaN(scalarFloat(infinity.getFirst())));
    }

    @Test
    void homogeneousRaw16ConvolutionAndAveragePoolUseOneFinalNarrow() {
        var convolution = node(
                MetalMpsGraphProgram.NodeKind.CONV2D,
                new int[] {0, 1, 2},
                new int[] {3},
                MetalMpsGraphProgram.AttributeKind.CONV_2D,
                1, 1, 0, 0, 1, 1, 1);
        var convolution3d = node(
                MetalMpsGraphProgram.NodeKind.CONV3D,
                new int[] {0, 1},
                new int[] {2},
                MetalMpsGraphProgram.AttributeKind.CONV_3D,
                1, 1, 1, 0, 0, 0, 1, 1, 1, 1);
        var average = node(
                MetalMpsGraphProgram.NodeKind.AVERAGE_POOL2D,
                new int[] {0},
                new int[] {1},
                MetalMpsGraphProgram.AttributeKind.WINDOW_2D,
                2, 2, 1, 1, 0, 0, 1, 1, 0);
        var average3d = node(
                MetalMpsGraphProgram.NodeKind.AVERAGE_POOL3D,
                new int[] {0},
                new int[] {1},
                MetalMpsGraphProgram.AttributeKind.WINDOW_3D,
                2, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0);
        for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
            List<byte[]> convolutionResult = execute(
                    new MetalMpsGraphProgram(List.of(convolution)),
                    List.of(
                            gradient(type, 1, 1, 2, 2),
                            gradient(type, 1, 1, 2, 2),
                            gradient(type, 1),
                            gradient(type, 1, 1, 1, 1)),
                    new int[] {0, 1, 2},
                    new int[] {3},
                    List.of(
                            lowWords(type, 1, 2, 3, 4),
                            lowWords(type, 1, 1, 1, 1),
                            lowWords(type, 0.5f)));
            assertArrayEquals(lowWords(type, 10.5f), convolutionResult.getFirst(), type.toString());

            List<byte[]> convolution3dResult = execute(
                    new MetalMpsGraphProgram(List.of(convolution3d)),
                    List.of(
                            gradient(type, 1, 1, 2, 2, 2),
                            gradient(type, 1, 1, 2, 2, 2),
                            gradient(type, 1, 1, 1, 1, 1)),
                    new int[] {0, 1},
                    new int[] {2},
                    List.of(
                            lowWords(type, 1, 2, 3, 4, 5, 6, 7, 8),
                            lowWords(type, 1, 1, 1, 1, 1, 1, 1, 1)));
            assertArrayEquals(
                    lowWords(type, 36.0f), convolution3dResult.getFirst(), type.toString());

            List<byte[]> averageResult = execute(
                    new MetalMpsGraphProgram(List.of(average)),
                    List.of(gradient(type, 1, 1, 2, 2), gradient(type, 1, 1, 1, 1)),
                    new int[] {0},
                    new int[] {1},
                    List.of(lowWords(type, 1, 2, 3, 4)));
            assertArrayEquals(lowWords(type, 2.5f), averageResult.getFirst(), type.toString());

            List<byte[]> average3dResult = execute(
                    new MetalMpsGraphProgram(List.of(average3d)),
                    List.of(gradient(type, 1, 1, 2, 1, 1), gradient(type, 1, 1, 1, 1, 1)),
                    new int[] {0},
                    new int[] {1},
                    List.of(lowWords(type, 2, 4)));
            assertArrayEquals(lowWords(type, 3.0f), average3dResult.getFirst(), type.toString());
        }
    }

    @Test
    void exactLocalSingletonHeightExpansionFeedsTwoDimensionalPoolWithoutCopy() {
        Shape viewShape = Shape.of(1, 1, 1, 4);
        var expanded = new MetalMpsGraphProgram.ValueDescriptor(
                DataType.FLOAT32,
                viewShape.toLongArray(),
                Optional.of(LayoutDescriptor.of(
                        viewShape, new long[] {4, 4, 4, 1}, 0, true)),
                false,
                true);
        var program = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.axis(
                        MetalMpsGraphProgram.NodeKind.EXPAND_DIMS, 0, 1, 2),
                node(
                        MetalMpsGraphProgram.NodeKind.MAX_POOL2D,
                        new int[] {1}, new int[] {2},
                        MetalMpsGraphProgram.AttributeKind.WINDOW_2D,
                        1, 2, 1, 1, 0, 0, 1, 1, 0)));
        List<byte[]> result = execute(
                program,
                List.of(
                        typed(DataType.FLOAT32, 1, 1, 4),
                        expanded,
                        typed(DataType.FLOAT32, 1, 1, 1, 3)),
                new int[] {0}, new int[] {2}, List.of(floatWords(1, 2, 3, 4)));
        assertArrayEquals(floatWords(2, 3, 4), result.getFirst());
    }

    private static MetalMpsGraphProgram.ValueDescriptor gradient(DataType type, long... shape) {
        return new MetalMpsGraphProgram.ValueDescriptor(type, shape, true);
    }

    private static long negativeZero(DataType type) {
        return switch (type) {
            case FLOAT64 -> 0x8000_0000_0000_0000L;
            case FLOAT32 -> 0x8000_0000L;
            case BFLOAT16, FLOAT16 -> 0x8000L;
            default -> throw new AssertionError(type);
        };
    }

    private static long positiveInfinity(DataType type) {
        return switch (type) {
            case FLOAT64 -> 0x7ff0_0000_0000_0000L;
            case FLOAT32 -> 0x7f80_0000L;
            case BFLOAT16 -> 0x7f80L;
            case FLOAT16 -> 0x7c00L;
            default -> throw new AssertionError(type);
        };
    }

    private static long negativeInfinity(DataType type) {
        return switch (type) {
            case FLOAT64 -> 0xfff0_0000_0000_0000L;
            case FLOAT32 -> 0xff80_0000L;
            case BFLOAT16 -> 0xff80L;
            case FLOAT16 -> 0xfc00L;
            default -> throw new AssertionError(type);
        };
    }

    private static long positiveOne(DataType type) {
        return switch (type) {
            case FLOAT64 -> 0x3ff0_0000_0000_0000L;
            case FLOAT32 -> 0x3f80_0000L;
            case BFLOAT16 -> 0x3f80L;
            case FLOAT16 -> 0x3c00L;
            default -> throw new AssertionError(type);
        };
    }

    private static long negativeOne(DataType type) {
        return switch (type) {
            case FLOAT64 -> 0xbff0_0000_0000_0000L;
            case FLOAT32 -> 0xbf80_0000L;
            case BFLOAT16 -> 0xbf80L;
            case FLOAT16 -> 0xbc00L;
            default -> throw new AssertionError(type);
        };
    }

    private static MetalMpsGraphProgram.Node node(
            MetalMpsGraphProgram.NodeKind kind,
            int[] inputs,
            int[] outputs,
            MetalMpsGraphProgram.AttributeKind attributeKind,
            long... words) {
        return MetalMpsGraphProgram.Node.generic(kind, inputs, outputs, attributeKind, words);
    }

    private static MetalMpsGraphProgram.ValueDescriptor typed(
            DataType type, long... dimensions) {
        return new MetalMpsGraphProgram.ValueDescriptor(type, dimensions, false);
    }

    private static List<byte[]> execute(
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            int[] feeds,
            int[] targets,
            List<byte[]> inputs) {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        assertEquals(feeds.length, inputs.size());
        int[] materialized = MetalPointwiseFusionPlanner.plan(program, values, feeds, targets,
                MetalPreparedRoute.CUSTOM_PROGRAM).materializedProgramValueIndices();
        Path library = Path.of(configured).toAbsolutePath().normalize();
        MetalNativeApi api = MetalNativeApi.open(library);
        MetalNativeApi.Handle context = null;
        MetalNativeApi.Handle executable = null;
        var buffers = new ArrayList<MetalNativeApi.Handle>();
        try {
            context = api.createContext();
            executable = api.createProgramExecutable(context, values, program, feeds, targets,
            MetalPreparedRoute.CUSTOM_PROGRAM);
            for (var value : values) buffers.add(api.createBuffer(context, value.byteCount()));
            for (int index = 0; index < feeds.length; index++) {
                upload(api, buffers.get(feeds[index]), inputs.get(index));
            }
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment addresses = arena.allocate(ADDRESS, materialized.length);
                for (int index = 0; index < materialized.length; index++) {
                    addresses.setAtIndex(ADDRESS, index, buffers.get(materialized[index]).carrier());
                }
                MemorySegment outputs = arena.allocate(ADDRESS, targets.length);
                for (int index = 0; index < targets.length; index++) {
                    outputs.setAtIndex(ADDRESS, index, buffers.get(targets[index]).carrier());
                }
                api.runExecutable(executable, materialized.length, addresses, targets.length, outputs);
                var result = new ArrayList<byte[]>(targets.length);
                for (int target : targets) {
                    result.add(download(api, buffers.get(target), values.get(target).byteCount()));
                }
                return List.copyOf(result);
            }
        } finally {
            for (int index = buffers.size(); index-- > 0;) api.releaseBuffer(buffers.get(index));
            if (executable != null) api.releaseExecutable(executable);
            if (context != null) api.releaseContext(context);
            api.close();
        }
    }

    private static void upload(MetalNativeApi api, MetalNativeApi.Handle buffer, byte[] bytes) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment source = arena.allocate(bytes.length, 1);
            source.copyFrom(MemorySegment.ofArray(bytes));
            api.upload(buffer, 0L, source, bytes.length);
        }
    }

    private static byte[] download(
            MetalNativeApi api, MetalNativeApi.Handle buffer, long byteCount) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment target = arena.allocate(byteCount, 1);
            api.download(buffer, 0L, target, byteCount);
            return target.toArray(java.lang.foreign.ValueLayout.JAVA_BYTE);
        }
    }

    private static byte[] rawWords(DataType type, long... words) {
        ByteBuffer bytes = ByteBuffer.allocate(Math.multiplyExact(words.length, type.byteWidth()))
                .order(ByteOrder.nativeOrder());
        for (long word : words) {
            switch (type) {
                case FLOAT64 -> bytes.putLong(word);
                case FLOAT32 -> bytes.putInt((int) word);
                case BFLOAT16, FLOAT16 -> bytes.putShort((short) word);
                default -> throw new AssertionError(type);
            }
        }
        return bytes.array();
    }

    private static byte[] floatWords(float... words) {
        ByteBuffer bytes = ByteBuffer.allocate(words.length * Float.BYTES)
                .order(ByteOrder.nativeOrder());
        for (float word : words) bytes.putFloat(word);
        return bytes.array();
    }

    private static byte[] intWords(int... words) {
        ByteBuffer bytes = ByteBuffer.allocate(words.length * Integer.BYTES)
                .order(ByteOrder.nativeOrder());
        for (int word : words) bytes.putInt(word);
        return bytes.array();
    }

    private static byte[] shortWords(int... words) {
        ByteBuffer bytes = ByteBuffer.allocate(words.length * Short.BYTES)
                .order(ByteOrder.nativeOrder());
        for (int word : words) bytes.putShort((short) word);
        return bytes.array();
    }

    private static byte[] lowWords(DataType type, float... words) {
        ByteBuffer bytes = ByteBuffer.allocate(words.length * Short.BYTES)
                .order(ByteOrder.nativeOrder());
        for (float word : words) {
            short raw = switch (type) {
                case BFLOAT16 -> BFloat16Bits.fromFloat(word);
                case FLOAT16 -> Float16Bits.fromFloat(word);
                default -> throw new AssertionError(type);
            };
            bytes.putShort(raw);
        }
        return bytes.array();
    }

    private static float scalarFloat(byte[] bytes) {
        return ByteBuffer.wrap(bytes).order(ByteOrder.nativeOrder()).getFloat();
    }
}
