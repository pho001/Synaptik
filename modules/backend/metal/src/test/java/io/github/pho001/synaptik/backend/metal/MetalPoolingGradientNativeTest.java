package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.backend.contract.BackendDeviceId;
import io.github.pho001.synaptik.backend.contract.DeviceClass;
import io.github.pho001.synaptik.compiler.FunctionalGradientRequest;
import io.github.pho001.synaptik.compiler.GraphCompilationPort;
import io.github.pho001.synaptik.config.compile.BackendIntent;
import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool3dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool3dAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.prepare.GraphPreparation;
import io.github.pho001.synaptik.prepare.PartitionPreparation;
import io.github.pho001.synaptik.runtime.run.PreparedExecutionRunner;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.UnaryOperator;
import org.junit.jupiter.api.Test;

class MetalPoolingGradientNativeTest {
    @Test
    void contractionAndGatherCotangentsStayOnMetalForEveryWorkingDomainCarrier() {
        assertPool(Shape.of(2, 2), x -> x.matmul(x),
                new float[] {1, 2, 3, 4}, new float[] {7, 10, 15, 22},
                new float[] {7, 11, 9, 13});
        assertPool(Shape.of(1, 1, 1, 2),
                x -> x.conv2d(x, new io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs(
                        1, 1, 0, 0, 1, 1, 1)),
                new float[] {1, 2}, new float[] {5}, new float[] {2, 4});
        assertPool(Shape.of(1, 1, 1, 1, 2),
                x -> x.conv3d(x, new io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs(
                        1, 1, 1, 0, 0, 0, 1, 1, 1, 1)),
                new float[] {1, 2}, new float[] {5}, new float[] {2, 4});
        assertPool(Shape.of(2, 3), x -> x.gather(x.argMax(0, true), 0),
                new float[] {1, 5, 2, 3, 4, 6}, new float[] {3, 4, 6, 1, 5, 2, 3, 4, 6},
                new float[] {1, 1, 1, 2, 2, 2});
    }

    @Test
    void overlappingVolumePoolingUsesActualCompilerCotangentsOnMetal() {
        float[] input = new float[27];
        input[13] = 9;
        float[] maxima = new float[8];
        Arrays.fill(maxima, 9);
        float[] maximumGradient = new float[27];
        maximumGradient[13] = 8;
        assertPool(Shape.of(1, 1, 3, 3, 3),
                x -> x.maxPool3d(new MaxPool3dAttrs(2, 2, 2, 1, 1, 1,
                        0, 0, 0, 1, 1, 1, false)),
                input, maxima, maximumGradient);

        Arrays.fill(input, 8);
        float[] averages = new float[8];
        Arrays.fill(averages, 8);
        float[] averageGradient = new float[27];
        for (int d = 0; d < 3; d++) {
            for (int h = 0; h < 3; h++) {
                for (int w = 0; w < 3; w++) {
                    averageGradient[(d * 3 + h) * 3 + w] =
                            (d == 1 ? 2 : 1) * (h == 1 ? 2 : 1) * (w == 1 ? 2 : 1) / 8.0f;
                }
            }
        }
        assertPool(Shape.of(1, 1, 3, 3, 3),
                x -> x.averagePool3d(new AveragePool3dAttrs(2, 2, 2, 1, 1, 1,
                        0, 0, 0, 1, 1, 1, false)),
                input, averages, averageGradient);
    }

    @Test
    void firstWinnerNaNSignedZeroAndRealNegativeInfinityExcludePaddingInActualBackward() {
        UnaryOperator<Tensor> max = x -> x.maxPool3d(new MaxPool3dAttrs(
                1, 1, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1, false));
        assertPool(Shape.of(1, 1, 1, 1, 3), max,
                new float[] {-0.0f, 0.0f, 0.0f}, new float[] {0, 0}, new float[] {0, 2, 0});
        float firstNaN = Float.intBitsToFloat(0x7fa12345);
        float secondNaN = Float.intBitsToFloat(0xffc54321);
        assertPool(Shape.of(1, 1, 1, 1, 3), max,
                new float[] {firstNaN, secondNaN, 1},
                new float[] {firstNaN, secondNaN}, new float[] {1, 1, 0});
        assertPool(Shape.of(1, 1, 1, 1, 3), max,
                new float[] {7, 7, 7}, new float[] {7, 7}, new float[] {1, 1, 0});
        assertPool(Shape.of(1, 1, 1, 1, 1),
                x -> x.maxPool3d(new MaxPool3dAttrs(1, 1, 2, 1, 1, 2,
                        0, 0, 2, 1, 1, 1, true)),
                new float[] {Float.NEGATIVE_INFINITY},
                new float[] {Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY},
                new float[] {1});
    }

    @Test
    void averageDilationPaddingAndLiteralCeilKeepTheFullKernelDivisor() {
        assertPool(Shape.of(1, 1, 1, 1, 3),
                x -> x.averagePool3d(new AveragePool3dAttrs(1, 1, 2, 1, 1, 2,
                        0, 0, 2, 1, 1, 2, true)),
                new float[] {2, 4, 6}, new float[] {1, 4, 3}, new float[] {1, 0, 1});
        assertPool(Shape.of(1, 1, 1, 1, 1),
                x -> x.averagePool3d(new AveragePool3dAttrs(1, 1, 2, 1, 1, 2,
                        0, 0, 2, 1, 1, 1, true)),
                new float[] {-0.0f}, new float[] {0, 0, 0}, new float[] {0.5f});
        assertPool(Shape.of(1, 1, 1, 1, 4),
                x -> x.averagePool3d(new AveragePool3dAttrs(1, 1, 2, 1, 1, 3,
                        0, 0, 0, 1, 1, 1, true)),
                new float[] {2, 4, 6, 8}, new float[] {3, 4},
                new float[] {0.5f, 0.5f, 0, 0.5f});
        assertPool(Shape.of(1, 1, 1, 1, 2),
                x -> x.averagePool3d(new AveragePool3dAttrs(1, 1, 2, 1, 1, 1,
                        0, 0, 0, 1, 1, 1, false)),
                new float[] {-0.0f, -0.0f}, new float[] {-0.0f}, new float[] {0.5f, 0.5f});
    }

    @Test
    void sharedTwoDimensionalOverlapHasTheSameActualGradientSemantics() {
        assertPool(Shape.of(1, 1, 1, 3),
                x -> x.maxPool2d(new MaxPool2dAttrs(1, 2, 1, 1, 0, 0, 1, 1, false)),
                new float[] {1, 5, 2}, new float[] {5, 5}, new float[] {0, 2, 0});
        assertPool(Shape.of(1, 1, 1, 3),
                x -> x.averagePool2d(new AveragePool2dAttrs(1, 2, 1, 1, 0, 0, 1, 1, false)),
                new float[] {2, 4, 6}, new float[] {3, 5}, new float[] {0.5f, 1, 0.5f});
    }

    private static void assertPool(
            Shape shape, UnaryOperator<Tensor> pool, float[] inputValues,
            float[] expectedForward, float[] expectedGradient) {
        for (DataType type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16))
            assertPool(type, shape, pool, inputValues, expectedForward, expectedGradient);
    }

    private static void assertPool(
            DataType type, Shape shape, UnaryOperator<Tensor> pool, float[] inputValues,
            float[] expectedForward, float[] expectedGradient) {
        String library = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(library != null && !library.isBlank(), "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        Tensor input = TensorFactory.create(new TensorDescriptor(type, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), true));
        Tensor output = pool.apply(input);
        Shape seedShape = output.descriptor().shape();
        Tensor seed = TensorFactory.create(new TensorDescriptor(type, seedShape,
                Optional.of(LayoutDescriptor.contiguous(seedShape)), false));
        var request = new FunctionalGradientRequest(List.of(new FunctionalGradientRequest.Stage(
                List.of(new FunctionalGradientRequest.ForwardTensorReference(output)),
                List.of(Optional.of(seed)), List.of(input), false,
                FunctionalGradientRequest.DisconnectedPolicy.ERROR)));
        var availability = new BackendAvailabilitySnapshot(MetalCapabilityProvider.METAL_BACKEND_ID,
                Map.of(new BackendDeviceId(MetalCapabilityProvider.METAL_BACKEND_ID, "default"),
                        DeviceClass.ACCELERATOR));
        var artifacts = GraphCompilationPort.compile(CompileMode.FORWARD_AND_BACKWARD,
                List.of(output), Optional.of(request), GraphOptimizationConfig.disabled(),
                BackendIntent.unconstrained(), PartitionScoringConfig.neutral(),
                List.of(new MetalCapabilityProvider()), List.of(availability));
        assertEquals(1, artifacts.partitions().size(), "the full primal/adjoint closure stays on Metal");
        assertEquals(2, artifacts.constants().bindableInputs().size());
        List<ValueId> publications = new ArrayList<>();
        artifacts.publication().forwardBindings().forEach(binding -> publications.add(binding.valueId()));
        artifacts.publication().gradientBindings().forEach(binding -> publications.add(binding.valueId()));
        assertEquals(2, publications.size());
        try (MetalDeviceContext context = MetalDeviceContext.open(Path.of(library));
                Arena arena = Arena.ofConfined();
                MetalBufferRepresentation inputBuffer = context.createBuffer((long) inputValues.length * type.byteWidth());
                MetalBufferRepresentation seedBuffer = context.createBuffer((long) expectedForward.length * type.byteWidth())) {
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
                var host = arena.allocateFrom(JAVA_BYTE, numericBytes(type, inputValues));
                inputBuffer.upload(0, host, 0, host.byteSize());
                float[] seedValues = new float[expectedForward.length];
                Arrays.fill(seedValues, 1);
                var seedHost = arena.allocateFrom(JAVA_BYTE, numericBytes(type, seedValues));
                seedBuffer.upload(0, seedHost, 0, seedHost.byteSize());
                try (var result = new PreparedExecutionRunner().run(execution, List.of(inputBuffer, seedBuffer))) {
                    assertRawValues(type, expectedForward,
                            (MetalBufferRepresentation) result.publicationRepresentation(0), arena);
                    assertRawValues(type, expectedGradient,
                            (MetalBufferRepresentation) result.publicationRepresentation(1), arena);
                }
            }
        }
    }

    private static void assertRawValues(DataType type, float[] expected, MetalBufferRepresentation actual, Arena arena) {
        var host = arena.allocate((long) expected.length * type.byteWidth(), type.byteWidth());
        actual.download(0, host, 0, host.byteSize());
        assertArrayEquals(numericBytes(type, expected), host.toArray(JAVA_BYTE), type.toString());
    }

    private static byte[] numericBytes(DataType type, float[] values) {
        var bytes = java.nio.ByteBuffer.allocate(values.length * type.byteWidth()).order(java.nio.ByteOrder.nativeOrder());
        for (float value : values) {
            if (type == DataType.FLOAT32) bytes.putFloat(value);
            else bytes.putShort(type == DataType.BFLOAT16
                    ? io.github.pho001.synaptik.model.datatype.BFloat16Bits.fromFloat(value)
                    : io.github.pho001.synaptik.model.datatype.Float16Bits.fromFloat(value));
        }
        return bytes.array();
    }
}
