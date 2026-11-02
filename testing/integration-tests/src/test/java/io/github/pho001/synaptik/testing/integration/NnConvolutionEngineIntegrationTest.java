package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.metal.MetalBackendConfiguration;
import io.github.pho001.synaptik.backend.metal.MetalBackendIntegration;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.engine.RunResult;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.initialization.ParameterInitialization;
import io.github.pho001.synaptik.nn.layers.Conv1d;
import io.github.pho001.synaptik.nn.layers.Conv2d;
import io.github.pho001.synaptik.nn.layers.Conv3d;
import io.github.pho001.synaptik.nn.module.Module;
import io.github.pho001.synaptik.nn.module.ModuleFactory;
import io.github.pho001.synaptik.nn.module.Parameter;
import io.github.pho001.synaptik.nn.module.StateDictionary;
import io.github.pho001.synaptik.nn.module.StateEntry;
import io.github.pho001.synaptik.nn.module.StateKind;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import java.util.function.Supplier;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/** Proves strict-loaded NN dimensional convolutions through the public Engine boundary. */
final class NnConvolutionEngineIntegrationTest {
    private static final float TOLERANCE = 1.0e-5f;
    private static final float[] BIAS = {0.5f, -0.25f, 1.0f, -1.0f};

    @Test
    void executesStrictLoadedConv1dThroughOneShotCompute() {
        float[] inputValues = pattern(12, 9, 4, 0.25f);
        float[] weightValues = pattern(12, 7, 3, 0.125f);
        HostTensorValue detached;
        try (Arena arena = Arena.ofShared()) {
            Tensor input = tensor(arena, Shape.of(1, 2, 6), false, inputValues);
            Tensor weight = tensor(arena, Shape.of(2, 2, 3), true, weightValues);
            Conv1d layer = new Conv1d(2, 3, 2, 1, 2, 1, false, DataType.FLOAT32,
                    ParameterInitialization.zeros(), 101L);

            assertThrows(IllegalStateException.class, layer::weight);
            assertThrows(IllegalStateException.class, layer::stateDictionary);
            layer.loadStateDictionary(dictionary(entry("weight", weight)));
            assertState(layer, weight, null, Shape.of(2, 2, 3));
            assertSame(layer.weight(), layer.parameters().getFirst());
            assertTrue(layer.bias().isEmpty());

            Tensor output = layer.forward(input);
            assertStorageBoundary(output, input, weight);
            assertEquals(Shape.of(1, 2, 2), output.descriptor().shape());

            Engine engine = Engine.standard();
            try {
                assertFalse(engine.isClosed());
                detached = engine.compute(output);
                assertValue(detached, Shape.of(1, 2, 2), conv1d(
                        inputValues, weightValues, null,
                        1, 2, 6, 2, 3, 2, 1, 2, 1));
                assertTrue(input.hostStorage().orElseThrow().isAlive());
                assertTrue(weight.hostStorage().orElseThrow().isAlive());
            } finally {
                engine.close();
            }
            assertTrue(engine.isClosed());
            assertThrows(IllegalStateException.class, () -> engine.compute(output));
        }
        assertValue(detached, Shape.of(1, 2, 2), conv1d(
                inputValues, weightValues, null,
                1, 2, 6, 2, 3, 2, 1, 2, 1));
    }

    @Test
    void reusesPreparedGroupedConv2dAndObservesPublicationCleanup() {
        float[] inputValues = pattern(80, 13, 6, 0.125f);
        float[] weightValues = pattern(48, 11, 5, 0.0625f);
        HostTensorValue firstDetached;
        HostTensorValue secondDetached;
        try (Arena arena = Arena.ofShared()) {
            Tensor input = tensor(arena, Shape.of(1, 4, 4, 5), false, inputValues);
            Tensor weight = tensor(arena, Shape.of(4, 2, 2, 3), true, weightValues);
            Tensor bias = tensor(arena, Shape.of(4), true, BIAS);
            Conv2d layer = ModuleFactory.standard().conv2d(
                    4, 2, 3, 2, 1, 1, 1, 1, 2, 2, true,
                    DataType.FLOAT32, ParameterInitialization.zeros(), 202L);
            layer.loadStateDictionary(dictionary(entry("bias", bias), entry("weight", weight)));
            assertState(layer, weight, bias, Shape.of(4, 2, 2, 3));
            assertSame(layer.weight(), layer.parameters().getFirst());
            assertSame(layer.bias().orElseThrow(), layer.parameters().get(1));

            Tensor output = layer.forward(input);
            assertStorageBoundary(output, input, weight, bias);
            assertEquals(Shape.of(1, 4, 3, 3), output.descriptor().shape());
            float[] expected = conv2d(inputValues, weightValues, BIAS,
                    1, 4, 4, 5, 4, 2, 3, 2, 1, 1, 1, 1, 2, 2);

            Engine engine = Engine.standard();
            try {
                var compiled = engine.compile(List.of(output));
                assertEquals(List.of(input.id(), weight.id(), bias.id()), compiled.inputs().stream()
                        .map(inputMetadata -> inputMetadata.tensorId()).toList());
                var prepared = engine.prepare(compiled);
                assertSame(compiled, prepared.compiledGraph());
                List<Tensor> suppliedOutOfCompiledOrder = List.of(bias, weight, input);
                RunResult first = engine.run(prepared, suppliedOutOfCompiledOrder);
                RunResult second = engine.run(prepared, suppliedOutOfCompiledOrder);
                assertNotSame(first, second);
                assertNotSame(first.publications().getFirst(), second.publications().getFirst());

                RunResult.Publication firstPublication = first.publications().getFirst();
                assertFalse(first.isClosed());
                assertFalse(firstPublication.isClosed());
                firstDetached = first.materialize(firstPublication, expected.length * Float.BYTES);
                assertValue(firstDetached, Shape.of(1, 4, 3, 3), expected);
                first.close();
                assertTrue(first.isClosed());
                assertTrue(firstPublication.isClosed());
                assertThrows(IllegalStateException.class,
                        () -> first.materialize(firstPublication, expected.length * Float.BYTES));
                assertCallerStorageAlive(input, weight, bias);

                RunResult.Publication secondPublication = second.publications().getFirst();
                assertFalse(second.isClosed());
                assertFalse(secondPublication.isClosed());
                secondDetached = second.materialize(
                        secondPublication, expected.length * Float.BYTES);
                assertValue(secondDetached, Shape.of(1, 4, 3, 3), expected);
                second.close();
                assertTrue(second.isClosed());
                assertTrue(secondPublication.isClosed());
                assertCallerStorageAlive(input, weight, bias);
            } finally {
                engine.close();
            }
            assertTrue(engine.isClosed());
            assertThrows(IllegalStateException.class,
                    () -> engine.run(engine.prepare(engine.compile(List.of(output))),
                            List.of(input, weight, bias)));
        }
        float[] expected = conv2d(inputValues, weightValues, BIAS,
                1, 4, 4, 5, 4, 2, 3, 2, 1, 1, 1, 1, 2, 2);
        assertValue(firstDetached, Shape.of(1, 4, 3, 3), expected);
        assertValue(secondDetached, Shape.of(1, 4, 3, 3), expected);
    }

    @Test
    void executesStrictLoadedGroupedConv3dAndDifferentiatesEveryRole() {
        try (Engine engine = Engine.standard()) {
            assertGroupedConv3dGradients(engine);
        }
    }

    @Test
    void groupedConv3dGradientsMatchFiniteDifferencesWithoutACpuBackend() {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        Assumptions.assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is required for real Metal integration");
        Path library = Path.of(configured).toAbsolutePath().normalize();
        assertTrue(Files.isRegularFile(library), "configured Metal library must exist: " + library);
        try (Engine.Builder builder = Engine.builder()) {
            builder.takeOwnership(MetalBackendIntegration.open(new MetalBackendConfiguration(library)));
            try (Engine engine = builder.build()) {
                assertGroupedConv3dGradients(engine);
            }
        }
    }

    private static void assertGroupedConv3dGradients(Engine engine) {
        float[] inputValues = pattern(240, 17, 8, 0.0625f);
        float[] weightValues = pattern(64, 13, 6, 0.03125f);
        float[] biasValues = BIAS.clone();
        Supplier<float[]> forward = () -> conv3d(inputValues, weightValues, biasValues,
                1, 4, 3, 4, 5, 4, 2, 2, 2,
                1, 2, 1, 1, 0, 1, 1, 1, 2, 2);
        HostTensorValue detached;
        try (Arena arena = Arena.ofShared()) {
            Tensor input = tensor(arena, Shape.of(1, 4, 3, 4, 5), true, inputValues);
            Tensor weight = tensor(arena, Shape.of(4, 2, 2, 2, 2), true, weightValues);
            Tensor bias = tensor(arena, Shape.of(4), true, biasValues);
            Conv3d layer = new Conv3d(
                    4, 2, 2, 2, 1, 2, 1, 1, 0, 1, 1, 1, 2, 2, true,
                    DataType.FLOAT32, ParameterInitialization.zeros(), 303L);
            layer.loadStateDictionary(dictionary(entry("weight", weight), entry("bias", bias)));
            assertState(layer, weight, bias, Shape.of(4, 2, 2, 2, 2));
            assertSame(layer.weight(), layer.parameters().getFirst());
            assertSame(layer.bias().orElseThrow(), layer.parameters().get(1));

            Tensor output = layer.forward(input);
            assertStorageBoundary(output, input, weight, bias);
            assertEquals(Shape.of(1, 4, 4, 2, 5), output.descriptor().shape());
            float[] expected = forward.get();

            detached = engine.compute(output);
            assertValue(detached, Shape.of(1, 4, 4, 2, 5), expected);
            List<Tensor> targets = List.of(input, weight, bias);
            var backward = engine.backward(output.sum(), targets, Long.MAX_VALUE);
            List<float[]> roles = List.of(inputValues, weightValues, biasValues);
            for (int role = 0; role < targets.size(); role++) {
                assertValue(backward.gradients().get(role), targets.get(role).descriptor().shape(),
                        finiteDifferences(roles.get(role), forward));
            }
            assertFalse(engine.isClosed());
            assertValue(engine.compute(output), Shape.of(1, 4, 4, 2, 5), expected);
            assertCallerStorageAlive(input, weight, bias);
        }
        assertValue(detached, Shape.of(1, 4, 4, 2, 5), forward.get());
    }

    /**
     * Differentiates the independent direct-loop scalar sum, restoring every perturbed value.
     * Binary fixtures and a binary step keep each forward contraction exactly representable.
     *
     * @param values non-null mutable caller-local role values, restored before returning
     * @param forward non-null independent convolution oracle reading those values
     * @return one central-difference cotangent per original role element
     */
    private static float[] finiteDifferences(float[] values, Supplier<float[]> forward) {
        float[] gradient = new float[values.length];
        float step = 0.125f;
        for (int lane = 0; lane < values.length; lane++) {
            float original = values[lane];
            try {
                values[lane] = original + step;
                double upper = sum(forward.get());
                values[lane] = original - step;
                double lower = sum(forward.get());
                gradient[lane] = (float) ((upper - lower) / (2 * step));
            } finally {
                values[lane] = original;
            }
        }
        return gradient;
    }

    private static double sum(float[] values) {
        double total = 0;
        for (float value : values) total += value;
        return total;
    }

    private static void assertState(
            Module layer, Tensor weight, Tensor bias, Shape expectedWeightShape) {
        StateDictionary exported = layer.stateDictionary();
        List<String> expectedPaths = bias == null ? List.of("weight") : List.of("weight", "bias");
        assertEquals(expectedPaths, exported.entries().stream().map(StateEntry::path).toList());
        assertTrue(exported.entries().stream().allMatch(entry -> entry.kind() == StateKind.PARAMETER));
        assertSame(weight, exported.entries().getFirst().value());
        assertEquals(expectedWeightShape, weight.descriptor().shape());
        assertSame(weight, layer.parameters().getFirst().value());
        if (bias == null) {
            assertEquals(1, layer.parameters().size());
        } else {
            assertEquals(2, layer.parameters().size());
            assertSame(bias, exported.entries().get(1).value());
            assertSame(bias, layer.parameters().get(1).value());
            assertEquals(Shape.of(4), bias.descriptor().shape());
        }
    }

    private static void assertStorageBoundary(Tensor output, Tensor... leaves) {
        assertTrue(output.hostStorage().isEmpty());
        assertTrue(output.provenance().isPresent());
        for (Tensor leaf : leaves) {
            assertTrue(leaf.hostStorage().isPresent());
            assertTrue(leaf.provenance().isEmpty());
        }
    }

    private static void assertCallerStorageAlive(Tensor... tensors) {
        for (Tensor tensor : tensors) {
            assertTrue(tensor.hostStorage().orElseThrow().isAlive());
        }
    }

    private static StateDictionary dictionary(StateEntry... entries) {
        return new StateDictionary(List.of(entries));
    }

    private static StateEntry entry(String path, Tensor value) {
        return new StateEntry(path, StateKind.PARAMETER, value);
    }

    private static Tensor tensor(
            Arena arena, Shape shape, boolean requiresGrad, float[] values) {
        assertEquals(shape.knownElementCount().orElseThrow(), values.length);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.of(LayoutDescriptor.contiguous(shape)),
                requiresGrad);
        MemorySegment source = MemorySegment.ofArray(values);
        MemorySegment segment = arena.allocate(source.byteSize(), Float.BYTES);
        MemorySegment.copy(source, 0, segment, 0, source.byteSize());
        return TensorFactory.create(descriptor, Optional.empty(), Optional.of(
                new MemorySegmentStorage(DataType.FLOAT32, values.length, segment)));
    }

    private static float[] pattern(int count, int modulus, int offset, float scale) {
        float[] values = new float[count];
        for (int index = 0; index < count; index++) {
            values[index] = (index % modulus - offset) * scale;
        }
        return values;
    }

    private static void assertValue(HostTensorValue value, Shape shape, float[] expected) {
        assertEquals(DataType.FLOAT32, value.dataType());
        assertEquals(shape, value.shape());
        FloatBuffer buffer = value.bytes().asFloatBuffer();
        float[] actual = new float[buffer.remaining()];
        buffer.get(actual);
        assertArrayEquals(expected, actual, TOLERANCE);
    }

    private static float[] conv1d(
            float[] input, float[] weight, float[] bias,
            int batches, int inputChannels, int inputWidth, int outputChannels,
            int kernelWidth, int stride, int padding, int dilation, int groups) {
        int outputWidth = (inputWidth + 2 * padding - dilation * (kernelWidth - 1) - 1)
                / stride + 1;
        int channelsPerGroup = inputChannels / groups;
        int outputsPerGroup = outputChannels / groups;
        float[] output = new float[batches * outputChannels * outputWidth];
        for (int batch = 0; batch < batches; batch++) {
            for (int outputChannel = 0; outputChannel < outputChannels; outputChannel++) {
                int inputChannelStart = (outputChannel / outputsPerGroup) * channelsPerGroup;
                for (int outputX = 0; outputX < outputWidth; outputX++) {
                    float sum = bias == null ? 0.0f : bias[outputChannel];
                    for (int localChannel = 0; localChannel < channelsPerGroup; localChannel++) {
                        int inputChannel = inputChannelStart + localChannel;
                        for (int kernelX = 0; kernelX < kernelWidth; kernelX++) {
                            int inputX = outputX * stride - padding + kernelX * dilation;
                            if (inputX >= 0 && inputX < inputWidth) {
                                int inputIndex = (batch * inputChannels + inputChannel)
                                        * inputWidth + inputX;
                                int weightIndex = (outputChannel * channelsPerGroup + localChannel)
                                        * kernelWidth + kernelX;
                                sum += input[inputIndex] * weight[weightIndex];
                            }
                        }
                    }
                    output[(batch * outputChannels + outputChannel) * outputWidth + outputX] = sum;
                }
            }
        }
        return output;
    }

    private static float[] conv2d(
            float[] input, float[] weight, float[] bias,
            int batches, int inputChannels, int inputHeight, int inputWidth,
            int outputChannels, int kernelHeight, int kernelWidth,
            int strideHeight, int strideWidth, int paddingHeight, int paddingWidth,
            int dilationHeight, int dilationWidth, int groups) {
        int outputHeight = (inputHeight + 2 * paddingHeight
                - dilationHeight * (kernelHeight - 1) - 1) / strideHeight + 1;
        int outputWidth = (inputWidth + 2 * paddingWidth
                - dilationWidth * (kernelWidth - 1) - 1) / strideWidth + 1;
        int channelsPerGroup = inputChannels / groups;
        int outputsPerGroup = outputChannels / groups;
        float[] output = new float[batches * outputChannels * outputHeight * outputWidth];
        for (int batch = 0; batch < batches; batch++) {
            for (int outputChannel = 0; outputChannel < outputChannels; outputChannel++) {
                int inputChannelStart = (outputChannel / outputsPerGroup) * channelsPerGroup;
                for (int outputY = 0; outputY < outputHeight; outputY++) {
                    for (int outputX = 0; outputX < outputWidth; outputX++) {
                        float sum = bias == null ? 0.0f : bias[outputChannel];
                        for (int localChannel = 0; localChannel < channelsPerGroup; localChannel++) {
                            int inputChannel = inputChannelStart + localChannel;
                            for (int kernelY = 0; kernelY < kernelHeight; kernelY++) {
                                int inputY = outputY * strideHeight - paddingHeight
                                        + kernelY * dilationHeight;
                                for (int kernelX = 0; kernelX < kernelWidth; kernelX++) {
                                    int inputX = outputX * strideWidth - paddingWidth
                                            + kernelX * dilationWidth;
                                    if (inputY >= 0 && inputY < inputHeight
                                            && inputX >= 0 && inputX < inputWidth) {
                                        int inputIndex = ((batch * inputChannels + inputChannel)
                                                * inputHeight + inputY) * inputWidth + inputX;
                                        int weightIndex = ((outputChannel * channelsPerGroup
                                                + localChannel) * kernelHeight + kernelY)
                                                * kernelWidth + kernelX;
                                        sum += input[inputIndex] * weight[weightIndex];
                                    }
                                }
                            }
                        }
                        output[((batch * outputChannels + outputChannel) * outputHeight + outputY)
                                * outputWidth + outputX] = sum;
                    }
                }
            }
        }
        return output;
    }

    private static float[] conv3d(
            float[] input, float[] weight, float[] bias,
            int batches, int inputChannels, int inputDepth, int inputHeight, int inputWidth,
            int outputChannels, int kernelDepth, int kernelHeight, int kernelWidth,
            int strideDepth, int strideHeight, int strideWidth,
            int paddingDepth, int paddingHeight, int paddingWidth,
            int dilationDepth, int dilationHeight, int dilationWidth, int groups) {
        int outputDepth = (inputDepth + 2 * paddingDepth
                - dilationDepth * (kernelDepth - 1) - 1) / strideDepth + 1;
        int outputHeight = (inputHeight + 2 * paddingHeight
                - dilationHeight * (kernelHeight - 1) - 1) / strideHeight + 1;
        int outputWidth = (inputWidth + 2 * paddingWidth
                - dilationWidth * (kernelWidth - 1) - 1) / strideWidth + 1;
        int channelsPerGroup = inputChannels / groups;
        int outputsPerGroup = outputChannels / groups;
        float[] output = new float[
                batches * outputChannels * outputDepth * outputHeight * outputWidth];
        for (int batch = 0; batch < batches; batch++) {
            for (int outputChannel = 0; outputChannel < outputChannels; outputChannel++) {
                int inputChannelStart = (outputChannel / outputsPerGroup) * channelsPerGroup;
                for (int outputZ = 0; outputZ < outputDepth; outputZ++) {
                    for (int outputY = 0; outputY < outputHeight; outputY++) {
                        for (int outputX = 0; outputX < outputWidth; outputX++) {
                            float sum = bias == null ? 0.0f : bias[outputChannel];
                            for (int localChannel = 0; localChannel < channelsPerGroup;
                                    localChannel++) {
                                int inputChannel = inputChannelStart + localChannel;
                                for (int kernelZ = 0; kernelZ < kernelDepth; kernelZ++) {
                                    int inputZ = outputZ * strideDepth - paddingDepth
                                            + kernelZ * dilationDepth;
                                    for (int kernelY = 0; kernelY < kernelHeight; kernelY++) {
                                        int inputY = outputY * strideHeight - paddingHeight
                                                + kernelY * dilationHeight;
                                        for (int kernelX = 0; kernelX < kernelWidth; kernelX++) {
                                            int inputX = outputX * strideWidth - paddingWidth
                                                    + kernelX * dilationWidth;
                                            if (inputZ >= 0 && inputZ < inputDepth
                                                    && inputY >= 0 && inputY < inputHeight
                                                    && inputX >= 0 && inputX < inputWidth) {
                                                int inputIndex = (((batch * inputChannels
                                                        + inputChannel) * inputDepth + inputZ)
                                                        * inputHeight + inputY) * inputWidth + inputX;
                                                int weightIndex = (((outputChannel * channelsPerGroup
                                                        + localChannel) * kernelDepth + kernelZ)
                                                        * kernelHeight + kernelY) * kernelWidth
                                                        + kernelX;
                                                sum += input[inputIndex] * weight[weightIndex];
                                            }
                                        }
                                    }
                                }
                            }
                            output[(((batch * outputChannels + outputChannel) * outputDepth
                                    + outputZ) * outputHeight + outputY) * outputWidth
                                    + outputX] = sum;
                        }
                    }
                }
            }
        }
        return output;
    }
}
