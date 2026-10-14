package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.layers.Activation;
import io.github.pho001.synaptik.nn.layers.Linear;
import io.github.pho001.synaptik.nn.module.DeclarativeModel;
import io.github.pho001.synaptik.nn.module.ModelSpecification;
import io.github.pho001.synaptik.nn.module.ModelSpecification.ActivationKind;
import io.github.pho001.synaptik.nn.module.Sequential;
import io.github.pho001.synaptik.nn.module.StateDictionary;
import io.github.pho001.synaptik.nn.module.StateEntry;
import io.github.pho001.synaptik.nn.module.StateKind;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.FloatBuffer;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Exercises descriptor reconstruction and separate strict value loading through the CPU Engine. */
final class NnDeclarativeModelEngineIntegrationTest {
    @Test
    void directAndStrictLoadedReconstructionMatchValuesWithChangedBatchesAndLeadingRanks() {
        float[] w1 = {1, -2, 0.5f, -1, 1, 2, 0.25f, 0.5f, -0.75f, 2, 0, -1};
        float[] b1 = {0.5f, -0.5f, 1, 0.25f};
        float[] w2 = {1, -0.5f, 2, 0.25f, -1, 1, 0.5f, 2};
        float[] b2 = {-0.25f, 0.5f};
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            Sequential direct = new Sequential(List.of(
                    new Linear(tensor(arena, Shape.of(4, 3), true, w1), tensor(arena, Shape.of(4), true, b1)),
                    new Activation(ActivationKind.RELU),
                    new Linear(tensor(arena, Shape.of(2, 4), true, w2), tensor(arena, Shape.of(2), true, b2))));
            ModelSpecification specification = ModelSpecification.describe(direct);
            DeclarativeModel reconstructed = DeclarativeModel.reconstruct(specification);
            assertTrue(reconstructed.stateDictionary().entries().stream()
                    .allMatch(entry -> entry.value().hostStorage().isEmpty()));
            reconstructed.loadStateDictionary(direct.stateDictionary());
            for (Shape shape : List.of(Shape.of(3), Shape.of(2, 3), Shape.of(5, 3), Shape.of(2, 3, 3))) {
                float[] inputValues = pattern(Math.toIntExact(shape.knownElementCount().orElseThrow()));
                Tensor input = tensor(arena, shape, false, inputValues);
                Tensor directOutput = direct.forward(input);
                Tensor rebuiltOutput = reconstructed.forward(input);
                assertEquals(directOutput.descriptor(), rebuiltOutput.descriptor());
                assertEquals(directOutput.descriptor(), specification.outputDescriptor(input.descriptor()));
                float[] expected = mlp(inputValues, w1, b1, w2, b2, 3, 4, 2);
                assertArrayEquals(expected, values(engine.compute(directOutput)), 1.0e-5f);
                assertArrayEquals(expected, values(engine.compute(rebuiltOutput)), 1.0e-5f);
            }
        }
    }

    @Test
    void tiedWeightsRemainExactAliasesAfterFreshValueReloadAndExecuteOrdinaryMlp() {
        float[] weightValues = {1, -0.5f, 2, 0.25f};
        float[] biasValues = {0.25f, -0.5f};
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            Tensor shared = tensor(arena, Shape.of(2, 2), true, weightValues);
            Tensor bias = tensor(arena, Shape.of(2), true, biasValues);
            Sequential direct = new Sequential(List.of(new Linear(shared, bias),
                    new Activation(ActivationKind.RELU), new Linear(shared, bias)));
            var spec = ModelSpecification.describe(direct);
            assertEquals(List.of(new ModelSpecification.AliasGroup(List.of("0.weight", "2.weight")),
                    new ModelSpecification.AliasGroup(List.of("0.bias", "2.bias"))), spec.aliases());
            var rebuilt = DeclarativeModel.reconstruct(spec);
            assertSame(rebuilt.parametersRecursively().get("0.weight").value(),
                    rebuilt.parametersRecursively().get("2.weight").value());
            Tensor freshWeight = tensor(arena, Shape.of(2, 2), true, weightValues);
            Tensor freshBias = tensor(arena, Shape.of(2), true, biasValues);
            rebuilt.loadStateDictionary(new StateDictionary(List.of(
                    new StateEntry("0.weight", StateKind.PARAMETER, freshWeight),
                    new StateEntry("0.bias", StateKind.PARAMETER, freshBias),
                    new StateEntry("2.weight", StateKind.PARAMETER, freshWeight),
                    new StateEntry("2.bias", StateKind.PARAMETER, freshBias))));
            assertSame(freshWeight, rebuilt.parametersRecursively().get("0.weight").value());
            assertSame(freshWeight, rebuilt.parametersRecursively().get("2.weight").value());
            assertSame(freshBias, rebuilt.parametersRecursively().get("2.bias").value());
            assertEquals(spec.aliases(), rebuilt.describe().aliases());
            float[] inputValues = {1, 2, -3, 1, 0.5f, -1};
            Tensor input = tensor(arena, Shape.of(3, 2), false, inputValues);
            assertArrayEquals(mlp(inputValues, weightValues, biasValues, weightValues, biasValues, 2, 2, 2),
                    values(engine.compute(rebuilt.forward(input))), 1.0e-5f);
            assertArrayEquals(values(engine.compute(direct.forward(input))),
                    values(engine.compute(rebuilt.forward(input))), 1.0e-5f);
        }
    }

    private static Tensor tensor(Arena arena, Shape shape, boolean grad, float[] values) {
        assertEquals(shape.knownElementCount().orElseThrow(), values.length);
        MemorySegment source = MemorySegment.ofArray(values);
        MemorySegment storage = arena.allocate(source.byteSize(), Float.BYTES);
        storage.copyFrom(source);
        return TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), grad), Optional.empty(),
                Optional.of(new MemorySegmentStorage(DataType.FLOAT32, values.length, storage)));
    }

    private static float[] values(HostTensorValue value) {
        FloatBuffer buffer = value.bytes().asFloatBuffer();
        float[] result = new float[buffer.remaining()];
        buffer.get(result);
        return result;
    }

    private static float[] pattern(int count) {
        float[] values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (i % 7 - 3) * 0.25f;
        return values;
    }

    private static float[] mlp(float[] input, float[] w1, float[] b1, float[] w2, float[] b2,
            int in, int hidden, int out) {
        int rows = input.length / in;
        float[] result = new float[rows * out];
        for (int row = 0; row < rows; row++) {
            float[] intermediate = new float[hidden];
            for (int h = 0; h < hidden; h++) {
                float value = 0;
                for (int i = 0; i < in; i++) value += input[row * in + i] * w1[h * in + i];
                intermediate[h] = Math.max(0, value + b1[h]);
            }
            for (int o = 0; o < out; o++) {
                float value = 0;
                for (int h = 0; h < hidden; h++) value += intermediate[h] * w2[o * hidden + h];
                result[row * out + o] = value + b2[o];
            }
        }
        return result;
    }
}
