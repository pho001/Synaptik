package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.metal.MetalBackendConfiguration;
import io.github.pho001.synaptik.backend.metal.MetalBackendIntegration;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool3dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool3dAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/** Independent spatial scatter/winner oracles exercise the public forward and gradient lifecycle. */
final class EnginePool3dTrainingIntegrationTest {
    @Test void cpuPoolingRoutesEveryCotangentAndExcludesConceptualPadding() {
        try (Engine engine = Engine.standard()) {
            for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) check(engine, type);
        }
    }

    @Test void metalPoolingRoutesEveryCotangentWithoutACpuBackend() {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        Assumptions.assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is required for real Metal integration");
        Path library = Path.of(configured).toAbsolutePath().normalize();
        assertTrue(Files.isRegularFile(library), "configured Metal library must exist");
        try (Engine.Builder builder = Engine.builder()) {
            builder.takeOwnership(MetalBackendIntegration.open(new MetalBackendConfiguration(library)));
            try (Engine engine = builder.build()) { check(engine, DataType.FLOAT32); }
        }
    }

    private static void check(Engine engine, DataType type) {
        double[] input = new double[27], seed = new double[8];
        for (int i = 0; i < input.length; i++) input[i] = i + 1;
        for (int i = 0; i < seed.length; i++) seed[i] = i + 1;
        for (boolean maximum : List.of(false, true)) {
            double[] expected = new double[8], gradient = new double[27];
            for (int z = 0; z < 2; z++) for (int h = 0; h < 2; h++) for (int w = 0; w < 2; w++) {
                int position = (z * 2 + h) * 2 + w;
                if (maximum) {
                    int winner = ((z + 1) * 3 + h + 1) * 3 + w + 1;
                    expected[position] = input[winner];
                    gradient[winner] += seed[position];
                } else {
                    for (int dz = 0; dz < 2; dz++) for (int dh = 0; dh < 2; dh++) for (int dw = 0; dw < 2; dw++) {
                        int source = ((z + dz) * 3 + h + dh) * 3 + w + dw;
                        expected[position] += input[source] / 8;
                        gradient[source] += seed[position] / 8;
                    }
                }
            }
            verify(engine, type, Shape.of(1, 1, 3, 3, 3), input, maximum, 0,
                    Shape.of(1, 1, 2, 2, 2), seed, expected, gradient);
        }
        verify(engine, type, Shape.of(1, 1, 2, 2, 2),
                new double[]{1, 2, Double.NaN, 3, Double.NaN, 4, 5, 6}, true, 0,
                Shape.of(1, 1, 1, 1, 1), new double[]{7}, new double[]{Double.NaN},
                new double[]{0, 0, 7, 0, 0, 0, 0, 0});
        verify(engine, type, Shape.of(1, 1, 2, 2, 2),
                new double[]{-0d, -0d, -1, -2, -3, +0d, -4, -5}, true, 0,
                Shape.of(1, 1, 1, 1, 1), new double[]{7}, new double[]{+0d},
                new double[]{0, 0, 0, 0, 0, 7, 0, 0});
        verify(engine, type, Shape.of(1, 1, 2, 2, 2),
                new double[]{2, 2, 1, 1, 1, 1, 1, 1}, true, 0,
                Shape.of(1, 1, 1, 1, 1), new double[]{7}, new double[]{2},
                new double[]{7, 0, 0, 0, 0, 0, 0, 0});
        for (boolean maximum : List.of(false, true)) {
            double[] paddingSeed = new double[64], expected = new double[64];
            for (int i = 0; i < paddingSeed.length; i++) paddingSeed[i] = i + 1;
            if (maximum) Arrays.fill(expected, Double.NEGATIVE_INFINITY);
            else for (int z = 1; z <= 2; z++) for (int h = 1; h <= 2; h++) for (int w = 1; w <= 2; w++) {
                expected[(z * 4 + h) * 4 + w] = 1;
            }
            verify(engine, type, Shape.of(1, 1, 1, 1, 1),
                    new double[]{maximum ? Double.NEGATIVE_INFINITY : 8}, maximum, 2,
                    Shape.of(1, 1, 4, 4, 4), paddingSeed, expected,
                    new double[]{maximum ? 260 : 32.5});
        }
    }

    private static void verify(Engine engine, DataType type, Shape inputShape, double[] values,
            boolean maximum, long padding, Shape outputShape, double[] seeds,
            double[] expectedOutput, double[] expectedGradient) {
        Tensor input = tensor(type, inputShape, true, values);
        Tensor output = maximum
                ? input.maxPool3d(new MaxPool3dAttrs(2, 2, 2, 1, 1, 1, padding, padding, padding, 1, 1, 1, false))
                : input.averagePool3d(new AveragePool3dAttrs(2, 2, 2, 1, 1, 1, padding, padding, padding, 1, 1, 1, false));
        Tensor seed = tensor(type, outputShape, false, seeds);
        var compiled = engine.compile(List.of(output), List.of(seed), List.of(input));
        try (var prepared = engine.prepare(compiled); var result = engine.run(prepared, List.of(input, seed))) {
            assertEquals(2, result.publications().size());
            compare(expectedOutput, result.materialize(result.publications().get(0), 1_000_000));
            compare(expectedGradient, result.materialize(result.publications().get(1), 1_000_000));
        }
    }

    private static Tensor tensor(DataType type, Shape shape, boolean gradient, double[] values) {
        var descriptor = new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), gradient);
        if (type == DataType.FLOAT64) return TensorFactory.fromFlatArray(descriptor, Optional.empty(), values);
        float[] floats = new float[values.length];
        for (int i = 0; i < floats.length; i++) floats[i] = (float) values[i];
        return TensorFactory.fromFlatArray(descriptor, Optional.empty(), floats);
    }

    private static void compare(double[] expected, HostTensorValue actual) {
        var bytes = actual.bytes();
        assertEquals(expected.length * actual.dataType().byteWidth(), bytes.remaining());
        for (double value : expected) {
            double observed = actual.dataType() == DataType.FLOAT64 ? bytes.getDouble() : bytes.getFloat();
            if (Double.isNaN(value)) assertTrue(Double.isNaN(observed));
            else if (value == 0) assertEquals(Double.doubleToRawLongBits(value), Double.doubleToRawLongBits(observed));
            else assertEquals(value, observed, 0);
        }
    }
}
