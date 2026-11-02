package io.github.pho001.synaptik.nn.module;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.checkpoint.artifact.ArtifactFile;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.training.Sgd;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Independent scalar equations for real Conv3d/pool/flatten/linear training across cold reload. */
final class Pool3dModelWorkflowTest {
    @TempDir Path directory;

    @Test void bothPoolsTrainSaveCloseReloadExactParametersAndTrainNewBatches() throws Exception {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (boolean average : List.of(false, true)) {
                Path initial = directory.resolve(type + "-" + average + "-initial");
                Path trained = directory.resolve(type + "-" + average + "-trained");
                double[] expected = {2, .5, .25, -.125};
                List<byte[]> saved;
                var builder = Model.builder().input(type, 1, 1, 2, 2, 2).conv3d(1, 1, 1, 1);
                if (average) builder.averagePool3d(2, 2, 2); else builder.maxPool3d(2, 2, 2);
                try (var model = builder.flatten().linear(1).build()) {
                    assertEquals(List.of("layers.0.weight", "layers.0.bias", "layers.3.weight", "layers.3.bias"),
                            model.stateDictionary().entries().stream().map(StateEntry::path).toList());
                    setParameters(model, type, expected);
                    model.save(initial); // Acquisition and saving do not require execution.
                    assertEquals("graph-model.v6", ArtifactFile.read(initial, ModelExecutionState.DEFAULT_LIMITS).kind());
                    try (var cold = Model.load(initial)) {
                        assertParameters(cold, type, expected);
                        checkPrediction(cold, type, expected, average ? 4.5 : 8,
                                batch(type, 1, 2, 3, 4, 5, 6, 7, 8));
                    }
                    model.configureTraining(descriptor(type, 1, 1), ModelBuilderExecutionTest::objective, new Sgd(.001));
                    step(model, type, expected, average ? 4.5 : 8,
                            batch(type, 1, 2, 3, 4, 5, 6, 7, 8), .25);
                    model.save(trained);
                    saved = model.stateDictionary().entries().stream().map(e -> bytes(e.value())).toList();
                }
                try (var loaded = Model.load(trained)) {
                    for (int i = 0; i < saved.size(); i++)
                        assertArrayEquals(saved.get(i), bytes(loaded.stateDictionary().entries().get(i).value()));
                    assertEquals(ForwardMode.EVALUATION, loaded.mode());
                    // Objectives and optimizer/session state are configured afresh, never restored.
                    loaded.configureTraining(descriptor(type, 1, 1), ModelBuilderExecutionTest::objective, new Sgd(.001));
                    step(loaded, type, expected, average ? 2 : 6,
                            batch(type, -2, -1, 0, 1, 3, 4, 5, 6), -.5);
                }
            }
    }

    private static void step(Model<Tensor, Tensor> model, DataType type, double[] parameters,
            double reducedInput, Tensor batch, double target) {
        double tolerance = type == DataType.FLOAT32 ? 2e-5 : 1e-12;
        checkPrediction(model, type, parameters, reducedInput, batch);
        double pooled = parameters[0] * reducedInput + parameters[1];
        double prediction = pooled * parameters[2] + parameters[3];
        double error = prediction - target, seed = 2 * error;
        double[] gradient = {seed * parameters[2] * reducedInput, seed * parameters[2],
                seed * pooled, seed};
        assertEquals(error * error, scalar(model.train(batch, value(type, new long[]{1, 1}, target)).objective(), type), tolerance);
        for (int i = 0; i < parameters.length; i++) {
            parameters[i] -= .001 * gradient[i];
            if (type == DataType.FLOAT32) parameters[i] = (float) parameters[i];
        }
        assertParameters(model, type, parameters);
        checkPrediction(model, type, parameters, reducedInput, batch);
    }

    private static void checkPrediction(Model<Tensor, Tensor> model, DataType type, double[] p,
            double reducedInput, Tensor batch) {
        assertTrue(p[0] > 0, "positive convolution weight retains the unique maximum's input position");
        double expected = (p[0] * reducedInput + p[1]) * p[2] + p[3];
        assertEquals(expected, scalar(model.compute(batch), type), type == DataType.FLOAT32 ? 2e-5 : 1e-12);
    }

    private static void setParameters(Model<Tensor, Tensor> model, DataType type, double[] values) {
        for (int i = 0; i < values.length; i++) {
            var segment = model.stateDictionary().entries().get(i).value().hostStorage().orElseThrow().segment();
            if (type == DataType.FLOAT32) segment.set(ValueLayout.JAVA_FLOAT_UNALIGNED, 0, (float) values[i]);
            else segment.set(ValueLayout.JAVA_DOUBLE_UNALIGNED, 0, values[i]);
        }
    }

    private static void assertParameters(Model<Tensor, Tensor> model, DataType type, double[] values) {
        for (int i = 0; i < values.length; i++) {
            var segment = model.stateDictionary().entries().get(i).value().hostStorage().orElseThrow().segment();
            double actual = type == DataType.FLOAT32 ? segment.get(ValueLayout.JAVA_FLOAT_UNALIGNED, 0)
                    : segment.get(ValueLayout.JAVA_DOUBLE_UNALIGNED, 0);
            assertEquals(values[i], actual, type == DataType.FLOAT32 ? 2e-5 : 1e-12, "parameter " + i);
        }
    }

    private static byte[] bytes(Tensor value) {
        return value.hostStorage().orElseThrow().segment().toArray(ValueLayout.JAVA_BYTE);
    }

    private static double scalar(HostTensorValue value, DataType type) {
        return type == DataType.FLOAT32 ? value.bytes().getFloat() : value.bytes().getDouble();
    }

    private static Tensor batch(DataType type, double... values) {
        return value(type, new long[]{1, 1, 2, 2, 2}, values);
    }

    private static Tensor value(DataType type, long[] axes, double... values) {
        if (type == DataType.FLOAT64)
            return TensorFactory.fromFlatArray(descriptor(type, axes), Optional.empty(), values);
        float[] floats = new float[values.length];
        for (int i = 0; i < floats.length; i++) floats[i] = (float) values[i];
        return TensorFactory.fromFlatArray(descriptor(type, axes), Optional.empty(), floats);
    }

    private static TensorDescriptor descriptor(DataType type, long... axes) {
        Shape shape = Shape.of(axes);
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), false);
    }
}
