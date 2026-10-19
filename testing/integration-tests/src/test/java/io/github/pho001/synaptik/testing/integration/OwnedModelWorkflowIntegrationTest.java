package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.layers.Linear;
import io.github.pho001.synaptik.nn.module.Model;
import io.github.pho001.synaptik.training.Sgd;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Application construction and file restoration share one owned compute/train/save lifecycle. */
final class OwnedModelWorkflowIntegrationTest {
    @TempDir Path directory;

    private static Tensor matrix(float value, boolean gradient) {
        Shape shape = Shape.of(1, 1);
        Tensor tensor = TensorFactory.allocate(new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), gradient));
        tensor.hostStorage().orElseThrow().segment().set(ValueLayout.JAVA_FLOAT_UNALIGNED, 0, value);
        return tensor;
    }

    @Test void manualAndRestoredTrainingShareApiButNeverCallerWeights() throws Exception {
        Tensor external = matrix(2, true);
        var originalStorage = external.hostStorage().orElseThrow();
        var signature = matrix(0, false).descriptor();
        Path initial = directory.resolve("initial"), trained = directory.resolve("trained");
        HostTensorValue retained;
        try (Engine borrowed = Engine.standard()) {
            try (Model<Tensor, Tensor> model = Model.defineTensor(signature, topology -> {
                var linear = topology.addModule("linear", new Linear(external));
                return linear::forward;
            }, borrowed)) {
                model.save(initial);
                retained = model.compute(matrix(3, false));
                assertEquals(6, retained.bytes().getFloat());
                Tensor owned = model.stateDictionary().entries().getFirst().value();
                model.configureTraining(signature, (p, t) -> {
                    Tensor error = p.sub(t); return error.mul(error).sum();
                }, new Sgd(0.1));
                assertEquals(25, model.train(matrix(3, false), matrix(1, false)).objective().bytes().getFloat());
                assertSame(owned, model.stateDictionary().entries().getFirst().value());
                assertEquals(-2, model.compute(matrix(2, false)).bytes().getFloat(), 1e-6);
                model.save(trained);
            }
            assertFalse(borrowed.isClosed());
            try (Model<Tensor, Tensor> loaded = Model.load(trained).withEngine(borrowed)) {
                assertEquals(-2, loaded.compute(matrix(2, false)).bytes().getFloat(), 1e-6);
                loaded.configureTraining(signature, (p, t) -> {
                    Tensor error = p.sub(t); return error.mul(error).sum();
                }, new Sgd(0.1));
                assertEquals(1, loaded.train(matrix(2, false), matrix(-3, false)).objective().bytes().getFloat(), 1e-6);
                assertEquals(-5.6f, loaded.compute(matrix(4, false)).bytes().getFloat(), 1e-5);
            }
            try (var untrained = Model.load(initial)) {
                assertEquals(6, untrained.compute(matrix(3, false)).bytes().getFloat());
            }
        }
        assertSame(originalStorage, external.hostStorage().orElseThrow());
        assertEquals(2, originalStorage.segment().get(ValueLayout.JAVA_FLOAT_UNALIGNED, 0));
        assertEquals(6, retained.bytes().getFloat());
    }
}
