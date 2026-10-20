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
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.nn.layers.Linear;
import io.github.pho001.synaptik.nn.module.Model;
import io.github.pho001.synaptik.training.Sgd;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.Optional;
import java.util.List;
import java.nio.file.Files;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
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

    @ParameterizedTest
    @EnumSource(value = UnaryElementwiseKind.class, names = {"RELU", "TANH", "SIGMOID"})
    void twoLinearActivationMlpMatchesIndependentTensorEngineForwardAndSgd(UnaryElementwiseKind activation)
            throws Exception {
        List<Tensor> external = weights();
        var stores = external.stream().map(t -> t.hostStorage().orElseThrow()).toList();
        var original = external.stream().map(OwnedModelWorkflowIntegrationTest::bits).toList();
        var signature = values(Shape.of(1, 2), false, 0, 0).descriptor();
        var targetSignature = matrix(0, false).descriptor();
        Path initial = directory.resolve("mlp"), trained = directory.resolve("trained-mlp");
        try (Engine engine = Engine.standard();
                var owned = Model.defineTensor(signature, topology -> {
                    var first = topology.addModule("first", new Linear(external.get(0), external.get(1)));
                    var last = topology.addModule("last", new Linear(external.get(2), external.get(3)));
                    return input -> last.forward(activate(first.forward(input), activation));
                })) {
            owned.save(initial); // No compute or Engine selection has occurred.
            try (var restored = Model.load(initial)) {
                for (var model : List.of(owned, restored)) {
                    // This oracle never calls Model, GraphCheckpoint or cold Compiler reconstruction.
                    List<Tensor> direct = weights();
                    for (float[] batch : List.of(new float[] {1, 2}, new float[] {-0.5f, 1})) {
                        Tensor input = values(Shape.of(1, 2), false, batch);
                        float expected = engine.compute(mlp(input, direct, activation)).bytes().getFloat();
                        if (batch[0] < 0) {
                            double a = switch (activation) {
                                case RELU -> 0; case TANH -> StrictMath.tanh(-0.3);
                                case SIGMOID -> 1 / (1 + StrictMath.exp(0.3));
                                default -> throw new AssertionError(activation);
                            };
                            double b = switch (activation) {
                                case RELU -> 0.275; case TANH -> StrictMath.tanh(0.275);
                                case SIGMOID -> 1 / (1 + StrictMath.exp(-0.275));
                                default -> throw new AssertionError(activation);
                            };
                            assertEquals(0.75 * a - 0.4 * b + 0.1, expected, 1e-5);
                            assertNotEquals(-0.235, expected, 1e-3, "identity is not the activation");
                        }
                        assertEquals(expected, model.compute(input).bytes().getFloat(), 1e-5);
                    }
                    model.configureTraining(targetSignature, OwnedModelWorkflowIntegrationTest::objective, new Sgd(0.01));
                    for (float[] batch : List.of(new float[] {1, 2}, new float[] {-0.5f, 1})) {
                        Tensor input = values(Shape.of(1, 2), false, batch), target = matrix(0.3f, false);
                        Tensor prediction = mlp(input, direct, activation);
                        var expected = engine.backward(objective(prediction, target), direct, 1000);
                        float loss = model.train(input, target).objective().bytes().getFloat();
                        assertEquals(expected.objective().bytes().getFloat(), loss, 1e-5);
                        assertTrue(loss > 0);
                        var entries = model.stateDictionary().entries();
                        assertEquals(4, entries.size());
                        for (int p = 0; p < direct.size(); p++) {
                            Tensor parameter = direct.get(p);
                            var memory = parameter.hostStorage().orElseThrow().segment();
                            var gradient = expected.gradients().get(p).bytes();
                            boolean changed = false;
                            for (int lane = 0; lane < memory.byteSize() / 4; lane++) {
                                float before = memory.get(ValueLayout.JAVA_FLOAT_UNALIGNED, lane * 4L);
                                float after = before - 0.01f * gradient.getFloat(lane * 4);
                                changed |= Math.abs(before - after) > 1e-8;
                                memory.set(ValueLayout.JAVA_FLOAT_UNALIGNED, lane * 4L, after);
                                assertEquals(after, entries.get(p).value().hostStorage().orElseThrow().segment()
                                        .get(ValueLayout.JAVA_FLOAT_UNALIGNED, lane * 4L), 1e-5);
                            }
                            assertTrue(changed, "each parameter must receive a nonzero update");
                        }
                        assertEquals(engine.compute(mlp(input, direct, activation)).bytes().getFloat(),
                                model.compute(input).bytes().getFloat(), 1e-5);
                    }
                    model.save(trained);
                    try (var loadedTrained = Model.load(trained)) {
                        for (int p = 0; p < 4; p++) assertArrayEquals(bits(model.stateDictionary().entries().get(p).value()),
                                bits(loadedTrained.stateDictionary().entries().get(p).value()));
                        Tensor input = values(Shape.of(1, 2), false, 1, 2);
                        assertEquals(engine.compute(mlp(input, direct, activation)).bytes().getFloat(),
                                loadedTrained.compute(input).bytes().getFloat(), 1e-5);
                    }
                }
            }
        }
        for (int p = 0; p < external.size(); p++) {
            assertSame(stores.get(p), external.get(p).hostStorage().orElseThrow());
            assertArrayEquals(original.get(p), bits(external.get(p)));
        }
        assertTrue(Files.size(trained) > 0);
    }

    /**
     * Supplies finite nonsaturated weights with positive and negative activation inputs across batches.
     * @return independent first weight/bias then last weight/bias, all FLOAT32 gradient leaves
     */
    private static List<Tensor> weights() {
        return List.of(values(Shape.of(2, 2), true, 0.5f, -0.25f, 0.25f, 0.5f),
                values(Shape.of(2), true, 0.2f, -0.1f), values(Shape.of(1, 2), true, 0.75f, -0.4f),
                values(Shape.of(1), true, 0.1f));
    }

    /**
     * @param input fresh [1,2] batch
     * @param parameters independent ordinary first/last weight/bias leaves
     * @param activation selected intervening unary kind
     * @return ordinary two-Linear Tensor expression, without any cold-source consumer
     */
    private static Tensor mlp(Tensor input, List<Tensor> parameters, UnaryElementwiseKind activation) {
        Tensor hidden = new Linear(parameters.get(0), parameters.get(1)).forward(input);
        return new Linear(parameters.get(2), parameters.get(3)).forward(activate(hidden, activation));
    }

    /**
     * @param input floating expression
     * @param kind one of the three tested activations
     * @return ordinary activation expression
     */
    private static Tensor activate(Tensor input, UnaryElementwiseKind kind) {
        return switch (kind) {
            case RELU -> input.relu(); case TANH -> input.tanh(); case SIGMOID -> input.sigmoid();
            default -> throw new AssertionError(kind);
        };
    }

    /**
     * @param prediction [1,1] prediction
     * @param target matching fresh target, not part of the persisted source
     * @return scalar squared-error objective using ordinary non-source SUB and SUM
     */
    private static Tensor objective(Tensor prediction, Tensor target) {
        Tensor error = prediction.sub(target); return error.mul(error).sum();
    }

    /**
     * @param shape static logical geometry matching lanes
     * @param gradient gradient eligibility
     * @param lanes row-major FLOAT32 initial values
     * @return mutable canonical heap leaf
     */
    private static Tensor values(Shape shape, boolean gradient, float... lanes) {
        Tensor tensor = TensorFactory.allocate(new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), gradient));
        for (int i = 0; i < lanes.length; i++) tensor.hostStorage().orElseThrow().segment()
                .set(ValueLayout.JAVA_FLOAT_UNALIGNED, i * 4L, lanes[i]);
        return tensor;
    }

    /**
     * @param tensor live stored value
     * @return exact native-order state bytes, not a numerical arithmetic oracle
     */
    private static byte[] bits(Tensor tensor) {
        return tensor.hostStorage().orElseThrow().segment().toArray(ValueLayout.JAVA_BYTE);
    }
}
