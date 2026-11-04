package io.github.pho001.synaptik.nn.module;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.initialization.ParameterInitializers;
import io.github.pho001.synaptik.nn.layers.BatchNorm;
import io.github.pho001.synaptik.nn.layers.LayerNorm;
import io.github.pho001.synaptik.training.Sgd;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.random.RandomGeneratorFactory;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** End-to-end low carriers through existing Model/Engine/Training/Checkpoint owners. */
final class LowPrecisionOwnedModelTest {
    @TempDir Path directory;
    private static final List<DataType> LOW = List.of(DataType.BFLOAT16, DataType.FLOAT16,
            DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2);

    @Test
    void repeatedTrainSaveLoadAndFreshTrainingKeepLowStorageAndBorrowedEngine() throws Exception {
        for (DataType type : LOW) {
            Path file = directory.resolve(type.name());
            Tensor weight = ParameterInitializers.ones(Shape.scalar(), type);
            Tensor x = TensorFactory.create(descriptor(type, false));
            Tensor target = TensorFactory.zeros(Shape.scalar(), DataType.FLOAT32, Optional.empty(), false);
            Tensor input = TensorFactory.ones(Shape.scalar(), type, Optional.empty(), false);
            byte[] saved;
            byte[] prediction;
            try (Engine engine = Engine.standard()) {
                try (var model = Model.graph(List.of(x), List.of(weight.mul(x)),
                        List.of(new GraphCheckpoint.Leaf("weight", GraphCheckpoint.Role.PARAMETER, weight)))) {
                    assertSame(model, model.withEngine(engine));
                    assertEquals(type, model.compute(input).dataType());
                    Tensor owned = model.stateDictionary().entries().getFirst().value();
                    AtomicInteger calls = new AtomicInteger();
                    model.configureTraining(target.descriptor(), (p, t) -> {
                        calls.incrementAndGet();
                        Tensor error = p.cast(DataType.FLOAT32).sub(t);
                        return error.mul(error);
                    }, new Sgd(0.125));
                    Object store = owned.hostStorage().orElseThrow();
                    for (int step = 1; step <= 3; step++) {
                        assertEquals(step, model.train(input, target).executionNumber());
                        assertEquals(type, owned.descriptor().dataType());
                        assertSame(store, owned.hostStorage().orElseThrow());
                    }
                    assertEquals(1, calls.get());
                    assertThrows(IllegalStateException.class, () -> model.withEngine(engine));
                    saved = owned.hostStorage().orElseThrow().segment().toArray(ValueLayout.JAVA_BYTE);
                    prediction = bytes(model.compute(input));
                    model.save(file);
                }
                assertFalse(engine.isClosed());
                try (var loaded = Model.load(file).withEngine(engine)) {
                    Tensor restored = loaded.stateDictionary().entries().getFirst().value();
                    assertEquals(type, restored.descriptor().dataType());
                    assertArrayEquals(saved, restored.hostStorage().orElseThrow().segment()
                            .toArray(ValueLayout.JAVA_BYTE));
                    assertArrayEquals(prediction, bytes(loaded.compute(input)));
                    loaded.configureTraining(target.descriptor(), (p, t) -> {
                        Tensor error = p.cast(DataType.FLOAT32).sub(t);
                        return error.mul(error);
                    }, new Sgd(0.125));
                    assertEquals(1, loaded.train(input, target).executionNumber());
                    assertFalse(java.util.Arrays.equals(saved, restored.hostStorage().orElseThrow()
                            .segment().toArray(ValueLayout.JAVA_BYTE)));
                }
                assertFalse(engine.isClosed());
            }
        }
    }

    @Test
    void lowObjectiveRejectionLeavesStorageUnchangedAndPermitsExplicitCastRetry() {
        for (DataType type : LOW) {
            Tensor weight = ParameterInitializers.ones(Shape.scalar(), type);
            Tensor x = TensorFactory.create(descriptor(type, false));
            Tensor target = TensorFactory.zeros(Shape.scalar(), type, Optional.empty(), false);
            try (var model = Model.graph(List.of(x), List.of(weight.mul(x)),
                    List.of(new GraphCheckpoint.Leaf("weight", GraphCheckpoint.Role.PARAMETER, weight)))) {
                Tensor owned = model.stateDictionary().entries().getFirst().value();
                Object before = owned.hostStorage().orElseThrow();
                assertThrows(IllegalArgumentException.class, () -> model.configureTraining(
                        target.descriptor(), Tensor::sub, new Sgd(0.125)));
                assertSame(before, owned.hostStorage().orElseThrow());
                assertThrows(IllegalStateException.class, () -> model.withEngine(null));
                model.configureTraining(target.descriptor(),
                        (p, t) -> p.sub(t).cast(DataType.FLOAT32), new Sgd(0.125));
                assertEquals(1, model.train(TensorFactory.ones(Shape.scalar(), type,
                        Optional.empty(), false), target).executionNumber());
            }
        }
    }

    @Test
    void initializersAndBuilderRetainEveryDeclaredLowCarrier() {
        for (DataType type : LOW) {
            for (Tensor initialized : List.of(ParameterInitializers.zeros(Shape.of(2, 2), type),
                    ParameterInitializers.ones(Shape.of(2, 2), type),
                    ParameterInitializers.glorotUniform(Shape.of(2, 2), type,
                            RandomGeneratorFactory.of("L64X128MixRandom").create(7)),
                    ParameterInitializers.kaimingReluNormal(Shape.of(2, 2), type,
                            RandomGeneratorFactory.of("L64X128MixRandom").create(7)))) {
                assertEquals(type, initialized.descriptor().dataType());
                assertEquals(4 * type.byteWidth(), initialized.hostStorage().orElseThrow().byteSize());
                assertTrue(initialized.descriptor().requiresGrad());
                assertTrue(initialized.provenance().isEmpty());
            }
            try (var model = Model.builder().input(type, 1, 2).linear(2).tanh().build()) {
                for (var entry : model.stateDictionary().entries()) {
                    assertEquals(type, entry.value().descriptor().dataType());
                }
                Tensor input = TensorFactory.ones(Shape.of(1, 2), type, Optional.empty(), false);
                HostTensorValue result = model.compute(input);
                assertEquals(type, result.dataType());
                assertEquals(2 * type.byteWidth(), result.byteSize());
            }
        }
    }

    @Test
    void fp8NormalizationRequiresFloat32MetadataWithoutChangingParameterType() {
        for (DataType type : List.of(DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2)) {
            ScalarValue eps = ScalarValue.float32(1e-5f);
            var layer = new LayerNorm(Shape.of(2), type, eps);
            assertEquals(type, layer.scale().value().descriptor().dataType());
            new ModelSpecification.LayerNormStep(Shape.of(2), type, eps);
            var batch = new BatchNorm(2, 1, type, ScalarValue.float32(0.25f), eps);
            assertEquals(type, batch.scale().value().descriptor().dataType());
            ScalarValue low = type == DataType.FLOAT8_E4M3
                    ? ScalarValue.float8E4m3(0.25f) : ScalarValue.float8E5m2(0.25f);
            assertThrows(IllegalArgumentException.class, () -> new LayerNorm(Shape.of(2), type, low));
            assertThrows(IllegalArgumentException.class,
                    () -> new BatchNorm(2, 1, type, low, low));
        }
    }

    private static TensorDescriptor descriptor(DataType type, boolean grad) {
        return new TensorDescriptor(type, Shape.scalar(),
                Optional.of(LayoutDescriptor.contiguous(Shape.scalar())), grad);
    }

    private static byte[] bytes(HostTensorValue value) {
        ByteBuffer source = value.bytes();
        byte[] bytes = new byte[source.remaining()];
        source.get(bytes);
        return bytes;
    }
}
