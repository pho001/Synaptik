package io.github.pho001.synaptik.training;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.Optional;

/** Native scalar problem and detached-state assertions shared by focused fresh-value tests. */
final class TrainingValueFixture {
    /** Prevents instantiation of the shared native-value fixture functions. */
    private TrainingValueFixture() { }

    /**
     * Creates one stored scalar, without sharing storage with other fixture calls.
     * @param arena caller-owned lifetime
     * @param value initial FLOAT32 value
     * @param grad whether the descriptor requests differentiation
     * @return fresh scalar Tensor borrowing the allocated native storage
     */
    static Tensor scalar(Arena arena, float value, boolean grad) {
        var shape = Shape.scalar();
        var descriptor = new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), grad);
        MemorySegment segment = arena.allocate(4, 4);
        segment.set(ValueLayout.JAVA_FLOAT, 0, value);
        return stored(descriptor, segment);
    }

    /**
     * Associates exact borrowed FLOAT32 storage without inferring layout or logical capacity.
     * @param descriptor exact descriptor, possibly unresolved or larger than storage
     * @param segment live exact physical region
     * @return fresh Tensor borrowing that segment
     */
    static Tensor stored(TensorDescriptor descriptor, MemorySegment segment) {
        return TensorFactory.create(descriptor, Optional.empty(), Optional.of(
                new MemorySegmentStorage(DataType.FLOAT32, segment.byteSize() / 4, segment)));
    }

    /**
     * Builds the scalar squared prediction error used by the guide and CPU regression.
     * @param weight fixed parameter
     * @param batch external stored input prototype
     * @param target external stored target prototype
     * @return scalar expression (weight * batch - target)^2
     */
    static Tensor objective(Tensor weight, Tensor batch, Tensor target) {
        Tensor error = weight.mul(batch).sub(target);
        return error.mul(error);
    }

    /**
     * Orders a batch/target pair using public final compiled metadata, not assumed graph order.
     * @param session open session over the two prototypes
     * @param batchPrototype exact original batch
     * @param targetPrototype exact original target
     * @param batch stored replacement batch
     * @param target stored replacement target
     * @return two values in session.inputs order
     */
    static List<Tensor> values(TrainingSession session, Tensor batchPrototype,
            Tensor targetPrototype, Tensor batch, Tensor target) {
        return session.inputs().stream().map(input -> {
            if (input.tensorId().equals(batchPrototype.id())) return batch;
            assertEquals(targetPrototype.id(), input.tensorId());
            return target;
        }).toList();
    }

    /**
     * Reads a scalar parameter without materializing an expression.
     * @param tensor stored FLOAT32 scalar
     * @return current physical value
     */
    static float value(Tensor tensor) {
        return tensor.hostStorage().orElseThrow().segment().get(ValueLayout.JAVA_FLOAT, 0);
    }

    /**
     * Inspects an existing private field without adding a production testing seam.
     * @param owner exact object declaring the field
     * @param name declared field name
     * @return current field value, possibly null
     * @throws ReflectiveOperationException if the named field is unavailable
     */
    static Object field(Object owner, String name) throws ReflectiveOperationException {
        Field field = owner.getClass().getDeclaredField(name);
        field.setAccessible(true);
        return field.get(owner);
    }

    /**
     * Counts Engine-retained results to prove synchronous failure/success result cleanup.
     * @param engine borrowed Engine with no unrelated active calls
     * @return number of currently registered internal results
     * @throws ReflectiveOperationException if the existing lifecycle fields change
     */
    static int openResults(Engine engine) throws ReflectiveOperationException {
        return ((List<?>) field(field(engine, "delegate"), "openResults")).size();
    }

    /**
     * Compares all detached continuation state, including adaptive slots and master bytes.
     * @param expected immutable baseline
     * @param actual immutable observed snapshot
     */
    static void assertState(TrainingState expected, TrainingState actual) {
        assertEquals(expected.optimizer(), actual.optimizer());
        assertEquals(expected.parameterGroups(), actual.parameterGroups());
        assertEquals(expected.executions(), actual.executions());
        assertEquals(expected.optimizerSteps(), actual.optimizerSteps());
        assertEquals(expected.accumulatedGradientRuns(), actual.accumulatedGradientRuns());
        assertEquals(expected.parameters().size(), actual.parameters().size());
        for (int i = 0; i < expected.parameters().size(); i++) {
            var left = expected.parameters().get(i);
            var right = actual.parameters().get(i);
            assertEquals(left.path(), right.path());
            assertEquals(left.dataType(), right.dataType());
            assertEquals(left.shape(), right.shape());
            assertArrayEquals(bytes(left.parameterBytes()), bytes(right.parameterBytes()));
            assertArrayEquals(bytes(left.masterParameterBytes()), bytes(right.masterParameterBytes()));
            assertArrayEquals(bytes(left.firstMomentBytes()), bytes(right.firstMomentBytes()));
            assertArrayEquals(bytes(left.accumulatedGradientBytes()), bytes(right.accumulatedGradientBytes()));
            assertEquals(left.secondMomentBytes().isPresent(), right.secondMomentBytes().isPresent());
            if (left.secondMomentBytes().isPresent()) {
                assertArrayEquals(bytes(left.secondMomentBytes().orElseThrow()),
                        bytes(right.secondMomentBytes().orElseThrow()));
            }
        }
    }

    /**
     * Copies a detached read-only payload for byte-exact assertions.
     * @param buffer independent canonical-byte view
     * @return exact remaining bytes
     */
    private static byte[] bytes(ByteBuffer buffer) {
        byte[] bytes = new byte[buffer.remaining()];
        buffer.get(bytes);
        return bytes;
    }
}
