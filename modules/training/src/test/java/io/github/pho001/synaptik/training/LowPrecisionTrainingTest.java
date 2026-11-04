package io.github.pho001.synaptik.training;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.Float16Bits;
import io.github.pho001.synaptik.model.datatype.Float8E4M3Bits;
import io.github.pho001.synaptik.model.datatype.Float8E5M2Bits;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.Test;

/** Logical low storage, FLOAT32 optimizer state and all-or-nothing installation. */
final class LowPrecisionTrainingTest {
    private static final List<DataType> LOW = List.of(DataType.BFLOAT16, DataType.FLOAT16,
            DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2);

    @Test
    void repeatedUpdatesUseFloatMastersAndRestoreEveryOptimizerStateExactly() {
        for (DataType type : LOW) for (Optimizer optimizer : List.of(
                new Sgd(0.01, 0.5, 0, false), new Adam(0.01, 0.5, 0.25, 0.01, 0),
                new AdamW(0.01, 0.5, 0.25, 0.01, 0.1))) {
            try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
                Tensor value = scalar(arena, type, encode(type, 2));
                var source = new TestParameterSource();
                source.parameter("value", value);
                try (var session = TrainingSession.open(engine, source,
                        value.cast(DataType.FLOAT32), optimizer)) {
                    TrainingState initial = session.state();
                    for (int step = 0; step < 4; step++) {
                        session.run(List.of(), GradientMode.RESET_AND_STEP);
                        var saved = session.state().parameters().getFirst();
                        assertEquals(4, saved.masterParameterBytes().remaining());
                        assertEquals(4, saved.momentumBytes().remaining());
                        float master = saved.masterParameterBytes().getFloat();
                        int actual = type.byteWidth() == 1
                                ? saved.parameterBytes().get() & 0xff
                                : saved.parameterBytes().getShort() & 0xffff;
                        assertEquals(encode(type, master), actual);
                    }
                    session.run(List.of(), GradientMode.ACCUMULATE);
                    TrainingState pending = session.state();
                    session.run(List.of(), GradientMode.ACCUMULATE_AND_STEP);
                    TrainingState advanced = session.state();
                    session.restore(pending);
                    assertState(pending, session.state());
                    session.run(List.of(), GradientMode.ACCUMULATE_AND_STEP);
                    assertState(advanced, session.state());
                    session.restore(initial);
                    assertState(initial, session.state());
                }
            }
        }
    }

    @Test
    void e4m3FiniteOverflowTieRetainsUnclippedMasterAndRejectsNextStep() {
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            Tensor value = scalar(arena, DataType.FLOAT8_E4M3, 0x7e);
            var source = new TestParameterSource();
            source.parameter("value", value);
            try (var session = TrainingSession.open(engine, source,
                    value.cast(DataType.FLOAT32).neg(), new Sgd(16))) {
                session.run(List.of(), GradientMode.RESET_AND_STEP);
                TrainingState before = session.state();
                var saved = before.parameters().getFirst();
                assertEquals(464.0f, saved.masterParameterBytes().getFloat());
                assertEquals((byte) 0x7e, saved.parameterBytes().get());
                assertThrows(ArithmeticException.class,
                        () -> session.run(List.of(), GradientMode.RESET_AND_STEP));
                assertState(before, session.state());
            }
        }
    }

    @Test
    void fp8RepresentedOverflowRejectsBeforeAnyParameterOrSlotPublication() {
        for (DataType type : List.of(DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2)) {
            int maximum = type == DataType.FLOAT8_E4M3 ? 0x7e : 0x7b;
            double rate = type == DataType.FLOAT8_E4M3 ? 32 : 8192;
            for (Optimizer optimizer : List.of(new Sgd(rate, 0.5, 0, false),
                    new Adam(rate, 0, 0, 0.01, 0), new AdamW(rate, 0, 0, 0.01, 0))) {
                try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
                    Tensor first = scalar(arena, type, encode(type, 2));
                    Tensor last = scalar(arena, type, maximum);
                    var source = new TestParameterSource();
                    source.parameter("first", first);
                    source.parameter("last", last);
                    Tensor objective = first.cast(DataType.FLOAT32).add(last.cast(DataType.FLOAT32)).neg();
                    try (var session = TrainingSession.open(engine, source, objective, optimizer)) {
                        session.run(List.of(), GradientMode.ACCUMULATE);
                        TrainingState before = session.state();
                        assertThrows(ArithmeticException.class,
                                () -> session.run(List.of(), GradientMode.RESET_AND_STEP));
                        assertState(before, session.state());
                        assertEquals(maximum, last.hostStorage().orElseThrow().segment()
                                .get(ValueLayout.JAVA_BYTE, 0) & 0xff);
                    }
                }
            }
        }
    }

    @Test
    void fp8WriteFailureRestoresRawSignedZeroMastersSlotsAndAllowsRetry() {
        for (DataType type : List.of(DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2)) {
            try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
                Tensor first = scalar(arena, type, 0x80);
                Tensor last = scalar(arena, type, encode(type, 1));
                var source = new TestParameterSource();
                source.parameter("first", first);
                source.parameter("last", last);
                AtomicBoolean fail = new AtomicBoolean(true);
                var hooks = new TrainingSession.Hooks() {
                    @Override void beforeParameterWrite(int index) {
                        if (index == 1 && fail.getAndSet(false)) throw new IllegalStateException("write");
                    }
                };
                try (var session = TrainingSession.openForTesting(engine, source,
                        first.cast(DataType.FLOAT32).add(last.cast(DataType.FLOAT32)),
                        new AdamW(0.01, 0.5, 0.25, 0.01, 0), hooks)) {
                    TrainingState before = session.state();
                    assertThrows(IllegalStateException.class,
                            () -> session.run(List.of(), GradientMode.RESET_AND_STEP));
                    assertState(before, session.state());
                    assertEquals((byte) 0x80, first.hostStorage().orElseThrow().segment()
                            .get(ValueLayout.JAVA_BYTE, 0));
                    session.run(List.of(), GradientMode.RESET_AND_STEP);
                    assertEquals(1, session.optimizerSteps());
                }
            }
        }
    }

    @Test
    void fp8NonfiniteRawStorageAndRestoredLogicalBytesRejectTransactionally() {
        for (DataType type : List.of(DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2)) {
            List<Integer> invalid = type == DataType.FLOAT8_E4M3
                    ? List.of(0x7f, 0xff) : List.of(0x7c, 0xfc, 0x7d, 0xff);
            try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
                Tensor value = scalar(arena, type, encode(type, 1));
                var source = new TestParameterSource();
                source.parameter("value", value);
                assertThrows(IllegalArgumentException.class,
                        () -> TrainingSession.open(engine, source, value, new Sgd(0.01)));
                try (var session = TrainingSession.open(engine, source,
                        value.cast(DataType.FLOAT32), new Sgd(0.01))) {
                    TrainingState before = session.state();
                    var entry = before.parameters().getFirst();
                    for (int raw : invalid) {
                        var corrupt = new TrainingState.ParameterState(entry.path(), type, entry.shape(),
                                new byte[] {(byte) raw}, bytes(entry.masterParameterBytes()),
                                bytes(entry.momentumBytes()), bytes(entry.accumulatedGradientBytes()));
                        var state = new TrainingState(before.optimizer(), 0, 0, 0, List.of(corrupt));
                        assertThrows(IllegalArgumentException.class, () -> session.restore(state));
                        assertState(before, session.state());
                    }
                }
                for (int raw : invalid) {
                    Tensor bad = scalar(arena, type, raw);
                    var badSource = new TestParameterSource();
                    badSource.parameter("bad", bad);
                    assertThrows(IllegalArgumentException.class,
                            () -> TrainingSession.open(engine, badSource,
                                    bad.cast(DataType.FLOAT32), new Sgd(0.01)));
                }
            }
        }
    }

    @Test
    void finiteObjectiveWithOverflowedFp8GradientRejectsWithoutMutation() {
        for (DataType type : List.of(DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2)) {
            try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
                Tensor value = scalar(arena, type, encode(type, 1));
                var source = new TestParameterSource();
                source.parameter("value", value);
                Tensor scale = TensorFactory.allocate(new TensorDescriptor(DataType.FLOAT32,
                        Shape.scalar(), Optional.of(LayoutDescriptor.contiguous(Shape.scalar())), false));
                scale.hostStorage().orElseThrow().segment()
                        .set(ValueLayout.JAVA_FLOAT_UNALIGNED, 0, 1e8f);
                try (var session = TrainingSession.open(engine, source,
                        value.cast(DataType.FLOAT32).mul(scale), new Sgd(0.01))) {
                    TrainingState before = session.state();
                    assertThrows(ArithmeticException.class,
                            () -> session.run(List.of(scale), GradientMode.RESET_AND_STEP));
                    assertState(before, session.state());
                }
            }
        }
    }

    private static Tensor scalar(Arena arena, DataType type, int bits) {
        MemorySegment segment = arena.allocate(type.byteWidth(), type.byteWidth());
        if (type.byteWidth() == 1) segment.set(ValueLayout.JAVA_BYTE, 0, (byte) bits);
        else segment.set(ValueLayout.JAVA_SHORT, 0, (short) bits);
        var descriptor = new TensorDescriptor(type, Shape.scalar(),
                Optional.of(LayoutDescriptor.contiguous(Shape.scalar())), true);
        return TensorFactory.create(descriptor, Optional.empty(),
                Optional.of(new MemorySegmentStorage(type, 1, segment)));
    }

    private static int encode(DataType type, float value) {
        return switch (type) {
            case BFLOAT16 -> BFloat16Bits.fromFloat(value) & 0xffff;
            case FLOAT16 -> Float16Bits.fromFloat(value) & 0xffff;
            case FLOAT8_E4M3 -> Float8E4M3Bits.fromFloat(value) & 0xff;
            case FLOAT8_E5M2 -> Float8E5M2Bits.fromFloat(value) & 0xff;
            default -> throw new AssertionError(type);
        };
    }

    private static byte[] bytes(ByteBuffer buffer) {
        byte[] bytes = new byte[buffer.remaining()];
        buffer.get(bytes);
        return bytes;
    }

    private static void assertState(TrainingState expected, TrainingState actual) {
        assertEquals(expected.executions(), actual.executions());
        assertEquals(expected.optimizerSteps(), actual.optimizerSteps());
        assertEquals(expected.accumulatedGradientRuns(), actual.accumulatedGradientRuns());
        assertEquals(expected.parameters().size(), actual.parameters().size());
        for (int index = 0; index < expected.parameters().size(); index++) {
            var a = expected.parameters().get(index);
            var b = actual.parameters().get(index);
            assertArrayEquals(bytes(a.parameterBytes()), bytes(b.parameterBytes()));
            assertArrayEquals(bytes(a.masterParameterBytes()), bytes(b.masterParameterBytes()));
            assertArrayEquals(bytes(a.momentumBytes()), bytes(b.momentumBytes()));
            assertArrayEquals(bytes(a.accumulatedGradientBytes()), bytes(b.accumulatedGradientBytes()));
            assertEquals(a.secondMomentBytes().isPresent(), b.secondMomentBytes().isPresent());
            if (a.secondMomentBytes().isPresent()) assertArrayEquals(
                    bytes(a.secondMomentBytes().orElseThrow()), bytes(b.secondMomentBytes().orElseThrow()));
        }
    }
}
