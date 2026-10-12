package io.github.pho001.synaptik.training;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.ReadOnlyBufferException;
import java.util.List;
import org.junit.jupiter.api.Test;

final class TrainingStateOwnershipTest {
    @Test
    void freshSnapshotPathAdoptsAllFourExactArraysForEverySupportedType() {
        for (DataType type : List.of(
                DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
            byte[][] payloads = payloads(type);
            TrainingState.ParameterState state = construct(true, type, payloads);

            assertSame(payloads[0], state.parameterBytesInternal());
            assertSame(payloads[1], state.masterParameterBytesInternal());
            assertSame(payloads[2], state.momentumBytesInternal());
            assertSame(payloads[3], state.accumulatedGradientBytesInternal());
            assertViewsAreIndependentAndReadOnly(state, payloads);
        }
    }

    @Test
    void ordinaryConstructorCopiesAllFourCallerRetainedArrays() {
        byte[][] payloads = payloads(DataType.FLOAT16);
        TrainingState.ParameterState state = construct(false, DataType.FLOAT16, payloads);

        assertNotSame(payloads[0], state.parameterBytesInternal());
        assertNotSame(payloads[1], state.masterParameterBytesInternal());
        assertNotSame(payloads[2], state.momentumBytesInternal());
        assertNotSame(payloads[3], state.accumulatedGradientBytesInternal());
        for (byte[] payload : payloads) {
            payload[0] = 99;
        }
        assertEquals(1, state.parameterBytes().get(0));
        assertEquals(2, state.masterParameterBytes().get(0));
        assertEquals(3, state.momentumBytes().get(0));
        assertEquals(4, state.accumulatedGradientBytes().get(0));
    }

    @Test
    void bothPathsRejectNullAndInexactPayloads() {
        for (boolean adopt : List.of(false, true)) {
            for (int index = 0; index < 4; index++) {
                byte[][] missing = payloads(DataType.FLOAT16);
                missing[index] = null;
                assertThrows(NullPointerException.class,
                        () -> construct(adopt, DataType.FLOAT16, missing));

                byte[][] wrongWidth = payloads(DataType.FLOAT16);
                wrongWidth[index] = new byte[wrongWidth[index].length + 1];
                assertThrows(IllegalArgumentException.class,
                        () -> construct(adopt, DataType.FLOAT16, wrongWidth));
            }
        }
    }

    private static byte[][] payloads(DataType type) {
        int logicalBytes = 2 * type.byteWidth();
        int optimizerBytes = 2 * (type == DataType.FLOAT64 ? Double.BYTES : Float.BYTES);
        byte[][] payloads = {
                new byte[logicalBytes],
                new byte[optimizerBytes],
                new byte[optimizerBytes],
                new byte[optimizerBytes]
        };
        for (int index = 0; index < payloads.length; index++) {
            payloads[index][0] = (byte) (index + 1);
        }
        return payloads;
    }

    private static TrainingState.ParameterState construct(
            boolean adopt, DataType type, byte[][] payloads) {
        return adopt
                ? TrainingState.ParameterState.adoptFreshSnapshotArrays(
                        "weight", type, Shape.of(2),
                        payloads[0], payloads[1], payloads[2], payloads[3])
                : new TrainingState.ParameterState(
                        "weight", type, Shape.of(2),
                        payloads[0], payloads[1], payloads[2], payloads[3]);
    }

    private static void assertViewsAreIndependentAndReadOnly(
            TrainingState.ParameterState state, byte[][] payloads) {
        List<ByteBuffer> views = List.of(
                state.parameterBytes(),
                state.masterParameterBytes(),
                state.momentumBytes(),
                state.accumulatedGradientBytes());
        for (int index = 0; index < views.size(); index++) {
            ByteBuffer view = views.get(index);
            assertTrue(view.isReadOnly());
            assertFalse(view.hasArray());
            assertEquals(ByteOrder.BIG_ENDIAN, view.order());
            assertEquals(0, view.position());
            assertEquals(payloads[index].length, view.limit());
            assertArrayEquals(payloads[index], bytes(view));
            assertThrows(ReadOnlyBufferException.class, () -> view.put(0, (byte) 0));
            view.position(1);
        }
        assertEquals(0, state.parameterBytes().position());
        assertEquals(0, state.masterParameterBytes().position());
        assertEquals(0, state.momentumBytes().position());
        assertEquals(0, state.accumulatedGradientBytes().position());
    }

    private static byte[] bytes(ByteBuffer source) {
        byte[] copy = new byte[source.remaining()];
        source.get(copy);
        return copy;
    }
}
