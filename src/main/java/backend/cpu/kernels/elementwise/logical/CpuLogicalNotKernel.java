package backend.cpu.kernels.elementwise.logical;

import backend.cpu.kernels.elementwise.ElementwiseOffsetCursor;
import backend.cpu.storage.CpuStorageView;
import operations.Operation;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;

public final class CpuLogicalNotKernel extends StorageAwareLogicalUnaryElementwiseKernel {
    @Override
    protected Operation.OpType opType() {
        return Operation.OpType.LOGICAL_NOT;
    }

    @Override
    protected String opLabel() {
        return "logical_not";
    }

    @Override
    protected void runArray(byte[] input, byte[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = apply(input[i]);
        }
    }

    @Override
    protected void runSegment(MemorySegment input, MemorySegment out, int start, int end) {
        for (int i = start; i < end; i++) {
            out.set(JAVA_BYTE, i, apply(input.get(JAVA_BYTE, i)));
        }
    }

    @Override
    protected void runIndexedArray(
            byte[] input,
            byte[] out,
            LogicalUnaryStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            out[cursor.offset(0)] = apply(input[cursor.offset(1)]);
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    @Override
    protected void runIndexedSegment(
            MemorySegment input,
            MemorySegment out,
            LogicalUnaryStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            out.set(JAVA_BYTE, cursor.offset(0), apply(input.get(JAVA_BYTE, cursor.offset(1))));
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    @Override
    protected void runIndexedMixed(
            CpuStorageView input,
            CpuStorageView out,
            LogicalUnaryStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            writeBool(out, cursor.offset(0), apply(readBool(input, cursor.offset(1))));
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    private static byte apply(byte value) {
        return bool(value == 0);
    }
}
