package backend.cpu.kernels.elementwise.logical;

import backend.cpu.kernels.elementwise.ElementwiseOffsetCursor;
import backend.cpu.storage.CpuStorageView;
import operations.Operation;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;

public final class CpuLogicalAndKernel extends StorageAwareLogicalBinaryElementwiseKernel {
    @Override
    protected Operation.OpType opType() {
        return Operation.OpType.LOGICAL_AND;
    }

    @Override
    protected String opLabel() {
        return "logical_and";
    }

    @Override
    protected void runArray(byte[] left, byte[] right, byte[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = apply(left[i], right[i]);
        }
    }

    @Override
    protected void runSegment(MemorySegment left, MemorySegment right, MemorySegment out, int start, int end) {
        for (int i = start; i < end; i++) {
            out.set(JAVA_BYTE, i, apply(left.get(JAVA_BYTE, i), right.get(JAVA_BYTE, i)));
        }
    }

    @Override
    protected void runIndexedArray(
            byte[] left,
            byte[] right,
            byte[] out,
            LogicalBinaryStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            out[cursor.offset(0)] = apply(left[cursor.offset(1)], right[cursor.offset(2)]);
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    @Override
    protected void runIndexedSegment(
            MemorySegment left,
            MemorySegment right,
            MemorySegment out,
            LogicalBinaryStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            out.set(JAVA_BYTE, cursor.offset(0), apply(
                    left.get(JAVA_BYTE, cursor.offset(1)),
                    right.get(JAVA_BYTE, cursor.offset(2))
            ));
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    @Override
    protected void runIndexedMixed(
            CpuStorageView left,
            CpuStorageView right,
            CpuStorageView out,
            LogicalBinaryStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            writeBool(out, cursor.offset(0), apply(readBool(left, cursor.offset(1)), readBool(right, cursor.offset(2))));
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    private static byte apply(byte left, byte right) {
        return bool(left != 0 && right != 0);
    }
}
