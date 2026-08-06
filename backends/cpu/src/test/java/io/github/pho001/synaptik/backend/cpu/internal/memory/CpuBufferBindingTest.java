package io.github.pho001.synaptik.backend.cpu.internal.memory;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import org.junit.jupiter.api.Test;

class CpuBufferBindingTest {
    @Test void classifiesNativeMemoryAsExactSegmentWithoutCopy() {
        try (var buffer = CpuNativeBuffer.allocate(DataType.FLOAT64, 16, 8)) {
            var argument = assertInstanceOf(CpuBufferArgument.Segment.class, buffer.argument());
            assertAll(
                    () -> assertSame(buffer.segment(), argument.segment()),
                    () -> assertEquals(16, argument.byteSize()),
                    () -> assertFalse(argument.readOnly()));
        }
    }
}
