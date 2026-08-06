package io.github.pho001.synaptik.backend.cpu.internal.memory;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import org.junit.jupiter.api.Test;

class CpuNativeRepresentationTest {
    @Test void ownsExactAlignedNativeLifetimeIncludingZeroSize() {
        var buffer = CpuNativeBuffer.allocate(DataType.FLOAT64, 0, 64);
        assertAll(
                () -> assertEquals(0, buffer.byteSize()),
                () -> assertEquals(64, buffer.byteAlignment()),
                () -> assertEquals(0, buffer.segment().address() % 64));
        buffer.close();
        assertThrows(IllegalStateException.class, buffer::segment);
        assertDoesNotThrow(buffer::close);
    }
}
