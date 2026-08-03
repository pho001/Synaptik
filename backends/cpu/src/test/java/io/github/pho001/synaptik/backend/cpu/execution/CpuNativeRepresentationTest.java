package io.github.pho001.synaptik.backend.cpu.execution;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.concurrent.Executors;
import org.junit.jupiter.api.Test;

class CpuNativeRepresentationTest {
    @Test void allocatesExactAlignedSharedNativeBufferIncludingZeroAndClosesOnce() throws Exception {
        var buffer = CpuNativeBuffer.allocate(DataType.FLOAT64, 64, 32);
        var zero = CpuNativeBuffer.allocate(DataType.BOOL, 0, 64);
        try (var executor = Executors.newSingleThreadExecutor()) {
            assertAll(
                    () -> assertEquals(64, buffer.byteSize()),
                    () -> assertEquals(0, buffer.segment().address() % 32),
                    () -> assertTrue(executor.submit(() -> buffer.segment().isAccessibleBy(
                            Thread.currentThread())).get()),
                    () -> assertEquals(0, zero.byteSize()),
                    () -> assertEquals(0, zero.segment().byteSize()),
                    () -> assertNotSame(buffer.segment().scope(), zero.segment().scope()));
        }
        try (var executor = Executors.newFixedThreadPool(2)) {
            executor.submit(buffer::close).get();
            executor.submit(buffer::close).get();
        }
        zero.close(); zero.close();
        assertAll(
                () -> assertTrue(buffer.isClosed()),
                () -> assertTrue(zero.isClosed()),
                () -> assertThrows(IllegalStateException.class, buffer::segment));
    }

    @Test void workspaceValidatesGeometryAndHasTheSameLifetimeRules() {
        assertEquals("byteSize must be non-negative", assertThrows(IllegalArgumentException.class,
                () -> CpuNativeWorkspace.allocate(-1, 0)).getMessage());
        assertEquals("byteAlignment must be a positive power of two",
                assertThrows(IllegalArgumentException.class,
                        () -> CpuNativeWorkspace.allocate(0, 3)).getMessage());
        var workspace = CpuNativeWorkspace.allocate(17, 16);
        assertAll(
                () -> assertEquals(17, workspace.byteSize()),
                () -> assertEquals(16, workspace.byteAlignment()),
                () -> assertEquals(0, workspace.segment().address() % 16));
        workspace.close(); workspace.close();
        assertTrue(workspace.isClosed());
    }

    @Test void bufferValidatesGeometryBeforeDataType() {
        assertEquals("byteSize must be non-negative", assertThrows(IllegalArgumentException.class,
                () -> CpuNativeBuffer.allocate(null, -1, 0)).getMessage());
        assertEquals("byteAlignment must be a positive power of two",
                assertThrows(IllegalArgumentException.class,
                        () -> CpuNativeBuffer.allocate(null, 0, 3)).getMessage());
        assertEquals("dataType", assertThrows(NullPointerException.class,
                () -> CpuNativeBuffer.allocate(null, 0, 1)).getMessage());
    }
}
