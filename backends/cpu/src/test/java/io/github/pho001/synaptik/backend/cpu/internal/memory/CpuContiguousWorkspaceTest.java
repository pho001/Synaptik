package io.github.pho001.synaptik.backend.cpu.internal.memory;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

class CpuContiguousWorkspaceTest {
    @Test void ownsAlignedSharedSegmentAndSupportsZeroBytes() {
        var workspace = CpuContiguousWorkspace.allocate(32, 8);
        assertAll(() -> assertEquals(32, workspace.byteSize()),
                () -> assertEquals(0, workspace.writableSegment().address() % 8),
                () -> assertTrue(workspace.isAccessible()));
        workspace.close();
        workspace.close();
        assertAll(() -> assertFalse(workspace.isAccessible()),
                () -> assertThrows(IllegalStateException.class, workspace::writableSegment),
                () -> assertDoesNotThrow(() -> CpuContiguousWorkspace.allocate(0, 8).close()),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> CpuContiguousWorkspace.allocate(1, 3)));
    }
}
