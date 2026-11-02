package io.github.pho001.synaptik.backend.cpu;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class CpuInternalPackageInventoryTest {
    @Test void adapterRepresentationCreatorsInitializeExactConstantsAndRemainFresh() {
        var entry = new io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan.BufferEntry(
                new io.github.pho001.synaptik.runtime.memory.BufferSlot(0), 16, 8);
        var creator = io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes
                .initializedBuffer(io.github.pho001.synaptik.model.datatype.DataType.FLOAT32,
                        entry, io.github.pho001.synaptik.model.datatype.ScalarValue.float32(-0.0f));
        try (var first = (io.github.pho001.synaptik.backend.cpu.internal.memory.CpuNativeBuffer)
                        creator.create(entry.byteSize(), entry.byteAlignment());
                var second = (io.github.pho001.synaptik.backend.cpu.internal.memory.CpuNativeBuffer)
                        creator.create(entry.byteSize(), entry.byteAlignment())) {
            assertNotSame(first, second);
            for (long index = 0; index < 4; index++) {
                assertEquals(Float.floatToRawIntBits(-0.0f), Float.floatToRawIntBits(
                        first.segment().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT,
                                index)));
            }
        }
        var workspace = new io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan.WorkspaceEntry(
                new io.github.pho001.synaptik.runtime.memory.WorkspaceSlot(0), 7, 16);
        var workspaceCreator = io.github.pho001.synaptik.backend.cpu.internal.memory
                .CpuRepresentationRecipes.workspace(workspace);
        try (var first = workspaceCreator.create(workspace.byteSize(), workspace.byteAlignment());
                var second = workspaceCreator.create(workspace.byteSize(), workspace.byteAlignment())) {
            assertNotSame(first, second);
        }
    }

}
