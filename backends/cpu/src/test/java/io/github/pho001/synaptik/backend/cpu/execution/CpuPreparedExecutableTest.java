package io.github.pho001.synaptik.backend.cpu.execution;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.runtime.execution.BoundInvocation;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable.*;
import io.github.pho001.synaptik.runtime.memory.*;
import io.github.pho001.synaptik.runtime.run.*;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.concurrent.Executors;
import org.junit.jupiter.api.Test;

class CpuPreparedExecutableTest {
    @Test void bindsMixedStorageToFreshTypedArraysAndInvocationRetainsDirectFields() {
        var plan = plan(16, 8, 16);
        var heap = CpuBorrowedBuffer.borrow(new MemorySegmentStorage(
                DataType.FLOAT32, 4, MemorySegment.ofArray(new float[4])));
        var nativeBuffer = CpuNativeBuffer.allocate(DataType.INT64, 8, 16);
        var workspace = CpuNativeWorkspace.allocate(8, 16);
        var state = new RunState(plan,
                List.of(List.of(borrowed(heap)), List.of(owned(nativeBuffer))), List.of(workspace));
        var executable = new ProbeExecutable(plan,
                List.of(new BufferSelection(0, 0), new BufferSelection(1, 0)),
                List.of(new WorkspaceSelection(0)),
                List.of(BufferAccess.READ_ONLY, BufferAccess.WRITE_ONLY),
                List.of(DataType.FLOAT32, DataType.INT64));
        ProbeInvocation invocation = (ProbeInvocation) executable.bind(state);
        invocation.execute();
        assertAll(
                () -> assertInstanceOf(CpuBufferArgument.Floats.class, invocation.input),
                () -> assertInstanceOf(CpuBufferArgument.Segment.class, invocation.output),
                () -> assertSame(workspace, invocation.workspace),
                () -> assertEquals(1, invocation.calls),
                () -> assertTrue(java.util.Arrays.stream(ProbeInvocation.class.getDeclaredFields())
                        .noneMatch(field -> field.getType().isArray())),
                () -> assertTrue(java.util.Arrays.stream(ProbeInvocation.class.getDeclaredFields())
                        .noneMatch(field -> io.github.pho001.synaptik.runtime.resource.BufferRepresentation.class
                                .isAssignableFrom(field.getType()))));
        state.close();
        assertAll(() -> assertFalse(heap.isClosed()), () -> assertTrue(nativeBuffer.isClosed()),
                () -> assertTrue(workspace.isClosed()));
    }

    @Test void rejectsReadOnlyWritesDataTypeSizeAndAlignmentThroughRuntimeMessage() {
        var plan = new PreparedMemoryPlan(List.of(new PreparedMemoryPlan.BufferEntry(
                new BufferSlot(1), 4, 8)), List.of());
        var readOnly = CpuBorrowedBuffer.borrow(new MemorySegmentStorage(DataType.FLOAT32, 1,
                MemorySegment.ofArray(new float[1]).asReadOnly()));
        var state = new RunState(plan, List.of(List.of(borrowed(readOnly))), List.of());
        var write = new ProbeExecutable(plan, List.of(new BufferSelection(0, 0)), List.of(),
                List.of(BufferAccess.WRITE_ONLY), List.of(DataType.FLOAT32));
        assertEquals("bufferSelections[0] is incompatible with prepared executable",
                assertThrows(IllegalArgumentException.class, () -> write.bind(state)).getMessage());
        state.close();
    }

    @Test void acceptsOpaqueReadOnlyHeapForReadsAndRejectsConfinedAccessFromAnotherThread()
            throws Exception {
        var plan = new PreparedMemoryPlan(List.of(new PreparedMemoryPlan.BufferEntry(
                new BufferSlot(1), 4, 1)), List.of());
        MemorySegment readOnlyHeap = MemorySegment.ofArray(new float[1]).asReadOnly();
        var opaque = CpuBorrowedBuffer.borrow(new MemorySegmentStorage(
                DataType.FLOAT32, 1, readOnlyHeap));
        var readState = new RunState(plan, List.of(List.of(borrowed(opaque))), List.of());
        var read = new ProbeExecutable(plan, List.of(new BufferSelection(0, 0)), List.of(),
                List.of(BufferAccess.READ_ONLY), List.of(DataType.FLOAT32));
        var invocation = (ProbeInvocation) read.bind(readState);
        assertSame(readOnlyHeap,
                assertInstanceOf(CpuBufferArgument.Segment.class, invocation.input).segment());
        readState.close();

        try (var arena = java.lang.foreign.Arena.ofConfined()) {
            var confined = CpuBorrowedBuffer.borrow(new MemorySegmentStorage(
                    DataType.FLOAT32, 1, arena.allocate(4, 4)));
            var state = new RunState(plan, List.of(List.of(borrowed(confined))), List.of());
            try (var executor = Executors.newSingleThreadExecutor()) {
                var failure = executor.submit(() -> assertThrows(IllegalArgumentException.class,
                        () -> read.bind(state))).get();
                assertEquals("bufferSelections[0] is incompatible with prepared executable",
                        failure.getMessage());
            }
            state.close();
        }
    }

    private static PreparedMemoryPlan plan(long firstSize, long secondSize, long alignment) {
        return new PreparedMemoryPlan(List.of(
                new PreparedMemoryPlan.BufferEntry(new BufferSlot(1), firstSize, 1),
                new PreparedMemoryPlan.BufferEntry(new BufferSlot(2), secondSize, alignment)),
                List.of(new PreparedMemoryPlan.WorkspaceEntry(new WorkspaceSlot(1), 8, alignment)));
    }
    private static BufferRepresentationBinding borrowed(CpuBufferRepresentation value) {
        return new BufferRepresentationBinding(value, RunResourceOwnership.BORROWED);
    }
    private static BufferRepresentationBinding owned(CpuBufferRepresentation value) {
        return new BufferRepresentationBinding(value, RunResourceOwnership.RUN_OWNED);
    }

    private static final class ProbeExecutable extends CpuPreparedExecutable {
        ProbeExecutable(PreparedMemoryPlan plan, List<BufferSelection> buffers,
                List<WorkspaceSelection> workspaces, List<BufferAccess> accesses,
                List<DataType> types) { super(plan, buffers, workspaces, accesses, types); }
        @Override protected BoundInvocation bindCpu(RunState state, CpuBufferArgument[] buffers,
                CpuNativeWorkspace[] workspaces) {
            return new ProbeInvocation(state, buffers[0], buffers.length > 1 ? buffers[1] : null,
                    workspaces.length > 0 ? workspaces[0] : null);
        }
    }
    private static final class ProbeInvocation extends BoundInvocation {
        final CpuBufferArgument input;
        final CpuBufferArgument output;
        final CpuNativeWorkspace workspace;
        int calls;
        ProbeInvocation(RunState state, CpuBufferArgument input, CpuBufferArgument output,
                CpuNativeWorkspace workspace) { super(state); this.input = input; this.output = output;
            this.workspace = workspace; }
        @Override protected void executeBound() { calls++; }
    }
}
