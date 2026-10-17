package io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBorrowedBuffer;
import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuConcurrencyBudget;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.memory.BufferSlot;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.run.BufferRepresentationBinding;
import io.github.pho001.synaptik.runtime.run.RunResourceOwnership;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class CpuOpenBlasPreparedExecutableTest {
    @Test void coordinatedBindingIsColdAndExecutionUsesOneFixedAdmission() {
        DataType type = DataType.FLOAT32;
        long bytes = 4L * type.byteWidth();
        var fake = new RecordingInvocation();
        var coordinator = new CpuOpenBlasCoordinator(fake, () -> fake.open = false,
                new CpuConcurrencyBudget(2));
        coordinator.configure(2);
        int queries = fake.queries;
        int sets = fake.sets;
        var memory = memory(type, bytes);
        var candidate = new CpuPartitionAnalysisInputs.OpenBlasRouteConfig.ThreadCandidate(2,
                CpuPartitionAnalysisInputs.CostTerms.complete(0, 0, 0));
        var route = new CpuOpenBlasRoutePlan(type, 2, 2, 2, 0, 1, 2, candidate,
                2, 2, 2, 0, CpuOpenBlasRoutePlan.Representation.DIRECT, Optional.empty(),
                Optional.empty(), Optional.empty(), List.of(), 1, 0, 0, 0,
                100, 10, 90, 9_000);
        var executable = new CpuOpenBlasPreparedExecutable(memory, selections(), List.of(), route,
                null, coordinator, List.of(), Optional.empty());
        try (Arena arena = Arena.ofConfined(); RunState state = state(memory, type,
                arena.allocate(bytes, type.byteWidth()), arena.allocate(bytes, type.byteWidth()),
                arena.allocate(bytes, type.byteWidth()))) {
            var bound = executable.bind(state);
            assertAll(() -> assertEquals(queries, fake.queries),
                    () -> assertEquals(sets, fake.sets));
            bound.execute();
            assertEquals(1, fake.sgemmCalls);
        } finally { coordinator.close(); }
    }

    @Test void bindsExactNativeSegmentsAndExecutesOneTypedCall() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            try (Arena arena = Arena.ofConfined()) {
                long bytes = 4L * type.byteWidth();
                MemorySegment left = arena.allocate(bytes, type.byteWidth());
                MemorySegment right = arena.allocate(bytes, type.byteWidth());
                MemorySegment output = arena.allocate(bytes, type.byteWidth());
                var fake = new RecordingInvocation();
                var executable = executable(type, fake, bytes);
                try (RunState state = state(executable.memoryPlan(), type, left, right, output)) {
                    executable.bind(state).execute();
                }
                assertAll(() -> assertEquals(type == DataType.FLOAT32 ? 1 : 0, fake.sgemmCalls),
                        () -> assertEquals(type == DataType.FLOAT64 ? 1 : 0, fake.dgemmCalls),
                        () -> assertEquals(2, fake.m), () -> assertEquals(2, fake.n),
                        () -> assertEquals(2, fake.k), () -> assertSame(left, fake.left),
                        () -> assertSame(right, fake.right), () -> assertSame(output, fake.output),
                        () -> assertEquals(1.0, fake.alpha),
                        () -> assertEquals(0.0, fake.beta));
            }
        }
    }

    @Test void failsBeforeProviderCallForHeapOutputOverlapClosureOrThreadDrift() {
        DataType type = DataType.FLOAT32;
        long bytes = 4L * type.byteWidth();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment region = arena.allocate(bytes * 3, type.byteWidth());
            MemorySegment left = region.asSlice(0, bytes);
            MemorySegment right = region.asSlice(bytes, bytes);
            MemorySegment overlappingOutput = region.asSlice(0, bytes);
            var fake = new RecordingInvocation();
            var executable = executable(type, fake, bytes);
            try (RunState state = state(executable.memoryPlan(), type, left, right,
                    overlappingOutput)) {
                assertThrows(IllegalArgumentException.class, () -> executable.bind(state));
            }
            fake.open = false;
            try (RunState state = state(executable.memoryPlan(), type, left, right,
                    region.asSlice(bytes * 2, bytes))) {
                assertThrows(IllegalStateException.class, () -> executable.bind(state));
            }
            fake.open = true; fake.threads = 2;
            try (RunState state = state(executable.memoryPlan(), type, left, right,
                    region.asSlice(bytes * 2, bytes))) {
                assertThrows(IllegalStateException.class, () -> executable.bind(state));
            }
            fake.threads = 1;
            MemorySegment misaligned = arena.allocate(bytes + 1, 1).asSlice(1, bytes);
            try (RunState state = state(executable.memoryPlan(), type, misaligned, right,
                    region.asSlice(bytes * 2, bytes))) {
                assertThrows(IllegalArgumentException.class, () -> executable.bind(state));
            }
            try (RunState state = state(executable.memoryPlan(), type, left, right,
                    region.asSlice(bytes * 2, bytes).asReadOnly())) {
                assertThrows(IllegalArgumentException.class, () -> executable.bind(state));
            }
            assertEquals(0, fake.sgemmCalls + fake.dgemmCalls);
        }
        var heapFake = new RecordingInvocation();
        var executable = executable(type, heapFake, bytes);
        var left = CpuBorrowedBuffer.borrow(new MemorySegmentStorage(type, 4,
                MemorySegment.ofArray(new float[4])));
        var right = CpuBorrowedBuffer.borrow(new MemorySegmentStorage(type, 4,
                MemorySegment.ofArray(new float[4])));
        var output = CpuBorrowedBuffer.borrow(new MemorySegmentStorage(type, 4,
                MemorySegment.ofArray(new float[4])));
        try (RunState state = state(executable.memoryPlan(), left, right, output)) {
            assertThrows(IllegalArgumentException.class, () -> executable.bind(state));
        }
        assertEquals(0, heapFake.sgemmCalls);
    }

    @Test void propagatesProviderFailureWithoutFallbackOrRetry() {
        DataType type = DataType.FLOAT64;
        long bytes = 4L * type.byteWidth();
        try (Arena arena = Arena.ofConfined()) {
            var fake = new RecordingInvocation();
            fake.failure = new IllegalStateException("provider failure");
            var executable = executable(type, fake, bytes);
            try (RunState state = state(executable.memoryPlan(), type,
                    arena.allocate(bytes, type.byteWidth()),
                    arena.allocate(bytes, type.byteWidth()),
                    arena.allocate(bytes, type.byteWidth()))) {
                var bound = executable.bind(state);
                assertSame(fake.failure, assertThrows(IllegalStateException.class,
                        bound::execute));
            }
            assertEquals(1, fake.dgemmCalls);
        }
    }

    @Test void reusesOneImmutableRecipeAcrossDistinctSequentialRuns() {
        DataType type = DataType.FLOAT32;
        long bytes = 4L * type.byteWidth();
        var fake = new RecordingInvocation();
        var executable = executable(type, fake, bytes);
        try (Arena arena = Arena.ofConfined()) {
            for (int run = 0; run < 2; run++) {
                try (RunState state = state(executable.memoryPlan(), type,
                        arena.allocate(bytes, type.byteWidth()),
                        arena.allocate(bytes, type.byteWidth()),
                        arena.allocate(bytes, type.byteWidth()))) {
                    executable.bind(state).execute();
                }
            }
        }
        assertEquals(2, fake.sgemmCalls);
    }

    private static CpuOpenBlasPreparedExecutable executable(DataType type,
            CpuOpenBlasInvocation invocation, long bytes) {
        var memory = memory(type, bytes);
        var route = new CpuOpenBlasRoutePlan(type, 2, 2, 2, 0, 1, 2,
                new CpuPartitionAnalysisInputs.OpenBlasRouteConfig.ThreadCandidate(1,
                        CpuPartitionAnalysisInputs.CostTerms.complete(0, 0, 0)),
                CpuOpenBlasRoutePlan.Representation.DIRECT, Optional.empty(), Optional.empty(),
                100, 10, 90, 9_000);
        return new CpuOpenBlasPreparedExecutable(memory, selections(), List.of(), route, invocation);
    }

    private static PreparedMemoryPlan memory(DataType type, long bytes) {
        return new PreparedMemoryPlan(List.of(
                new PreparedMemoryPlan.BufferEntry(new BufferSlot(0), bytes, type.byteWidth()),
                new PreparedMemoryPlan.BufferEntry(new BufferSlot(1), bytes, type.byteWidth()),
                new PreparedMemoryPlan.BufferEntry(new BufferSlot(2), bytes, type.byteWidth())),
                List.of());
    }

    private static List<PreparedExecutable.BufferSelection> selections() {
        return List.of(
                new PreparedExecutable.BufferSelection(0, 0),
                new PreparedExecutable.BufferSelection(1, 0),
                new PreparedExecutable.BufferSelection(2, 0));
    }

    private static RunState state(PreparedMemoryPlan plan, DataType type, MemorySegment left,
            MemorySegment right, MemorySegment output) {
        return state(plan, borrowed(type, left), borrowed(type, right), borrowed(type, output));
    }

    private static RunState state(PreparedMemoryPlan plan, CpuBorrowedBuffer left,
            CpuBorrowedBuffer right, CpuBorrowedBuffer output) {
        return new RunState(plan, List.of(binding(left), binding(right), binding(output)),
                List.of());
    }

    private static CpuBorrowedBuffer borrowed(DataType type, MemorySegment segment) {
        return CpuBorrowedBuffer.borrow(new MemorySegmentStorage(type,
                segment.byteSize() / type.byteWidth(), segment));
    }

    private static List<BufferRepresentationBinding> binding(CpuBorrowedBuffer buffer) {
        return List.of(new BufferRepresentationBinding(buffer, RunResourceOwnership.BORROWED));
    }

    private static final class RecordingInvocation implements CpuOpenBlasInvocation {
        boolean open = true;
        int threads = 1;
        int sgemmCalls;
        int dgemmCalls;
        int m;
        int n;
        int k;
        double alpha;
        double beta;
        MemorySegment left;
        MemorySegment right;
        MemorySegment output;
        RuntimeException failure;
        int queries;
        int sets;

        @Override public boolean isOpen() { return open; }
        @Override public int threadCount() { queries++; return threads; }
        @Override public void setThreadCount(int threadCount) { sets++; threads = threadCount; }
        @Override public void sgemm(int m, int n, int k, float alpha, MemorySegment a,
                MemorySegment b, float beta, MemorySegment c) {
            sgemmCalls++; record(m, n, k, alpha, a, b, beta, c);
        }
        @Override public void dgemm(int m, int n, int k, double alpha, MemorySegment a,
                MemorySegment b, double beta, MemorySegment c) {
            dgemmCalls++; record(m, n, k, alpha, a, b, beta, c);
        }
        private void record(int m, int n, int k, double alpha, MemorySegment left,
                MemorySegment right, double beta, MemorySegment output) {
            this.m = m; this.n = n; this.k = k; this.alpha = alpha; this.beta = beta;
            this.left = left; this.right = right; this.output = output;
            if (failure != null) throw failure;
        }
    }
}
