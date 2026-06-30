package backend.cpu1;

import backend.cpu1.exec.Cpu1ExecutableUnit;
import backend.cpu1.exec.Cpu1ElementwiseMemorySegmentExecutableUnit;
import backend.cpu1.exec.Cpu1ProviderCache;
import backend.cpu1.exec.Cpu1ScratchBuffer;
import backend.cpu1.exec.Cpu1ScratchBufferSpec;
import backend.cpu1.kernels.Cpu1KernelRegistry;
import backend.cpu1.kernels.Cpu1LayoutKind;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.launch.Cpu1SingleThreadLaunch;
import backend.cpu1.plan.Cpu1IterationPlan;
import backend.cpu1.prepare.Cpu1PreparedArtifact;
import backend.cpu1.prepare.Cpu1PreparedElementwiseUnit;
import backend.cpu1.storage.Cpu1StorageKind;
import runtime.execution.ExecutionContext;
import runtime.execution.PreparedRuntimeStateAllocator;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.lang.foreign.MemorySegment;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class Cpu1ScratchBufferTest {

    @Test
    void noneSpecAllocatesTypedEmptyWorkspace() {
        Cpu1ScratchBufferSpec spec = Cpu1ScratchBufferSpec.none();

        assertTrue(spec.isEmpty());
        Cpu1ScratchBuffer scratchBuffer = Cpu1ScratchBuffer.allocate(spec);
        assertSame(spec, scratchBuffer.spec());
        assertThrows(IllegalStateException.class, scratchBuffer::requireF32Array);
    }

    @Test
    void scratchBufferAllocatesExactArrayAndSegmentCapacities() {
        Cpu1ScratchBufferSpec spec = new Cpu1ScratchBufferSpec(7, 5, 3, 128L, true);

        Cpu1ScratchBuffer scratchBuffer = Cpu1ScratchBuffer.allocate(spec);

        assertSame(spec, scratchBuffer.spec());
        assertEquals(7, scratchBuffer.requireF32Array(7).length);
        assertEquals(5, scratchBuffer.requireF64Array(5).length);
        assertEquals(3, scratchBuffer.requireI32Array(3).length);
        assertEquals(128L, scratchBuffer.requireSegment(128L).byteSize());
        assertNotNull(scratchBuffer.providerCache());
    }

    @Test
    void scratchBufferRejectsUnavailableOrTooSmallScratch() {
        Cpu1ScratchBuffer scratchBuffer = Cpu1ScratchBuffer.allocate(Cpu1ScratchBufferSpec.arrays(2, 0, 0));

        assertThrows(IllegalStateException.class, () -> scratchBuffer.requireF32Array(3));
        assertThrows(IllegalStateException.class, scratchBuffer::requireF64Array);
        assertThrows(IllegalStateException.class, scratchBuffer::requireI32Array);
        assertThrows(IllegalStateException.class, scratchBuffer::requireSegment);
        assertThrows(IllegalStateException.class, scratchBuffer::providerCache);
    }

    @Test
    void scratchBufferSpecRejectsNegativeSizesAndOversizedHeapSegment() {
        assertThrows(IllegalArgumentException.class, () -> new Cpu1ScratchBufferSpec(-1, 0, 0, 0L, false));
        assertThrows(IllegalArgumentException.class, () -> new Cpu1ScratchBufferSpec(0, -1, 0, 0L, false));
        assertThrows(IllegalArgumentException.class, () -> new Cpu1ScratchBufferSpec(0, 0, -1, 0L, false));
        assertThrows(IllegalArgumentException.class, () -> new Cpu1ScratchBufferSpec(0, 0, 0, -1L, false));
        assertThrows(IllegalArgumentException.class, () -> new Cpu1ScratchBufferSpec(0, 0, 0, (long) Integer.MAX_VALUE + 1L, false));
    }

    @Test
    void memorySegmentElementwiseAllocatesTypedEmptyWorkspace() {
        Cpu1KernelRegistry registry = new Cpu1KernelRegistry();
        Cpu1PreparedElementwiseUnit unit = new Cpu1PreparedElementwiseUnit(
                42,
                List.of(1),
                42,
                Operation.OpType.RELU,
                DataType.FLOAT32,
                Cpu1IterationPlan.contiguous(6, new int[]{6}),
                Cpu1LayoutKind.CONTIGUOUS,
                Cpu1StorageKind.MEMORY_SEGMENT,
                registry.resolve(
                        Operation.OpType.RELU,
                        DataType.FLOAT32,
                        Cpu1LayoutKind.CONTIGUOUS,
                        Cpu1StorageKind.MEMORY_SEGMENT,
                        Cpu1VectorizationKind.SCALAR
                ),
                new Cpu1SingleThreadLaunch()
        );
        Cpu1PreparedArtifact artifact = new Cpu1PreparedArtifact(unit);
        RecordingAllocator allocator = new RecordingAllocator();

        artifact.allocateRuntimeState(42, allocator);

        Cpu1ElementwiseMemorySegmentExecutableUnit executable = assertInstanceOf(
                Cpu1ElementwiseMemorySegmentExecutableUnit.class,
                artifact.executableUnit()
        );
        assertSame(unit, executable.preparedUnit());
        assertTrue(artifact.scratchBufferSpec().isEmpty());
        assertInstanceOf(Cpu1ScratchBuffer.class, allocator.workspaces.get(42));
    }

    @Test
    void providerCacheStoresRunScopedValues() {
        Cpu1ProviderCache cache = new Cpu1ProviderCache();
        Object key = new Object();
        Object value = new Object();

        cache.put(key, value);

        assertSame(value, cache.get(key));
        assertSame(value, cache.computeIfAbsent(key, Object::new));
    }

    @Test
    void preparedArtifactAllocatesTypedWorkspaceForDefaultExecutable() {
        RecordingAllocator allocator = new RecordingAllocator();
        Cpu1PreparedArtifact artifact = new Cpu1PreparedArtifact(new NoopExecutable());

        artifact.allocateRuntimeState(42, allocator);

        assertInstanceOf(Cpu1ScratchBuffer.class, allocator.workspaces.get(42));
    }

    @Test
    void preparedArtifactAllocatesScratchBufferFromExecutableSpec() {
        RecordingAllocator allocator = new RecordingAllocator();
        Cpu1ScratchBufferSpec spec = new Cpu1ScratchBufferSpec(4, 0, 2, 64L, true);
        Cpu1PreparedArtifact artifact = new Cpu1PreparedArtifact(new ScratchBufferExecutable(spec));

        artifact.allocateRuntimeState(42, allocator);

        Object scratchBufferObject = allocator.workspaces.get(42);
        Cpu1ScratchBuffer scratchBuffer = assertInstanceOf(Cpu1ScratchBuffer.class, scratchBufferObject);
        assertEquals(4, scratchBuffer.requireF32Array(4).length);
        assertEquals(2, scratchBuffer.requireI32Array(2).length);
        MemorySegment segment = scratchBuffer.requireSegment(64L);
        assertEquals(64L, segment.byteSize());
        assertNotNull(scratchBuffer.providerCache());
    }

    private static final class NoopExecutable implements Cpu1ExecutableUnit {
        @Override
        public void run(ExecutionContext context) {
        }
    }

    private static final class ScratchBufferExecutable implements Cpu1ExecutableUnit {
        private final Cpu1ScratchBufferSpec scratchBufferSpec;

        private ScratchBufferExecutable(Cpu1ScratchBufferSpec scratchBufferSpec) {
            this.scratchBufferSpec = scratchBufferSpec;
        }

        @Override
        public Cpu1ScratchBufferSpec scratchBufferSpec() {
            return scratchBufferSpec;
        }

        @Override
        public void run(ExecutionContext context) {
        }
    }

    private static final class RecordingAllocator implements PreparedRuntimeStateAllocator {
        private final Map<Integer, Object> workspaces = new HashMap<>();

        @Override
        public Object forkWorkspace(Object template, Supplier<?> forkFactory) {
            return forkFactory.get();
        }

        @Override
        public void putWorkspace(int nodeId, Object scratchBuffer) {
            workspaces.put(nodeId, scratchBuffer);
        }

        @Override
        public void putPreparedInputTensor(int nodeId, int inputIndex, Tensor tensor) {
        }
    }
}
