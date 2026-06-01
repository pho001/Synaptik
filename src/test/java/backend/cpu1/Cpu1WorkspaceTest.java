package backend.cpu1;

import backend.cpu1.exec.Cpu1ExecutableUnit;
import backend.cpu1.exec.Cpu1ProviderCache;
import backend.cpu1.exec.Cpu1Workspace;
import backend.cpu1.exec.Cpu1WorkspaceSpec;
import backend.cpu1.prepare.Cpu1PreparedArtifact;
import backend.runtime.ExecutionContext;
import graph.execution.plan.PreparedRuntimeStateAllocator;
import org.junit.jupiter.api.Test;
import tensor.Tensor;

import java.lang.foreign.MemorySegment;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class Cpu1WorkspaceTest {

    @Test
    void noneSpecIsEmptyAndDoesNotAllocate() {
        Cpu1WorkspaceSpec spec = Cpu1WorkspaceSpec.none();

        assertTrue(spec.isEmpty());
        assertThrows(IllegalArgumentException.class, () -> Cpu1Workspace.allocate(spec));
    }

    @Test
    void workspaceAllocatesExactArrayAndSegmentCapacities() {
        Cpu1WorkspaceSpec spec = new Cpu1WorkspaceSpec(7, 5, 3, 128L, true);

        Cpu1Workspace workspace = Cpu1Workspace.allocate(spec);

        assertSame(spec, workspace.spec());
        assertEquals(7, workspace.requireF32Array(7).length);
        assertEquals(5, workspace.requireF64Array(5).length);
        assertEquals(3, workspace.requireI32Array(3).length);
        assertEquals(128L, workspace.requireSegment(128L).byteSize());
        assertNotNull(workspace.providerCache());
    }

    @Test
    void workspaceRejectsUnavailableOrTooSmallScratch() {
        Cpu1Workspace workspace = Cpu1Workspace.allocate(Cpu1WorkspaceSpec.arrays(2, 0, 0));

        assertThrows(IllegalStateException.class, () -> workspace.requireF32Array(3));
        assertThrows(IllegalStateException.class, workspace::requireF64Array);
        assertThrows(IllegalStateException.class, workspace::requireI32Array);
        assertThrows(IllegalStateException.class, workspace::requireSegment);
        assertThrows(IllegalStateException.class, workspace::providerCache);
    }

    @Test
    void workspaceSpecRejectsNegativeSizesAndOversizedHeapSegment() {
        assertThrows(IllegalArgumentException.class, () -> new Cpu1WorkspaceSpec(-1, 0, 0, 0L, false));
        assertThrows(IllegalArgumentException.class, () -> new Cpu1WorkspaceSpec(0, -1, 0, 0L, false));
        assertThrows(IllegalArgumentException.class, () -> new Cpu1WorkspaceSpec(0, 0, -1, 0L, false));
        assertThrows(IllegalArgumentException.class, () -> new Cpu1WorkspaceSpec(0, 0, 0, -1L, false));
        assertThrows(IllegalArgumentException.class, () -> new Cpu1WorkspaceSpec(0, 0, 0, (long) Integer.MAX_VALUE + 1L, false));
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
    void preparedArtifactDoesNotAllocateWorkspaceForDefaultExecutable() {
        RecordingAllocator allocator = new RecordingAllocator();
        Cpu1PreparedArtifact artifact = new Cpu1PreparedArtifact(new NoopExecutable());

        artifact.allocateRuntimeState(42, allocator);

        assertFalse(allocator.workspaces.containsKey(42));
    }

    @Test
    void preparedArtifactAllocatesWorkspaceFromExecutableSpec() {
        RecordingAllocator allocator = new RecordingAllocator();
        Cpu1WorkspaceSpec spec = new Cpu1WorkspaceSpec(4, 0, 2, 64L, true);
        Cpu1PreparedArtifact artifact = new Cpu1PreparedArtifact(new WorkspaceExecutable(spec));

        artifact.allocateRuntimeState(42, allocator);

        Object workspaceObject = allocator.workspaces.get(42);
        Cpu1Workspace workspace = assertInstanceOf(Cpu1Workspace.class, workspaceObject);
        assertEquals(4, workspace.requireF32Array(4).length);
        assertEquals(2, workspace.requireI32Array(2).length);
        MemorySegment segment = workspace.requireSegment(64L);
        assertEquals(64L, segment.byteSize());
        assertNotNull(workspace.providerCache());
    }

    private static final class NoopExecutable implements Cpu1ExecutableUnit {
        @Override
        public void run(ExecutionContext context) {
        }
    }

    private static final class WorkspaceExecutable implements Cpu1ExecutableUnit {
        private final Cpu1WorkspaceSpec workspaceSpec;

        private WorkspaceExecutable(Cpu1WorkspaceSpec workspaceSpec) {
            this.workspaceSpec = workspaceSpec;
        }

        @Override
        public Cpu1WorkspaceSpec workspaceSpec() {
            return workspaceSpec;
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
        public void putWorkspace(int nodeId, Object workspace) {
            workspaces.put(nodeId, workspace);
        }

        @Override
        public void putPreparedInputTensor(int nodeId, int inputIndex, Tensor tensor) {
        }
    }
}
