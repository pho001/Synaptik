package runtime.state;

import runtime.memory.nativecpu.NativeCpuAllocator;
import runtime.memory.nativecpu.NativeCpuMemoryPool;
import runtime.memory.nativecpu.NativeCpuMemoryStats;
import runtime.memory.nativecpu.NativeCpuStorageFactory;
import runtime.memory.ExecutionResource;
import config.runtime.NativeCpuMemoryConfig;
import trace.execution.NativeCpuMemoryTrace;
import tensor.DataType;
import tensor.storage.NativeTensorStorage;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Run-scoped owned native/backend resources.
 */
public final class RuntimeResourceRegistry {
    private final List<ExecutionResource> executionResources = new ArrayList<>();
    private final Set<ExecutionResource> registeredResources = Collections.newSetFromMap(new IdentityHashMap<>());
    private NativeCpuAllocator nativeCpuAllocator = new NativeCpuAllocator();
    private NativeCpuStorageFactory nativeCpuStorageFactory = new NativeCpuStorageFactory(nativeCpuAllocator);

    public void configureNativeCpuMemory(NativeCpuMemoryConfig config, NativeCpuMemoryPool preparedPool) {
        nativeCpuAllocator = new NativeCpuAllocator(config, new NativeCpuMemoryStats(), preparedPool);
        nativeCpuStorageFactory = new NativeCpuStorageFactory(nativeCpuAllocator);
    }

    public NativeTensorStorage allocateNativeStorage(DataType dataType, int elements, String label) {
        return nativeCpuStorageFactory.allocate(dataType, elements, label);
    }

    public void registerResource(ExecutionResource resource) {
        Objects.requireNonNull(resource, "resource cannot be null");
        if (registeredResources.add(resource)) {
            executionResources.add(resource);
        }
    }

    public void closeResources() {
        RuntimeException closeFailure = null;
        for (int i = executionResources.size() - 1; i >= 0; i--) {
            try {
                executionResources.get(i).close();
            } catch (RuntimeException ex) {
                if (closeFailure == null) {
                    closeFailure = new RuntimeException("One or more execution resources failed to close.");
                }
                closeFailure.addSuppressed(ex);
            }
        }
        executionResources.clear();
        registeredResources.clear();
        nativeCpuAllocator.drainRunLocalPool();
        if (closeFailure != null) {
            throw closeFailure;
        }
    }

    public NativeCpuMemoryTrace nativeCpuMemoryTrace() {
        var snapshot = nativeCpuAllocator.statsSnapshot();
        return new NativeCpuMemoryTrace(
                snapshot.allocationCount(),
                snapshot.releaseCount(),
                snapshot.retainCount(),
                snapshot.allocationFailureCount(),
                nativeCpuAllocator.requestedPoolPolicy().name(),
                nativeCpuAllocator.effectivePoolPolicy().name(),
                snapshot.requestedBytes(),
                snapshot.allocatedBytes(),
                snapshot.currentLiveBytes(),
                snapshot.peakLiveBytes(),
                snapshot.retainedBytes(),
                snapshot.poolHitCount(),
                snapshot.poolMissCount(),
                snapshot.pooledBytes(),
                snapshot.reusedBytes(),
                snapshot.discardedBytes(),
                snapshot.wastedBytes()
        );
    }
}
