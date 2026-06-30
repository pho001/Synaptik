package runtime.residency;

import tensor.storage.NativeTensorStorage;

import java.util.HashMap;
import java.util.Map;

/**
 * Run-scoped native CPU storage bindings by compiled node id.
 */
public final class NativeCpuStorageRegistry {
    private final Map<Integer, NativeTensorStorage> nativeStorageByNodeId = new HashMap<>();

    public NativeTensorStorage get(int nodeId) {
        return nativeStorageByNodeId.get(nodeId);
    }

    public void put(int nodeId, NativeTensorStorage storage) {
        nativeStorageByNodeId.put(nodeId, storage);
    }

    public void remove(int nodeId) {
        nativeStorageByNodeId.remove(nodeId);
    }

    public void clear() {
        nativeStorageByNodeId.clear();
    }
}
