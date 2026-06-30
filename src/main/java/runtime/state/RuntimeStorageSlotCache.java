package runtime.state;

import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.storage.BFloat16Storage;
import tensor.storage.BoolStorage;
import tensor.storage.Int32Storage;
import tensor.storage.Int64Storage;
import tensor.storage.NativeTensorStorage;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Run-scoped reusable storage cache keyed by planned runtime slot identity.
 */
public final class RuntimeStorageSlotCache {
    private final RuntimeResourceRegistry resourceRegistry;
    private final Map<RuntimeStorageSlotKey, Object> storageByKey = new HashMap<>();

    public RuntimeStorageSlotCache(RuntimeResourceRegistry resourceRegistry) {
        this.resourceRegistry = Objects.requireNonNull(resourceRegistry, "resourceRegistry cannot be null");
    }

    public void bindJavaStorage(Tensor tensor, RuntimeStorageSlotKey key) {
        Objects.requireNonNull(tensor, "tensor cannot be null");
        Object storage = javaStorage(key);
        if (tensor.getDataType() != key.dataType()) {
            throw new IllegalArgumentException("Runtime slot dtype mismatch. tensorType=" + tensor.getDataType()
                    + ", slotType=" + key.dataType());
        }
        if (tensor.getFlatDataSize() != key.elements()) {
            throw new IllegalArgumentException("Runtime slot size mismatch. tensorElements=" + tensor.getFlatDataSize()
                    + ", slotElements=" + key.elements());
        }
        switch (key.dataType()) {
            case FLOAT64 -> tensor.setData((double[]) storage);
            case FLOAT32 -> tensor.setFloat32Data((float[]) storage);
            case BFLOAT16 -> TensorInternalAccess.replaceStorage(tensor, new BFloat16Storage((short[]) storage));
            case INT32 -> TensorInternalAccess.replaceStorage(tensor, new Int32Storage((int[]) storage));
            case INT64 -> TensorInternalAccess.replaceStorage(tensor, new Int64Storage((long[]) storage));
            case BOOL -> TensorInternalAccess.replaceStorage(tensor, new BoolStorage((byte[]) storage));
        }
    }

    public NativeTensorStorage nativeCpuStorage(RuntimeStorageSlotKey key, String label) {
        requireKind(key, RuntimeStorageKind.NATIVE_CPU);
        Object storage = storageByKey.computeIfAbsent(key, ignored ->
                resourceRegistry.allocateNativeStorage(key.dataType(), key.elements(), label));
        if (!(storage instanceof NativeTensorStorage nativeStorage)) {
            throw new IllegalStateException("Runtime slot key does not hold native CPU storage: " + key);
        }
        if (nativeStorage.getType() != key.dataType() || nativeStorage.getSize() != key.elements()) {
            throw new IllegalStateException("Runtime native CPU slot storage is invalid. key=" + key
                    + ", storageType=" + nativeStorage.getType()
                    + ", storageElements=" + nativeStorage.getSize());
        }
        nativeStorage.ensureOpen();
        return nativeStorage;
    }

    private Object javaStorage(RuntimeStorageSlotKey key) {
        requireKind(key, RuntimeStorageKind.JAVA_ARRAY);
        return storageByKey.computeIfAbsent(key, RuntimeStorageSlotCache::allocateJavaStorage);
    }

    private static Object allocateJavaStorage(RuntimeStorageSlotKey key) {
        return switch (key.dataType()) {
            case FLOAT64 -> new double[key.elements()];
            case FLOAT32 -> new float[key.elements()];
            case BFLOAT16 -> new short[key.elements()];
            case INT32 -> new int[key.elements()];
            case INT64 -> new long[key.elements()];
            case BOOL -> new byte[key.elements()];
        };
    }

    private static void requireKind(RuntimeStorageSlotKey key, RuntimeStorageKind expected) {
        Objects.requireNonNull(key, "key cannot be null");
        if (key.kind() != expected) {
            throw new IllegalArgumentException("Runtime storage slot kind mismatch. expected="
                    + expected + ", actual=" + key.kind());
        }
    }
}
