package backend.cpu1.exec;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.function.Supplier;

/**
 * Run-scoped cache for provider-prepared cpu1 data such as packed weights.
 */
public final class Cpu1ProviderCache {
    private final ConcurrentMap<Object, Object> values = new ConcurrentHashMap<>();

    public Object get(Object key) {
        if (key == null) {
            throw new IllegalArgumentException("key cannot be null");
        }
        return values.get(key);
    }

    public void put(Object key, Object value) {
        if (key == null) {
            throw new IllegalArgumentException("key cannot be null");
        }
        if (value == null) {
            throw new IllegalArgumentException("value cannot be null");
        }
        values.put(key, value);
    }

    public Object computeIfAbsent(Object key, Supplier<?> supplier) {
        if (key == null) {
            throw new IllegalArgumentException("key cannot be null");
        }
        if (supplier == null) {
            throw new IllegalArgumentException("supplier cannot be null");
        }
        return values.computeIfAbsent(key, ignored -> {
            Object value = supplier.get();
            if (value == null) {
                throw new IllegalArgumentException("cached value cannot be null");
            }
            return value;
        });
    }
}
