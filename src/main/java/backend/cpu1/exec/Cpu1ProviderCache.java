package backend.cpu1.exec;

import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.function.Supplier;

/**
 * Run-scoped cache for provider-prepared cpu1 data such as packed weights.
 */
public final class Cpu1ProviderCache {
    private final ConcurrentMap<Object, Object> values = new ConcurrentHashMap<>();

    public Object get(Object key) {
        return values.get(Objects.requireNonNull(key, "key cannot be null"));
    }

    public void put(Object key, Object value) {
        values.put(
                Objects.requireNonNull(key, "key cannot be null"),
                Objects.requireNonNull(value, "value cannot be null")
        );
    }

    public Object computeIfAbsent(Object key, Supplier<?> supplier) {
        Objects.requireNonNull(key, "key cannot be null");
        Objects.requireNonNull(supplier, "supplier cannot be null");
        return values.computeIfAbsent(key, ignored -> Objects.requireNonNull(supplier.get(), "cached value cannot be null"));
    }
}
