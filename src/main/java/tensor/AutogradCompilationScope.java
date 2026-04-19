package tensor;

import java.util.IdentityHashMap;
import java.util.Map;

/**
 * Thread-local compile-time autograd scope.
 *
 * While active, gradient bindings are captured outside the semantic Tensor objects so backward-graph
 * assembly does not leak compile-time state into the user-visible graph.
 */
public final class AutogradCompilationScope implements AutoCloseable {
    private static final ThreadLocal<AutogradCompilationScope> CURRENT = new ThreadLocal<>();

    private final AutogradCompilationScope previous;
    private final IdentityHashMap<Tensor, Tensor> gradients = new IdentityHashMap<>();
    private boolean closed;

    private AutogradCompilationScope(AutogradCompilationScope previous) {
        this.previous = previous;
    }

    public static AutogradCompilationScope open() {
        AutogradCompilationScope scope = new AutogradCompilationScope(CURRENT.get());
        CURRENT.set(scope);
        return scope;
    }

    static AutogradCompilationScope current() {
        return CURRENT.get();
    }

    Tensor gradientOf(Tensor tensor) {
        return gradients.get(tensor);
    }

    void setGradient(Tensor tensor, Tensor gradient) {
        if (gradient == null) {
            gradients.remove(tensor);
            return;
        }
        gradients.put(tensor, gradient);
    }

    public Map<Tensor, Tensor> snapshot() {
        if (gradients.isEmpty()) {
            return Map.of();
        }
        return Map.copyOf(new IdentityHashMap<>(gradients));
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        CURRENT.set(previous);
    }
}
