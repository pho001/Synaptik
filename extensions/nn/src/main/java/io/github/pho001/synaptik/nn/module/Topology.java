package io.github.pho001.synaptik.nn.module;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Short-lived named-child collector used while defining one functional {@link Model}.
 *
 * <p>An open topology retains exact name/module pairs in declaration order without attaching a
 * parent. The Model factory seals it after the definition callback on every success or failure
 * path. A sealed topology rejects every later registration and exposes no lookup, iteration,
 * removal, rename, or other structural mutation surface.</p>
 *
 * <p>The collector is callback-confined and not thread-safe. It defines module ownership and
 * stable state paths, not Tensor graph topology, parameter initialization or lazy binding,
 * automatic differentiation, training, compilation, or execution.</p>
 */
public final class Topology {
    private final Map<String, Module> modules = new LinkedHashMap<>();
    private final IdentityHashMap<Module, Boolean> identities = new IdentityHashMap<>();
    private boolean sealed;

    Topology() {
    }

    /**
     * Adds one exact module candidate under a descriptive local name.
     *
     * <p>This method validates facts available during collection and returns the exact supplied
     * module for strongly typed local use. It does not attach the module or call its forward
     * method. If the complete definition succeeds, the resulting Model permanently owns the
     * candidate under this name; if collection, the callback, its result, or final validation
     * fails, a previously unowned candidate remains unattached. Complete ownership and cycle
     * validation occurs atomically after the definition callback has returned a valid forward
     * body.</p>
     *
     * @param name non-null, non-blank local name without {@code .}; unique in this topology
     * @param module non-null exact module candidate whose identity is not already present
     * @param <M> the concrete module type preserved by the return value
     * @return the exact supplied {@code module}; never {@code null}
     * @throws IllegalStateException if this topology is already sealed; this check precedes all
     *     argument validation
     * @throws NullPointerException if {@code name} or {@code module} is {@code null}
     * @throws IllegalArgumentException if {@code name} is blank, contains {@code .}, or is
     *     already present, or if the exact module identity was already added
     */
    public <M extends Module> M addModule(String name, M module) {
        if (sealed) {
            throw new IllegalStateException("topology is sealed");
        }
        Objects.requireNonNull(name, "name");
        Objects.requireNonNull(module, "module");
        if (name.isBlank()) {
            throw new IllegalArgumentException("name must not be blank");
        }
        if (name.contains(".")) {
            throw new IllegalArgumentException("name must not contain '.'");
        }
        if (modules.containsKey(name)) {
            throw new IllegalArgumentException("module name is already declared: " + name);
        }
        if (identities.containsKey(module)) {
            throw new IllegalArgumentException("module identity is repeated");
        }
        modules.put(name, module);
        identities.put(module, Boolean.TRUE);
        return module;
    }

    void seal() {
        sealed = true;
    }

    Map<String, Module> snapshot() {
        return Collections.unmodifiableMap(new LinkedHashMap<>(modules));
    }
}
