package io.github.pho001.synaptik.nn.module;

import io.github.pho001.synaptik.model.tensor.Tensor;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Owner of one module's direct state, exclusively owned child modules, and forward mode.
 *
 * <p>A subclass declares its direct state through {@link #parameter(String, Tensor)} and
 * {@link #buffer(String, Tensor)}, and {@link #child(String, Module)}, then uses retained tensor
 * references while constructing a layer-specific expression. A child is permanently owned by
 * exactly one parent. This foundation deliberately has no universal {@code forward} method: each
 * future layer defines a typed signature appropriate to its inputs and outputs.</p>
 *
 * <p>Parameters, buffers, and children share one local-name namespace. Direct declaration order
 * and child-registration order determine immutable discovery snapshots. Recursive state uses
 * dot-separated paths and depth-first traversal: each module contributes parameters, then
 * buffers, then its children. Module instances are mutable through {@link #train()} and
 * {@link #eval()} and are not thread-safe; callers must synchronize concurrent declaration,
 * traversal, ownership, or mode operations. This type provides neither binding replacement,
 * checkpoints, optimizer behavior, nor execution lifecycle.</p>
 */
public abstract class Module {
    private final Map<String, Parameter> parameters = new LinkedHashMap<>();
    private final Map<String, Buffer> buffers = new LinkedHashMap<>();
    private final Map<String, Module> children = new LinkedHashMap<>();
    private Module parent;
    private ForwardMode mode = ForwardMode.TRAINING;

    /**
     * Creates a module with no direct state and local {@link ForwardMode#TRAINING} mode.
     *
     * <p>Subclasses normally declare their direct parameters and buffers during construction.
     * Declaration does not evaluate a tensor expression or create optimizer, compiler, runtime,
     * or backend state.</p>
     */
    protected Module() {
    }

    /**
     * Declares a named trainable tensor binding owned directly by this module.
     *
     * @param name the non-null, non-blank local name without {@code .}; it must be unused by all
     *     direct parameters, buffers, and children of this module
     * @param value the non-null tensor reference to retain exactly; this method neither copies nor
     *     evaluates it
     * @return the newly declared parameter, never {@code null}
     * @throws NullPointerException if {@code name} or {@code value} is {@code null}
     * @throws IllegalArgumentException if {@code name} is blank, contains {@code .}, or already
     *     names direct state or a child
     */
    protected final Parameter parameter(String name, Tensor value) {
        validateAvailableName(name);
        Parameter parameter = new Parameter(name, Objects.requireNonNull(value, "value"));
        parameters.put(name, parameter);
        return parameter;
    }

    /**
     * Declares a named persistent non-trainable tensor binding owned directly by this module.
     *
     * @param name the non-null, non-blank local name without {@code .}; it must be unused by all
     *     direct parameters, buffers, and children of this module
     * @param value the non-null tensor reference to retain exactly; this method neither copies nor
     *     evaluates it
     * @return the newly declared buffer, never {@code null}
     * @throws NullPointerException if {@code name} or {@code value} is {@code null}
     * @throws IllegalArgumentException if {@code name} is blank, contains {@code .}, or already
     *     names direct state or a child
     */
    protected final Buffer buffer(String name, Tensor value) {
        validateAvailableName(name);
        Buffer buffer = new Buffer(name, Objects.requireNonNull(value, "value"));
        buffers.put(name, buffer);
        return buffer;
    }

    /**
     * Registers one named child that this module owns permanently.
     *
     * <p>The child must not already have a parent and must not be this module or an ancestor of
     * this module. Every validation completes before the name is installed or the child's parent
     * is set, so a failed registration leaves both modules unchanged. There is no detach, rename,
     * reparent, or shared-child operation.</p>
     *
     * @param name the non-null, non-blank local path segment without {@code .}; it must be unused
     *     by all direct parameters, buffers, and children of this module
     * @param child the non-null module to own exclusively; it must have no parent and must not
     *     create a self or ancestor cycle
     * @param <T> the concrete child type, retained to support typed subclass fields
     * @return {@code child}, after successful permanent registration; never {@code null}
     * @throws NullPointerException if {@code name} or {@code child} is {@code null}
     * @throws IllegalArgumentException if {@code name} is blank, contains {@code .}, or already
     *     names direct state or a child, or if {@code child} is this module or an ancestor of this
     *     module
     * @throws IllegalStateException if {@code child} is already owned by another module
     */
    protected final <T extends Module> T child(String name, T child) {
        validateAvailableName(name);
        Objects.requireNonNull(child, "child");
        Module moduleChild = child;
        if (moduleChild == this || isAncestor(moduleChild)) {
            throw new IllegalArgumentException("child must not be this module or an ancestor");
        }
        if (moduleChild.parent != null) {
            throw new IllegalStateException("child is already owned by a module");
        }
        children.put(name, moduleChild);
        moduleChild.parent = this;
        return child;
    }

    /**
     * Returns this module's direct trainable declarations in declaration order.
     *
     * @return an unmodifiable snapshot of direct parameters; never {@code null}
     */
    public final List<Parameter> parameters() {
        return List.copyOf(parameters.values());
    }

    /**
     * Returns this module's direct persistent non-trainable declarations in declaration order.
     *
     * @return an unmodifiable snapshot of direct buffers; never {@code null}
     */
    public final List<Buffer> buffers() {
        return List.copyOf(buffers.values());
    }

    /**
     * Returns this module's direct children in child-registration order.
     *
     * <p>The returned map is an independent, insertion-ordered, unmodifiable snapshot. Its keys
     * are local path segments and its values are the exact owned child instances; it does not
     * include descendants.</p>
     *
     * @return an unmodifiable insertion-ordered snapshot from local child name to exact child;
     *     never {@code null}
     */
    public final Map<String, Module> children() {
        return immutableSnapshot(children);
    }

    /**
     * Returns every reachable parameter under this module in deterministic depth-first order.
     *
     * <p>Each module contributes its direct parameters before its direct buffers and before its
     * children. This parameter-only result therefore visits a module's parameters, then descends
     * through children in registration order. Keys are paths relative to this receiving module:
     * a direct parameter uses its local name and a descendant parameter uses dot-separated child
     * names, such as {@code encoder.layer1.weight}. The result contains the exact declared
     * {@link Parameter} objects, not copies or replacement bindings.</p>
     *
     * @return an unmodifiable insertion-ordered snapshot from relative parameter path to exact
     *     parameter; never {@code null}
     */
    public final Map<String, Parameter> parametersRecursively() {
        Map<String, Parameter> result = new LinkedHashMap<>();
        collectParameters("", result);
        return immutableSnapshot(result);
    }

    /**
     * Returns every reachable buffer under this module in deterministic depth-first order.
     *
     * <p>Each module contributes its direct parameters before its direct buffers and before its
     * children. This buffer-only result therefore visits a module's buffers, then descends through
     * children in registration order. Keys are paths relative to this receiving module: a direct
     * buffer uses its local name and a descendant buffer uses dot-separated child names, such as
     * {@code encoder.layer1.runningMean}. The result contains the exact declared {@link Buffer}
     * objects, not copies or replacement bindings.</p>
     *
     * @return an unmodifiable insertion-ordered snapshot from relative buffer path to exact
     *     buffer; never {@code null}
     */
    public final Map<String, Buffer> buffersRecursively() {
        Map<String, Buffer> result = new LinkedHashMap<>();
        collectBuffers("", result);
        return immutableSnapshot(result);
    }

    /**
     * Returns this module's current local forward mode.
     *
     * @return the non-null current mode; it is this module's local field, which is changed with
     *     every owned descendant by {@link #train()} and {@link #eval()}
     */
    public final ForwardMode mode() {
        return mode;
    }

    /**
     * Returns an immutable snapshot of this module's current local forward mode.
     *
     * @return a non-null context representing the mode at this call; it is unaffected by later
     *     calls to {@link #train()} or {@link #eval()}
     */
    public final ForwardContext forwardContext() {
        return new ForwardContext(mode);
    }

    /**
     * Sets this module and every reachable owned descendant to {@link ForwardMode#TRAINING}.
     *
     * <p>The operation first verifies that the owned tree contains no repeated module identity.
     * If defensive verification detects a malformed cycle or shared child, it throws before
     * changing any mode. Otherwise it changes all collected modules in deterministic preorder.
     * The change affects only contexts obtained after this call. It has no Tensor mutation,
     * expression evaluation, or execution effect.</p>
     *
     * @throws IllegalStateException if the owned tree has a repeated module identity
     */
    public final void train() {
        changeModeRecursively(ForwardMode.TRAINING);
    }

    /**
     * Sets this module and every reachable owned descendant to {@link ForwardMode#EVALUATION}.
     *
     * <p>The operation first verifies that the owned tree contains no repeated module identity.
     * If defensive verification detects a malformed cycle or shared child, it throws before
     * changing any mode. Otherwise it changes all collected modules in deterministic preorder.
     * The change affects only contexts obtained after this call. It has no Tensor mutation,
     * expression evaluation, or execution effect.</p>
     *
     * @throws IllegalStateException if the owned tree has a repeated module identity
     */
    public final void eval() {
        changeModeRecursively(ForwardMode.EVALUATION);
    }

    private void validateAvailableName(String name) {
        Objects.requireNonNull(name, "name");
        if (name.isBlank()) {
            throw new IllegalArgumentException("name must not be blank");
        }
        if (name.contains(".")) {
            throw new IllegalArgumentException("name must not contain '.'");
        }
        if (parameters.containsKey(name) || buffers.containsKey(name) || children.containsKey(name)) {
            throw new IllegalArgumentException("direct module name is already declared: " + name);
        }
    }

    private boolean isAncestor(Module possibleAncestor) {
        for (Module module = this; module != null; module = module.parent) {
            if (module == possibleAncestor) {
                return true;
            }
        }
        return false;
    }

    private void collectParameters(String prefix, Map<String, Parameter> result) {
        parameters.forEach((name, parameter) -> result.put(path(prefix, name), parameter));
        children.forEach((name, child) -> child.collectParameters(path(prefix, name), result));
    }

    private void collectBuffers(String prefix, Map<String, Buffer> result) {
        buffers.forEach((name, buffer) -> result.put(path(prefix, name), buffer));
        children.forEach((name, child) -> child.collectBuffers(path(prefix, name), result));
    }

    private void changeModeRecursively(ForwardMode requestedMode) {
        List<Module> modules = new java.util.ArrayList<>();
        collectOwnedModules(modules, new IdentityHashMap<>());
        modules.forEach(module -> module.mode = requestedMode);
    }

    private void collectOwnedModules(List<Module> result, IdentityHashMap<Module, Boolean> visited) {
        if (visited.put(this, Boolean.TRUE) != null) {
            throw new IllegalStateException("owned module tree contains a repeated module identity");
        }
        result.add(this);
        children.values().forEach(child -> child.collectOwnedModules(result, visited));
    }

    private static String path(String prefix, String localName) {
        return prefix.isEmpty() ? localName : prefix + "." + localName;
    }

    private static <T> Map<String, T> immutableSnapshot(Map<String, T> source) {
        return Collections.unmodifiableMap(new LinkedHashMap<>(source));
    }
}
