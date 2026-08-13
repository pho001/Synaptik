package io.github.pho001.synaptik.nn.module;

import io.github.pho001.synaptik.model.tensor.Tensor;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.IdentityHashMap;
import java.util.Iterator;
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
 * buffers, then its children. Module instances are mutable through {@link #train()},
 * {@link #eval()}, direct buffer replacement, and compatible parameter replacement and are not
 * thread-safe; callers must synchronize concurrent declaration, traversal, ownership, mode,
 * replacement, or forward construction when a consistent view matters. This type provides no
 * version counter, checkpoints, optimizer behavior, transaction across bindings, or execution
 * lifecycle.</p>
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
     * @param value the non-null floating, gradient-eligible tensor reference to retain exactly;
     *     this method neither copies nor evaluates it
     * @return the newly declared parameter, never {@code null}
     * @throws NullPointerException if {@code name} or {@code value} is {@code null}
     * @throws IllegalArgumentException if {@code name} is blank, contains {@code .}, or already
     *     names direct state or a child, or if {@code value} is not floating or does not have
     *     {@code requiresGrad == true}
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
     * Replaces the current Tensor binding of one parameter declared directly by this module.
     *
     * <p>The supplied local name never traverses a child or interprets a dot path. This method
     * first rejects a null {@code name}, then a null {@code value}, then a name that is not a
     * direct parameter of this module; only then does it delegate to
     * {@link Parameter#replace(Tensor)}. A direct buffer with the requested name is not a
     * parameter. The parameter validates its declaration-time exact data type, structural Shape,
     * and gradient eligibility while deliberately allowing different Tensor identity, layout,
     * storage, provenance, and label. A successful replacement retains the exact supplied Tensor
     * without copying or evaluation. It neither replaces the {@link Parameter} wrapper nor
     * changes discovery order, child ownership, or mode.</p>
     *
     * <p>This individual operation is not a versioned snapshot, checkpoint, or transaction with
     * other bindings. It is not thread-safe; callers must externally synchronize concurrent
     * replacement, declaration, traversal, mode changes, and forward construction when they need
     * a consistent view.</p>
     *
     * @param name the non-null local name of a parameter directly declared by this module
     * @param value the non-null exact Tensor reference to become that parameter's current binding
     * @throws NullPointerException if {@code name} is null, or if {@code value} is null after a
     *     non-null {@code name}
     * @throws IllegalArgumentException if {@code name} does not identify a direct parameter of
     *     this module, including when it identifies a direct buffer, or if {@code value} has an
     *     incompatible data type, Shape, or gradient-eligibility value
     */
    protected final void replaceParameter(String name, Tensor value) {
        Objects.requireNonNull(name, "name");
        Objects.requireNonNull(value, "value");
        Parameter parameter = parameters.get(name);
        if (parameter == null) {
            throw new IllegalArgumentException("no direct parameter declared with name: " + name);
        }
        parameter.replace(value);
    }

    /**
     * Replaces the current Tensor binding of one buffer declared directly by this module.
     *
     * <p>The supplied local name never traverses a child or interprets a dot path. This method
     * first rejects a null {@code name}, then a null {@code value}, then a name that is not a
     * direct buffer of this module; only then does it replace the current binding. A direct
     * parameter with the requested name is not a buffer. Replacement retains the exact supplied
     * Tensor reference without copying or evaluation. Because this module declares no binding
     * schema, it performs no descriptor, data-type, shape, layout, provenance,
     * gradient-eligibility, or storage compatibility validation. It neither replaces the
     * {@link Buffer} wrapper nor changes discovery order, child ownership, or mode.</p>
     *
     * <p>This individual operation is not a versioned snapshot, checkpoint, or transaction with
     * other bindings. It is not thread-safe; callers must externally synchronize concurrent
     * replacement, declaration, traversal, mode changes, and forward construction when they need
     * a consistent view.</p>
     *
     * @param name the non-null local name of a buffer directly declared by this module
     * @param value the non-null exact Tensor reference to become that buffer's current binding
     * @throws NullPointerException if {@code name} is null, or if {@code value} is null after a
     *     non-null {@code name}
     * @throws IllegalArgumentException if {@code name} does not identify a direct buffer of this
     *     module, including when it identifies a direct parameter
     */
    protected final void replaceBuffer(String name, Tensor value) {
        Objects.requireNonNull(name, "name");
        Objects.requireNonNull(value, "value");
        Buffer buffer = buffers.get(name);
        if (buffer == null) {
            throw new IllegalArgumentException("no direct buffer declared with name: " + name);
        }
        buffer.replaceValue(value);
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
     * @return an unmodifiable structural snapshot of direct parameter wrappers; never {@code
     *     null}. A wrapper's {@link Parameter#value()} observes its current binding when read.
     */
    public final List<Parameter> parameters() {
        return List.copyOf(parameters.values());
    }

    /**
     * Returns this module's direct persistent non-trainable declarations in declaration order.
     *
     * @return an unmodifiable structural snapshot of direct buffer wrappers; never {@code null}.
     *     A wrapper's {@link Buffer#value()} observes its current binding when read.
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
     * {@link Parameter} objects, not copies or historical Tensor bindings. The map is a
     * structural snapshot: a wrapper it contains observes its current binding when
     * {@link Parameter#value()} is later called. Traversal uses an explicit stack rather than the
     * Java call stack and has no arbitrary depth limit. Defensive identity tracking rejects a
     * malformed cycle or shared child before any snapshot is returned.</p>
     *
     * @return an unmodifiable insertion-ordered snapshot from relative parameter path to exact
     *     parameter; never {@code null}
     * @throws IllegalStateException if defensive traversal encounters a repeated module identity
     */
    public final Map<String, Parameter> parametersRecursively() {
        Map<String, Parameter> result = new LinkedHashMap<>();
        collectParameters(result);
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
     * objects, not copies or historical Tensor bindings. The map is a structural snapshot: a
     * wrapper it contains observes its current binding when {@link Buffer#value()} is later
     * called. Traversal uses an explicit stack rather than the Java call stack and has no
     * arbitrary depth limit. Defensive identity tracking rejects a malformed cycle or shared
     * child before any snapshot is returned.</p>
     *
     * @return an unmodifiable insertion-ordered snapshot from relative buffer path to exact
     *     buffer; never {@code null}
     * @throws IllegalStateException if defensive traversal encounters a repeated module identity
     */
    public final Map<String, Buffer> buffersRecursively() {
        Map<String, Buffer> result = new LinkedHashMap<>();
        collectBuffers(result);
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
     * Verification uses an explicit stack rather than the Java call stack and has no arbitrary
     * depth limit.
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
     * Verification uses an explicit stack rather than the Java call stack and has no arbitrary
     * depth limit.
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

    private void changeModeRecursively(ForwardMode requestedMode) {
        List<Module> modules = collectOwnedModules();
        modules.forEach(module -> module.mode = requestedMode);
    }

    private void collectParameters(Map<String, Parameter> result) {
        traverseState((module, pathSegments) -> module.parameters.forEach(
                (name, parameter) -> result.put(path(pathSegments, name), parameter)));
    }

    private void collectBuffers(Map<String, Buffer> result) {
        traverseState((module, pathSegments) -> module.buffers.forEach(
                (name, buffer) -> result.put(path(pathSegments, name), buffer)));
    }

    private void traverseState(StateVisitor visitor) {
        IdentityHashMap<Module, Boolean> visited = new IdentityHashMap<>();
        List<String> pathSegments = new ArrayList<>();
        Deque<TraversalFrame> stack = new ArrayDeque<>();

        visitOnce(this, visited);
        visitor.visit(this, pathSegments);
        stack.push(new TraversalFrame(this, null));

        while (!stack.isEmpty()) {
            TraversalFrame frame = stack.peek();
            if (frame.children.hasNext()) {
                Map.Entry<String, Module> childEntry = frame.children.next();
                Module child = childEntry.getValue();
                visitOnce(child, visited);
                pathSegments.add(childEntry.getKey());
                visitor.visit(child, pathSegments);
                stack.push(new TraversalFrame(child, childEntry.getKey()));
            } else {
                TraversalFrame completed = stack.pop();
                if (completed.incomingName != null) {
                    pathSegments.removeLast();
                }
            }
        }
    }

    private List<Module> collectOwnedModules() {
        List<Module> result = new ArrayList<>();
        IdentityHashMap<Module, Boolean> visited = new IdentityHashMap<>();
        Deque<TraversalFrame> stack = new ArrayDeque<>();

        visitOnce(this, visited);
        result.add(this);
        stack.push(new TraversalFrame(this, null));

        while (!stack.isEmpty()) {
            TraversalFrame frame = stack.peek();
            if (frame.children.hasNext()) {
                Module child = frame.children.next().getValue();
                visitOnce(child, visited);
                result.add(child);
                stack.push(new TraversalFrame(child, null));
            } else {
                stack.pop();
            }
        }
        return result;
    }

    private static void visitOnce(Module module, IdentityHashMap<Module, Boolean> visited) {
        if (visited.put(module, Boolean.TRUE) != null) {
            throw new IllegalStateException("owned module tree contains a repeated module identity");
        }
    }

    private static String path(List<String> pathSegments, String localName) {
        if (pathSegments.isEmpty()) {
            return localName;
        }
        int length = localName.length() + pathSegments.size();
        for (String pathSegment : pathSegments) {
            length += pathSegment.length();
        }
        StringBuilder qualifiedPath = new StringBuilder(length);
        for (String pathSegment : pathSegments) {
            qualifiedPath.append(pathSegment).append('.');
        }
        return qualifiedPath.append(localName).toString();
    }

    private static <T> Map<String, T> immutableSnapshot(Map<String, T> source) {
        return Collections.unmodifiableMap(new LinkedHashMap<>(source));
    }

    @FunctionalInterface
    private interface StateVisitor {
        void visit(Module module, List<String> pathSegments);
    }

    private static final class TraversalFrame {
        private final Iterator<Map.Entry<String, Module>> children;
        private final String incomingName;

        private TraversalFrame(Module module, String incomingName) {
            this.children = module.children.entrySet().iterator();
            this.incomingName = incomingName;
        }
    }
}
