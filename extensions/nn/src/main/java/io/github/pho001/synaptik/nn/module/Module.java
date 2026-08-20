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
import java.util.function.Consumer;

/**
 * Owner of one module's direct state, exclusively owned child modules, and forward mode.
 *
 * <p>A subclass declares its direct state through {@link #parameter(String, Tensor)} and
 * {@link #buffer(String, Tensor)}, and {@link #child(String, Module)}, then uses retained tensor
 * references while constructing a layer-specific expression. A concrete input-dependent layer
 * may instead reserve private future parameter names and publish their complete real wrappers as
 * one direct group. A child is permanently owned by exactly one parent. This foundation
 * deliberately has no universal {@code forward} method: subclasses choose truthful typed
 * signatures, while {@link UnaryTensorModule} names only the narrower one-Tensor-to-one-Tensor
 * case.</p>
 *
 * <p>Parameters, buffers, and children share one local-name namespace. Direct declaration order
 * and child-registration order determine immutable discovery snapshots. Recursive state uses
 * dot-separated paths and depth-first traversal: each module contributes parameters, then
 * buffers, then its children. Module instances are mutable through {@link #train()},
 * {@link #eval()}, direct buffer replacement, and compatible parameter replacement and are not
 * thread-safe; callers must synchronize concurrent declaration, traversal, ownership, mode,
 * replacement, state loading, or general forward construction when a consistent view matters.
 * Reserved direct-parameter publication uses one release/acquire completion gate, so a racing
 * accessor or discovery call observes an unbound failure or the complete direct published set,
 * never a partial direct set. This narrow guarantee does not make tree operations generally
 * thread-safe. This type provides no
 * version counter, persistent checkpoint format, optimizer behavior, or execution lifecycle.
 * {@link #loadStateDictionary(StateDictionary)} provides validate-before-install atomicity for
 * ordinary caller-coordinated schema failures, not a lock or a simultaneous snapshot for racing
 * readers.</p>
 */
public abstract class Module {
    private final Map<String, ParameterSlot> parameterDeclarations = new LinkedHashMap<>();
    private final Map<String, Buffer> buffers = new LinkedHashMap<>();
    private final Map<String, Module> children = new LinkedHashMap<>();
    private volatile boolean parameterReservationsBound = true;
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
        parameterDeclarations.put(name, ParameterSlot.bound(name, parameter));
        return parameter;
    }

    /**
     * Reserves one direct parameter name whose complete Tensor is supplied later by the subclass.
     *
     * <p>The validator is retained exactly and is called before any wrapper from the reservation
     * group is published. It must either return normally or throw without an externally visible
     * side effect. A reservation occupies the shared direct namespace and its declaration-order
     * position immediately, but it is not an incomplete {@link Parameter}.</p>
     *
     * @param name the non-null, non-blank unused local name without {@code .}
     * @param validator non-null deterministic validator for the eventual exact Tensor binding
     * @throws NullPointerException if {@code name} or {@code validator} is null
     * @throws IllegalArgumentException if {@code name} is invalid or unavailable
     */
    protected final void reserveParameter(String name, Consumer<Tensor> validator) {
        validateAvailableName(name);
        Consumer<Tensor> suppliedValidator = Objects.requireNonNull(validator, "validator");
        parameterDeclarations.put(name, ParameterSlot.reserved(name, suppliedValidator));
        parameterReservationsBound = false;
    }

    /**
     * Validates and publishes every outstanding direct parameter reservation as one group.
     *
     * <p>All values and retained validators are checked and all real {@link Parameter} wrappers
     * are constructed before any slot changes. Success publishes the complete direct group
     * through the reservation completion gate; ordinary failure publishes none. This operation
     * is not recursive and provides no rollback for fatal virtual-machine failure.</p>
     *
     * @param values non-null ordered values, one per outstanding reservation in declaration order
     * @throws NullPointerException if {@code values} or one of its elements is null
     * @throws IllegalArgumentException if the value count or a reserved schema is invalid
     */
    protected final void bindReservedParameters(List<Tensor> values) {
        Objects.requireNonNull(values, "values");
        List<ParameterSlot> reservations = outstandingParameterReservations();
        if (values.size() != reservations.size()) {
            throw new IllegalArgumentException(
                    "reserved parameter value count mismatch: expected="
                            + reservations.size() + ", actual=" + values.size());
        }

        List<Parameter> prepared = new ArrayList<>(reservations.size());
        for (int index = 0; index < reservations.size(); index++) {
            Tensor value = Objects.requireNonNull(values.get(index), "values[" + index + "]");
            ParameterSlot reservation = reservations.get(index);
            reservation.validator.accept(value);
            prepared.add(new Parameter(reservation.name, value));
        }
        for (int index = 0; index < reservations.size(); index++) {
            reservations.get(index).parameter = prepared.get(index);
        }
        parameterReservationsBound = true;
    }

    /**
     * Reports whether every direct parameter reservation has a published wrapper.
     *
     * <p>The completion read has acquire semantics paired with successful group publication. It
     * is lifecycle support for a concrete module, not a public initialization-status API.</p>
     *
     * @return {@code true} when no direct reservation is outstanding and all corresponding
     *     wrappers are visible to the current thread
     */
    protected final boolean parameterReservationsBound() {
        return parameterReservationsBound;
    }

    /**
     * Returns one exact direct parameter wrapper after all direct reservations are complete.
     *
     * <p>The completion check has acquire semantics. It therefore returns only a fully published
     * wrapper set or fails; it never exposes an incomplete placeholder.</p>
     *
     * @param name non-null local parameter name
     * @return the exact published wrapper, never {@code null}
     * @throws NullPointerException if {@code name} is null
     * @throws IllegalArgumentException if the name is not a direct parameter declaration
     * @throws IllegalStateException if any direct parameter reservation is outstanding
     */
    protected final Parameter boundParameter(String name) {
        Objects.requireNonNull(name, "name");
        requireDirectParameterReservationsBound();
        ParameterSlot slot = parameterDeclarations.get(name);
        if (slot == null) {
            throw new IllegalArgumentException("no direct parameter declared with name: " + name);
        }
        return slot.parameter;
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
        ParameterSlot slot = parameterDeclarations.get(name);
        if (slot == null) {
            throw new IllegalArgumentException("no direct parameter declared with name: " + name);
        }
        requireDirectParameterReservationsBound();
        slot.parameter.replace(value);
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
        validateChildCandidate(moduleChild);
        children.put(name, moduleChild);
        moduleChild.parent = this;
        return child;
    }

    /**
     * Registers an immutable snapshot of children under zero-based decimal names.
     *
     * <p>The supplied list is traversed once to capture its exact elements. Complete null,
     * identity-duplicate, local-name, cycle, and existing-ownership validation precedes every
     * child-map or parent-link change. Installation then follows snapshot order. This
     * package-private construction primitive does not provide synchronization or recovery from
     * fatal virtual-machine failure.</p>
     *
     * @param modules non-null ordered candidate list; every element must be non-null, have a
     *     distinct identity, and be available for permanent ownership by this module
     * @param <T> the common module subtype retained in the immutable result
     * @return the immutable ordered snapshot whose exact instances were registered; never
     *     {@code null}
     * @throws NullPointerException if {@code modules} or its first null element is null
     * @throws IllegalArgumentException if a candidate identity is repeated, a numeric name is
     *     unavailable, or a candidate is this module or an ancestor
     * @throws IllegalStateException if a candidate is already owned by a module
     */
    final <T extends Module> List<T> registerIndexedChildren(List<? extends T> modules) {
        Objects.requireNonNull(modules, "modules");
        List<T> candidates = new ArrayList<>();
        int index = 0;
        for (T candidate : modules) {
            candidates.add(Objects.requireNonNull(candidate, "modules[" + index + "]"));
            index++;
        }
        List<T> snapshot = List.copyOf(candidates);

        IdentityHashMap<Module, Boolean> identities = new IdentityHashMap<>();
        for (T candidate : snapshot) {
            if (identities.put(candidate, Boolean.TRUE) != null) {
                throw new IllegalArgumentException("module identity is repeated");
            }
        }

        for (int childIndex = 0; childIndex < snapshot.size(); childIndex++) {
            validateAvailableName(Integer.toString(childIndex));
            validateChildCandidate(snapshot.get(childIndex));
        }
        for (int childIndex = 0; childIndex < snapshot.size(); childIndex++) {
            Module candidate = snapshot.get(childIndex);
            children.put(Integer.toString(childIndex), candidate);
            candidate.parent = this;
        }
        return snapshot;
    }

    /**
     * Atomically registers one encounter-ordered snapshot of named children.
     *
     * <p>Every null, local-name, collision, repeated-identity, cycle, and existing-ownership
     * check completes for the full supplied snapshot before this module's child map or any
     * candidate's parent link changes. The supplied map and its entries are not retained. This
     * protected construction primitive exists for subclasses whose complete structure needs
     * several descriptive child names installed atomically; it is not a public child-mutation or
     * reparenting API. Success establishes permanent exclusive ownership in map encounter order.
     * The method is not thread-safe and provides no rollback for concurrent races or fatal
     * virtual-machine failure. Subclasses must call it only while establishing their structure,
     * before publishing the owning module for concurrent use.</p>
     *
     * @param children non-null encounter-ordered mapping from valid available local names to
     *     non-null, identity-distinct modules available for permanent ownership
     * @throws NullPointerException if {@code children}, a key, or a value is {@code null}
     * @throws IllegalArgumentException if a name is blank, contains {@code .}, or collides with
     *     direct state or a child; a module identity is repeated; or a candidate is this module
     *     or an ancestor
     * @throws IllegalStateException if a candidate is already owned by a module
     */
    protected final void registerNamedChildren(Map<String, ? extends Module> children) {
        Objects.requireNonNull(children, "children");
        Map<String, Module> snapshot = new LinkedHashMap<>();
        int entryIndex = 0;
        for (Map.Entry<String, ? extends Module> entry : children.entrySet()) {
            String name = Objects.requireNonNull(entry.getKey(), "children key[" + entryIndex + "]");
            Module candidate = Objects.requireNonNull(
                    entry.getValue(), "children[" + name + "]");
            if (snapshot.put(name, candidate) != null) {
                throw new IllegalArgumentException("child name is repeated: " + name);
            }
            entryIndex++;
        }

        for (String name : snapshot.keySet()) {
            validateAvailableName(name);
        }

        IdentityHashMap<Module, Boolean> identities = new IdentityHashMap<>();
        for (Module candidate : snapshot.values()) {
            if (identities.put(candidate, Boolean.TRUE) != null) {
                throw new IllegalArgumentException("module identity is repeated");
            }
        }
        for (Module candidate : snapshot.values()) {
            validateChildCycle(candidate);
        }
        for (Module candidate : snapshot.values()) {
            validateChildOwnership(candidate);
        }

        snapshot.forEach((name, candidate) -> {
            this.children.put(name, candidate);
            candidate.parent = this;
        });
    }

    /**
     * Returns this module's direct trainable declarations in declaration order.
     *
     * @return an unmodifiable structural snapshot of direct parameter wrappers; never {@code
     *     null}. A wrapper's {@link Parameter#value()} observes its current binding when read.
     * @throws IllegalStateException if the first direct parameter reservation is not yet bound
     */
    public final List<Parameter> parameters() {
        requireDirectParameterReservationsBound();
        return parameterDeclarations.values().stream()
                .map(slot -> slot.parameter)
                .toList();
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
     *     or the first qualified parameter reservation is not yet bound
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
     * Captures the complete module tree's current parameter and buffer bindings.
     *
     * <p>Traversal is deterministic depth first. Each module emits direct parameters in declaration
     * order, then direct buffers in declaration order, then visits children in registration order.
     * Paths are relative dot-separated names. Each entry retains the exact Tensor returned by its
     * wrapper during this call, so later replacement does not change an earlier dictionary.
     * Wrapper discovery snapshots behave differently: they retain stable wrappers whose later
     * {@code value()} reads observe replacement.</p>
     *
     * <p>The result is an immutable shallow in-memory snapshot. No Tensor is copied, evaluated,
     * materialized, detached, or mutated, and no Tensor storage is read. The call creates no Tensor
     * or expression producer and performs no compiler, runtime, or backend work. Traversal uses an
     * explicit stack and rejects a malformed cycle or shared module identity before returning a
     * result. This operation is not thread-safe or linearizable; callers must coordinate it with
     * declaration, replacement, loading, mode changes, and forward construction when they require
     * a consistent view.</p>
     *
     * @return non-null immutable ordered dictionary of exact current Tensor references; empty for
     *     a tree with no parameters or buffers
     * @throws IllegalStateException if defensive traversal encounters a repeated module identity
     *     or the first qualified parameter reservation is not yet bound
     */
    public final StateDictionary stateDictionary() {
        List<StateBinding> bindings = collectStateBindings(false);
        List<StateEntry> entries = new ArrayList<>(bindings.size());
        for (StateBinding binding : bindings) {
            entries.add(new StateEntry(binding.path(), binding.kind(), binding.currentValue()));
        }
        return new StateDictionary(entries);
    }

    /**
     * Strictly installs a complete in-memory state dictionary after validating the whole tree.
     *
     * <p>Paths identify entries independent of candidate list order. Validation first captures one
     * identity-defended target-tree snapshot, then rejects the first missing target path, the first
     * unexpected candidate path, and finally the first target-order mismatch in kind, exact data
     * type, structural Shape, or parameter gradient eligibility, in that order. Parameter
     * compatibility therefore matches the permanent declaration schema. Buffer data type and
     * Shape are compared with the exact current target binding captured at load start; buffer
     * gradient eligibility is deliberately ignored because buffer role, rather than a Tensor flag,
     * excludes it from optimizer discovery.</p>
     *
     * <p>Only after every target validates are the exact candidate Tensor references installed in
     * target traversal order. Existing Parameter and Buffer wrapper identities, names, paths,
     * declaration order, child ownership, and mode remain unchanged. Tensors and expressions
     * obtained before this call also remain unchanged; later wrapper reads and later expression
     * construction observe the new bindings. Ordinary validation failure changes no binding.
     * Current immutable descriptors make the prevalidated parameter replacements and non-null
     * buffer assignments non-throwing by construction.</p>
     *
     * <p>A reserved parameter path participates in the same complete target schema. Its retained
     * validator checks the candidate and a real wrapper is prepared before installation. A
     * successful load can therefore initialize a reserved group without calling a layer
     * initializer, creating a random generator, or constructing another Tensor. Until every
     * reserved group in the owned tree is bound, ordinary discovery and state export fail rather
     * than return a partial schema.</p>
     *
     * <p>This mutable operation is not thread-safe or linearizable and provides no Java-memory-
     * model visibility guarantee. Atomicity means complete ordinary validation before sequential
     * installation under external caller coordination; it does not promise a simultaneous view to
     * racing readers, synchronization, rollback for concurrent races, or recovery from JVM-fatal
     * failure. Loading creates, copies, evaluates, and mutates no Tensor and performs no compiler,
     * runtime, or backend work. The dictionary contains no persistent checkpoint bytes,
     * optimizer/session state, graph random state, mode, or execution state.</p>
     *
     * @param dictionary non-null complete candidate dictionary whose exact Tensor references will
     *     be installed when every target validates
     * @throws NullPointerException if {@code dictionary} is null
     * @throws IllegalArgumentException if state is missing or unexpected, or a matched entry has
     *     incompatible kind, data type, Shape, or parameter gradient eligibility, checked in the
     *     documented order before any installation
     * @throws IllegalStateException if defensive target traversal encounters a repeated module
     *     identity before any installation
     */
    public final void loadStateDictionary(StateDictionary dictionary) {
        StateDictionary suppliedDictionary = Objects.requireNonNull(dictionary, "dictionary");
        List<StateBinding> targets = collectStateBindings(true);
        Map<String, StateEntry> candidates = new LinkedHashMap<>();
        for (StateEntry entry : suppliedDictionary.entries()) {
            candidates.put(entry.path(), entry);
        }

        Map<String, StateBinding> targetsByPath = new LinkedHashMap<>();
        for (StateBinding target : targets) {
            targetsByPath.put(target.path(), target);
            if (!candidates.containsKey(target.path())) {
                throw new IllegalArgumentException("missing state path: " + target.path());
            }
        }
        for (StateEntry candidate : suppliedDictionary.entries()) {
            if (!targetsByPath.containsKey(candidate.path())) {
                throw new IllegalArgumentException("unexpected state path: " + candidate.path());
            }
        }

        List<StateInstall> installs = new ArrayList<>(targets.size());
        List<Module> modulesWithReservations = new ArrayList<>();
        IdentityHashMap<Module, Boolean> reservationOwners = new IdentityHashMap<>();
        for (StateBinding target : targets) {
            StateEntry candidate = candidates.get(target.path());
            if (candidate.kind() != target.kind()) {
                throw new IllegalArgumentException(
                        "state kind mismatch at path " + target.path() + ": expected="
                                + target.kind() + ", actual=" + candidate.kind());
            }
            if (target.reserved()) {
                Parameter prepared = prepareReservedParameter(target, candidate.value());
                installs.add(new StateInstall(target, candidate.value(), prepared));
                if (reservationOwners.put(target.owner(), Boolean.TRUE) == null) {
                    modulesWithReservations.add(target.owner());
                }
            } else {
                validateCompatible(target, candidate);
                installs.add(new StateInstall(target, candidate.value(), null));
            }
        }
        for (StateInstall install : installs) {
            StateBinding target = install.target();
            if (target.reserved()) {
                target.parameterSlot().parameter = install.preparedParameter();
            } else if (target.kind() == StateKind.PARAMETER) {
                target.parameter().replace(install.value());
            } else {
                target.buffer().replaceValue(install.value());
            }
        }
        for (Module module : modulesWithReservations) {
            module.publishParameterReservationCompletion();
        }
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
        if (parameterDeclarations.containsKey(name)
                || buffers.containsKey(name)
                || children.containsKey(name)) {
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

    private void validateChildCandidate(Module candidate) {
        validateChildCycle(candidate);
        validateChildOwnership(candidate);
    }

    private void validateChildCycle(Module candidate) {
        if (candidate == this || isAncestor(candidate)) {
            throw new IllegalArgumentException("child must not be this module or an ancestor");
        }
    }

    private static void validateChildOwnership(Module candidate) {
        if (candidate.parent != null) {
            throw new IllegalStateException("child is already owned by a module");
        }
    }

    private void changeModeRecursively(ForwardMode requestedMode) {
        List<Module> modules = collectOwnedModules();
        modules.forEach(module -> module.mode = requestedMode);
    }

    private void collectParameters(Map<String, Parameter> result) {
        traverseState((module, pathSegments) -> {
            module.requireParameterReservationsBound(pathSegments);
            module.parameterDeclarations.forEach((name, slot) ->
                    result.put(path(pathSegments, name), slot.parameter));
        });
    }

    private void collectBuffers(Map<String, Buffer> result) {
        traverseState((module, pathSegments) -> module.buffers.forEach(
                (name, buffer) -> result.put(path(pathSegments, name), buffer)));
    }

    private List<StateBinding> collectStateBindings(boolean includeReservations) {
        List<StateBinding> result = new ArrayList<>();
        traverseState((module, pathSegments) -> {
            if (!includeReservations) {
                module.requireParameterReservationsBound(pathSegments);
            }
            module.parameterDeclarations.forEach((name, slot) -> {
                String qualifiedPath = path(pathSegments, name);
                if (slot.parameter == null) {
                    result.add(StateBinding.reservedParameter(qualifiedPath, module, slot));
                } else {
                    result.add(StateBinding.parameter(
                            qualifiedPath, module, slot, slot.parameter, slot.parameter.value()));
                }
            });
            module.buffers.forEach((name, buffer) -> result.add(StateBinding.buffer(
                    path(pathSegments, name), module, buffer, buffer.value())));
        });
        return result;
    }

    private static void validateCompatible(StateBinding target, StateEntry candidate) {
        Tensor expected = target.currentValue();
        Tensor actual = candidate.value();
        if (actual.descriptor().dataType() != expected.descriptor().dataType()) {
            throw new IllegalArgumentException(
                    "state data type mismatch at path " + target.path() + ": expected="
                            + expected.descriptor().dataType() + ", actual="
                            + actual.descriptor().dataType());
        }
        if (!actual.descriptor().shape().equals(expected.descriptor().shape())) {
            throw new IllegalArgumentException(
                    "state shape mismatch at path " + target.path() + ": expected="
                            + expected.descriptor().shape() + ", actual="
                            + actual.descriptor().shape());
        }
        if (target.kind() == StateKind.PARAMETER && !actual.descriptor().requiresGrad()) {
            throw new IllegalArgumentException(
                    "state parameter requiresGrad mismatch at path " + target.path()
                            + ": expected=true, actual=false");
        }
    }

    private static Parameter prepareReservedParameter(StateBinding target, Tensor value) {
        try {
            target.parameterSlot().validator.accept(value);
            return new Parameter(target.parameterSlot().name, value);
        } catch (IllegalArgumentException exception) {
            throw new IllegalArgumentException(
                    "state parameter mismatch at path " + target.path() + ": "
                            + exception.getMessage(), exception);
        }
    }

    private List<ParameterSlot> outstandingParameterReservations() {
        List<ParameterSlot> reservations = new ArrayList<>();
        for (ParameterSlot slot : parameterDeclarations.values()) {
            if (slot.parameter == null) {
                reservations.add(slot);
            }
        }
        return reservations;
    }

    private void requireDirectParameterReservationsBound() {
        if (parameterReservationsBound) {
            return;
        }
        for (ParameterSlot slot : parameterDeclarations.values()) {
            if (slot.parameter == null) {
                throw new IllegalStateException(
                        "parameter reservation is not bound: " + slot.name);
            }
        }
        for (ParameterSlot slot : parameterDeclarations.values()) {
            if (slot.validator != null) {
                throw new IllegalStateException(
                        "parameter reservation is not bound: " + slot.name);
            }
        }
        throw new IllegalStateException("parameter reservations are not bound");
    }

    private void requireParameterReservationsBound(List<String> pathSegments) {
        if (parameterReservationsBound) {
            return;
        }
        for (ParameterSlot slot : parameterDeclarations.values()) {
            if (slot.parameter == null) {
                throw new IllegalStateException(
                        "parameter reservation is not bound: " + path(pathSegments, slot.name));
            }
        }
        for (ParameterSlot slot : parameterDeclarations.values()) {
            if (slot.validator != null) {
                throw new IllegalStateException(
                        "parameter reservation is not bound: " + path(pathSegments, slot.name));
            }
        }
        throw new IllegalStateException("parameter reservations are not bound");
    }

    private void publishParameterReservationCompletion() {
        for (ParameterSlot slot : parameterDeclarations.values()) {
            if (slot.parameter == null) {
                throw new IllegalStateException(
                        "parameter reservation is not bound: " + slot.name);
            }
        }
        parameterReservationsBound = true;
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

    private record StateBinding(
            String path,
            StateKind kind,
            Module owner,
            ParameterSlot parameterSlot,
            Parameter parameter,
            Buffer buffer,
            Tensor currentValue) {
        private static StateBinding parameter(
                String path,
                Module owner,
                ParameterSlot slot,
                Parameter parameter,
                Tensor value) {
            return new StateBinding(
                    path, StateKind.PARAMETER, owner, slot, parameter, null, value);
        }

        private static StateBinding reservedParameter(
                String path, Module owner, ParameterSlot slot) {
            return new StateBinding(path, StateKind.PARAMETER, owner, slot, null, null, null);
        }

        private static StateBinding buffer(
                String path, Module owner, Buffer buffer, Tensor value) {
            return new StateBinding(path, StateKind.BUFFER, owner, null, null, buffer, value);
        }

        private boolean reserved() {
            return kind == StateKind.PARAMETER && parameter == null;
        }
    }

    private record StateInstall(
            StateBinding target, Tensor value, Parameter preparedParameter) {
    }

    private static final class ParameterSlot {
        private final String name;
        private final Consumer<Tensor> validator;
        private Parameter parameter;

        private ParameterSlot(String name, Consumer<Tensor> validator, Parameter parameter) {
            this.name = name;
            this.validator = validator;
            this.parameter = parameter;
        }

        private static ParameterSlot bound(String name, Parameter parameter) {
            return new ParameterSlot(name, null, parameter);
        }

        private static ParameterSlot reserved(String name, Consumer<Tensor> validator) {
            return new ParameterSlot(name, validator, null);
        }
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
