package io.github.pho001.synaptik.engine;

import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.model.tensor.Tensor;
import java.util.List;
import java.util.Objects;

/**
 * Owns the ordinary standard Synaptik Engine composition and its lifetime.
 *
 * <p>The current standard composition contains exactly one freshly opened CPU backend integration
 * per Engine. Each instance owns an independent composition; closing one instance does not affect
 * another, and no standard instance is cached or stored in process-global state. The CPU
 * integration and lower-level lifecycle owner remain private implementation details and never
 * transfer to the caller.</p>
 *
 * <p>The ordinary surface compiles Tensor expressions, prepares immutable reusable recipes, binds
 * logical input Tensors by identity in arbitrary order, and returns publication leases whose
 * exact occurrences can be materialized explicitly as detached immutable host values. A
 * one-shot compute convenience discovers reachable expression leaves, then performs fresh
 * compilation, preparation, execution, complete publication and aggregate-byte preflight,
 * ordered host materialization, and cleanup in one synchronous call. Forward publications retain
 * requested-output identity; gradient publications
 * retain explicit target identity and position even when roles alias inwardly. Implicit targets
 * and backward convenience remain outside the current surface. Lifecycle observation and closure
 * are thread-safe and inherit the owned lifecycle's idempotent, failure-retaining semantics.</p>
 */
public final class Engine implements AutoCloseable {
    private final AdvancedEngine delegate;

    /**
     * Opens a new independent Engine with the fixed current built-in composition.
     *
     * <p>The fixed inventory contains CPU only. Every call opens one fresh composition without
     * discovery or global reuse. If construction fails after that opening, the partially
     * transferred owner is closed exactly once. The original unchecked failure is preserved, and
     * a distinct rollback failure is suppressed on it.</p>
     *
     * @return a new non-null open Engine owning an independent CPU-only composition
     * @throws RuntimeException if CPU composition construction fails
     * @throws Error if construction or rollback reports a fatal failure
     */
    public static Engine standard() {
        CpuBackendIntegration integration = CpuBackendIntegration.open();
        AdvancedEngine owner = null;
        try {
            owner = AdvancedEngine.takeOwnership(integration);
            return new Engine(owner);
        } catch (RuntimeException | Error failure) {
            try {
                if (owner == null) {
                    integration.close();
                } else {
                    owner.close();
                }
            } catch (RuntimeException | Error cleanupFailure) {
                if (cleanupFailure != failure) {
                    failure.addSuppressed(cleanupFailure);
                }
            }
            throw failure;
        }
    }

    /**
     * Creates an ordinary Engine that takes cleanup ownership of one exact lifecycle owner.
     *
     * @param delegate non-null lifecycle owner whose cleanup ownership transfers on success;
     *     this constructor does not inspect its current lifecycle state
     * @throws NullPointerException if {@code delegate} is null, with message {@code delegate}
     */
    Engine(AdvancedEngine delegate) {
        this.delegate = Objects.requireNonNull(delegate, "delegate");
    }

    /**
     * Compiles one non-empty identity-unique ordered forward boundary with standard settings.
     *
     * @param forwardOutputs non-null non-empty list of non-null identity-unique output Tensors;
     *     the container is snapshotted and not retained
     * @return a fresh non-null owner-bound immutable compile handle
     * @throws NullPointerException if the list or an element is {@code null}
     * @throws IllegalArgumentException if the list is empty, repeats an exact Tensor, or Compiler
     *     rejects the graph
     * @throws IllegalStateException if closure has begun or no backend is eligible
     * @throws RuntimeException if inward compilation reports another unchecked failure
     * @throws Error if inward compilation reports a fatal failure
     */
    public CompiledGraph compile(List<Tensor> forwardOutputs) {
        return delegate.compileOrdinary(this, forwardOutputs);
    }

    /**
     * Compiles one explicitly seeded first-order reverse-mode request with standard settings.
     * Seeds align one-for-one with forward outputs and may repeat or be expressions; targets are
     * never inferred.
     *
     * @param forwardOutputs non-null non-empty identity-unique ordered forward outputs
     * @param cotangentSeeds non-null output-aligned explicit non-null cotangent seeds
     * @param targets non-null non-empty identity-unique ordered differentiation targets
     * @return a fresh non-null owner-bound immutable compile handle
     * @throws NullPointerException if a list or element is {@code null}
     * @throws IllegalArgumentException if local structure or Compiler semantics reject the request
     * @throws IllegalStateException if closure has begun or no backend is eligible
     * @throws RuntimeException if inward compilation reports another unchecked failure
     * @throws Error if inward compilation reports a fatal failure
     */
    public CompiledGraph compile(
            List<Tensor> forwardOutputs,
            List<Tensor> cotangentSeeds,
            List<Tensor> targets) {
        return delegate.compileOrdinary(this, forwardOutputs, cotangentSeeds, targets);
    }

    /**
     * Prepares one compile handle created by this exact Engine.
     * Current CPU preparation requires one non-empty maximal CPU partition and may therefore
     * reject an artifact that compiled successfully.
     *
     * @param compiledGraph non-null owner-bound compile handle from this Engine
     * @return a fresh non-null immutable reusable prepared handle
     * @throws NullPointerException if {@code compiledGraph} is {@code null}
     * @throws IllegalArgumentException if ownership or inward preparation is invalid
     * @throws IllegalStateException if closure has begun
     * @throws RuntimeException if inward preparation reports another unchecked failure
     * @throws Error if inward preparation reports a fatal failure
     */
    public PreparedExecution prepare(CompiledGraph compiledGraph) {
        return delegate.prepareOrdinary(this, compiledGraph);
    }

    /**
     * Runs one prepared recipe using current caller-owned host associations matched by Tensor ID.
     *
     * <p>The input list may use any order. Associations are snapshotted once after complete
     * logical validation; caller storage must remain usable until the returned result closes.</p>
     *
     * @param preparedExecution non-null prepared handle created by this exact Engine
     * @param inputs non-null list supplying every required logical Tensor exactly once
     * @return a fresh non-null publication lease supporting metadata access and explicit
     *     per-occurrence host materialization while open
     * @throws NullPointerException if an argument or input element is {@code null}
     * @throws IllegalArgumentException if ownership, identity, descriptor, storage geometry, or
     *     inward binding is invalid
     * @throws IllegalStateException if closure has begun or storage is absent, dead, inaccessible,
     *     or inward execution state is invalid
     * @throws RuntimeException if inward execution reports another unchecked failure
     * @throws Error if inward execution reports a fatal failure
     */
    public RunResult run(PreparedExecution preparedExecution, List<Tensor> inputs) {
        return delegate.runOrdinary(this, preparedExecution, inputs);
    }

    /**
     * Computes one forward output without an aggregate caller byte limit.
     *
     * <p>This overload delegates to {@link #compute(Tensor, long)} with
     * {@link Long#MAX_VALUE}. Checked aggregate arithmetic, the per-value JVM array ceiling, and
     * all descriptor, storage, and inward execution constraints still apply.</p>
     *
     * @param output non-null forward output Tensor
     * @return a fresh non-null detached immutable host value
     * @throws NullPointerException if {@code output} is null
     * @throws IllegalArgumentException if output metadata is invalid, a payload exceeds the JVM
     *     array ceiling, or an inward compile, prepare, binding, or CPU validation rejects the
     *     request
     * @throws IllegalStateException if Engine closure has begun, reachable Tensor identity is
     *     inconsistent, an authoritative compiled input has no reachable leaf, selected caller
     *     storage is absent, dead, or inaccessible, or result metadata is inconsistent
     * @throws ArithmeticException if checked logical byte-count arithmetic overflows
     * @throws RuntimeException if inward execution, copying, or cleanup reports another unchecked
     *     failure
     * @throws Error if inward work, allocation, copying, or cleanup reports a fatal failure
     */
    public HostTensorValue compute(Tensor output) {
        return compute(output, Long.MAX_VALUE);
    }

    /**
     * Computes one forward output through a fresh complete CPU-only lifecycle and returns its
     * detached canonical host value.
     *
     * <p>This is the singleton specialization of the ordered-output overload. One Engine
     * admission spans argument validation, transient iterative discovery of provenance-free
     * leaves, compilation, Compiler-authoritative input selection, preparation, logical input
     * binding, execution, complete publication and aggregate-size preflight, copying, and
     * temporary-result cleanup. Discovery follows immutable Tensor provenance by exact object
     * identity; final {@link CompiledGraph#inputs()} metadata alone determines which discovered
     * leaves are bound and in what order. No Tensor or provenance state is retained after the
     * synchronous call. The call does not cache or reuse a compiled or prepared recipe; callers
     * performing repeated runs or selective materialization should use {@link #compile(List)},
     * {@link #prepare(CompiledGraph)}, {@link #run(PreparedExecution, List)}, and
     * {@link RunResult#materialize}.</p>
     *
     * <p>The byte limit covers only the returned canonical payload, not inputs, intermediate or
     * peak memory, recipes, object overhead, or backend workspace. The caller retains ownership
     * of selected leaf host storage and must keep it live, accessible, and free from conflicting
     * mutation through synchronous completion. Storage of an unselected discovered leaf is not
     * inspected. The returned value is immutable, owns no closeable
     * resource, and remains readable after this Engine closes. On failure, no value is returned;
     * temporary-result cleanup still runs, and a distinct cleanup failure is suppressed on the
     * primary failure.</p>
     *
     * @param output non-null forward output Tensor
     * @param maximumTotalBytes non-negative aggregate upper bound, in bytes, for the returned
     *     canonical payload
     * @return a fresh non-null detached immutable host value
     * @throws NullPointerException if {@code output} is null
     * @throws IllegalArgumentException if the byte limit is negative, logical binding or output
     *     metadata is invalid, the payload exceeds the limit or JVM array ceiling, or an inward
     *     compile, prepare, binding, or CPU validation rejects the request
     * @throws IllegalStateException if Engine closure has begun, reachable Tensor identity is
     *     inconsistent, an authoritative compiled input has no reachable leaf, selected caller
     *     storage is absent, dead, or inaccessible, or result metadata is inconsistent
     * @throws ArithmeticException if checked logical byte-count arithmetic overflows
     * @throws RuntimeException if inward execution, copying, or cleanup reports another unchecked
     *     failure
     * @throws Error if inward work, allocation, copying, or cleanup reports a fatal failure
     */
    public HostTensorValue compute(Tensor output, long maximumTotalBytes) {
        return delegate.computeOrdinary(this, output, maximumTotalBytes);
    }

    /**
     * Computes an ordered non-empty forward boundary without an aggregate caller byte limit.
     *
     * <p>This overload delegates to {@link #compute(List, long)} with
     * {@link Long#MAX_VALUE}. Checked aggregate arithmetic, each value's JVM array ceiling, and all
     * descriptor, storage, and inward execution constraints still apply.</p>
     *
     * @param outputs non-null non-empty ordered list of non-null identity-unique output Tensors;
     *     the container is snapshotted and not retained or mutated
     * @return a fresh non-null immutable ordered list of detached immutable host values
     * @throws NullPointerException if {@code outputs} or an output element is null
     * @throws IllegalArgumentException if outputs are empty or repeat an exact Tensor, output
     *     metadata is invalid, an individual payload exceeds the JVM array ceiling, or inward
     *     validation rejects the request
     * @throws IllegalStateException if Engine closure has begun, reachable Tensor identity is
     *     inconsistent, an authoritative compiled input has no reachable leaf, selected caller
     *     storage is absent, dead, or inaccessible, or result metadata is inconsistent
     * @throws ArithmeticException if checked logical or aggregate byte-count arithmetic overflows
     * @throws RuntimeException if inward compilation, preparation, execution, copying, or cleanup
     *     reports another unchecked failure
     * @throws Error if inward work, allocation, copying, or cleanup reports a fatal failure
     */
    public List<HostTensorValue> compute(List<Tensor> outputs) {
        return compute(outputs, Long.MAX_VALUE);
    }

    /**
     * Computes an ordered non-empty forward boundary through one fresh complete CPU-only
     * lifecycle and returns detached canonical host values in exact requested publication order.
     *
     * <p>One Engine admission spans argument validation, one transient identity-safe inventory of
     * reachable provenance-free leaves, one compilation, Compiler-authoritative ordered input
     * selection, one preparation, one run, complete
     * aggregate-size preflight before any physical copy, one copy per forward occurrence, and
     * temporary-result cleanup. Inventory traversal is iterative and carries no liveness or input
     * ordering meaning; {@link CompiledGraph#inputs()} supplies both final membership and order.
     * Unselected leaves are ignored without storage access, and no Tensor or provenance reference
     * escapes the call. The returned list and values are immutable and lifecycle-independent.
     * Distinct occurrences are copied independently even if they alias an inward representation.
     * No compilation, preparation, result, Tensor-inventory, or byte cache is created. A
     * failure returns no partial list; temporary-result cleanup still runs, and a distinct cleanup
     * failure is suppressed on the primary failure.</p>
     *
     * <p>The aggregate byte limit covers only the sum of canonical returned payload lengths, not
     * inputs, Runtime buffers or workspaces, recipes, object overhead, defensive copies, peak
     * memory, or other allocation. Selected leaf storage remains caller-owned and must stay live,
     * accessible, and free from conflicting mutation through synchronous completion. Current
     * execution and host copying use the fixed CPU-only composition and require supported fully
     * static outputs with resolved final layouts.</p>
     *
     * @param outputs non-null non-empty ordered list of non-null identity-unique output Tensors;
     *     the container is snapshotted and not retained or mutated
     * @param maximumTotalBytes non-negative aggregate upper bound, in bytes, for the sum of all
     *     returned canonical payloads
     * @return a fresh non-null immutable ordered list of detached immutable host values
     * @throws NullPointerException if {@code outputs} or an output element is null
     * @throws IllegalArgumentException if outputs are empty or repeat an exact Tensor, the byte
     *     limit is negative, logical binding or output metadata is invalid, the aggregate exceeds
     *     the limit, an individual payload exceeds the JVM array ceiling, or inward validation
     *     rejects the request
     * @throws IllegalStateException if Engine closure has begun, reachable Tensor identity is
     *     inconsistent, an authoritative compiled input has no reachable leaf, selected caller
     *     storage is absent, dead, or inaccessible, or result metadata is inconsistent
     * @throws ArithmeticException if checked logical or aggregate byte-count arithmetic overflows
     * @throws RuntimeException if inward compilation, preparation, execution, copying, or cleanup
     *     reports another unchecked failure
     * @throws Error if inward work, allocation, copying, or cleanup reports a fatal failure
     */
    public List<HostTensorValue> compute(
            List<Tensor> outputs, long maximumTotalBytes) {
        return delegate.computeOrdinary(this, outputs, maximumTotalBytes);
    }

    /**
     * Reports whether closure of this Engine has begun.
     *
     * <p>This is a thread-safe point-in-time observation; another thread may begin closure
     * immediately after a {@code false} result.</p>
     *
     * @return {@code true} from the instant the first close call starts closure
     */
    public boolean isClosed() {
        return delegate.isClosed();
    }

    /**
     * Closes the owned composition through its concurrent, idempotent lifecycle protocol.
     * Repeated and concurrent callers wait for the first cleanup attempt and observe its exact
     * retained failure, if any.
     *
     * @throws RuntimeException if owned cleanup reports an unchecked failure
     * @throws Error if owned cleanup reports a fatal failure
     */
    @Override
    public void close() {
        delegate.close();
    }
}
