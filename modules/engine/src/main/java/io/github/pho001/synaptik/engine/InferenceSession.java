package io.github.pho001.synaptik.engine;

import io.github.pho001.synaptik.model.tensor.Tensor;
import java.util.List;
import java.util.Objects;

/**
 * Reusable public inference owner for one exact compiled graph and prepared execution.
 *
 * <p>An {@link Engine} creates a session from an owner-bound {@link CompiledGraph}, performs the
 * graph's one preparation before publishing the session, and lends its composition lifecycle to
 * the session. Engine ownership is not transferred: the caller must keep the Engine open and close
 * the session when reuse ends. Engine closure remains the final safety boundary and closes a
 * retained session after its retained run results.</p>
 *
 * <p>Every {@link #run(List)} delegates to the existing Engine run contract over the same prepared
 * execution. The compiled graph's immutable {@link CompiledGraph#inputs()} list is the stable
 * input-occurrence authority, while each call may supply its matching Tensors in any order and
 * snapshots their current caller-owned host-storage associations. The prepared recipe and direct
 * input/publication adapters are reused without compilation, preparation, provider interrogation,
 * registry lookup, or backend selection. Runtime creates one isolated mutable run state and fresh
 * run-owned resources for each call, so the session supports concurrent runs subject to the
 * caller-owned storage's own lifetime, accessibility, and mutation constraints.</p>
 *
 * <p>A run returns the existing {@link RunResult} publication lease. Its ordered forward-then-
 * gradient occurrences, exact occurrence authentication, explicit materialization, caller-storage
 * borrow lifetime, and detached {@link HostTensorValue} lifetime are unchanged. Closing this
 * session atomically prevents later run admission and closes its one prepared execution exactly
 * once. A Runtime run that already acquired the prepared lease may complete, and an already
 * returned result remains independently usable until it or the Engine closes.</p>
 */
public final class InferenceSession implements AutoCloseable {
    private final Engine engine;
    private final CompiledGraph compiledGraph;
    private final PreparedExecution preparedExecution;

    /**
     * Creates one Engine-owned session around an unpublished prepared handle.
     *
     * @param engine non-null borrowed Engine composition owner
     * @param compiledGraph non-null exact graph compiled by {@code engine}
     * @param preparedExecution non-null exact prepared handle owned by this session
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if the graph or preparation belongs to another Engine or
     *     the preparation does not retain the exact graph
     */
    InferenceSession(
            Engine engine,
            CompiledGraph compiledGraph,
            PreparedExecution preparedExecution) {
        this.engine = Objects.requireNonNull(engine, "engine");
        this.compiledGraph = Objects.requireNonNull(compiledGraph, "compiledGraph");
        this.preparedExecution = Objects.requireNonNull(
                preparedExecution, "preparedExecution");
        if (compiledGraph.owner() != engine || preparedExecution.owner() != engine) {
            throw new IllegalArgumentException("session state belongs to another engine");
        }
        if (preparedExecution.compiledGraph() != compiledGraph) {
            throw new IllegalArgumentException(
                    "prepared execution belongs to another compiled graph");
        }
    }

    /**
     * Returns the exact immutable graph fixed by this session.
     *
     * <p>Its {@link CompiledGraph#inputs()} list remains readable after session or Engine closure
     * and defines stable caller-input membership and Compiler occurrence order.</p>
     *
     * @return the non-null originating compile handle by reference identity
     */
    public CompiledGraph compiledGraph() {
        return compiledGraph;
    }

    /**
     * Runs the fixed prepared graph with current caller-owned input-storage associations.
     *
     * <p>The list may use any order. Every Tensor must match one required input identity and exact
     * compiled descriptor, and every required input must occur exactly once. The call validates
     * storage data type, resolved-layout capacity, liveness, and current-thread accessibility
     * before borrowing. Associations are snapshotted once; the caller keeps every storage alive,
     * accessible where used, and free from conflicting mutation until the returned result closes.
     * Separate calls receive isolated Runtime state and may execute concurrently.</p>
     *
     * <p>Engine closure admission has first precedence. Under an open Engine, the session checks
     * its hidden preparation's close gate before inspecting {@code inputs}, Tensor metadata,
     * storage, or adapters. A closed session therefore reports {@code prepared execution is
     * closed} without borrowing caller storage, even when the supplied inputs are otherwise
     * invalid.</p>
     *
     * @param inputs non-null list supplying every required logical Tensor exactly once
     * @return a fresh non-null publication lease with deterministic occurrence order and explicit
     *     detached host materialization
     * @throws NullPointerException if {@code inputs} or an input element is {@code null}
     * @throws IllegalArgumentException if an input identity, descriptor, storage data type,
     *     capacity, or inward binding is invalid
     * @throws IllegalStateException if Engine or session closure has begun, storage is absent,
     *     dead, or inaccessible, or inward execution state is invalid
     * @throws RuntimeException if inward execution reports another unchecked failure
     * @throws Error if inward execution reports a fatal failure
     */
    public RunResult run(List<Tensor> inputs) {
        return engine.runSession(preparedExecution, inputs);
    }

    /**
     * Reports whether closure of this session has begun.
     *
     * <p>A true result prevents new runs. An earlier admitted run or its returned result may still
     * be alive under Runtime's existing lease protocol.</p>
     *
     * @return {@code true} after explicit session close or owner-Engine cleanup reaches this
     *     session; otherwise {@code false}
     */
    public boolean isClosed() {
        return preparedExecution.isClosed();
    }

    /**
     * Closes this session's exact prepared execution once and unregisters the session from its
     * Engine owner even when cleanup fails.
     *
     * <p>Closure does not wait for a Runtime run that already acquired its prepared lease and does
     * not close an already returned {@link RunResult}. Repeated and concurrent calls inherit the
     * prepared handle's idempotent, waiting, interruption-restoring, and failure-retaining close
     * behavior.</p>
     *
     * @throws RuntimeException if prepared-resource cleanup reports an unchecked failure
     * @throws Error if prepared-resource cleanup reports a fatal failure
     */
    @Override
    public void close() {
        try {
            preparedExecution.close();
        } finally {
            engine.lifecycleOwner().unregister(this);
        }
    }
}
