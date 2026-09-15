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
 * logical input Tensors by identity in arbitrary order, and returns metadata-only publication
 * leases. Forward publications retain requested-output identity; gradient publications retain
 * explicit target identity and position even when roles alias inwardly. Host values, implicit
 * targets, and one-shot execution remain outside the current surface. Lifecycle observation and
 * closure are thread-safe and inherit the owned lifecycle's idempotent, failure-retaining
 * semantics.</p>
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
     * @return a fresh non-null metadata-only result lease
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
