package io.github.pho001.synaptik.engine;

import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
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
 * <p>This initial ordinary surface is construction-only. Typed compile, prepare, input-binding,
 * run, publication, and result APIs are not yet part of this type. Lifecycle observation and
 * closure are thread-safe and inherit the owned lifecycle's idempotent, failure-retaining close
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
