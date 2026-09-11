package io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas;

import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicReference;

/**
 * CPU-private owner that keeps one successfully discovered provider lifetime alive.
 *
 * <p>The immutable result is always available. A loaded session additionally exposes one
 * borrowed invocation and initially owns its matching close action. The owner may atomically
 * transfer that pair once to a coordinator; immutable result and invocation views remain readable,
 * while later session close cannot affect the transferred resource. Without transfer, closing
 * atomically claims the action once and does not query or restore thread state.</p>
 */
final class CpuOpenBlasDiscoverySession implements AutoCloseable {
    private final CpuOpenBlasDiscoveryResult result;
    private final Optional<CpuOpenBlasInvocation> invocation;
    private final AtomicReference<OwnedResource> ownedResource;

    /**
     * Creates one session and validates that live ownership agrees with the immutable status.
     *
     * @param result the immutable completed discovery result
     * @param ownedResource the optional loaded provider-equivalent resource
     * @throws NullPointerException if a required reference is {@code null}
     * @throws IllegalArgumentException if loaded status and resource presence disagree
     */
    CpuOpenBlasDiscoverySession(CpuOpenBlasDiscoveryResult result,
            Optional<OwnedResource> ownedResource) {
        this.result = Objects.requireNonNull(result, "result");
        ownedResource = Objects.requireNonNull(ownedResource, "ownedResource");
        if ((result.status() == CpuOpenBlasDiscoveryResult.Status.LOADED)
                != ownedResource.isPresent()) {
            throw new IllegalArgumentException(
                    "loaded discovery status and owned resource must agree");
        }
        this.invocation = ownedResource.map(OwnedResource::invocation);
        this.ownedResource = new AtomicReference<>(ownedResource.orElse(null));
    }

    /**
     * Returns the immutable discovery metadata before or after close.
     *
     * @return the same immutable result supplied at construction; never {@code null}
     */
    CpuOpenBlasDiscoveryResult result() {
        return result;
    }

    /**
     * Returns the borrowed invocation for a loaded session.
     *
     * <p>The optional remains present after close so existing borrowers observe the underlying
     * provider's ordinary closed-state behavior. Presence does not imply route qualification.</p>
     *
     * @return the immutable optional borrowed invocation, empty when no provider loaded
     */
    Optional<CpuOpenBlasInvocation> invocation() {
        return invocation;
    }

    /**
     * Transfers the loaded invocation and its close action exactly once into a coordinator.
     *
     * @param budget non-null borrowed budget used by the coordinator
     * @return the new sole provider-lifetime owner
     * @throws NullPointerException if {@code budget} is {@code null}
     * @throws IllegalStateException if this session has no unclaimed loaded resource
     */
    CpuOpenBlasCoordinator transferToCoordinator(
            io.github.pho001.synaptik.backend.cpu.internal.executable.CpuConcurrencyBudget budget) {
        Objects.requireNonNull(budget, "budget");
        OwnedResource resource = ownedResource.getAndSet(null);
        if (resource == null) throw new IllegalStateException(
                "OpenBLAS discovery resource is unavailable, closed, or already transferred");
        return new CpuOpenBlasCoordinator(resource.invocation(), resource.closeAction(), budget);
    }

    /**
     * Atomically closes the owned resource at most once.
     *
     * <p>The winning close propagates an unchecked cleanup failure. Ownership remains claimed,
     * so a later close does not retry. No thread count is queried, changed, or restored.</p>
     *
     * @throws RuntimeException if the owned close action fails with a runtime exception
     * @throws Error if the owned close action fails with an error
     */
    @Override
    public void close() {
        OwnedResource resource = ownedResource.getAndSet(null);
        if (resource != null) resource.closeAction().close();
    }

    /**
     * One inseparable loaded invocation and close action used by production and native-free tests.
     * The close action is expected to retain the same lifetime owner as the invocation.
     *
     * @param invocation the borrowed invocation view
     * @param closeAction the one-shot unchecked close action
     */
    record OwnedResource(CpuOpenBlasInvocation invocation, CloseAction closeAction) {
        /**
         * Validates one owned resource pair.
         *
         * @param invocation the required borrowed invocation
         * @param closeAction the required close action for its retained owner
         * @throws NullPointerException if either component is {@code null}
         */
        OwnedResource {
            Objects.requireNonNull(invocation, "invocation");
            Objects.requireNonNull(closeAction, "closeAction");
        }
    }

    /** Unchecked resource close operation retained by one loaded session. */
    @FunctionalInterface
    interface CloseAction {
        /**
         * Closes the retained provider-equivalent lifetime.
         *
         * @throws RuntimeException if cleanup fails with a runtime exception
         * @throws Error if cleanup fails with an error
         */
        void close();
    }
}
