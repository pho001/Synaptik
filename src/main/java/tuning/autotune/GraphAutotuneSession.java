package tuning.autotune;

import tuning.autotune.AutotuneSession;

import java.util.Objects;

/**
 * Convenience session for graph-policy autotune.
 *
 * <p>The session delegates execution to {@link AutotuneSession} after converting
 * {@link GraphAutotuneRequest} to a generic request. It does not perform
 * platform calibration and does not mutate the supplied runtime profile.</p>
 *
 * <p>Instances are lightweight and intended for one {@link #run()} call.</p>
 */
public final class GraphAutotuneSession {
    private final GraphAutotuneRequest request;

    private GraphAutotuneSession(GraphAutotuneRequest request) {
        this.request = Objects.requireNonNull(request, "request cannot be null");
    }

    /**
     * Creates a graph autotune session.
     *
     * @param request graph autotune request; must not be {@code null}
     * @return new session
     */
    public static GraphAutotuneSession create(GraphAutotuneRequest request) {
        return new GraphAutotuneSession(request);
    }

    /**
     * Runs graph autotune and wraps the generic tuning result with graph mode
     * metadata.
     *
     * @return graph autotune result
     */
    public GraphAutotuneResult run() {
        return new GraphAutotuneResult(
                request.mode(),
                AutotuneSession.create(request.toAutotuneRequest()).run()
        );
    }
}
