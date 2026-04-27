package tuning.autotune;

import tuning.autotune.AutotuneSession;

import java.util.Objects;

public final class GraphAutotuneSession {
    private final GraphAutotuneRequest request;

    private GraphAutotuneSession(GraphAutotuneRequest request) {
        this.request = Objects.requireNonNull(request, "request cannot be null");
    }

    public static GraphAutotuneSession create(GraphAutotuneRequest request) {
        return new GraphAutotuneSession(request);
    }

    public GraphAutotuneResult run() {
        return new GraphAutotuneResult(
                request.mode(),
                AutotuneSession.create(request.toAutotuneRequest()).run()
        );
    }
}
