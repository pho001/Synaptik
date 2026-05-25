package backend.cpu.plan.layout;

import java.util.Objects;

public record StridedLayoutDecision(
        Kind kind
) {
    public enum Kind {
        NONE,
        KEEP_STRIDED,
        MATERIALIZE_ALL,
        MATERIALIZE_INPUT_0,
        MATERIALIZE_INPUT_1
    }

    public static final StridedLayoutDecision NONE = new StridedLayoutDecision(Kind.NONE);
    public static final StridedLayoutDecision KEEP_STRIDED = new StridedLayoutDecision(Kind.KEEP_STRIDED);
    public static final StridedLayoutDecision MATERIALIZE_ALL = new StridedLayoutDecision(Kind.MATERIALIZE_ALL);
    public static final StridedLayoutDecision MATERIALIZE_INPUT_0 = new StridedLayoutDecision(Kind.MATERIALIZE_INPUT_0);
    public static final StridedLayoutDecision MATERIALIZE_INPUT_1 = new StridedLayoutDecision(Kind.MATERIALIZE_INPUT_1);

    public StridedLayoutDecision {
        Objects.requireNonNull(kind, "kind cannot be null");
    }

    public boolean useStridedPath() {
        return kind == Kind.KEEP_STRIDED;
    }

    public boolean forcesMaterialization() {
        return switch (kind) {
            case MATERIALIZE_ALL, MATERIALIZE_INPUT_0, MATERIALIZE_INPUT_1 -> true;
            case NONE, KEEP_STRIDED -> false;
        };
    }

    public boolean shouldForcePrepareInput(int inputIndex) {
        return switch (kind) {
            case MATERIALIZE_ALL -> true;
            case MATERIALIZE_INPUT_0 -> inputIndex == 0;
            case MATERIALIZE_INPUT_1 -> inputIndex == 1;
            case NONE, KEEP_STRIDED -> false;
        };
    }
}
