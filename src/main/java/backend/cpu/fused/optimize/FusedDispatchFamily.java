package backend.cpu.fused.optimize;

/**
 * Cost and access family assigned to a fused expression for scheduling.
 */
public enum FusedDispatchFamily {
    CHEAP_CONTIGUOUS(true, true, "cheap-contiguous"),
    CHEAP_STRIDED(true, false, "cheap-strided"),
    NON_CHEAP_CONTIGUOUS(false, true, "noncheap-contiguous"),
    NON_CHEAP_STRIDED(false, false, "noncheap-strided");

    private final boolean cheap;
    private final boolean contiguous;
    private final String id;

    FusedDispatchFamily(boolean cheap, boolean contiguous, String id) {
        this.cheap = cheap;
        this.contiguous = contiguous;
        this.id = id;
    }

    /**
     * Returns whether the family contains only cheap operations.
     */
    public boolean cheap() {
        return cheap;
    }

    /**
     * Returns whether the family uses direct or offset contiguous inputs only.
     */
    public boolean contiguous() {
        return contiguous;
    }

    /**
     * Returns the stable diagnostic id for the family.
     */
    public String id() {
        return id;
    }
}
