package backend.cpu.fused.optimize;

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

    public boolean cheap() {
        return cheap;
    }

    public boolean contiguous() {
        return contiguous;
    }

    public String id() {
        return id;
    }
}
