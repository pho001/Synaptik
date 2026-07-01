package backend.lowering;

public enum LoweringFamily {
    FUSED_NATIVE("FUSED_NATIVE"),
    DIRECT_KERNEL("DIRECT_KERNEL"),
    BLAS("BLAS"),
    METAL_GRAPH_PARTITION("METAL_GRAPH_PARTITION"),
    CUDA_GRAPH_PARTITION("CUDA_GRAPH_PARTITION");

    private final String id;

    LoweringFamily(String id) {
        this.id = id;
    }

    public String id() {
        return id;
    }
}
