package backend.lowering;

public enum LoweringFamily {
    FUSED_NATIVE("FUSED_NATIVE"),
    DIRECT_KERNEL("DIRECT_KERNEL"),
    BLAS("BLAS"),
    METAL_GRAPH_REGION("METAL_GRAPH_REGION"),
    METAL_FUSED_ELEMENTWISE_GRAPH("METAL_FUSED_ELEMENTWISE_GRAPH"),
    CUDA_GRAPH_REGION("CUDA_GRAPH_REGION"),
    CUDA_FUSED_ELEMENTWISE_GRAPH("CUDA_FUSED_ELEMENTWISE_GRAPH");

    private final String id;

    LoweringFamily(String id) {
        this.id = id;
    }

    public String id() {
        return id;
    }
}
