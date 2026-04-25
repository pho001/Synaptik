package backend.cuda.bridge;

public record CudaBridgeContext(
        boolean available,
        String reason
) {
    public CudaBridgeContext {
        reason = reason == null ? "" : reason;
    }

    public static CudaBridgeContext unavailable(String reason) {
        return new CudaBridgeContext(false, reason);
    }
}
