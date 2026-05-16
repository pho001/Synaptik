package backend.lowering.region;

public record EmptyRegionPayload() implements RegionBackendPayload {
    public static final EmptyRegionPayload INSTANCE = new EmptyRegionPayload();
}
