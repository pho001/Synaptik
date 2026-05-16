package backend.lowering.region;

public enum RegionRole {
    PROVIDER,
    LOCAL_KERNEL,
    VIEW_ALIAS,
    BOUNDARY_OUTPUT,
    CONTROL,
    FALLBACK_ONLY,
    UNKNOWN
}
