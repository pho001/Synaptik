package backend.lowering.region;

public enum RegionStorageContract {
    CPU_ARRAY,
    CPU_NATIVE,
    DEVICE_BUFFER,
    VIEW_ALIAS,
    MIXED_BOUNDARY,
    UNKNOWN
}
