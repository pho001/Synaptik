package backend.lowering.partition;

public enum PartitionLegalityStatus {
    SELECTED,
    LEGAL,
    REJECTED,
    FALLBACK_ONLY,
    UNKNOWN
}
