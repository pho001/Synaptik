package backend.cpu1.storage;

/**
 * Prepare-time storage access class for cpu1 tensors.
 */
public enum Cpu1StorageAccessKind {
    DENSE_CONTIGUOUS,
    DENSE_WITH_OFFSET,
    STRIDED,
    BROADCAST,
    UNSUPPORTED
}
