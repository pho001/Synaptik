package planning.partition.specialization;

/**
 * Gradient output produced by a specialized canonical scaled-dot-product-attention backward partition.
 */
public enum SdpaBackwardOutputKind {
    QUERY,
    KEY,
    VALUE
}
