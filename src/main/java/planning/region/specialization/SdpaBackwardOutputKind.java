package planning.region.specialization;

/**
 * Gradient output produced by a specialized canonical scaled-dot-product-attention backward region.
 */
public enum SdpaBackwardOutputKind {
    QUERY,
    KEY,
    VALUE
}
