package operations.linalg;
import operations.Operation;

/**
 * Computes one requested gradient output of scaled dot-product attention.
 *
 * <p>The same backward descriptor type is used for query, key, and value
 * gradients; {@link OutputKind} selects which output this instance represents.</p>
 */
public final class scaledDotProductAttentionBackward implements Operation {
    /**
     * Gradient output produced by an attention backward descriptor.
     */
    public enum OutputKind {
        QUERY,
        KEY,
        VALUE
    }

    private final OutputKind outputKind;

    /**
     * Creates an attention backward descriptor for one gradient output.
     *
     * @param outputKind query, key, or value gradient to produce
     * @throws IllegalArgumentException if {@code outputKind} is {@code null}
     */
    public scaledDotProductAttentionBackward(OutputKind outputKind) {
        if (outputKind == null) {
            throw new IllegalArgumentException("scaledDotProductAttentionBackward outputKind cannot be null");
        }
        this.outputKind = outputKind;
    }

    @Override
    public OpType opType() {
        return OpType.SCALED_DOT_PRODUCT_ATTENTION_BACKWARD;
    }

    @Override
    public String getExpression() {
        return "scaledDotProductAttentionBackward[" + outputKind.name().toLowerCase() + "]";
    }

    /**
     * Returns the selected gradient output.
     *
     * @return gradient output kind produced by this descriptor
     */
    public OutputKind getOutputKind() {
        return outputKind;
    }
}
