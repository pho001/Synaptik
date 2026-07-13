package io.github.pho001.synaptik.model.tensor;

import java.util.Objects;

/**
 * Attention output and normalized weights produced by one exact attention occurrence.
 *
 * <p>When returned by {@link Tensor#scaledDotProductAttentionWithWeights(Tensor, Tensor)}, the
 * output occupies producer slot zero and has Shape {@code [..., L, Ev]}; the normalized weights
 * occupy slot one and have score Shape {@code [..., L, S]}. Both use the promoted
 * query/key/value data type, have unresolved layout, and retain the same exact producer,
 * operation, attributes, ordered inputs, and corresponding output-descriptor references. Output
 * gradient eligibility is the query/key/value request OR, while weights eligibility is the
 * query/key request OR.</p>
 *
 * <p>This shallowly immutable carrier retains the exact Tensor wrappers and has ordinary record
 * value semantics. Its public constructor checks only component nullability; it does not validate
 * descriptors or producer agreement for independently supplied Tensors. The carrier owns no
 * execution, storage, gradient rule, saved-value lifetime, compiler, backend, or runtime
 * lifecycle behavior.</p>
 *
 * @param output non-null attention output Tensor retained by exact reference; attention
 *     construction supplies Shape {@code [..., L, Ev]} at producer slot zero
 * @param weights non-null normalized attention-weights Tensor retained by exact reference;
 *     attention construction supplies Shape {@code [..., L, S]} at producer slot one
 */
public record ScaledDotProductAttentionResult(Tensor output, Tensor weights) {
    /**
     * Retains the two exact output wrappers in producer slot order.
     *
     * @param output non-null attention output wrapper at slot zero
     * @param weights non-null normalized weights wrapper at slot one
     * @throws NullPointerException if {@code output} or {@code weights} is null, checked in
     *     declaration order with message {@code output} or {@code weights}
     */
    public ScaledDotProductAttentionResult {
        output = Objects.requireNonNull(output, "output");
        weights = Objects.requireNonNull(weights, "weights");
    }
}
