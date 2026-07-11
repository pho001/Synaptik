package io.github.pho001.synaptik.model.operation.random;

import io.github.pho001.synaptik.model.operation.OperationAttrs;

/**
 * Carries the two raw 64-bit words of an explicit graph RNG state initializer.
 *
 * <p>Both components are unsigned bit patterns represented by Java {@code long}; every pattern is
 * valid, and signed rendering or ordering has no semantic meaning. The key identifies a
 * caller-selected stream or domain, while the counter identifies the next abstract logical sample
 * position. A future consuming operation retains the key and advances the counter modulo
 * {@code 2^64} by its specified logical draw count. This record performs no normalization, seed
 * expansion, hashing, or key derivation.</p>
 *
 * <p>Record equality and hashing compare the exact bits of both components. A future graph
 * serializer must preserve those bits losslessly, but this value defines no encoding, parser,
 * schema, or stable serialized token.</p>
 *
 * @param key caller-selected stream/domain identity as an unsigned 64-bit bit pattern
 * @param counter next abstract logical sample position as an unsigned 64-bit bit pattern
 */
public record GraphRngStateAttrs(long key, long counter) implements OperationAttrs {
    /**
     * Creates immutable explicit graph RNG state attributes.
     *
     * <p>Both words are retained bit-for-bit. Construction accepts every pair and performs no
     * signedness check, ordering, normalization, derivation, allocation, or random sampling.</p>
     *
     * @param key caller-selected stream/domain identity as an unsigned 64-bit bit pattern
     * @param counter next abstract logical sample position as an unsigned 64-bit bit pattern
     */
    public GraphRngStateAttrs {
        // Every pair of raw 64-bit words is valid and is retained by record assignment.
    }

    /**
     * Returns the caller-selected stream/domain word.
     *
     * @return the exact 64 bits supplied for {@code key}, interpreted as unsigned
     */
    @Override
    public long key() {
        return key;
    }

    /**
     * Returns the next abstract logical sample-position word.
     *
     * @return the exact 64 bits supplied for {@code counter}, interpreted as unsigned
     */
    @Override
    public long counter() {
        return counter;
    }
}
