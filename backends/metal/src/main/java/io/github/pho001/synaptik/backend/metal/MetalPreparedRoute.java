package io.github.pho001.synaptik.backend.metal;

import java.util.Objects;
import java.util.Optional;

/**
 * Closed immutable identity for one Metal implementation route selected during cold preparation.
 *
 * <p>Each identity owns its stable candidate wire and broad implementation family. A prepared
 * plan retains exactly one identity before shared requirements escape analysis; finalization and
 * execution may inspect that retained identity but must never replace it, retry another identity,
 * or make a hot-path selection.</p>
 */
enum MetalPreparedRoute {
    /** Exact singleton FLOAT32 NEG custom pipeline. */
    CUSTOM_SINGLE_NEG(1, Family.CUSTOM_KERNEL),

    /** Whole-partition typed MPSGraph executable. */
    MPSGRAPH(2, Family.MPSGRAPH),

    /** Whole-partition Task-0052 custom program. */
    CUSTOM_TASK0052(3, Family.CUSTOM_KERNEL);

    /** Closed implementation family used by trace and route-independent policy. */
    enum Family {
        /** MPSGraph graph/executable resources. */
        MPSGRAPH,
        /** Explicit custom Metal pipeline/program resources. */
        CUSTOM_KERNEL
    }

    private final int wireIdentity;
    private final Family family;

    MetalPreparedRoute(int wireIdentity, Family family) {
        this.wireIdentity = wireIdentity;
        this.family = Objects.requireNonNull(family, "family");
    }

    /**
     * Returns the stable positive candidate wire owned by this route.
     *
     * @return wire {@code 1}, {@code 2}, or {@code 3}
     */
    int wireIdentity() {
        return wireIdentity;
    }

    /**
     * Returns the route's immutable implementation family.
     *
     * @return non-null closed family
     */
    Family family() {
        return family;
    }

    /**
     * Resolves an encoded candidate wire without accepting aliases.
     *
     * @param wireIdentity encoded positive candidate wire
     * @return matching canonical route, or empty for an unknown wire
     */
    static Optional<MetalPreparedRoute> fromWireIdentity(int wireIdentity) {
        return switch (wireIdentity) {
            case 1 -> Optional.of(CUSTOM_SINGLE_NEG);
            case 2 -> Optional.of(MPSGRAPH);
            case 3 -> Optional.of(CUSTOM_TASK0052);
            default -> Optional.empty();
        };
    }
}
