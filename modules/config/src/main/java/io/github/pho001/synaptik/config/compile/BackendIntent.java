package io.github.pho001.synaptik.config.compile;

import io.github.pho001.synaptik.backend.contract.BackendRequirement;
import java.util.Optional;

/**
 * Records whether backend planning is constrained by one hard eligibility target.
 *
 * <p>An unconstrained intent contains no hard target. This absence does not promise discovery,
 * fallback, preference, availability, capability, or successful ownership selection. A
 * constrained intent retains one {@link BackendRequirement} for current Planning to evaluate
 * without weakening it. The optional container itself is always non-null.</p>
 *
 * <p>The current package-private compiler artifact entry passes this value to Planning for every
 * final graph node. This immutable value does not itself evaluate requirements, rank candidates,
 * contain calibrated profile data, select ownership, locate a backend service, or perform
 * lifecycle work. Equality, hashing, and diagnostic text follow ordinary record semantics over
 * the optional target.</p>
 *
 * @param hardRequirement optional hard eligibility target; must not be {@code null}; the exact
 *     optional reference and, when present, exact requirement reference are retained
 */
public record BackendIntent(Optional<BackendRequirement> hardRequirement) {
    /**
     * Creates backend intent from an exact caller-supplied optional hard target.
     *
     * @param hardRequirement optional hard eligibility target; must not be {@code null}; the
     *     exact optional reference and, when present, exact requirement reference are retained
     * @throws NullPointerException if {@code hardRequirement} is {@code null}; the exception
     *     message is {@code hardRequirement}
     */
    public BackendIntent(Optional<BackendRequirement> hardRequirement) {
        if (hardRequirement == null) {
            throw new NullPointerException("hardRequirement");
        }
        this.hardRequirement = hardRequirement;
    }

    /**
     * Creates intent with no hard backend eligibility target.
     *
     * @return a new intent containing an empty optional; never {@code null}
     */
    public static BackendIntent unconstrained() {
        return new BackendIntent(Optional.empty());
    }

    /**
     * Creates intent constrained by one exact hard backend eligibility target.
     *
     * @param requirement hard eligibility target to retain by reference for later planning; must
     *     not be {@code null}
     * @return a new intent containing the exact supplied requirement reference; never
     *     {@code null}
     * @throws NullPointerException if {@code requirement} is {@code null}; the exception message
     *     is {@code requirement}
     */
    public static BackendIntent requiring(BackendRequirement requirement) {
        if (requirement == null) {
            throw new NullPointerException("requirement");
        }
        return new BackendIntent(Optional.of(requirement));
    }

    /**
     * Returns the optional hard eligibility target retained for planning.
     *
     * @return the exact non-null optional reference supplied at construction, containing the
     *     exact supplied requirement reference when present
     */
    public Optional<BackendRequirement> hardRequirement() {
        return hardRequirement;
    }
}
