package tuning.validate;

/**
 * Reference source used by validation.
 *
 * <p>A reference can be absent, embedded as a tensor snapshot, or generated from
 * a baseline execution profile. Payload type is determined by {@link #kind()}:
 * {@link ValidationReferenceKind#SNAPSHOT} uses
 * {@link SnapshotValidationReference}; {@link ValidationReferenceKind#BASELINE_PROFILE}
 * uses {@link BaselineProfileValidationReference}.</p>
 *
 * @param kind reference kind
 * @param payload reference payload matching the kind, or {@code null} for none
 */
public record ValidationReference(
        ValidationReferenceKind kind,
        Object payload
) {
    public ValidationReference {
        kind = kind == null ? ValidationReferenceKind.NONE : kind;
    }

    /**
     * @return reference that disables external comparison
     */
    public static ValidationReference none() {
        return new ValidationReference(ValidationReferenceKind.NONE, null);
    }

    /**
     * Creates a snapshot reference.
     *
     * @param output expected output tensor snapshot
     * @param gradients expected gradient snapshots by label
     * @param gradientTargetLabels gradient labels to compare
     * @return snapshot reference
     */
    public static ValidationReference snapshot(
            TensorSnapshot output,
            java.util.Map<String, TensorSnapshot> gradients,
            java.util.List<String> gradientTargetLabels
    ) {
        return new ValidationReference(
                ValidationReferenceKind.SNAPSHOT,
                new SnapshotValidationReference(output, gradients, gradientTargetLabels)
        );
    }

    /**
     * Creates a reference that executes a baseline profile for comparison.
     *
     * @param profile baseline execution profile
     * @param gradientTargetLabels gradient labels to compare
     * @return baseline-profile reference
     */
    public static ValidationReference baselineProfile(
            config.profile.ExecutionProfile profile,
            java.util.List<String> gradientTargetLabels
    ) {
        return new ValidationReference(
                ValidationReferenceKind.BASELINE_PROFILE,
                new BaselineProfileValidationReference(profile, gradientTargetLabels)
        );
    }
}
