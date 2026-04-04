package tuning.validate;

public record ValidationReference(
        ValidationReferenceKind kind,
        Object payload
) {
    public ValidationReference {
        kind = kind == null ? ValidationReferenceKind.NONE : kind;
    }

    public static ValidationReference none() {
        return new ValidationReference(ValidationReferenceKind.NONE, null);
    }

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
