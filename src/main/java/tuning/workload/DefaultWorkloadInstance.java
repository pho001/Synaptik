package tuning.workload;

import tensor.Tensor;
import tuning.validate.ValidationReference;
import tuning.validate.ValidationTarget;

import java.util.Objects;

/**
 * Default immutable {@link WorkloadInstance} implementation.
 *
 * @param root executable root tensor
 * @param validationTarget target selected for validation; {@code null} means root
 * @param reference validation reference; {@code null} means none
 * @param metadata workload metadata; {@code null} is derived from the root label
 */
public record DefaultWorkloadInstance(
        Tensor root,
        ValidationTarget validationTarget,
        ValidationReference reference,
        WorkloadMetadata metadata
) implements WorkloadInstance {
    public DefaultWorkloadInstance {
        Objects.requireNonNull(root, "root cannot be null");
        validationTarget = validationTarget == null ? ValidationTarget.root() : validationTarget;
        reference = reference == null ? ValidationReference.none() : reference;
        metadata = metadata == null ? WorkloadMetadata.of(root.getLabel(), WorkloadKind.GENERIC) : metadata;
    }

    /**
     * Creates a workload instance that validates the root tensor.
     *
     * @param root executable root tensor
     * @param reference validation reference
     * @param metadata workload metadata
     */
    public DefaultWorkloadInstance(
            Tensor root,
            ValidationReference reference,
            WorkloadMetadata metadata
    ) {
        this(root, ValidationTarget.root(), reference, metadata);
    }
}
