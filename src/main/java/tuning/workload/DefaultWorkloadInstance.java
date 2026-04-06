package tuning.workload;

import tensor.Tensor;
import tuning.validate.ValidationReference;
import tuning.validate.ValidationTarget;

import java.util.Objects;

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

    public DefaultWorkloadInstance(
            Tensor root,
            ValidationReference reference,
            WorkloadMetadata metadata
    ) {
        this(root, ValidationTarget.root(), reference, metadata);
    }
}
