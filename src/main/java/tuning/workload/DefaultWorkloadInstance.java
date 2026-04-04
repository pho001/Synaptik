package tuning.workload;

import tensor.Tensor;
import tuning.validate.ValidationReference;

import java.util.Objects;

public record DefaultWorkloadInstance(
        Tensor root,
        ValidationReference reference,
        WorkloadMetadata metadata
) implements WorkloadInstance {
    public DefaultWorkloadInstance {
        Objects.requireNonNull(root, "root cannot be null");
        reference = reference == null ? ValidationReference.none() : reference;
        metadata = metadata == null ? WorkloadMetadata.of(root.getLabel(), WorkloadKind.GENERIC) : metadata;
    }
}
