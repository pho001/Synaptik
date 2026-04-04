package tuning.workload;

import tensor.Tensor;
import tuning.validate.ValidationReference;

import java.util.Objects;
import java.util.function.Function;

public final class TensorRootWorkloadSpec implements WorkloadSpec {
    private final String name;
    private final WorkloadKind kind;
    private final Function<WorkloadEnvironment, Tensor> rootFactory;
    private final Function<WorkloadEnvironment, ValidationReference> referenceFactory;
    private final Function<WorkloadEnvironment, WorkloadMetadata> metadataFactory;

    public TensorRootWorkloadSpec(
            String name,
            WorkloadKind kind,
            Function<WorkloadEnvironment, Tensor> rootFactory
    ) {
        this(
                name,
                kind,
                rootFactory,
                environment -> ValidationReference.none(),
                environment -> WorkloadMetadata.of(name, kind)
        );
    }

    public TensorRootWorkloadSpec(
            String name,
            WorkloadKind kind,
            Function<WorkloadEnvironment, Tensor> rootFactory,
            Function<WorkloadEnvironment, ValidationReference> referenceFactory,
            Function<WorkloadEnvironment, WorkloadMetadata> metadataFactory
    ) {
        this.name = (name == null || name.isBlank()) ? "workload" : name;
        this.kind = kind == null ? WorkloadKind.GENERIC : kind;
        this.rootFactory = Objects.requireNonNull(rootFactory, "rootFactory cannot be null");
        this.referenceFactory = referenceFactory == null ? environment -> ValidationReference.none() : referenceFactory;
        this.metadataFactory = metadataFactory == null ? environment -> WorkloadMetadata.of(this.name, this.kind) : metadataFactory;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public WorkloadKind kind() {
        return kind;
    }

    @Override
    public WorkloadInstance instantiate(WorkloadEnvironment environment) {
        Objects.requireNonNull(environment, "environment cannot be null");
        Tensor root = rootFactory.apply(environment);
        ValidationReference reference = referenceFactory.apply(environment);
        WorkloadMetadata metadata = metadataFactory.apply(environment);
        return new DefaultWorkloadInstance(root, reference, metadata);
    }
}
