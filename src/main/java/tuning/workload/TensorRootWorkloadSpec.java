package tuning.workload;

import tensor.Tensor;
import tuning.validate.ValidationReference;
import tuning.validate.ValidationTarget;

import java.util.Objects;
import java.util.function.Function;

public final class TensorRootWorkloadSpec implements WorkloadSpec {
    private final String name;
    private final WorkloadKind kind;
    private final WorkloadGraphFactory graphFactory;
    private final Function<WorkloadEnvironment, ValidationTarget> validationTargetFactory;
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
                (WorkloadGraphFactory) environment -> WorkloadGraph.of(rootFactory.apply(environment)),
                environment -> ValidationTarget.root(),
                environment -> ValidationReference.none(),
                environment -> WorkloadMetadata.of(name, kind),
                true
        );
    }

    public static TensorRootWorkloadSpec fromGraphFactory(
            String name,
            WorkloadKind kind,
            WorkloadGraphFactory graphFactory
    ) {
        return new TensorRootWorkloadSpec(
                name,
                kind,
                graphFactory,
                environment -> ValidationTarget.root(),
                environment -> ValidationReference.none(),
                environment -> WorkloadMetadata.of(name, kind),
                true
        );
    }

    public TensorRootWorkloadSpec(
            String name,
            WorkloadKind kind,
            Function<WorkloadEnvironment, Tensor> rootFactory,
            Function<WorkloadEnvironment, ValidationTarget> validationTargetFactory,
            Function<WorkloadEnvironment, ValidationReference> referenceFactory,
            Function<WorkloadEnvironment, WorkloadMetadata> metadataFactory
    ) {
        this(
                name,
                kind,
                (WorkloadGraphFactory) environment -> WorkloadGraph.of(rootFactory.apply(environment)),
                validationTargetFactory,
                referenceFactory,
                metadataFactory,
                true
        );
    }

    public static TensorRootWorkloadSpec fromGraphFactory(
            String name,
            WorkloadKind kind,
            WorkloadGraphFactory graphFactory,
            Function<WorkloadEnvironment, ValidationTarget> validationTargetFactory,
            Function<WorkloadEnvironment, ValidationReference> referenceFactory,
            Function<WorkloadEnvironment, WorkloadMetadata> metadataFactory
    ) {
        return new TensorRootWorkloadSpec(
                name,
                kind,
                graphFactory,
                validationTargetFactory,
                referenceFactory,
                metadataFactory,
                true
        );
    }

    private TensorRootWorkloadSpec(
            String name,
            WorkloadKind kind,
            WorkloadGraphFactory graphFactory,
            Function<WorkloadEnvironment, ValidationTarget> validationTargetFactory,
            Function<WorkloadEnvironment, ValidationReference> referenceFactory,
            Function<WorkloadEnvironment, WorkloadMetadata> metadataFactory,
            boolean graphFactoryConstructor
    ) {
        this.name = (name == null || name.isBlank()) ? "workload" : name;
        this.kind = kind == null ? WorkloadKind.GENERIC : kind;
        this.graphFactory = Objects.requireNonNull(graphFactory, "graphFactory cannot be null");
        this.validationTargetFactory = validationTargetFactory == null ? environment -> ValidationTarget.root() : validationTargetFactory;
        this.referenceFactory = referenceFactory == null ? environment -> ValidationReference.none() : referenceFactory;
        this.metadataFactory = metadataFactory == null ? environment -> WorkloadMetadata.of(this.name, this.kind) : metadataFactory;
    }

    public TensorRootWorkloadSpec(
            String name,
            WorkloadKind kind,
            Function<WorkloadEnvironment, Tensor> rootFactory,
            Function<WorkloadEnvironment, ValidationReference> referenceFactory,
            Function<WorkloadEnvironment, WorkloadMetadata> metadataFactory
    ) {
        this(name, kind, rootFactory, environment -> ValidationTarget.root(), referenceFactory, metadataFactory);
    }

    public static TensorRootWorkloadSpec fromGraphFactory(
            String name,
            WorkloadKind kind,
            WorkloadGraphFactory graphFactory,
            Function<WorkloadEnvironment, ValidationReference> referenceFactory,
            Function<WorkloadEnvironment, WorkloadMetadata> metadataFactory
    ) {
        return fromGraphFactory(name, kind, graphFactory, environment -> ValidationTarget.root(), referenceFactory, metadataFactory);
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
        WorkloadGraph graph = graphFactory.create(environment);
        ValidationTarget validationTarget = validationTargetFactory.apply(environment);
        ValidationReference reference = referenceFactory.apply(environment);
        WorkloadMetadata metadata = metadataFactory.apply(environment);
        return new DefaultWorkloadInstance(
                graph.root(),
                validationTarget,
                reference,
                metadata,
                graph.backendIntentPlan()
        );
    }
}
