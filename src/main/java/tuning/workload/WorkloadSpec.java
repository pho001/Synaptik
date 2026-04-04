package tuning.workload;

public interface WorkloadSpec {
    String name();

    WorkloadKind kind();

    WorkloadInstance instantiate(WorkloadEnvironment environment);
}
