package tuning.workload;

import tensor.Tensor;
import tuning.validate.ValidationReference;

public interface WorkloadInstance {
    Tensor root();

    ValidationReference reference();

    WorkloadMetadata metadata();
}
