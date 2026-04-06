package tuning.workload;

import tensor.Tensor;
import tuning.validate.ValidationReference;
import tuning.validate.ValidationTarget;

public interface WorkloadInstance {
    Tensor root();

    ValidationTarget validationTarget();

    ValidationReference reference();

    WorkloadMetadata metadata();
}
