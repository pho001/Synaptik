package tuning.validate;

import java.util.List;
import java.util.Map;

public record SnapshotValidationReference(
        TensorSnapshot output,
        Map<String, TensorSnapshot> gradients,
        List<String> gradientTargetLabels
) {
    public SnapshotValidationReference {
        if (output == null) {
            throw new IllegalArgumentException("output snapshot cannot be null");
        }
        gradients = gradients == null ? Map.of() : Map.copyOf(gradients);
        gradientTargetLabels = gradientTargetLabels == null ? List.of() : List.copyOf(gradientTargetLabels);
    }
}
