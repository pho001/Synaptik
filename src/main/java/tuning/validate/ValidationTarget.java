package tuning.validate;

import tensor.Tensor;

import java.util.Objects;

public record ValidationTarget(
        ValidationTargetKind kind,
        String label
) {
    public ValidationTarget {
        kind = kind == null ? ValidationTargetKind.ROOT : kind;
        label = label == null ? "" : label;
        if (kind == ValidationTargetKind.LABEL && label.isBlank()) {
            throw new IllegalArgumentException("ValidationTarget label cannot be blank for LABEL kind");
        }
    }

    public static ValidationTarget root() {
        return new ValidationTarget(ValidationTargetKind.ROOT, "");
    }

    public static ValidationTarget label(String label) {
        return new ValidationTarget(ValidationTargetKind.LABEL, Objects.requireNonNull(label, "label cannot be null"));
    }

    public Tensor resolve(Tensor executionRoot) {
        if (executionRoot == null) {
            throw new IllegalArgumentException("executionRoot cannot be null");
        }
        return switch (kind) {
            case ROOT -> executionRoot;
            case LABEL -> resolveByLabel(executionRoot, label);
        };
    }

    private static Tensor resolveByLabel(Tensor root, String label) {
        Tensor found = null;
        for (Tensor tensor : root.topologicalSort()) {
            if (label.equals(tensor.getLabel())) {
                found = tensor;
            }
        }
        if (found == null) {
            throw new IllegalArgumentException("Validation target label not found in graph: " + label);
        }
        return found;
    }
}
