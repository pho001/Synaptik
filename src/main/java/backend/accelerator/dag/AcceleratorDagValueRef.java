package backend.accelerator.dag;

import java.util.Objects;

public record AcceleratorDagValueRef(
        AcceleratorDagValueRefKind kind,
        int index
) {
    public AcceleratorDagValueRef {
        Objects.requireNonNull(kind, "kind cannot be null");
        if (kind == AcceleratorDagValueRefKind.NONE) {
            index = -1;
        } else if (index < 0) {
            throw new IllegalArgumentException("non-empty dag ref requires non-negative index");
        }
    }

    public static AcceleratorDagValueRef none() {
        return new AcceleratorDagValueRef(AcceleratorDagValueRefKind.NONE, -1);
    }

    public static AcceleratorDagValueRef externalInput(int index) {
        return new AcceleratorDagValueRef(AcceleratorDagValueRefKind.EXTERNAL_INPUT, index);
    }

    public static AcceleratorDagValueRef nodeOutput(int index) {
        return new AcceleratorDagValueRef(AcceleratorDagValueRefKind.NODE_OUTPUT, index);
    }
}
