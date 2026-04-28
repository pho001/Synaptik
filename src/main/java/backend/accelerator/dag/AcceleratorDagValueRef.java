package backend.accelerator.dag;

import java.util.Objects;

/**
 * Reference to a value visible inside a lowered accelerator DAG.
 *
 * @param kind namespace of the referenced value
 * @param index zero-based index inside the selected namespace, or {@code -1} for {@link AcceleratorDagValueRefKind#NONE}
 */
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

    /**
     * Returns an empty operand reference.
     */
    public static AcceleratorDagValueRef none() {
        return new AcceleratorDagValueRef(AcceleratorDagValueRefKind.NONE, -1);
    }

    /**
     * References an entry in {@link AcceleratorDagSpec#externalInputs()}.
     */
    public static AcceleratorDagValueRef externalInput(int index) {
        return new AcceleratorDagValueRef(AcceleratorDagValueRefKind.EXTERNAL_INPUT, index);
    }

    /**
     * References an output from an earlier entry in {@link AcceleratorDagSpec#nodes()}.
     */
    public static AcceleratorDagValueRef nodeOutput(int index) {
        return new AcceleratorDagValueRef(AcceleratorDagValueRefKind.NODE_OUTPUT, index);
    }
}
