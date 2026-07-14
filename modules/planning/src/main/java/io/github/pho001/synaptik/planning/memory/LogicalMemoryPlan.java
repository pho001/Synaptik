package io.github.pho001.synaptik.planning.memory;

import io.github.pho001.synaptik.model.graph.ValueId;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;

/**
 * Collects immutable logical memory requirements in deterministic value order.
 *
 * <p>A generated plan contains one requirement per graph value in graph-value encounter order.
 * The public value can also stand alone and therefore permits an empty list. It neither owns a
 * graph nor describes physical memory, allocation, transfers, schedules, or runtime state.</p>
 *
 * <p>Record-generated equality and hashing compare the ordered requirements by value. The
 * generated {@link #toString()} is diagnostic text and is not a serialization or execution
 * format.</p>
 *
 * @param requirements non-null ordered logical requirements with distinct value identities;
 *     membership is copied, elements must be non-null, and exact requirement references are
 *     retained
 */
public record LogicalMemoryPlan(List<LogicalMemoryRequirement> requirements) {
    /**
     * Creates an immutable ordered logical-memory recipe.
     *
     * <p>The list is scanned in encounter order for the first null or later requirement whose
     * value identity equals one already seen. Membership is then copied with
     * {@link List#copyOf(java.util.Collection)}, preserving exact requirement references without
     * retaining the supplied list container.</p>
     *
     * @param requirements non-null ordered requirements to snapshot; elements must be non-null
     *     and their {@link LogicalMemoryRequirement#valueId()} values must be unique by equality
     * @throws NullPointerException if {@code requirements} is {@code null}; the message is
     *     {@code requirements}
     * @throws NullPointerException if an element is {@code null}; the message is
     *     {@code requirements[index]} with its zero-based encounter index
     * @throws IllegalArgumentException if a later requirement repeats an earlier value identity;
     *     the message is {@code requirements[index] duplicates <valueId>} with the later
     *     zero-based encounter index and duplicate {@link ValueId} diagnostic text
     */
    public LogicalMemoryPlan {
        Objects.requireNonNull(requirements, "requirements");

        var observedValueIds = new HashSet<ValueId>();
        for (int index = 0; index < requirements.size(); index++) {
            LogicalMemoryRequirement requirement = Objects.requireNonNull(
                    requirements.get(index), "requirements[" + index + "]");
            if (!observedValueIds.add(requirement.valueId())) {
                throw new IllegalArgumentException(
                        "requirements[" + index + "] duplicates " + requirement.valueId());
            }
        }
        requirements = List.copyOf(requirements);
    }

    /**
     * Returns the immutable ordered snapshot of per-value logical requirements.
     *
     * <p>The result may be empty, contains no null element or duplicate value identity, and
     * cannot be mutated. Exact requirement references are retained; list-container identity with
     * the constructor argument is not promised.</p>
     *
     * @return the non-null immutable ordered snapshot of logical memory requirements; generated
     *     plans follow their owning graph's value order
     */
    @Override
    public List<LogicalMemoryRequirement> requirements() {
        return requirements;
    }
}
