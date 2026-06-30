package planning.region.specialization;

import planning.value.GraphValueRef;

import java.util.List;

/**
 * Backend-neutral graph-level specialization candidate.
 *
 * @param kind specialization family
 * @param orderedNodeIds nodes covered by the candidate in graph order
 * @param inputValueRefs values consumed from outside the candidate
 * @param outputValueRef candidate output value
 * @param anchorNodeId node that anchors the specialized unit
 * @param summary short diagnostic summary
 * @param payload structured specialization metadata
 */
public record RegionSpecializationCandidate(
        RegionSpecializationKind kind,
        List<Integer> orderedNodeIds,
        List<GraphValueRef> inputValueRefs,
        GraphValueRef outputValueRef,
        int anchorNodeId,
        String summary,
        RegionSpecializationPayload payload
) {
    public RegionSpecializationCandidate(
            RegionSpecializationKind kind,
            List<Integer> orderedNodeIds,
            List<GraphValueRef> inputValueRefs,
            GraphValueRef outputValueRef,
            int anchorNodeId,
            String summary
    ) {
        this(
                kind,
                orderedNodeIds,
                inputValueRefs,
                outputValueRef,
                anchorNodeId,
                summary,
                RegionSpecializationPayload.empty()
        );
    }

    public RegionSpecializationCandidate {
        if (kind == null) {
            throw new IllegalArgumentException("kind cannot be null");
        }
        orderedNodeIds = List.copyOf(orderedNodeIds == null ? List.of() : orderedNodeIds);
        inputValueRefs = List.copyOf(inputValueRefs == null ? List.of() : inputValueRefs);
        if (outputValueRef == null) {
            throw new IllegalArgumentException("outputValueRef cannot be null");
        }
        if (anchorNodeId < 0) {
            throw new IllegalArgumentException("anchorNodeId must be >= 0");
        }
        summary = summary == null ? "" : summary;
        payload = payload == null ? RegionSpecializationPayload.empty() : payload;
    }
}
