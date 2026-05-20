package graph.compile.planning.memory;

import graph.compile.planning.value.GraphValueRef;

import java.util.Map;

record RegionBindingAssignment(
        Map<GraphValueRef, RegionMemoryBinding> bindingsByValueRef,
        Map<GraphValueRef, Integer> slotByValueRef,
        Map<Integer, Integer> slotSizes
) {
}
