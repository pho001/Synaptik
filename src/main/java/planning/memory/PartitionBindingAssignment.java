package planning.memory;

import planning.value.GraphValueRef;

import java.util.Map;

record PartitionBindingAssignment(
        Map<GraphValueRef, PartitionMemoryBinding> bindingsByValueRef,
        Map<GraphValueRef, Integer> slotByValueRef,
        Map<Integer, Integer> slotSizes
) {
}
