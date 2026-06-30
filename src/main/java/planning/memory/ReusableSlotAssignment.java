package planning.memory;

import tensor.Tensor;

import java.util.Map;

record ReusableSlotAssignment(
        Map<Tensor, Integer> slotByOwner,
        Map<Integer, Integer> slotSizes
) {
}
