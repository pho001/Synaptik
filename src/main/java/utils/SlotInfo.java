package utils;

import java.util.Collections;
import java.util.List;

/**
 * Package-private local-slot allocation record used by {@link SlotManager}.
 */
class SlotInfo {
    final List<Integer> slots;

    SlotInfo(int singleSlot) {
        this.slots = Collections.singletonList(singleSlot);
    }

    /**
     * Creates a grouped slot allocation record.
     *
     * @param groupSlots local-variable slots in the group
     */
    public SlotInfo(List<Integer> groupSlots) {
        if (groupSlots == null || groupSlots.isEmpty()) {
            throw new IllegalArgumentException("Slot group cannot be null or empty");
        }
        this.slots = groupSlots;
    }


    boolean isGroup() {
        return slots.size() > 1;
    }

    int getSlot() {
        return slots.get(0);
    }

    List<Integer> getSlots() {
        return slots;
    }
}
