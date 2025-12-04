package Utils;

import java.util.Collections;
import java.util.List;

class SlotInfo {
    final List<Integer> slots;

    SlotInfo(int singleSlot) {
        this.slots = Collections.singletonList(singleSlot);
    }

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