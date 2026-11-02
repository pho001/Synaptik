package io.github.pho001.synaptik.model.tensor;

import static io.github.pho001.synaptik.model.tensor.ScanBodyTest.*;
import static org.junit.jupiter.api.Assertions.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class ScanResultTest {
    @Test
    void snapshotsGroupsWithoutMergingRepeatedWrappers() {
        Tensor value = leaf(false);
        List<Tensor> list = new ArrayList<>(List.of(value, value));
        ScanResult result = new ScanResult(list, list, list);
        list.clear();
        assertEquals(List.of(value, value), result.finalCarry());
        assertSame(value, result.outputs().getFirst());
        assertSame(value, result.residuals().getLast());
        assertThrows(UnsupportedOperationException.class, () -> result.outputs().clear());
        assertThrows(UnsupportedOperationException.class, () -> result.residuals().clear());
        assertEquals(result, new ScanResult(result.finalCarry(), result.outputs(), result.residuals()));
        assertEquals(result.hashCode(), new ScanResult(result.finalCarry(), result.outputs(), result.residuals()).hashCode());
    }

    @Test
    void rejectsNullContainersAndElementsAndAllowsEmptyCarrierGroups() {
        assertThrows(NullPointerException.class, () -> new ScanResult(null, List.of(), List.of()));
        assertThrows(NullPointerException.class, () -> new ScanResult(List.of(), null, List.of()));
        assertThrows(NullPointerException.class, () -> new ScanResult(List.of(), List.of(), null));
        assertThrows(NullPointerException.class, () -> new ScanResult(Arrays.asList((Tensor) null), List.of(), List.of()));
        assertTrue(new ScanResult(List.of(), List.of(), List.of()).finalCarry().isEmpty());
    }
}
