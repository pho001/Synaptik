package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class MetalOperationRouteCatalogTest {
    @Test
    void catalogReturnsSharedDescriptiveFactsForEveryRegisteredKind() {
        for (var kind : MetalMpsGraphProgram.NodeKind.values()) {
            var entry = MetalOperationRouteCatalog.entry(kind);
            assertSame(entry, MetalOperationRouteCatalog.entry(kind));
            assertNotNull(entry.mpsGraphState());
            assertNotNull(entry.mpsGraphReason());
            assertNotNull(entry.customKernelState());
            assertNotNull(entry.customKernelReason());
        }
        assertThrows(NullPointerException.class, () -> MetalOperationRouteCatalog.entry(null));
    }
}
