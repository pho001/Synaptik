package backend.cpu1;

import operations.Operation.OpType;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class Cpu1CpuParityInventoryTest {
    @Test
    void cpu1CoverageGateListsAllOldCpuDirectOps() {
        Cpu1CoverageReport report = Cpu1CoverageReport.current();

        assertTrue(
                report.missingRequiredOps().isEmpty(),
                report::gateReport
        );
    }

    @Test
    void oldCpuNonDirectOpsAreExplicitlyClassified() {
        Cpu1CoverageReport report = Cpu1CoverageReport.current();

        assertTrue(
                report.unclassifiedNonOldCpuDirectOps().isEmpty(),
                report::gateReport
        );
        assertTrue(
                report.legacyOrSpecialWithoutOldCpuDirectKernelOps()
                        .containsKey(OpType.SCALED_DOT_PRODUCT_ATTENTION_BACKWARD),
                report::gateReport
        );
    }

    @Test
    void coverageBucketsAreExplicitAndDisjoint() {
        Cpu1CoverageReport report = Cpu1CoverageReport.current();

        assertNoOverlap(report.cpu1PreparedFamilyRoutes(), report.allowedMissingOrDeferredOps(), report);
        assertNoOverlap(report.cpu1PreparedFamilyRoutes(), report.intentionallyGraphLoweredOrNotDirectOps(), report);
        assertNoOverlap(report.allowedMissingOrDeferredOps(), report.intentionallyGraphLoweredOrNotDirectOps(), report);
        assertTrue(
                report.oldCpuDirectKernelClasses().keySet().containsAll(report.cpu1PreparedFamilyRoutes().keySet()),
                report::gateReport
        );
        assertTrue(
                report.oldCpuDirectKernelClasses().keySet().containsAll(report.allowedMissingOrDeferredOps().keySet()),
                report::gateReport
        );
        assertTrue(
                report.oldCpuDirectKernelClasses().keySet()
                        .containsAll(report.intentionallyGraphLoweredOrNotDirectOps().keySet()),
                report::gateReport
        );
    }

    @Test
    void requiredSentinelOpsArePresent() {
        Cpu1CoverageReport report = Cpu1CoverageReport.current();

        assertTrue(report.oldCpuDirectKernelClasses().containsKey(OpType.CROSS_ENTROPY_LOSS_INDICES), report::gateReport);
        assertEquals("matmul", report.cpu1PreparedFamilyRoutes().get(OpType.MATMUL), report.gateReport());
        assertEquals("matmul", report.cpu1PreparedFamilyRoutes().get(OpType.LINEAR), report.gateReport());
        assertEquals("index", report.cpu1PreparedFamilyRoutes().get(OpType.GATHER), report.gateReport());
        assertEquals("conv2d", report.cpu1PreparedFamilyRoutes().get(OpType.CONV2D), report.gateReport());
        assertEquals("attention", report.cpu1PreparedFamilyRoutes().get(OpType.SCALED_DOT_PRODUCT_ATTENTION), report.gateReport());
        assertTrue(report.intentionallyGraphLoweredOrNotDirectOps().containsKey(OpType.FUSED), report::gateReport);
    }

    @Test
    void coverageGateReportNamesRequiredAndDeferredBuckets() {
        String text = Cpu1CoverageReport.current().gateReport();

        assertTrue(text.contains("missingRequiredOps="), text);
        assertTrue(text.contains("allowedMissingOrDeferredOps="), text);
        assertTrue(text.contains("intentionallyGraphLoweredOrNotDirectOps="), text);
        assertTrue(text.contains("legacyOrSpecialWithoutOldCpuDirectKernelOps="), text);
    }

    private static void assertNoOverlap(
            java.util.Map<OpType, String> left,
            java.util.Map<OpType, String> right,
            Cpu1CoverageReport report
    ) {
        java.util.EnumSet<OpType> overlap = java.util.EnumSet.noneOf(OpType.class);
        overlap.addAll(left.keySet());
        overlap.retainAll(right.keySet());
        assertTrue(overlap.isEmpty(), report::gateReport);
    }
}
