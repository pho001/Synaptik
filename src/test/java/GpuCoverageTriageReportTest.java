import backend.ComputeBackend;
import org.junit.jupiter.api.Test;
import tuning.benchmark.report.BenchmarkReport;
import tuning.benchmark.report.BenchmarkSuiteReport;
import tuning.benchmark.report.GpuCoverageGapCategory;
import tuning.benchmark.report.GpuCoverageTriageReport;
import tuning.benchmark.report.JsonGpuCoverageTriageReportRenderer;
import tuning.benchmark.report.TextGpuCoverageTriageReportRenderer;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class GpuCoverageTriageReportTest {
    @Test
    void buildsTriageReportFromSuiteAndDefaultTargets() {
        GpuCoverageTriageReport report = GpuCoverageTriageReport.fromSuite(suiteReport(), 20);

        assertEquals(4, report.hotPathTargets().size());
        assertFalse(report.topGaps().isEmpty());
        assertTrue(report.gapCountsByCategory().containsKey(GpuCoverageGapCategory.DEVICE_HANDOFF));
        assertTrue(report.gapCountsByRequirementFamily().containsKey("GPUHARDEN"));
    }

    @Test
    void textRendererPrintsStableTriageSections() {
        String text = TextGpuCoverageTriageReportRenderer.render(GpuCoverageTriageReport.fromSuite(suiteReport(), 20));

        assertTrue(text.contains("GPU Coverage Gap Triage"));
        assertTrue(text.contains("Hot Path Targets"));
        assertTrue(text.contains("Top Coverage Gaps"));
        assertTrue(text.contains("Requirement Family Ranking"));
        assertTrue(text.contains("Downstream Phase Targets"));
        assertTrue(text.contains("transformer_block_hot_path"));
    }

    @Test
    void jsonRendererPrintsStableTriageFields() {
        String json = JsonGpuCoverageTriageReportRenderer.render(GpuCoverageTriageReport.fromSuite(suiteReport(), 20));

        assertTrue(json.contains("\"hotPathTargets\""));
        assertTrue(json.contains("\"topGaps\""));
        assertTrue(json.contains("\"gapCountsByCategory\""));
        assertTrue(json.contains("\"gapCountsByRequirementFamily\""));
        assertTrue(json.contains("\"downstreamPhaseTargets\""));
        assertTrue(json.contains("\"transformer_block_hot_path\""));
    }

    @Test
    void reportMapsTopGapsToDownstreamPhaseTargets() {
        GpuCoverageTriageReport report = GpuCoverageTriageReport.fromSuite(suiteReport(), 20);
        String text = TextGpuCoverageTriageReportRenderer.render(report);
        String json = JsonGpuCoverageTriageReportRenderer.render(report);

        assertTrue(report.downstreamPhaseTargets().containsKey(15));
        assertTrue(report.downstreamPhaseTargets().containsKey(16));
        assertTrue(report.downstreamPhaseTargets().containsKey(17));
        assertTrue(report.downstreamPhaseTargets().containsKey(18));
        assertTrue(report.downstreamPhaseTargets().containsKey(19));
        assertTrue(report.downstreamPhaseTargets().containsKey(20));
        assertTrue(text.contains("Phase 15"));
        assertTrue(text.contains("Phase 16"));
        assertTrue(text.contains("Phase 17"));
        assertTrue(text.contains("Phase 18"));
        assertTrue(text.contains("Phase 19"));
        assertTrue(text.contains("Phase 20"));
        assertTrue(json.contains("Phase 15"));
        assertTrue(json.contains("Phase 20"));
    }

    private static BenchmarkSuiteReport suiteReport() {
        return new BenchmarkSuiteReport(null, List.of(
                BenchmarkReport.of("transformer_block_hot_path", List.of(
                        GpuCoverageGapTriageTest.candidate(
                                "metal-triage",
                                GpuCoverageSummaryTest.traceFor("GPU_METAL", ComputeBackend.GPU_METAL)
                        )
                )),
                BenchmarkReport.of("mlp_classifier_small", List.of(
                        GpuCoverageGapTriageTest.candidate(
                                "cuda-triage",
                                GpuCoverageSummaryTest.traceFor("GPU_CUDA", ComputeBackend.GPU_CUDA)
                        )
                ))
        ));
    }
}
