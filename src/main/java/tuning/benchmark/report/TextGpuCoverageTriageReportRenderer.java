package tuning.benchmark.report;

public final class TextGpuCoverageTriageReportRenderer {
    private TextGpuCoverageTriageReportRenderer() {
    }

    public static String render(GpuCoverageTriageReport report) {
        if (report == null) {
            throw new IllegalArgumentException("report cannot be null");
        }
        StringBuilder sb = new StringBuilder();
        sb.append("GPU Coverage Gap Triage\n\n");

        sb.append("Hot Path Targets\n");
        for (GpuHotPathCoverageTarget target : report.hotPathTargets()) {
            sb.append("- ")
                    .append(target.workloadName())
                    .append(" kind=")
                    .append(target.targetKind())
                    .append(" ownerPhase=Phase ")
                    .append(target.ownerPhase())
                    .append(" families=")
                    .append(String.join(",", target.requirementFamilies()))
                    .append('\n');
        }
        sb.append('\n');

        sb.append("cudaHotPathBlockers:\n");
        for (GpuCoverageTriageReport.CudaHotPathBlockerEntry blocker : report.cudaHotPathBlockers()) {
            sb.append("- ")
                    .append(blocker.workloadName())
                    .append(" class=")
                    .append(blocker.blockerClass().name())
                    .append(" detail=")
                    .append(blocker.detail())
                    .append('\n');
        }
        sb.append('\n');

        sb.append("Top Coverage Gaps\n");
        for (GpuCoverageGap gap : report.topGaps()) {
            sb.append("- ")
                    .append(gap.workloadName())
                    .append(" / ")
                    .append(gap.backend())
                    .append(" / ")
                    .append(gap.category().name())
                    .append(" reason=")
                    .append(gap.reason())
                    .append(" count=")
                    .append(gap.count())
                    .append(" score=")
                    .append(gap.severityScore())
                    .append(" family=")
                    .append(gap.requirementFamily())
                    .append('\n');
        }
        sb.append('\n');

        sb.append("Requirement Family Ranking\n");
        for (var entry : report.gapCountsByRequirementFamily().entrySet()) {
            sb.append("- ").append(entry.getKey()).append(": ").append(entry.getValue()).append('\n');
        }
        sb.append('\n');

        sb.append("Downstream Phase Targets\n");
        for (var entry : report.downstreamPhaseTargets().entrySet()) {
            sb.append("- Phase ")
                    .append(entry.getKey())
                    .append(": ")
                    .append(String.join(", ", entry.getValue()))
                    .append('\n');
        }
        return sb.toString();
    }
}
