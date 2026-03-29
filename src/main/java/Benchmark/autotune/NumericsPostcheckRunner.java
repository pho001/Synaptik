package Benchmark.autotune;

import Benchmark.OptimizerCandidate;
import Numerics.NumericsMetrics;
import Numerics.NumericsPolicy;
import Numerics.NumericsReport;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.function.Function;

public final class NumericsPostcheckRunner {
    private NumericsPostcheckRunner() {}

    @FunctionalInterface
    public interface Probe {
        NumericsReport run(OptimizerCandidate baselineNoOptSameKnobs, OptimizerCandidate finalist);
    }

    public static NumericsPostcheckResult run(
            List<OptimizerCandidate> finalists,
            NumericsPostcheckConfig config,
            Probe probe,
            UnsafeCandidateHistory history,
            Function<OptimizerCandidate, String> fingerprintFn
    ) {
        int topN = Math.max(0, config.topN());
        List<OptimizerCandidate> kept = new ArrayList<>();
        List<NumericsPostcheckRow> rows = new ArrayList<>();
        List<NumericsPostcheckDrop> dropped = new ArrayList<>();

        int checked = 0;
        int markedUnsafe = 0;
        for (OptimizerCandidate finalist : finalists) {
            if (checked >= topN) {
                kept.add(finalist);
                continue;
            }
            OptimizerCandidate baselineNoOptSameKnobs = new OptimizerCandidate(
                    finalist.name() + "_NUM_NOOPT",
                    List.of(),
                    finalist.knobs()
            );
            NumericsReport report = probe.run(baselineNoOptSameKnobs, finalist);
            checked++;
            rows.add(new NumericsPostcheckRow(finalist.name(), report));
            if (report.verdict.status == NumericsPolicy.Status.UNSAFE) {
                history.markUnsafe(
                        fingerprintFn.apply(finalist),
                        finalist.name(),
                        "NUMERICS_POSTCHECK_UNSAFE: " + report.verdict.reason
                );
                markedUnsafe++;
                dropped.add(new NumericsPostcheckDrop(finalist.name(), report.verdict.reason));
                continue;
            }
            kept.add(finalist);
        }

        Path reportPath = writeReport(config, rows);
        return new NumericsPostcheckResult(List.copyOf(kept), checked, markedUnsafe, reportPath, List.copyOf(dropped));
    }

    private static Path writeReport(NumericsPostcheckConfig config, List<NumericsPostcheckRow> rows) {
        if (rows.isEmpty()) {
            return null;
        }
        String ts = LocalDateTime.now().format(config.timestampFormat());
        String dtype = config.dtype().name().toLowerCase(Locale.ROOT);
        Path path = config.reportDir().resolve("autotune-postcheck-" + dtype + "-" + ts + ".tsv");
        List<String> lines = new ArrayList<>(rows.size() + 1);
        lines.add("candidate\tstatus\treason\taggMaxAbs\taggMaxRel\taggMaxUlp\taggInvalid"
                + "\toutMaxAbs\toutMaxRel\toutMaxUlp\toutInvalid"
                + "\tgradAMaxAbs\tgradAMaxRel\tgradAMaxUlp\tgradAInvalid"
                + "\tgradBMaxAbs\tgradBMaxRel\tgradBMaxUlp\tgradBInvalid"
                + "\tgradCMaxAbs\tgradCMaxRel\tgradCMaxUlp\tgradCInvalid"
                + "\tbroadcastMaxAbs\tbroadcastMaxRel\tbroadcastMaxUlp\tbroadcastInvalid");
        for (NumericsPostcheckRow row : rows) {
            lines.add(row.toLine());
        }
        try {
            Files.createDirectories(path.getParent());
            Files.write(path, lines, StandardCharsets.UTF_8);
            return path;
        } catch (IOException e) {
            throw new IllegalStateException("Failed to write numerics post-check report", e);
        }
    }

    private static String sanitizeTsv(String value) {
        if (value == null) {
            return "";
        }
        return value.replace('\t', ' ')
                .replace('\n', ' ')
                .replace('\r', ' ');
    }

    private static String fmtDouble(double value) {
        return String.format(Locale.US, "%.12e", value);
    }

    private record NumericsPostcheckRow(String candidateName, NumericsReport report) {
        private String toLine() {
            StringBuilder sb = new StringBuilder(512);
            sb.append(sanitizeTsv(candidateName)).append('\t');
            sb.append(report.verdict.status).append('\t');
            sb.append(sanitizeTsv(report.verdict.reason)).append('\t');
            appendAggregate(sb, report.aggregate);
            appendSignal(sb, report.out);
            appendSignal(sb, report.gradA);
            appendSignal(sb, report.gradB);
            appendSignal(sb, report.gradC);
            appendSignal(sb, report.broadcast);
            return sb.toString();
        }

        private static void appendAggregate(StringBuilder sb, NumericsMetrics.AggregateMetrics m) {
            sb.append(fmtDouble(m.maxAbs)).append('\t');
            sb.append(fmtDouble(m.maxRel)).append('\t');
            sb.append(m.maxUlp).append('\t');
            sb.append(m.invalidCount).append('\t');
        }

        private static void appendSignal(StringBuilder sb, NumericsMetrics.SignalMetrics m) {
            sb.append(fmtDouble(m.maxAbs)).append('\t');
            sb.append(fmtDouble(m.maxRel)).append('\t');
            sb.append(m.maxUlp).append('\t');
            sb.append(m.invalidCount).append('\t');
        }
    }
}
