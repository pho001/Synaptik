package Benchmark.autotune;

import Benchmark.OptimizerCandidate;
import Tensor.DataType;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.OffsetDateTime;
import java.util.Locale;

public final class AutoTuneProgressTracker {
    private static final String RESET = "\u001B[0m";
    private static final String GRAY = "\u001B[90m";

    private final Path progressPath;
    private final Path rowsPath;
    private final DataType dtype;
    private final int candidateStart;
    private final int totalCandidates;
    private final int progressLogEvery;
    private final long progressMinIntervalMs;
    private final long startedNs;
    private long cumulativeRowNs;
    private int recordedRows;
    private long lastLogNs;

    public AutoTuneProgressTracker(
            Path progressPath,
            Path rowsPath,
            DataType dtype,
            int candidateStart,
            int totalCandidates,
            int progressLogEvery,
            long progressMinIntervalMs
    ) {
        this.progressPath = progressPath;
        this.rowsPath = rowsPath;
        this.dtype = dtype;
        this.candidateStart = candidateStart;
        this.totalCandidates = totalCandidates;
        this.progressLogEvery = Math.max(1, progressLogEvery);
        this.progressMinIntervalMs = Math.max(0L, progressMinIntervalMs);
        this.startedNs = System.nanoTime();
        this.lastLogNs = this.startedNs;
        ensureRowsHeader();
        writeProgressJson(
                "STARTED",
                0,
                0,
                0,
                0,
                null,
                Double.NaN,
                Double.NaN,
                null,
                null
        );
    }

    public void recordPhase1(
            String status,
            OptimizerCandidate candidate,
            int processed,
            int valid,
            int mismatch,
            int skipped,
            int safetyMismatch,
            int fullMismatch,
            AutoTuneResult bestTraining,
            AutoTuneResult bestInference,
            double rowMs,
            double fwdMs,
            double trainMs,
            double broadcastMs,
            int graphInfSize,
            int graphTrnSize
    ) {
        recordRowTiming(rowMs);
        appendRow(
                "phase1",
                status,
                processed,
                valid,
                mismatch,
                skipped,
                candidate,
                rowMs,
                avgRowMs(),
                fwdMs,
                trainMs,
                broadcastMs,
                graphInfSize,
                graphTrnSize
        );
        writeProgressJson(
                "PHASE1",
                processed,
                valid,
                mismatch,
                skipped,
                candidate,
                rowMs,
                avgRowMs(),
                bestTraining,
                bestInference
        );
        maybeLog(
                "Phase1",
                status,
                processed,
                valid,
                mismatch,
                skipped,
                candidate,
                rowMs,
                bestTraining,
                bestInference
        );
    }

    public void recordRefine(
            OptimizerCandidate candidate,
            int refinedIndex,
            int finalists,
            AutoTuneResult bestTraining,
            AutoTuneResult bestInference,
            double rowMs,
            double fwdMs,
            double trainMs,
            double broadcastMs
    ) {
        appendRow(
                "refine",
                "REFINE_DONE " + refinedIndex + "/" + finalists,
                refinedIndex,
                refinedIndex,
                0,
                0,
                candidate,
                rowMs,
                Double.NaN,
                fwdMs,
                trainMs,
                broadcastMs,
                -1,
                -1
        );
        writeProgressJson(
                "REFINE",
                refinedIndex,
                refinedIndex,
                0,
                0,
                candidate,
                rowMs,
                Double.NaN,
                bestTraining,
                bestInference
        );
        System.out.println(GRAY + "Refine progress: " + refinedIndex + "/" + finalists
                + " | candidate=" + candidate.name()
                + " | rowMs=" + fmtMillis(rowMs)
                + " | bestTrain=" + bestName(bestTraining)
                + " | bestInf=" + bestName(bestInference)
                + RESET);
    }

    public void complete(
            String phase,
            int processed,
            int valid,
            int mismatch,
            int skipped,
            AutoTuneResult bestTraining,
            AutoTuneResult bestInference
    ) {
        writeProgressJson(
                phase,
                processed,
                valid,
                mismatch,
                skipped,
                null,
                Double.NaN,
                avgRowMs(),
                bestTraining,
                bestInference
        );
    }

    private void recordRowTiming(double rowMs) {
        if (!Double.isFinite(rowMs)) {
            return;
        }
        cumulativeRowNs += (long) (rowMs * 1_000_000.0);
        recordedRows++;
    }

    private double avgRowMs() {
        if (recordedRows == 0) {
            return Double.NaN;
        }
        return (cumulativeRowNs / 1_000_000.0) / recordedRows;
    }

    private void maybeLog(
            String phaseLabel,
            String status,
            int processed,
            int valid,
            int mismatch,
            int skipped,
            OptimizerCandidate candidate,
            double rowMs,
            AutoTuneResult bestTraining,
            AutoTuneResult bestInference
    ) {
        long now = System.nanoTime();
        boolean byCount = processed <= 1 || processed % progressLogEvery == 0;
        boolean byTime = ((now - lastLogNs) / 1_000_000L) >= progressMinIntervalMs;
        if (!byCount && !byTime) {
            return;
        }
        lastLogNs = now;
        double elapsedSec = (now - startedNs) / 1_000_000_000.0;
        double etaSec = processed > 0 ? ((elapsedSec / processed) * Math.max(0, totalCandidates - processed)) : Double.NaN;
        System.out.println(GRAY + phaseLabel + " progress: " + processed + "/" + totalCandidates
                + " | valid=" + valid
                + ", mismatch=" + mismatch
                + ", skipped=" + skipped
                + " | rowMs=" + fmtMillis(rowMs)
                + ", avgRowMs=" + fmtMillis(avgRowMs())
                + ", etaSec=" + fmtSeconds(etaSec)
                + " | status=" + status
                + " | candidate=" + (candidate == null ? "n/a" : candidate.name())
                + " | bestTrain=" + bestName(bestTraining)
                + " | bestInf=" + bestName(bestInference)
                + RESET);
    }

    private void ensureRowsHeader() {
        try {
            if (rowsPath.getParent() != null) {
                Files.createDirectories(rowsPath.getParent());
            }
            if (!Files.exists(rowsPath)) {
                Files.writeString(
                        rowsPath,
                        "timestamp\tphase\tstatus\tdtype\tcandidateStart\tprocessed\ttotal\tvalid\tmismatch\tskipped\tcandidate\trowMs\tavgRowMs\tfwdMs\ttrainMs\tbroadcastMs\tgraphInfSize\tgraphTrnSize\n",
                        StandardCharsets.UTF_8,
                        StandardOpenOption.CREATE,
                        StandardOpenOption.TRUNCATE_EXISTING
                );
            }
        } catch (IOException e) {
            throw new IllegalStateException("Failed to initialize autotune progress rows", e);
        }
    }

    private void appendRow(
            String phase,
            String status,
            int processed,
            int valid,
            int mismatch,
            int skipped,
            OptimizerCandidate candidate,
            double rowMs,
            double avgRowMs,
            double fwdMs,
            double trainMs,
            double broadcastMs,
            int graphInfSize,
            int graphTrnSize
    ) {
        String line = OffsetDateTime.now() + "\t"
                + sanitizeTsv(phase) + "\t"
                + sanitizeTsv(status) + "\t"
                + dtype + "\t"
                + candidateStart + "\t"
                + processed + "\t"
                + totalCandidates + "\t"
                + valid + "\t"
                + mismatch + "\t"
                + skipped + "\t"
                + sanitizeTsv(candidate == null ? "" : candidate.name()) + "\t"
                + fmtMillis(rowMs) + "\t"
                + fmtMillis(avgRowMs) + "\t"
                + fmtMillis(fwdMs) + "\t"
                + fmtMillis(trainMs) + "\t"
                + fmtMillis(broadcastMs) + "\t"
                + graphInfSize + "\t"
                + graphTrnSize + "\n";
        try {
            Files.writeString(rowsPath, line, StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to append autotune progress row", e);
        }
    }

    private void writeProgressJson(
            String phase,
            int processed,
            int valid,
            int mismatch,
            int skipped,
            OptimizerCandidate candidate,
            double rowMs,
            double avgRowMs,
            AutoTuneResult bestTraining,
            AutoTuneResult bestInference
    ) {
        double elapsedSec = (System.nanoTime() - startedNs) / 1_000_000_000.0;
        double etaSec = processed > 0 ? ((elapsedSec / processed) * Math.max(0, totalCandidates - processed)) : Double.NaN;
        StringBuilder sb = new StringBuilder(1024);
        sb.append("{\n");
        sb.append("  \"timestamp\": ").append(jsonString(OffsetDateTime.now().toString())).append(",\n");
        sb.append("  \"phase\": ").append(jsonString(phase)).append(",\n");
        sb.append("  \"dtype\": ").append(jsonString(dtype.name())).append(",\n");
        sb.append("  \"candidateStart\": ").append(candidateStart).append(",\n");
        sb.append("  \"processed\": ").append(processed).append(",\n");
        sb.append("  \"total\": ").append(totalCandidates).append(",\n");
        sb.append("  \"valid\": ").append(valid).append(",\n");
        sb.append("  \"mismatch\": ").append(mismatch).append(",\n");
        sb.append("  \"skipped\": ").append(skipped).append(",\n");
        sb.append("  \"currentCandidate\": ").append(jsonString(candidate == null ? null : candidate.name())).append(",\n");
        sb.append("  \"rowMs\": ").append(Double.isFinite(rowMs) ? fmtMillis(rowMs) : "null").append(",\n");
        sb.append("  \"avgRowMs\": ").append(Double.isFinite(avgRowMs) ? fmtMillis(avgRowMs) : "null").append(",\n");
        sb.append("  \"elapsedSec\": ").append(fmtSeconds(elapsedSec)).append(",\n");
        sb.append("  \"etaSec\": ").append(Double.isFinite(etaSec) ? fmtSeconds(etaSec) : "null").append(",\n");
        sb.append("  \"bestTrainingCandidate\": ").append(jsonString(bestName(bestTraining))).append(",\n");
        sb.append("  \"bestTrainingScore\": ").append(bestTraining == null ? "null" : fmtDouble(bestTraining.score())).append(",\n");
        sb.append("  \"bestInferenceCandidate\": ").append(jsonString(bestName(bestInference))).append(",\n");
        sb.append("  \"bestInferenceScore\": ").append(bestInference == null ? "null" : fmtDouble(bestInference.score())).append("\n");
        sb.append("}\n");
        try {
            if (progressPath.getParent() != null) {
                Files.createDirectories(progressPath.getParent());
            }
            Files.writeString(
                    progressPath,
                    sb.toString(),
                    StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING
            );
        } catch (IOException e) {
            throw new IllegalStateException("Failed to write autotune progress json", e);
        }
    }

    private static String bestName(AutoTuneResult best) {
        return best == null ? null : best.candidate().name();
    }

    private static String sanitizeTsv(String value) {
        if (value == null) {
            return "";
        }
        return value.replace('\t', ' ')
                .replace('\n', ' ')
                .replace('\r', ' ');
    }

    private static String fmtDouble(double v) {
        return String.format(Locale.US, "%.12e", v);
    }

    private static String fmtMillis(double ms) {
        if (!Double.isFinite(ms)) {
            return "n/a";
        }
        return String.format(Locale.US, "%.3f", ms);
    }

    private static String fmtSeconds(double seconds) {
        if (!Double.isFinite(seconds)) {
            return "n/a";
        }
        return String.format(Locale.US, "%.1f", seconds);
    }

    private static String jsonString(String value) {
        if (value == null) {
            return "null";
        }
        String escaped = value
                .replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
        return "\"" + escaped + "\"";
    }
}
