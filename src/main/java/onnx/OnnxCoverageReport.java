package onnx;

import operations.Operation;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.EnumMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * Markdown renderer for the ONNX interchange coverage matrix.
 */
public final class OnnxCoverageReport {
    private OnnxCoverageReport() {
    }

    public static String renderMarkdown() {
        StringBuilder out = new StringBuilder();
        out.append("# ONNX Coverage Report\n\n");
        out.append("Generated from `OnnxCoverageMatrix`; do not hand-edit status rows.\n\n");
        appendSummary(out);
        out.append("## Matrix\n\n");
        out.append("| ONNX op | Synaptik mapping | Import | Export | CPU | Metal | CUDA | Round-trip evidence | Mapped op types | Limitations |\n");
        out.append("|---|---|---|---|---|---|---|---|---|---|\n");
        for (OnnxCoverageMatrix.Entry entry : OnnxCoverageMatrix.entries()) {
            out.append("| ")
                    .append(escape(entry.onnxOp()))
                    .append(" | ")
                    .append(escape(entry.synaptikMapping()))
                    .append(" | ")
                    .append(status(entry.importStatus()))
                    .append(" | ")
                    .append(status(entry.exportStatus()))
                    .append(" | ")
                    .append(status(entry.cpuStatus()))
                    .append(" | ")
                    .append(status(entry.metalStatus()))
                    .append(" | ")
                    .append(status(entry.cudaStatus()))
                    .append(" | ")
                    .append(evidence(entry.roundTripEvidence()))
                    .append(" | ")
                    .append(escape(mappedOps(entry)))
                    .append(" | ")
                    .append(escape(entry.limitations()))
                    .append(" |\n");
        }
        return out.toString();
    }

    public static void write(Path path) {
        Objects.requireNonNull(path, "path cannot be null");
        try {
            Path parent = path.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            Files.writeString(path, renderMarkdown());
        } catch (IOException e) {
            throw new OnnxException("Failed to write ONNX coverage report to " + path + ".", e);
        }
    }

    public static void main(String[] args) {
        if (args.length == 0) {
            System.out.print(renderMarkdown());
            return;
        }
        if (args.length != 1) {
            throw new IllegalArgumentException("Usage: OnnxCoverageReport [output-path]");
        }
        write(Path.of(args[0]));
    }

    private static void appendSummary(StringBuilder out) {
        out.append("## Summary\n\n");
        appendStatusSummary(out, "Import", OnnxCoverageMatrix.Entry::importStatus);
        appendStatusSummary(out, "Export", OnnxCoverageMatrix.Entry::exportStatus);
        appendStatusSummary(out, "CPU", OnnxCoverageMatrix.Entry::cpuStatus);
        appendStatusSummary(out, "Metal", OnnxCoverageMatrix.Entry::metalStatus);
        appendStatusSummary(out, "CUDA", OnnxCoverageMatrix.Entry::cudaStatus);
        appendEvidenceSummary(out);
        out.append('\n');
    }

    private static void appendStatusSummary(
            StringBuilder out,
            String label,
            java.util.function.Function<OnnxCoverageMatrix.Entry, OnnxCoverageMatrix.CoverageStatus> selector
    ) {
        EnumMap<OnnxCoverageMatrix.CoverageStatus, Integer> counts = new EnumMap<>(OnnxCoverageMatrix.CoverageStatus.class);
        for (OnnxCoverageMatrix.CoverageStatus status : OnnxCoverageMatrix.CoverageStatus.values()) {
            counts.put(status, 0);
        }
        for (OnnxCoverageMatrix.Entry entry : OnnxCoverageMatrix.entries()) {
            counts.compute(selector.apply(entry), (ignored, value) -> value == null ? 1 : value + 1);
        }
        out.append("- ")
                .append(label)
                .append(": supported=")
                .append(counts.get(OnnxCoverageMatrix.CoverageStatus.SUPPORTED))
                .append(", partial=")
                .append(counts.get(OnnxCoverageMatrix.CoverageStatus.PARTIAL))
                .append(", unsupported=")
                .append(counts.get(OnnxCoverageMatrix.CoverageStatus.UNSUPPORTED))
                .append('\n');
    }

    private static void appendEvidenceSummary(StringBuilder out) {
        EnumMap<OnnxCoverageMatrix.RoundTripEvidence, Integer> counts = new EnumMap<>(OnnxCoverageMatrix.RoundTripEvidence.class);
        for (OnnxCoverageMatrix.RoundTripEvidence evidence : OnnxCoverageMatrix.RoundTripEvidence.values()) {
            counts.put(evidence, 0);
        }
        for (OnnxCoverageMatrix.Entry entry : OnnxCoverageMatrix.entries()) {
            counts.compute(entry.roundTripEvidence(), (ignored, value) -> value == null ? 1 : value + 1);
        }
        out.append("- Round-trip evidence: round_trip_tested=")
                .append(counts.get(OnnxCoverageMatrix.RoundTripEvidence.ROUND_TRIP_TESTED))
                .append(", explicitly_classified=")
                .append(counts.get(OnnxCoverageMatrix.RoundTripEvidence.EXPLICITLY_CLASSIFIED))
                .append(", import_only_tested=")
                .append(counts.get(OnnxCoverageMatrix.RoundTripEvidence.IMPORT_ONLY_TESTED))
                .append(", rejection_tested=")
                .append(counts.get(OnnxCoverageMatrix.RoundTripEvidence.REJECTION_TESTED))
                .append(", not_applicable=")
                .append(counts.get(OnnxCoverageMatrix.RoundTripEvidence.NOT_APPLICABLE))
                .append('\n');
    }

    private static String mappedOps(OnnxCoverageMatrix.Entry entry) {
        if (entry.mappedOpTypes().isEmpty()) {
            return "";
        }
        return entry.mappedOpTypes().stream()
                .map(Operation.OpType::name)
                .collect(Collectors.joining(", "));
    }

    private static String status(OnnxCoverageMatrix.CoverageStatus status) {
        return status.name().toLowerCase(Locale.ROOT);
    }

    private static String evidence(OnnxCoverageMatrix.RoundTripEvidence evidence) {
        return evidence.name().toLowerCase(Locale.ROOT);
    }

    private static String escape(String text) {
        return (text == null ? "" : text).replace("|", "\\|");
    }
}
