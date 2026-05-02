package backend.accelerator.lowering;

import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * Deterministic derived report comparing CUDA lowering coverage against Metal.
 */
public record GpuBackendParityReport(List<GpuBackendParityRow> rows) {
    private static final Comparator<GpuBackendParityRow> ROW_ORDER = Comparator
            .comparing((GpuBackendParityRow row) -> row.family().name())
            .thenComparing(row -> row.opType().name());

    public GpuBackendParityReport {
        rows = rows == null ? List.of() : rows.stream()
                .filter(Objects::nonNull)
                .sorted(ROW_ORDER)
                .toList();
    }

    /**
     * Returns rows where Metal is supported and CUDA is not yet supported.
     */
    public List<GpuBackendParityRow> gapRows() {
        return rows.stream()
                .filter(GpuBackendParityRow::parityGap)
                .toList();
    }

    /**
     * Returns rows supported by both Metal and CUDA.
     */
    public List<GpuBackendParityRow> supportedOnBothRows() {
        return rows.stream()
                .filter(row -> row.metalSupported() && row.cudaSupported())
                .toList();
    }

    /**
     * Returns rows where CUDA is fallback or unsupported.
     */
    public List<GpuBackendParityRow> cudaUnsupportedRows() {
        return rows.stream()
                .filter(row -> !row.cudaSupported())
                .toList();
    }

    /**
     * Groups rows by the evidence required before a CUDA parity claim can be made.
     */
    public Map<String, List<GpuBackendParityRow>> rowsByRequiredEvidence() {
        return rows.stream()
                .collect(Collectors.groupingBy(
                        GpuBackendParityRow::requiredEvidence,
                        LinkedHashMap::new,
                        Collectors.toList()
                ));
    }
}
