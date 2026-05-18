package backend.cpu.nativecpu;

import operations.Operation;
import tensor.DataType;

import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Executable operation/storage/layout row for one operation and output dtype.
 */
public record NativeCpuParityEntry(
        Operation.OpType opType,
        DataType dataType,
        boolean logicalOperationDefined,
        Set<NativeCpuStoragePath> storagePaths,
        Set<NativeCpuLayoutCapability> layoutCapabilities,
        Set<NativeCpuResultResidency> resultResidencies,
        NativeCpuKernelPerformanceStatus status,
        NativeCpuKernelFamily family,
        boolean autoEligible,
        String reason
) {
    public NativeCpuParityEntry {
        opType = Objects.requireNonNull(opType, "opType cannot be null");
        dataType = Objects.requireNonNull(dataType, "dataType cannot be null");
        storagePaths = Set.copyOf(storagePaths == null ? Set.of() : storagePaths);
        layoutCapabilities = Set.copyOf(layoutCapabilities == null ? Set.of() : layoutCapabilities);
        resultResidencies = Set.copyOf(resultResidencies == null ? Set.of() : resultResidencies);
        status = Objects.requireNonNull(status, "status cannot be null");
        family = Objects.requireNonNull(family, "family cannot be null");
        reason = reason == null ? "" : reason;
    }

    public boolean hasStoragePath(NativeCpuStoragePath path) {
        return storagePaths.contains(path);
    }

    public boolean hasLayoutCapability(NativeCpuLayoutCapability capability) {
        return layoutCapabilities.contains(capability);
    }

    public boolean hasResultResidency(NativeCpuResultResidency residency) {
        return resultResidencies.contains(residency);
    }

    String reportRow() {
        return opType.name()
                + "," + dataType.name()
                + "," + logicalOperationDefined
                + "," + status.name()
                + "," + family.name()
                + "," + autoEligible
                + "," + sortedNames(storagePaths)
                + "," + sortedNames(layoutCapabilities)
                + "," + sortedNames(resultResidencies)
                + "," + reason;
    }

    private static <E extends Enum<E>> String sortedNames(Set<E> values) {
        if (values == null || values.isEmpty()) {
            return "-";
        }
        return values.stream()
                .sorted(Comparator.comparingInt(Enum::ordinal))
                .map(Enum::name)
                .collect(java.util.stream.Collectors.joining("|"));
    }
}
