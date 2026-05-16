package tuning.benchmark.report;

import graph.execution.trace.HostDeviceTransferKind;
import graph.execution.trace.HostDeviceTransferTrace;
import tuning.benchmark.NativeDeviceBridgeBenchmark;

import java.util.ArrayList;
import java.util.List;

/**
 * Evidence gate for the Wave 7 native CPU to Metal bridge benchmark.
 */
public final class NativeDeviceBridgeBenchmarkGate {
    private NativeDeviceBridgeBenchmarkGate() {
    }

    /**
     * Evaluates whether a benchmark report contains the three expected host/device transfer routes.
     *
     * @param report benchmark report
     * @return immutable failure list; empty means the evidence gate passed
     */
    public static List<String> evaluate(BenchmarkReport report) {
        var failures = new ArrayList<String>();
        if (report == null) {
            return List.of("missing native device bridge benchmark report");
        }
        requireCandidate(
                report,
                NativeDeviceBridgeBenchmark.CPU_ARRAY_METAL,
                HostDeviceTransferKind.CPU_ARRAY_TO_DEVICE_COPY,
                failures
        );
        requireCandidate(
                report,
                NativeDeviceBridgeBenchmark.CPU_NATIVE_ARRAY_BRIDGE_METAL,
                HostDeviceTransferKind.NATIVE_TO_ARRAY_TO_DEVICE_BRIDGE,
                failures
        );
        requireCandidate(
                report,
                NativeDeviceBridgeBenchmark.CPU_NATIVE_DIRECT_METAL,
                HostDeviceTransferKind.NATIVE_SEGMENT_TO_DEVICE_COPY,
                failures
        );
        return List.copyOf(failures);
    }

    /**
     * Throws if {@link #evaluate(BenchmarkReport)} finds missing bridge evidence.
     *
     * @param report benchmark report
     */
    public static void requirePass(BenchmarkReport report) {
        List<String> failures = evaluate(report);
        if (!failures.isEmpty()) {
            throw new IllegalStateException(String.join("; ", failures));
        }
    }

    private static void requireCandidate(
            BenchmarkReport report,
            String candidateName,
            HostDeviceTransferKind expectedKind,
            List<String> failures
    ) {
        BenchmarkCandidateReport candidate = report.candidates().stream()
                .filter(entry -> candidateName.equals(entry.entry().name()))
                .findFirst()
                .orElse(null);
        if (candidate == null) {
            failures.add("missing native device bridge candidate " + candidateName);
            return;
        }
        if (!candidate.success() || candidate.measurement() == null) {
            failures.add("native device bridge candidate failed " + candidateName);
            return;
        }
        List<HostDeviceTransferTrace> transfers = candidate.measurement().trace().run().hostDeviceTransfers();
        HostDeviceTransferTrace match = transfers.stream()
                .filter(transfer -> transfer.transferKind() == expectedKind)
                .findFirst()
                .orElse(null);
        if (match == null) {
            failures.add("missing transfer route " + expectedKind + " for " + candidateName);
            return;
        }
        if (expectedKind == HostDeviceTransferKind.NATIVE_TO_ARRAY_TO_DEVICE_BRIDGE
                && match.fallbackReason().isBlank()) {
            failures.add("missing native-device fallback reason for " + candidateName);
        }
        if (expectedKind == HostDeviceTransferKind.NATIVE_SEGMENT_TO_DEVICE_COPY) {
            if (match.javaArrayBytes() != 0L) {
                failures.add("direct native-device route used Java array bytes for " + candidateName);
            }
            if (!match.directTransferSupported()) {
                failures.add("direct native-device route was not marked supported for " + candidateName);
            }
        }
    }
}
