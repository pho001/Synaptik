package backend.accelerator.exec;

import backend.contract.ComputeBackend;
import runtime.device.buffer.AcceleratorBufferDecision;
import runtime.device.buffer.AcceleratorBufferExecutionPath;
import backend.accelerator.lowering.GpuCompoundRegionSummary;
import backend.accelerator.lowering.GpuLoweredRegionManifest;
import backend.lowering.region.RegionExecutionPlan;
import backend.runtime.ExecutionContext;

import java.util.LinkedHashMap;
import java.util.List;

/**
 * Prepared runtime artifact for an accelerator-backed partition.
 *
 * <p>Implementations own any native bridge context or compiled executable needed
 * for a lowered partition and must fall back to CPU execution when the bridge is
 * unavailable or rejects the prepared artifact.</p>
 */
public interface PreparedAcceleratorExecutable {
    /**
     * Returns the accelerator backend this executable targets.
     */
    ComputeBackend backend();

    /**
     * Executes the prepared partition against tensors resolved from the runtime context.
     *
     * @param context runtime tensor lookup and execution flags for the current graph run
     */
    void execute(ExecutionContext context);

    /**
     * Returns CPU fallback steps prepared for this accelerator executable.
     *
     * <p>Runtime state uses these plans to allocate per-run prepared-input tensors for
     * accelerator partitions whose CPU fallback metadata can also be used to prepare
     * external inputs for native execution.</p>
     */
    default List<PreparedAcceleratorExecutionSupport.CpuFallbackStep> cpuFallbackSteps() {
        return List.of();
    }

    /**
     * Returns buffer-binding diagnostics from the most recent execution attempt.
     *
     * @return latest buffer decision, or a not-evaluated decision when the backend has no buffer path
     */
    default AcceleratorBufferDecision lastAcceleratorBufferDecision() {
        return AcceleratorBufferDecision.notEvaluated(backend());
    }

    /**
     * Returns the residency reason to record after this executable runs.
     *
     * <p>This keeps graph execution backend-neutral while still letting concrete accelerator executables expose more
     * precise CPU fallback or device-buffer write reasons.</p>
     *
     * @return concise residency reason for runtime state diagnostics
     */
    default String outputResidencyReason() {
        AcceleratorBufferDecision decision = lastAcceleratorBufferDecision();
        if (decision != null && decision.path() == AcceleratorBufferExecutionPath.BUFFER_BINDING) {
            return "accelerator buffer binding execution wrote device buffer";
        }
        if (decision != null && decision.path() == AcceleratorBufferExecutionPath.CPU_FALLBACK) {
            return "accelerator cpu fallback wrote CPU array";
        }
        return "backend wrote CPU array";
    }

    /**
     * Returns compound GPU region metadata for this executable.
     *
     * <p>The default is a non-compound summary so existing accelerator executables can opt in
     * without changing their runtime behavior.</p>
     *
     * @return compound summary, or a non-compound summary when none was lowered
     */
    default GpuCompoundRegionSummary compoundSummary() {
        return GpuCompoundRegionSummary.none(backend(), List.of());
    }

    /**
     * Returns lowered-region manifest metadata for this executable, if available.
     *
     * @return lowered-region manifest, or {@code null} when absent
     */
    default GpuLoweredRegionManifest gpuLoweredRegionManifest() {
        return null;
    }

    default RegionExecutionPlan regionExecutionPlan() {
        return null;
    }

    /**
     * Adds backend-specific execution trace attributes after a run.
     *
     * <p>The graph execution layer owns trace assembly, but concrete accelerator backends own their bridge, route, and
     * native-copy vocabulary. Implementations may leave this empty when they expose no backend-specific attributes.</p>
     *
     * @param attrs mutable trace attribute map for the current step
     */
    default void contributeRunTraceAttributes(LinkedHashMap<String, Object> attrs) {
    }
}
