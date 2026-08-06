package io.github.pho001.synaptik.backend.cpu.internal.prepare;

import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;

/**
 * Checked immutable CPU inputs that do not contain graph semantics or instance resources.
 *
 * @param loweringManifestEnabled whether cold diagnostics should retain a lowering manifest
 */
public record CpuPartitionAnalysisInputs(boolean loweringManifestEnabled)
        implements BackendAnalysisInputs {
    /** Default exact CPU analysis inputs with cold lowering-manifest retention disabled. */
    public static final CpuPartitionAnalysisInputs DEFAULT = new CpuPartitionAnalysisInputs(false);
}
