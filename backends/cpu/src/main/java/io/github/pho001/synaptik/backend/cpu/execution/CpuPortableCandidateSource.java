package io.github.pho001.synaptik.backend.cpu.execution;

import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.List;

/**
 * Direct family-owned source of complete portable candidates in deterministic preference order.
 *
 * <p>The collaboration is injected directly into CPU analysis. It is not a registry, discovery
 * mechanism, tuning search, or shared parameter map. Implementations own operation-family
 * semantics in later tasks; the current production package supplies no family implementation.</p>
 */
@FunctionalInterface
interface CpuPortableCandidateSource {
    /**
     * Produces complete candidates for one already CPU-owned partition projection.
     *
     * @param context non-null complete validated analysis context
     * @return non-null deterministic ordered candidate list; entries must be non-null
     */
    List<CpuPortableKernelCandidate> candidates(
            PrepareContext<CpuPortableAnalysisInputs> context);
}
