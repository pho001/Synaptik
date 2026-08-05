package io.github.pho001.synaptik.backend.cpu.execution;

import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.List;

/**
 * Direct family-owned source of complete portable partition candidates in deterministic
 * preference order.
 *
 * <p>The collaboration is injected directly into CPU analysis. It is not a registry, discovery
 * mechanism, tuning search, or shared parameter map. An implementation owns the exact family
 * semantics and must fail closed unless every node in the partition can be represented by the
 * returned complete candidate. The current pointwise implementation supplies only the bounded
 * dense {@code ADD} matrix.</p>
 */
@FunctionalInterface
interface CpuPortableCandidateSource {
    /**
     * Produces complete candidates for one already CPU-owned partition projection.
     *
     * @param context non-null complete validated analysis context
     * @return non-null deterministic ordered candidate list; entries must be non-null and each
     *     candidate must cover the complete partition
     */
    List<CpuPortablePartitionCandidate> candidates(
            PrepareContext<CpuPortableAnalysisInputs> context);
}
