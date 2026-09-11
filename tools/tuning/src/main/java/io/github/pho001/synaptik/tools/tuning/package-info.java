/**
 * Coordinates explicit cache-first cold workload tuning while treating backend candidates and
 * decisions as opaque typed values.
 *
 * <p>The current package accepts caller-supplied occurrences, stable model/profile evidence
 * identities, a backend-owned candidate/decision collaboration, complete candidate execution, and
 * an explicit cache path. It provides deterministic deduplication, bounded warmup and sampling,
 * integer-median selection with encounter-order ties, atomic persistent cache publication, and
 * separate rich evidence. Model extraction, a supported CPU adapter, Config or Engine facades,
 * and complete graph/plan tuning remain outside the current package.
 */
package io.github.pho001.synaptik.tools.tuning;
