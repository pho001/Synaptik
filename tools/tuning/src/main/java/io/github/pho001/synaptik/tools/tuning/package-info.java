/**
 * Coordinates explicit cold workload tuning and bounded complete-plan tuning while treating
 * backend candidates and decisions as opaque typed values.
 *
 * <p>Phase 1 accepts caller-supplied workload occurrences, stable model and representative-profile
 * evidence identities, backend-owned compatibility and candidate identities, complete candidate
 * enumeration and execution, and an explicit persistent cache path. It deduplicates compatible
 * occurrences, measures cache misses with bounded warmups and samples, selects the lowest
 * integer-middle median with encounter-order ties, atomically publishes compact decisions, and
 * returns richer evidence separately.
 *
 * <p>Phase 2 is a separate generic transaction over a producer-supplied complete-plan batch. It
 * checks every candidate for exact caller-defined correctness before timing any candidate,
 * preflights the checked {@code N * (1 + W + S)} execution bound, and returns a decision-present
 * Prepare handoff, a compact selected-plan record, and rich in-memory evidence. A
 * session-scoped producer causes no filesystem operation. A persistent hit is usable only after
 * the producer's codec authenticates it against the current batch; persistent misses are
 * published only after an equal compatible codec round trip.
 *
 * <p>The caller still owns representative inputs, preparation, execution, publication copying,
 * correctness-reference bytes, cleanup, and later preparation of the selected decision. The
 * concrete producer owns candidate meaning, legality, compatibility, ordering, reuse scope,
 * decision construction, and codecs. Current {@code Engine.prepareTuned(...)} composes both
 * phases for one eligible CPU workload occurrence and one CPU complete-plan candidate batch with
 * session-scoped reuse. Broader occurrence extraction, graph, ownership, or partition
 * alternatives, mixed-backend composition, persistent CPU complete-plan reuse, and executable
 * persistence remain outside the current public workflow.
 *
 * <p>{@link io.github.pho001.synaptik.tools.tuning.TuningInspection} supplies a separate cold,
 * read-only view of both compact schema-1 artifacts and of already-created rich evidence. It
 * reports opaque values only by schema, length, and SHA-256 digest. Exact stored-key equality is
 * not a compatibility verdict: backend-owned decision decoding and fresh preparation remain
 * required, and inspection performs neither operation.
 */
package io.github.pho001.synaptik.tools.tuning;
