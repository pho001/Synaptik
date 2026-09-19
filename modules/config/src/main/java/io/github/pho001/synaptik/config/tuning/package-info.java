/**
 * Defines immutable declarative inputs for requesting later model-autotuning composition.
 *
 * <p>The package contains only
 * {@link io.github.pho001.synaptik.config.tuning.ModelAutotuningConfig}, which records a selection
 * objective, Phase-1 sampling policy, representative-profile identity, fallback policy, explicit
 * workload-cache path, independent Phase-2 complete-plan bounds, and explicit model-plan-cache
 * path. Possessing that value expresses a request; it does not execute tuning or enable it through
 * a hidden default.</p>
 *
 * <p>Tuning tooling owns candidate measurement and selection, cache parsing and persistence, and
 * evidence. Backends own candidate vocabulary and eligibility. Later composition owns translation,
 * representative inputs, model identity, complete-plan correctness execution, preparation, and
 * fallback control flow. The current Engine consumes only the Phase-1 values; the current CPU
 * complete-plan producer is session-scoped, so the supplied model-plan path is not accessed. This
 * package performs no I/O, discovery, backend work, Engine orchestration, or Runtime work.</p>
 */
package io.github.pho001.synaptik.config.tuning;
