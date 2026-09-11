/**
 * Defines immutable declarative inputs for requesting later model-autotuning composition.
 *
 * <p>The package contains only
 * {@link io.github.pho001.synaptik.config.tuning.ModelAutotuningConfig}, which records a selection
 * objective, bounded sampling policy, representative-profile identity, fallback policy, and
 * explicit workload-cache path. Possessing that value expresses a request; it does not execute
 * tuning or enable it through a hidden default.</p>
 *
 * <p>Tuning tooling owns candidate measurement and selection, cache parsing and persistence, and
 * evidence. Backends own candidate vocabulary and eligibility. Later composition owns translation,
 * representative inputs, model identity, preparation, and fallback control flow. This package
 * performs no I/O, discovery, backend work, Engine orchestration, or Runtime work.</p>
 */
package io.github.pho001.synaptik.config.tuning;
