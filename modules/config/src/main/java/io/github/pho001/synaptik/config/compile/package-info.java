/**
 * Defines immutable declarative configuration values for graph compilation.
 *
 * <p>The current package contains four standalone values:
 * {@link io.github.pho001.synaptik.config.compile.BackendIntent} records optionality for one hard
 * backend requirement, {@link io.github.pho001.synaptik.config.compile.CompileMode} records the
 * requested compile-time graph scope,
 * {@link io.github.pho001.synaptik.config.compile.GraphOptimizationConfig} records permission for
 * optional semantics-preserving compiler optimization, and
 * {@link io.github.pho001.synaptik.config.compile.PartitionScoringConfig} records an optional soft
 * device-class preference for ranking after hard eligibility. These values describe declarative
 * compile requests. The complete artifact entry passes backend intent and scoring preference to
 * Planning once per final graph node; no numerical selector is carried.</p>
 *
 * <p>Model owns numerical semantics, while backend capability remains an occurrence-specific
 * provider answer. This package contains no compiler pass API, numerical-policy evaluator,
 * scoring evaluator, live service, runtime state, or concrete backend implementation.</p>
 */
package io.github.pho001.synaptik.config.compile;
