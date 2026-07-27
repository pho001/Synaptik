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
 * device-class preference for ranking after hard eligibility. These values describe requests
 * only. Current package-private compiler entries consume mode and optimization permission; the
 * complete artifact entry additionally passes backend intent and scoring preference to Planning
 * once per final graph node. No public compile aggregate or public compiler entry point consumes
 * these values.</p>
 *
 * <p>Later configuration work owns immutable profile inputs and the aggregate compile
 * configuration. Current Planning evaluates hard eligibility and applies the cost-free
 * preferred-class/provider-order owner-selection baseline. Numeric cost scoring, preparation,
 * runtime, engine composition, and concrete backend implementation remain outside this package.
 * This package contains no compiler pass API, profile data, scoring evaluator, live service,
 * runtime state, or concrete backend implementation.</p>
 */
package io.github.pho001.synaptik.config.compile;
