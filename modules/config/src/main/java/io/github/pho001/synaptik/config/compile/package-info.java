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
 * device-class preference for later ranking after hard eligibility. These values describe
 * requests only. The current package-private compiler transformation boundary consumes only the
 * optimization permission. No current compile aggregate or public compiler entry point consumes
 * these values, and compile mode, backend intent, and scoring orchestration remain planned.</p>
 *
 * <p>Later configuration work owns immutable profile inputs and the aggregate compile
 * configuration. Later planning evaluates hard eligibility, candidates, and scoring and chooses
 * ownership; compiler orchestration, preparation, runtime, engine composition, and concrete
 * backend implementation remain outside this package. This package contains no compiler pass
 * API, profile data, scoring evaluator, live service, runtime state, or concrete backend
 * implementation.</p>
 */
package io.github.pho001.synaptik.config.compile;
