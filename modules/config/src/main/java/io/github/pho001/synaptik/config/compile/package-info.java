/**
 * Defines immutable declarative configuration values for graph compilation.
 *
 * <p>The current package contains five standalone values:
 * {@link io.github.pho001.synaptik.config.compile.BackendIntent} records optionality for one hard
 * backend requirement, {@link io.github.pho001.synaptik.config.compile.CompileMode} records the
 * requested compile-time graph scope,
 * {@link io.github.pho001.synaptik.config.compile.GraphOptimizationConfig} records permission for
 * optional semantics-preserving compiler optimization, and
 * {@link io.github.pho001.synaptik.config.compile.PartitionScoringConfig} records an optional soft
 * device-class preference for ranking after hard eligibility. {@link
 * io.github.pho001.synaptik.config.compile.NumericalProfile} records only the immutable graph-wide
 * numerical-profile identity. These values describe requests or identity only. Current
 * package-private compiler entries consume all five values; the complete artifact entry passes
 * backend intent, scoring preference, and the exact profile to Planning once per final graph node.
 * Engine construction supplies the current default of {@code STRICT_IEEE}; Config supplies no
 * default.</p>
 *
 * <p>Config owns no profile semantics: Model remains the sole authority for profile-indexed
 * allowed-result sets. Planning, Compiler, Prepare, Engine, and concrete backends may transport or
 * realize the identity without reinterpreting it. This package contains no compiler pass API,
 * profile evaluator, scoring evaluator, live service, runtime state, or concrete backend
 * implementation.</p>
 */
package io.github.pho001.synaptik.config.compile;
