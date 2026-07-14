/**
 * Defines immutable declarative configuration values for graph compilation.
 *
 * <p>The current package contains three standalone values:
 * {@link io.github.pho001.synaptik.config.compile.BackendIntent} records optionality for one hard
 * backend requirement, {@link io.github.pho001.synaptik.config.compile.CompileMode} records the
 * requested compile-time graph scope, and
 * {@link io.github.pho001.synaptik.config.compile.GraphOptimizationConfig} records permission for
 * optional semantics-preserving compiler optimization. These values describe requests only; no
 * current aggregate or compiler entry point consumes them.</p>
 *
 * <p>Later configuration work owns backend-neutral scoring policy, immutable profile inputs, and
 * the aggregate compile configuration. Later planning evaluates requirements and chooses
 * ownership; compiler, prepare, runtime, engine, training, and concrete backend layers own their
 * respective lifecycle behavior. This package contains no compiler pass API, live service,
 * runtime state, or concrete backend implementation.</p>
 */
package io.github.pho001.synaptik.config.compile;
