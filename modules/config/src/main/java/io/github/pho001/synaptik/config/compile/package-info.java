/**
 * Defines immutable declarative configuration values for graph compilation.
 *
 * <p>The current package contains only
 * {@link io.github.pho001.synaptik.config.compile.BackendIntent}, which records optionality for
 * one hard backend requirement. It neither evaluates the requirement nor expresses backend
 * preference, and an unconstrained intent promises no default backend or fallback.</p>
 *
 * <p>Later configuration work owns compile modes, graph optimization, backend-neutral scoring
 * policy, and immutable profile inputs. Later planning evaluates requirements and chooses
 * ownership; compiler, prepare, runtime, engine, and concrete backend layers own their respective
 * lifecycle behavior. This package contains no live service or concrete backend implementation.</p>
 */
package io.github.pho001.synaptik.config.compile;
