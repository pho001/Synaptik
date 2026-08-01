/**
 * Defines the immutable prepared-execution root, cold-bound executable boundary, and per-run
 * invocation guard.
 *
 * <p>A {@link io.github.pho001.synaptik.runtime.execution.PreparedExecution} retains one exact
 * prepared memory plan and one exact same-plan prepared schedule as the complete current
 * immutable reusable Runtime recipe. It owns no closeable or per-run resource and may be shared
 * while distinct logical runs use isolated run state.
 *
 * <p>A {@link io.github.pho001.synaptik.runtime.execution.PreparedExecutable} is an immutable
 * reusable backend-owned recipe associated with that exact prepared memory plan. It resolves its
 * ordered dense buffer/workspace selections from one matching open run state, delegates explicit
 * checked compatibility to the concrete backend, and produces a backend-owned {@link
 * io.github.pho001.synaptik.runtime.execution.BoundInvocation}. That invocation retains the exact
 * state and direct concrete typed resource references for its prepared region.
 *
 * <p>Binding is the cold path and may allocate ordinary JVM arrays and the invocation object.
 * Bound execution performs only a run-open guard and the backend call. These contracts add no
 * prepared unit, allocation or physical access, auxiliary binding-resource lifecycle, validity
 * or residency, transfer, publication or result, backend discovery, route selection, tuning,
 * tracing, schedule consumption, or runner behavior.
 */
package io.github.pho001.synaptik.runtime.execution;
