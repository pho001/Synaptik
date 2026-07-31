/**
 * Defines the cold-bound prepared executable boundary and per-run invocation guard.
 *
 * <p>A {@link io.github.pho001.synaptik.runtime.execution.PreparedExecutable} is an immutable
 * reusable backend-owned recipe associated with one exact prepared memory plan. It resolves its
 * ordered dense buffer/workspace selections from one matching open run state, delegates explicit
 * checked compatibility to the concrete backend, and produces a backend-owned
 * {@link io.github.pho001.synaptik.runtime.execution.BoundInvocation}. That invocation retains
 * the exact state and direct concrete typed resource references for its prepared region.
 *
 * <p>Binding is the cold path and may allocate ordinary JVM arrays and the invocation object.
 * Bound execution performs only a run-open guard and the backend call. These contracts add no
 * prepared unit or schedule, allocation or physical access, auxiliary binding-resource
 * lifecycle, validity or residency, transfer, publication or result, backend discovery, route
 * selection, tuning, tracing, or runner behavior.
 */
package io.github.pho001.synaptik.runtime.execution;
