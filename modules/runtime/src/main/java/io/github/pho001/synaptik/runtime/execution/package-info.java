/**
 * Defines the immutable prepared-execution root, cold-bound executable boundary, and per-run
 * invocation guard, and prepared/bound buffer-transfer boundary.
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
 * <p>A {@link io.github.pho001.synaptik.runtime.execution.PreparedBufferTransfer} similarly
 * resolves one source and destination representation during cold binding and produces a {@link
 * io.github.pho001.synaptik.runtime.execution.BoundBufferTransfer} whose backend subclass retains
 * direct concrete references. Its final action owns the destination-no-op, source-valid, and
 * success-only destination-valid transition. Materialization of an equivalent already-created
 * representation is this same transfer operation.
 *
 * <p>Binding is the cold path and may allocate ordinary JVM arrays and the invocation object.
 * Bound invocation execution performs only a run-open guard and the backend call; bound transfer
 * execution uses only dense validity operations and one backend call. These contracts add no
 * prepared unit, allocation or physical access, auxiliary binding-resource lifecycle, validity
 * coherence or invalidation policy, publication or result, backend discovery, route selection,
 * tuning, tracing, schedule consumption, or runner behavior.
 */
package io.github.pho001.synaptik.runtime.execution;
