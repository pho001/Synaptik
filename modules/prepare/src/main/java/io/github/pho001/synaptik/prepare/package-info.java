/**
 * Defines complete shared graph preparation, assignment, typed backend finalization, and
 * schedule-validation contracts.
 *
 * <p>The {@code analysis} subpackage supplies validated backend analysis inputs and exact resource
 * declarations. This root package projects complete partition contexts, associates declarations
 * with Runtime-owned slots, hands each opaque typed plan back to its owning backend for immutable
 * executable construction, and validates one explicitly assembled complete schedule. The
 * complete-graph facade can also validate an exact producerless published-constant contribution
 * and append its backend-supplied buffer geometry after ordinary partition declarations without
 * projecting it into a partition or assigning it to a finalizer. Backend finalizers may return
 * acquisition-ordered persistent resources with their executables. Shared Prepare owns successful
 * results transactionally, rolls resources back across later finalization and assembly failures,
 * and transfers ownership only to a successfully constructed Runtime prepared execution. The
 * stateless public facade does not physically allocate resources; constant initialization or
 * materialization, mutable per-run state, execution, backend discovery, and Engine composition
 * remain outside this package.</p>
 */
package io.github.pho001.synaptik.prepare;
