/**
 * Defines complete shared graph preparation, assignment, typed backend finalization, and
 * schedule-validation contracts.
 *
 * <p>The {@code analysis} subpackage supplies validated backend analysis inputs and exact resource
 * declarations. This root package projects complete partition contexts, associates declarations
 * with Runtime-owned slots, hands each opaque typed plan back to its owning backend for immutable
 * executable construction, and validates one explicitly assembled complete schedule. The
 * complete-graph facade authenticates producerless, consumerless published caller or constant
 * graph inputs before analysis. Exact caller geometry leads in Compiler bindable occurrence
 * order; source-only constants follow ordinary declarations in graph-value order. The supplied
 * size recipe and backend owner are retained without projecting sources into a partition or
 * assigning them to a finalizer. Shared fixed-or-sole-named-B validation includes complete source
 * values, and named sources retain symbolic mode even when their physical size is always zero.
 * Constants remain static, resolved, nongradient and exact-typed; static caller views retain their
 * resolved affine layout while named callers retain unresolved templates. These geometry-only
 * declarations neither grant no-node support nor change borrowed caller ownership or per-run
 * constant initialization. Backend finalizers may return
 * acquisition-ordered persistent resources with their executables. Shared Prepare owns successful
 * results transactionally, rolls resources back across later finalization and assembly failures,
 * and transfers ownership only to a successfully constructed Runtime prepared execution. The
 * stateless public facade does not physically allocate resources. Projected contexts contain
 * partition facts and backend inputs without a numerical selector. Constant initialization or
 * materialization, mutable per-run state, execution, backend discovery, and Engine composition
 * remain outside this package.</p>
 *
 * <p>General scans, including length-aware forward and generated first-order reverse occurrences,
 * use the same projection and declaration/assignment/finalization transaction. Shared Prepare
 * preserves exact scoped bodies, ordered row roles and complete-graph output demand before any
 * analyzer. Backend analysis owns mixed-boundary byte geometry, carry/staging/temporary needs
 * and demanded histories; an undemanded residual port creates no history storage obligation.
 * These static metadata and recipe contracts neither allocate physical histories nor validate
 * runtime length values or implement activity skipping.</p>
 */
package io.github.pho001.synaptik.prepare;
