/**
 * CPU-private discovery, selection, lifetime composition, and execution for the narrow OpenBLAS
 * MATMUL route.
 *
 * <p>Cold analysis consumes the already-complete portable MATMUL lowering plus immutable
 * qualification, storage, thread, materialization, and cost facts. It can select only one
 * positive rank-two same-type FLOAT32 or FLOAT64 product. Configuration accepts at most 32 unique
 * positive thread counts with per-count cost terms; an incomplete configuration fails closed.
 * Counts above the analysis capacity are filtered, then lower complete cost, lower thread count,
 * and stable input order select among eligible native candidates. Each of the left input, right
 * input, and output is either already a canonical native matrix or receives one distinct declared
 * run-owned workspace and one generated affine copy. Copy-out is a route-local result transition,
 * not an external-read materialization. Ineligibility, overflow, a tie with portable cost, or
 * insufficient benefit leaves the portable route selected. A selected native plan is immutable
 * and retains no provider, segment, slot, or physical run resource.</p>
 *
 * <p>Before preparation, an explicit CPU-internal composition call may disable loading, try one
 * exact name or absolute path, or try a fixed platform candidate table. Its immutable result
 * records loading and binding only; it does not prove compatibility, qualify the provider, or
 * enable this route. A separate caller-owned discovery session retains the loaded provider and a
 * borrowed invocation until composition either closes it or atomically transfers its ownership
 * to one coordinator. Discovery performs no thread query, mutation, restoration, filesystem
 * search, or hot-path work.</p>
 *
 * <p>Compatibility finalization consumes an already-borrowed provider-free count-one invocation.
 * Coordinated finalization instead validates the selected bounded thread candidate against the
 * shared CPU budget and installs it before creating artifacts. The coordinator serializes thread
 * configuration against admitted native calls, admits complete copy-in/GEMM/copy-out sequences
 * at their fixed plan demand, and restores verified original provider state before closing its
 * transferred owner. Prepared execution borrows that lifetime and, during cold binding, checks
 * coordinator availability or the compatibility invocation's open single-thread state, exact
 * carriers, generated-copy identities, spans, alignment, writability, and every buffer/workspace
 * overlap before any write. Each invocation copies left then right when selected, performs
 * exactly one typed GEMM, and finally copies the result when selected. Prepared recipes may be
 * reused, but every {@code RunState} owns distinct physical workspaces. Execution never performs
 * discovery, configuration, restoration, late route selection, or fallback.</p>
 */
package io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas;
