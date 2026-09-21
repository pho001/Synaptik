/**
 * Provides ordinary standard construction and the advanced owner-bound Engine lifecycle facade.
 *
 * <p>The ordinary {@code Engine} currently opens one fresh independent CPU-only composition and
 * exposes owner-bound compile and prepared handles, Tensor-ID-based host-input binding, synchronous
 * execution, publication occurrences, explicit detached host materialization, one-shot compute
 * execution, lifecycle observation, and closure without backend discovery or process-global reuse.
 * One-shot compute transiently traverses immutable Tensor-expression provenance by exact object
 * identity, then uses final Compiler input metadata as the sole binding-membership and ordering
 * authority. Each
 * run snapshots caller-owned host associations; those storage lifetimes extend through result
 * closure even though execution is synchronous. Host
 * materialization performs a synchronous CPU copy and provides neither caching nor implicit
 * transfer. One-shot compute freshly compiles, prepares, runs, preflights the complete publication
 * set and aggregate returned canonical byte count, materializes every ordered output, and closes
 * each temporary result before its temporary preparation without retaining its leaf inventory.
 * Reusable prepared handles are explicit closeable outward owners and remain registered for
 * final Engine shutdown until explicitly closed. One-shot scalar-objective backward
 * execution uses the same discovery and lifecycle seams, fixes Compiler's absent unit seed and
 * disconnected-target error policy, and returns a detached objective plus target-aligned first
 * derivatives. Repeated execution, explicit seeds, and selective output access should use the
 * reusable explicit-input lifecycle instead. Optional model-autotuning accepts one caller-defined
 * model identity and one live representative input set. It first completes bounded CPU-local
 * workload selection, then preserves that exact decision while checking and measuring the
 * session-scoped complete CPU plan alternatives. Every correctness, warmup, and timed action uses
 * fresh preparation and Runtime state; the returned handle is another fresh preparation of the
 * authenticated complete-plan winner. The result reports immutable evidence for both phases, or
 * explicitly reports one fresh safe-heuristic fallback preparation. Every temporary trial
 * preparation closes after its result, while the selected or fallback handle is the sole outward
 * owner retained for caller close or final Engine shutdown.</p>
 *
 * <p>The advanced surface owns one explicitly supplied CPU integration and coordinates the
 * advanced {@code compile -> prepare -> run} lifecycle. Compiled and prepared recipes remain
 * behind owner-bound opaque handles; each synchronous run has isolated mutable Runtime state and
 * returns only a publication count plus cleanup lifecycle. Prepared handles are explicitly
 * closeable; Engine shutdown closes results first, then retained preparations, then composition.
 * Caller storage and borrowed input representations remain caller-owned.</p>
 *
 * <p>Neither surface performs backend discovery, mixed-backend composition, or successful
 * zero-node preparation. Current complete-plan tuning is exact-byte, session-scoped, CPU-only,
 * and performs no model-plan-cache file access. Inferred backward targets, multi-occurrence,
 * persistent complete-plan reuse, general Compiler/Planning graph-plan alternatives, and
 * cross-backend materialization are not current APIs.</p>
 */
package io.github.pho001.synaptik.engine;
