/**
 * Provides ordinary standard construction and the advanced owner-bound Engine lifecycle facade.
 *
 * <p>The ordinary {@code Engine} supports explicit ownership composition of opened CPU and Metal
 * integrations. Registration freezes each backend's capability provider and point-in-time
 * availability in order. Compilation considers that immutable inventory; cold preparation accepts
 * a non-empty single-owner plan or a mixed CPU/Metal plan in the bounded static canonical
 * contiguous {@code FLOAT32} transfer domain. Mixed preparation assigns deterministic
 * owner-indexed representations, assembles one shared ordered schedule, and captures exact
 * adapters per caller-input and publication occurrence for run and materialization.
 * {@code Engine.standard()} remains the CPU-only convenience and uses the same builder path. There
 * is no discovery, process-global registry, or implicit fallback. The ordinary surface exposes
 * owner-bound compiled graphs and a reusable {@code InferenceSession} that prepares once, owns
 * exactly one inward execution, and borrows its Engine composition lifetime. Sequential and
 * concurrent session runs reuse the immutable recipe and captured adapters while Runtime creates
 * one isolated mutable state per run. The lower-level closeable prepared handle remains available
 * for direct ownership. Both forms use Tensor-ID-based host-input binding, synchronous execution,
 * publication occurrences, and explicit detached host materialization. Each run snapshots current
 * caller-owned host associations; those storage lifetimes extend through result closure even
 * though execution is synchronous. Host materialization performs a synchronous selected-backend
 * copy and provides neither caching nor implicit transfer. One-shot compute transiently traverses
 * immutable Tensor-expression provenance by exact object identity, then uses final Compiler input
 * metadata as the sole binding-membership and ordering authority. It freshly compiles, prepares,
 * runs, preflights the complete publication set and aggregate returned canonical byte count,
 * materializes every ordered output, and closes each temporary result before its temporary
 * preparation without retaining its leaf inventory.
 * Sessions and standalone prepared handles remain registered for final Engine shutdown until
 * explicitly closed. One-shot scalar-objective backward execution uses the same discovery and
 * lifecycle seams, fixes Compiler's absent unit seed and disconnected-target error policy, and
 * returns a detached objective plus target-aligned first derivatives. Explicit seeds and
 * selective output access are also available through inference sessions. Optional
 * model-autotuning accepts one caller-defined
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
 * <p>Both surfaces capture one exact graph-wide {@link
 * io.github.pho001.synaptik.config.compile.NumericalProfile} at construction. Builders default to
 * {@code STRICT_IEEE}; an explicit selection is passed unchanged through compilation and
 * preparation. CPU and Metal currently fail closed for {@code ACCELERATOR}, and Runtime performs
 * no per-run profile lookup.</p>
 *
 * <p>Neither surface performs backend discovery or successful zero-node preparation. Mixed-backend
 * execution is confined to ordinary Engine's explicit CPU/Metal composition and exact transfer
 * domain; the advanced surface remains owner-bound to one CPU integration. Current complete-plan
 * tuning is exact-byte, session-scoped, CPU-only, and performs no model-plan-cache file access.
 * Inferred backward targets, multi-occurrence, persistent complete-plan reuse, general
 * Compiler/Planning graph-plan alternatives, and unsupported transfer or conversion domains are
 * not current APIs.</p>
 */
package io.github.pho001.synaptik.engine;
