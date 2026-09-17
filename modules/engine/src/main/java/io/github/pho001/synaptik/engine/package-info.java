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
 * set and aggregate returned canonical byte count, materializes every ordered output, and cleans
 * up before return without retaining its leaf inventory; repeated execution and selective output
 * access should use the reusable explicit-input lifecycle instead.</p>
 *
 * <p>The advanced surface owns one explicitly supplied CPU integration and coordinates the
 * advanced {@code compile -> prepare -> run} lifecycle. Compiled and prepared recipes remain
 * behind owner-bound opaque handles; each synchronous run has isolated mutable Runtime state and
 * returns only a publication count plus cleanup lifecycle. Caller storage and borrowed input
 * representations remain caller-owned.</p>
 *
 * <p>Neither surface performs backend discovery, mixed-backend composition, or successful
 * zero-node preparation. Scalar backward convenience, tuning, and
 * cross-backend materialization are not current APIs.</p>
 */
package io.github.pho001.synaptik.engine;
