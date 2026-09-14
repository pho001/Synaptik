/**
 * Provides ordinary standard construction and the advanced owner-bound Engine lifecycle facade.
 *
 * <p>The ordinary {@code Engine} currently opens one fresh independent CPU-only composition and
 * exposes construction, lifecycle observation, and closure without discovery or process-global
 * reuse. Its typed compile, prepare, input, run, publication, and result surface remains planned;
 * host materialization is a separate later boundary.</p>
 *
 * <p>The advanced surface owns one explicitly supplied CPU integration and coordinates the
 * advanced {@code compile -> prepare -> run} lifecycle. Compiled and prepared recipes remain
 * behind owner-bound opaque handles; each synchronous run has isolated mutable Runtime state and
 * returns only a publication count plus cleanup lifecycle. Caller storage and borrowed input
 * representations remain caller-owned.</p>
 *
 * <p>Neither surface performs backend discovery, mixed-backend composition, or successful
 * zero-node preparation. One-shot execution, backward convenience, tuning, typed ordinary
 * binding and result access, and host materialization are not current APIs.</p>
 */
package io.github.pho001.synaptik.engine;
