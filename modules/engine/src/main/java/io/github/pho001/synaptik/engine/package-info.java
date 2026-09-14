/**
 * Provides the advanced owner-bound Engine composition and lifecycle facade.
 *
 * <p>The current surface owns one explicitly supplied CPU integration and coordinates the
 * advanced {@code compile -> prepare -> run} lifecycle. Compiled and prepared recipes remain
 * behind owner-bound opaque handles; each synchronous run has isolated mutable Runtime state and
 * returns only a publication count plus cleanup lifecycle. Caller storage and borrowed input
 * representations remain caller-owned.</p>
 *
 * <p>This package is intentionally lower level than the future ordinary typed binding, published
 * value, host materialization, standard composition, one-shot execution, backward convenience,
 * and tuning APIs. It supports neither mixed-backend composition nor successful zero-node
 * preparation.</p>
 */
package io.github.pho001.synaptik.engine;
