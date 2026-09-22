/**
 * Implements Metal's package-private native, preparation, storage, and execution boundary.
 *
 * <p>One context owns the default Metal device, command queue, and native-library lifetime.
 * Backend analysis validates and lowers one complete maximal partition of supported unary
 * {@code NEG} occurrences and declares its boundary buffers plus one address workspace before
 * shared slot assignment. Backend finalization then compiles one shape-specialized persistent
 * MPSGraph executable and returns it as a prepared resource.</p>
 *
 * <p>Each run borrows caller input buffers, creates fresh initialized constant buffers and output
 * buffers, and owns native-address workspace storage through its {@code RunState}. Cold binding
 * validates concrete representations and fills that workspace once. The bound hot invocation
 * retains direct references and makes one synchronous native downcall into the reusable
 * executable, which writes the caller-supplied output destinations.</p>
 *
 * <p>All types remain package-private. Shared Runtime sees only its nominal resource,
 * representation, executable, and schedule contracts; no native handle, Objective-C value,
 * MPSGraph type, operation, graph node, backend lookup, or route choice crosses the hot-path
 * boundary.</p>
 */
package io.github.pho001.synaptik.backend.metal.internal;
