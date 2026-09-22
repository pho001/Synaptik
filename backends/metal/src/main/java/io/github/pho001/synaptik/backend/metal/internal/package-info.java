/**
 * Implements Metal's package-private native, preparation, storage, and execution boundary.
 *
 * <p>One context owns the default Metal device, command queue, and native-library lifetime.
 * Backend analysis validates and lowers one complete maximal partition of supported unary
 * {@code NEG} occurrences, selects one of two private routes, and declares exact resources before
 * shared slot assignment. An exact singleton in the custom 32-bit index domain has only feed and
 * target buffers; every other supported partition also has the MPSGraph address workspace.
 * Backend finalization compiles the selected custom pipeline or MPSGraph executable and returns
 * its typed persistent owner. The choice is a deterministic implementation-domain boundary, not
 * tuning, capability narrowing, fallback, retry, or repartitioning.</p>
 *
 * <p>Each run borrows caller input buffers, creates fresh initialized constant buffers and output
 * buffers. MPSGraph runs also own native-address workspace storage through their
 * {@code RunState}; custom runs do not. Cold binding creates a route-specific invocation with
 * direct references. The bound hot invocation makes one synchronous native downcall into the
 * selected reusable resource, which writes the assigned output destination directly without an
 * explicit host-staging or intermediate-copy step.</p>
 *
 * <p>All types remain package-private. Shared Runtime sees only its nominal resource,
 * representation, executable, and schedule contracts; no native handle, Objective-C value,
 * MPSGraph type, operation, graph node, backend lookup, or route choice crosses the hot-path
 * boundary.</p>
 */
package io.github.pho001.synaptik.backend.metal.internal;
