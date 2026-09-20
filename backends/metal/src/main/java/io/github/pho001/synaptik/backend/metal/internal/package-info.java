/**
 * Implements the non-executing Metal native and run-owned storage foundation.
 *
 * <p>Types in this package are deliberately package-private. One context owns a default Metal
 * device and command queue, while fresh buffer and workspace wrappers own shared-storage native
 * resources and child leases. The package submits no command, selects no operation route, and
 * exposes no native handle or physical storage API to shared Runtime.</p>
 */
package io.github.pho001.synaptik.backend.metal.internal;
