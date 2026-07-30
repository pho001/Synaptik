/**
 * Defines runtime-owned identity vocabulary for prepared memory.
 *
 * <p>The current package surface contains only
 * {@link io.github.pho001.synaptik.runtime.memory.BufferSlot}: a deeply immutable, non-negative
 * identity interpreted within one prepared-memory-plan context. It is distinct from a graph value
 * identity and carries no physical buffer, storage handle, address, allocation, resource, device,
 * or residency fact.
 *
 * <p>Prepared-memory plans, workspace identities, slot access, per-run bindings, and resource
 * lifetime remain planned contracts. This package currently allocates no memory and provides no
 * storage or execution access.
 */
package io.github.pho001.synaptik.runtime.memory;
