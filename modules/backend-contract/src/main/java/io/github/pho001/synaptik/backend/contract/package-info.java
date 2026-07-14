/**
 * Defines the minimal backend-neutral vocabulary shared by planning and later lifecycle layers.
 *
 * <p>A {@link io.github.pho001.synaptik.backend.contract.BackendId} names a backend ownership
 * domain. A {@link io.github.pho001.synaptik.backend.contract.BackendDeviceId} names one device
 * within that backend's opaque device-token namespace. These immutable identities allow shared
 * contracts to describe ownership without retaining a concrete backend implementation or live
 * service.</p>
 *
 * <p>Identity does not establish registration, discovery, availability, capability, resource
 * access, preparation, or execution. Concrete backend implementations and the engine composition
 * root own those later concerns.</p>
 */
package io.github.pho001.synaptik.backend.contract;
