/**
 * Defines the minimal backend-neutral vocabulary shared by planning and later lifecycle layers.
 *
 * <p>A {@link io.github.pho001.synaptik.backend.contract.BackendId} names a backend ownership
 * domain. A {@link io.github.pho001.synaptik.backend.contract.BackendDeviceId} names one device
 * within that backend's opaque device-token namespace. A
 * {@link io.github.pho001.synaptik.backend.contract.DeviceClass} describes only whether a device
 * belongs to the coarse CPU or accelerator category. A
 * {@link io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot} associates one
 * backend's currently reported available device identities with those classes. These immutable
 * identities and declarative facts allow shared contracts to name ownership and supplied
 * availability without retaining a concrete backend implementation or live service.</p>
 *
 * <p>A backend identity, backend-scoped device identity, and device class are distinct from a
 * backend-internal implementation route. Device class is not stored by the device identity; the
 * availability snapshot supplies that association for one backend at one point in time. It does
 * not discover devices, register backends, monitor liveness, or establish capability,
 * configuration preference, selected ownership, resource access, preparation, or execution.
 * Concrete backend implementations and the engine composition root own those later concerns.</p>
 */
package io.github.pho001.synaptik.backend.contract;
