package io.github.pho001.synaptik.backend.cpu;

import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas.CpuBackendComposition;
import io.github.pho001.synaptik.compiler.CompileArtifacts;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.prepare.PartitionPreparation;
import io.github.pho001.synaptik.prepare.PreparedScheduleAssembler;
import io.github.pho001.synaptik.runtime.execution.PreparedExecution;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import java.util.List;
import java.util.Objects;

/**
 * Supported CPU service-provider integration boundary for lifecycle composition.
 *
 * <p>The adapter is intended for Engine and other module-integration code, not as an ordinary
 * application facade. One instance owns optional automatically discovered OpenBLAS state and
 * must remain open while preparing or running recipes obtained from it. Portable CPU preparation
 * remains available when the bounded optional-provider discovery or qualification attempt fails.
 * The adapter and its returned immutable collaborators may be used concurrently; each Runtime
 * invocation still owns an isolated run state and fresh run-owned representations. Preparation
 * retains only immutable recipes, including initializer recipes for eligible source-only
 * published constants; it never retains a run-owned physical representation.</p>
 */
public final class CpuBackendIntegration implements AutoCloseable {
    private final CpuCapabilityProvider capabilityProvider = new CpuCapabilityProvider();
    private final CpuBackendComposition composition;
    private final CpuLocalWorkloadTuning localWorkloadTuning;
    private final CpuCompletePlanTuning completePlanTuning;

    /**
     * Retains one CPU-private composition owner for the lifetime of this adapter.
     *
     * @param composition the non-null open composition owner; ownership transfers to this adapter
     * @throws NullPointerException if {@code composition} is {@code null}
     */
    private CpuBackendIntegration(CpuBackendComposition composition) {
        this.composition = Objects.requireNonNull(composition, "composition");
        this.localWorkloadTuning = new CpuLocalWorkloadTuning(composition);
        this.completePlanTuning = new CpuCompletePlanTuning(composition, localWorkloadTuning);
    }

    /**
     * Opens the exact/default CPU composition and attempts bounded automatic OpenBLAS setup.
     *
     * <p>Automatic discovery is limited to the CPU backend's fixed platform table. A load or
     * qualification attempt may make one OpenBLAS route eligible, but does not guarantee that the
     * route is present or selected. Failure of the optional setup leaves the portable route
     * available.</p>
     *
     * @return a new non-null open CPU integration owner with portable execution available
     * @throws RuntimeException if required CPU composition construction fails
     * @throws Error if construction or cleanup reports a fatal error
     */
    public static CpuBackendIntegration open() {
        return new CpuBackendIntegration(CpuBackendComposition.openAutomatic());
    }

    /**
     * Returns the stable stateless CPU capability provider retained by this adapter.
     *
     * @return the non-null retained CPU capability provider; repeated calls return the same
     *     instance and transfer no ownership
     */
    public CpuCapabilityProvider capabilityProvider() {
        return capabilityProvider;
    }

    /**
     * Returns the fixed portable CPU availability fact without performing discovery.
     *
     * @return the retained immutable snapshot containing exactly the {@code cpu/host} CPU device;
     *     it is independent of optional OpenBLAS eligibility
     */
    public BackendAvailabilitySnapshot availabilitySnapshot() {
        return composition.availabilitySnapshot();
    }

    /**
     * Creates the sole positional preparation for exactly one non-empty CPU partition.
     *
     * <p>A zero-node pass-through graph has no prepared buffer assignment under the current
     * Prepare contract and is rejected. A mixed-owner or multi-partition schedule is also outside
     * this CPU-owned complete-schedule boundary and must be composed by later Engine/Prepare
     * integration.</p>
     *
     * @param artifacts non-null immutable compile artifacts containing exactly one non-empty
     *     maximal partition owned by CPU; the artifacts are inspected but not mutated
     * @return a non-null immutable singleton whose CPU-private input and plan types remain hidden
     *     behind shared Prepare roles
     * @throws NullPointerException if {@code artifacts} is {@code null}
     * @throws IllegalArgumentException if the artifacts have zero, multiple, empty, or non-CPU
     *     partitions, or do not provide complete resolved CPU preparation facts
     * @throws IllegalStateException if this adapter is closed
     */
    public List<PartitionPreparation<?, ?>> preparations(CompileArtifacts artifacts) {
        return composition.preparations(artifacts);
    }

    /**
     * Prepares the complete immutable Runtime recipe for exactly one non-empty CPU partition.
     *
     * <p>CPU derives physical geometry for any fully static canonical source-only published
     * constants and supplies those declarations to shared Prepare. The returned execution retains
     * initializer recipes only: every run creates and initializes fresh run-owned CPU
     * representations exactly once while creating its new run state. A source-only constant adds
     * no executable schedule occurrence. Pure zero-node constant graphs remain outside this
     * adapter's supported one-partition domain.</p>
     *
     * @param artifacts non-null immutable compile artifacts containing exactly one non-empty
     *     maximal partition owned by CPU; inspected but not mutated
     * @return a non-null immutable reusable prepared recipe containing no run-owned CPU resource
     * @throws NullPointerException if {@code artifacts} is {@code null}
     * @throws IllegalArgumentException if the artifacts or a required source-only constant role,
     *     descriptor, scalar type, or physical geometry is unsupported or inconsistent
     * @throws ArithmeticException if canonical layout or byte-size arithmetic overflows
     * @throws IllegalStateException if this adapter is closed
     */
    public PreparedExecution prepare(CompileArtifacts artifacts) {
        return composition.prepare(artifacts);
    }

    /**
     * Returns the retained CPU-owned cold local-workload tuning collaboration.
     *
     * <p>The collaboration is an integration building block, not a tuning runner. It borrows this
     * adapter's lifetime and provider coordination, performs no measurement or cache access, and
     * has no independent close operation.</p>
     *
     * @return the same non-null collaboration on every call; ownership is not transferred
     * @throws IllegalStateException if this adapter is closed
     */
    public CpuLocalWorkloadTuning localWorkloadTuning() {
        composition.assertOpen();
        return localWorkloadTuning;
    }

    /**
     * Returns the retained CPU-owned complete-plan candidate collaboration.
     *
     * <p>The collaboration borrows this integration's lifetime and performs only cold candidate
     * exposure, decision encoding, and fresh recipe preparation. It does not execute, measure,
     * rerank Phase-1 choices, access a cache, or apply fallback policy. Its current candidates
     * vary only retained CPU topology and representation choices for the same sole partition and
     * exact Phase-1 state.</p>
     *
     * @return the same non-null collaboration on every call; ownership is not transferred
     * @throws IllegalStateException if this adapter is closed
     */
    public CpuCompletePlanTuning completePlanTuning() {
        composition.assertOpen();
        return completePlanTuning;
    }

    /**
     * Returns the retained immutable CPU schedule assembler.
     *
     * <p>The assembler independently accepts only a complete context for one non-empty CPU
     * partition and rejects assembly after this adapter closes.</p>
     *
     * @return the non-null retained immutable shared schedule-assembler collaboration; repeated
     *     calls return the same instance and transfer no ownership
     * @throws IllegalStateException if this adapter is closed
     */
    public PreparedScheduleAssembler scheduleAssembler() {
        return composition.scheduleAssembler();
    }

    /**
     * Borrows intrinsically valid CPU-compatible host storage without taking ownership.
     *
     * <p>This operation validates only facts available from the storage itself. It does not
     * compare the storage with a logical binding's type, required span, capacity, or access role.
     * Closing the returned wrapper never closes caller-owned storage.</p>
     *
     * @param storage non-null live storage whose exact segment is accessible to the current
     *     thread and whose element capacity, data type, byte size, and observable heap carrier are
     *     intrinsically consistent; the caller retains storage ownership
     * @return a new non-owning CPU buffer representation retaining the exact {@code storage};
     *     never {@code null}
     * @throws NullPointerException if {@code storage} is {@code null}
     * @throws IllegalArgumentException if intrinsic byte geometry, data type, or an observable
     *     heap carrier is inconsistent or unsupported
     * @throws IllegalStateException if this adapter or the storage is closed, or the segment is
     *     inaccessible to the current thread
     * @throws ArithmeticException if capacity-to-byte conversion overflows
     */
    public BufferRepresentation borrow(HostTensorStorage storage) {
        return composition.borrow(storage);
    }

    /**
     * Copies one borrowed CPU publication representation into canonical detached host bytes.
     *
     * <p>Logical elements are traversed in row-major coordinate order through the supplied fully
     * static, resolved descriptor, including its non-negative offset and positive or zero strides.
     * {@code FLOAT64}, {@code FLOAT32}, {@code BFLOAT16}, {@code INT64}, and {@code INT32}
     * represented bits are encoded big-endian without conversion; floating-point NaN payloads and
     * signed zeros are preserved. {@code BOOL} accepts and copies only stored bytes {@code 0} and
     * {@code 1}. Source segment values are interpreted in native byte order. The fresh mutable
     * result belongs exclusively to the caller and is unaffected by later representation, result,
     * or adapter closure.</p>
     *
     * <p>The call is synchronous and does not retain, close, mutate, or transfer the source. The
     * caller must keep this adapter and the Runtime result lease open, keep the representation
     * accessible to the calling thread, and prevent source mutation or closure for the complete
     * call. Concurrent mutation has no atomic-snapshot guarantee.</p>
     *
     * @param representation non-null exact borrowed publication representation implemented by the
     *     current CPU backend; ownership remains with its existing owner
     * @param descriptor non-null exact resolved logical descriptor paired with that publication by
     *     the caller; its shape must be fully static, its layout present, and its data type equal to
     *     the representation data type; the descriptor is inspected but not retained or mutated
     * @param maximumBytes non-negative caller limit for the canonical payload size in bytes
     * @return a fresh non-null caller-owned mutable byte array in canonical row-major big-endian
     *     form; rank-zero descriptors copy their offset element, and zero-element shapes return a
     *     fresh empty array without source-element access
     * @throws NullPointerException if {@code representation} or {@code descriptor} is {@code null}
     * @throws IllegalStateException if this adapter is closed, or the CPU representation is closed
     *     or inaccessible to the current thread
     * @throws IllegalArgumentException if the byte limit is negative; shape or layout is
     *     unsupported; the payload exceeds the caller limit or JVM array ceiling; the concrete
     *     representation, data type, element geometry, carrier, or capacity is incompatible; or a
     *     BOOL element is not canonical
     * @throws ArithmeticException if logical count, byte count, or source/destination address
     *     arithmetic overflows
     * @throws OutOfMemoryError if the JVM cannot allocate the otherwise valid result array
     */
    public byte[] copyToCanonicalHostBytes(BufferRepresentation representation,
            TensorDescriptor descriptor, long maximumBytes) {
        return composition.copyToCanonicalHostBytes(representation, descriptor, maximumBytes);
    }

    /**
     * Prevents new preparation, borrowing, and assembler access, then quiesces and closes optional
     * native ownership. Repeated and concurrent calls are idempotent. Prepared recipes and runs
     * must not outlive this adapter; callers must coordinate closure with their use.
     *
     * @throws RuntimeException if native restoration or cleanup reports an unchecked failure
     * @throws Error if native restoration or cleanup reports an error
     */
    @Override
    public void close() {
        composition.close();
    }
}
