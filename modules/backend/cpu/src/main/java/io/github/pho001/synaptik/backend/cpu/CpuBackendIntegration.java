package io.github.pho001.synaptik.backend.cpu;

import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferRepresentation;
import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas.CpuBackendComposition;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnRoutePlan;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.prepare.PartitionPreparation;
import io.github.pho001.synaptik.prepare.PreparedScheduleAssembler;
import io.github.pho001.synaptik.prepare.PreparedScheduleContributor;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import java.util.Objects;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;

/**
 * Supported CPU service-provider integration boundary for lifecycle composition.
 *
 * <p>The adapter is intended for Engine and other module-integration code, not as an ordinary
 * application facade. One instance owns optional automatically discovered OpenBLAS state and,
 * when explicitly configured, one required exact oneDNN library. Required oneDNN loading errors
 * are never converted into portable execution. Unsupported operations remain ordinary portable
 * CPU units selected during preparation. The adapter must remain open for new preparation and
 * run admission; native resources retain leases for already-admitted runs after owner closure.
 * The adapter and its returned immutable collaborators may be used concurrently; each Runtime
 * invocation still owns an isolated run state and fresh run-owned representations. Preparation
 * retains only immutable recipes, including initializer recipes for eligible source-only
 * published constants; it never retains a run-owned physical representation.</p>
 */
public final class CpuBackendIntegration implements AutoCloseable {
    /**
     * Freezes the oneDNN API policy before CPU analysis, not provider selection.
     * Portable/OpenBLAS may still win whole-plan comparison; no policy changes during execution.
     */
    public enum OneDnnApi {
        /** Lets cold analysis compare the qualified primitive and Graph implementations. */
        AUTO,
        /** Requires the oneDNN primitive API for selected native units. */
        PRIMITIVES,
        /** Requires the oneDNN Graph API for selected native units. */
        GRAPH
    }

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
        return takeComposition(CpuBackendComposition.openAutomatic());
    }

    /**
     * Opens CPU with an explicitly required exact oneDNN library and automatic API selection.
     *
     * @param absolutePath non-null absolute path to a oneDNN 3.14.0 LP64 SEQ shared library
     * @return new CPU integration owning the library; ordinary unsupported units remain portable
     * @throws NullPointerException if {@code absolutePath} is null
     * @throws IllegalArgumentException if the path, ABI, version, runtime, or symbols are invalid
     * @throws RuntimeException if required native loading or composition construction fails
     * @throws Error if construction or cleanup reports an error
     */
    public static CpuBackendIntegration openOneDnn(Path absolutePath) {
        return openOneDnn(absolutePath, OneDnnApi.AUTO);
    }

    /**
     * Opens CPU with one required exact oneDNN library and an immutable API preference.
     *
     * <p>The library path is never discovered or substituted. Failure to load or authenticate
     * oneDNN fails this call; independent optional OpenBLAS discovery cannot mask that failure.
     * Only qualified bare homogeneous FLOAT32 MATMUL with K greater than one is native-eligible.
     * Convolution, external bias, activation and other unsupported units retain the existing CPU
     * lowering and automatic-differentiation path. Native selection and all resource requirements
     * are fixed before execution, with no late fallback or retry.</p>
     *
     * <p>Both APIs consume common CPU lowering and canonical external buffers; Graph is not
     * automatic differentiation and is not forced by node count. The preference restricts native
     * API candidates but does not force a native provider to win. The exact binary must target
     * little-endian LP64, with Graph support and sequential CPU runtime (SEQ, code 1). No binary
     * is bundled. Classpath deployments require {@code --enable-native-access=ALL-UNNAMED}
     * for the JDK Foreign Function and Memory API.</p>
     *
     * @param absolutePath non-null absolute path to a oneDNN 3.14.0 LP64 SEQ shared library
     * @param api non-null API preference frozen for this integration's lifetime
     * @return new CPU integration owning the exact native library and optional OpenBLAS provider
     * @throws NullPointerException if an argument is null
     * @throws IllegalArgumentException if the path, ABI, version, runtime, or symbols are invalid
     * @throws RuntimeException if required native loading or composition construction fails
     * @throws Error if construction or cleanup reports an error
     */
    public static CpuBackendIntegration openOneDnn(Path absolutePath, OneDnnApi api) {
        Objects.requireNonNull(api, "api");
        CpuOneDnnRoutePlan.Api internalApi = switch (api) {
            case AUTO -> CpuOneDnnRoutePlan.Api.AUTO;
            case PRIMITIVES -> CpuOneDnnRoutePlan.Api.PRIMITIVE;
            case GRAPH -> CpuOneDnnRoutePlan.Api.GRAPH;
        };
        return takeComposition(CpuBackendComposition.openOneDnn(absolutePath, internalApi));
    }

    private static CpuBackendIntegration takeComposition(CpuBackendComposition composition) {
        try {
            return new CpuBackendIntegration(composition);
        } catch (RuntimeException | Error failure) {
            try { composition.close(); } catch (RuntimeException | Error cleanup) {
                if (cleanup != failure) failure.addSuppressed(cleanup);
            }
            throw failure;
        }
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
     *     it is independent of optional OpenBLAS and explicit oneDNN route eligibility
     */
    public BackendAvailabilitySnapshot availabilitySnapshot() {
        return composition.availabilitySnapshot();
    }

    /**
     * Creates one positional preparation for a non-empty CPU-owned partition.
     *
     * <p>A zero-node pass-through graph has no partition or prepared buffer assignment and remains
     * rejected by ordinary preparation. For a mixed or multi-partition plan, Engine and shared
     * Prepare combine this partition collaboration with
     * {@link #scheduleContributor() CPU's physical contribution} and the other registered
     * owners' contributions into one complete schedule.</p>
     *
     * @return a non-null immutable preparation whose CPU-private input and plan types remain hidden
     *     behind shared Prepare roles
     * @throws IllegalStateException if this adapter is closed
     */
    public PartitionPreparation<?, ?> partitionPreparation() {
        return composition.preparation();
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
     * Returns the retained CPU physical-creation contributor used by shared mixed-owner assembly.
     *
     * @return the non-null retained immutable contributor; ownership remains with this integration
     * @throws IllegalStateException if this integration is closed
     */
    public PreparedScheduleContributor scheduleContributor() {
        return (PreparedScheduleContributor) composition.scheduleAssembler();
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
     * Reports whether one nominal representation can be cold-bound as exact native host staging
     * for a prepared CPU/Metal storage-layout transfer of any model data type.
     *
     * <p>The admitted descriptor has positive static extents and a resolved, positive-stride
     * layout proved injective by the shared CPU output proof. Unresolved, negative,
     * zero-stride broadcast, overlapping and unproved layouts are rejected at the boundary.</p>
     *
     * @param representation non-null candidate CPU representation
     * @param descriptor non-null exact logical descriptor
     * @param writable whether the transfer writes into the CPU representation
     * @return {@code true} only for a live current-thread-accessible native, exact-span,
     *     rank-0..16 supported storage representation with matching type and mutability
     * @throws NullPointerException if an object argument is null
     * @throws IllegalStateException if this integration is closed
     */
    public boolean acceptsStorageLayoutTransfer(
            BufferRepresentation representation,
            TensorDescriptor descriptor,
            boolean writable) {
        composition.assertOpen();
        Objects.requireNonNull(representation, "representation");
        Objects.requireNonNull(descriptor, "descriptor");
        if (!isStorageTransfer(descriptor)
                || !(representation instanceof CpuBufferRepresentation cpu)
                || cpu.dataType() != descriptor.dataType()
                || !cpu.isAccessible()) {
            return false;
        }
        MemorySegment segment = cpu.segment();
        long byteCount = transferByteCount(descriptor);
        return cpu.byteSize() == byteCount
                && segment.byteSize() == byteCount
                && segment.isNative()
                && (!writable || !segment.isReadOnly());
    }

    /**
     * Cold-binds one exact storage-layout CPU representation as reusable native host staging.
     *
     * @param representation non-null candidate CPU representation
     * @param descriptor non-null exact logical descriptor
     * @param writable whether the transfer will write into the returned segment
     * @return the exact non-owning live native physical-span segment retained by the representation
     * @throws NullPointerException if an object argument is null
     * @throws IllegalArgumentException if representation, type, layout, byte extent, native
     *     carrier, accessibility, or mutability is incompatible
     * @throws IllegalStateException if this integration or representation is closed
     */
    public MemorySegment bindStorageLayoutTransfer(
            BufferRepresentation representation,
            TensorDescriptor descriptor,
            boolean writable) {
        if (!acceptsStorageLayoutTransfer(representation, descriptor, writable)) {
            throw new IllegalArgumentException(
                    "CPU transfer requires an exact live native storage representation");
        }
        return ((CpuBufferRepresentation) representation).segment();
    }

    /**
     * Copies one borrowed CPU publication representation into canonical detached host bytes.
     *
     * <p>Logical elements are traversed in row-major coordinate order through the supplied fully
     * static, resolved descriptor, including its non-negative offset and positive or zero strides.
     * {@code FLOAT64}, {@code FLOAT32}, {@code BFLOAT16}, {@code FLOAT16}, {@code INT64}, and
     * {@code INT32} represented bits are encoded big-endian without conversion. Both FP8 types
     * copy their represented bytes verbatim, including every NaN encoding and signed zero;
     * floating-point NaN payloads and signed zeros are preserved for wider types as well.
     * Only {@code BOOL} restricts stored bytes to {@code 0} and {@code 1}. Source segment values
     * are interpreted in native byte order. The fresh mutable
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

    private static boolean isStorageTransfer(TensorDescriptor descriptor) {
        if (!descriptor.shape().isFullyStatic()
                || descriptor.shape().rank() > 16
                || descriptor.layout().isEmpty()) {
            return false;
        }
        long[] dimensions = descriptor.shape().toLongArray();
        LayoutDescriptor layout = descriptor.layout().orElseThrow();
        long[] strides = layout.strides();
        long covered = 1L;
        try {
            for (int axis = 0; axis < dimensions.length; axis++) {
                if (dimensions[axis] <= 0L || strides[axis] <= 0L) return false;
                covered = Math.addExact(covered,
                        Math.multiplyExact(dimensions[axis] - 1L, strides[axis]));
            }
            if (!io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuOutputInjectivity
                    .isInjective(dimensions, strides)) return false;
            return layout.referencedElementSpan()
                    == Math.addExact(layout.storageOffset(), covered)
                    && Math.multiplyExact(
                            layout.referencedElementSpan(),
                            descriptor.dataType().byteWidth()) > 0L;
        } catch (ArithmeticException overflow) {
            return false;
        }
    }

    private static long transferByteCount(TensorDescriptor descriptor) {
        return Math.multiplyExact(
                descriptor.layout().orElseThrow().referencedElementSpan(),
                descriptor.dataType().byteWidth());
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
