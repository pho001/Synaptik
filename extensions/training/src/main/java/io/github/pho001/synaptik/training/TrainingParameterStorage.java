package io.github.pho001.synaptik.training;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutKind;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.nn.module.Module;
import io.github.pho001.synaptik.nn.module.Parameter;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Explicitly prepares a complete Module parameter tree for the native-storage training domain.
 *
 * <p>This backend-neutral operation copies the exact represented bytes of finite values,
 * including signed zero, into fresh caller-owned shared native memory. It changes only each
 * existing Tensor's borrowed host-storage association: Parameter and Tensor identity, descriptor,
 * and graph provenance stay fixed. It does not open a session or transfer values to a backend. A
 * caller must give the operation exclusive use of the Module and parameter storage and must never
 * call it while a {@link TrainingSession} is open on that Module.</p>
 */
public final class TrainingParameterStorage {
    private TrainingParameterStorage() {
    }

    /**
     * Copies every recursively discovered parameter into fresh exact-capacity, aligned, writable
     * native storage allocated from a caller-owned shareable Arena.
     *
     * <p>A zero-byte scope probe checks destination sharing without native allocation. Complete
     * source validation precedes parameter allocation or association changes. All source
     * regions are copied before any association changes; successful installs follow recursive
     * parameter path order. A source may be heap or native, read-only, and unaligned; it must be
     * live and accessible to this thread. The destination Arena must be live and accessible to
     * other operation threads. A confined Arena owned by this thread fails the shareability
     * check; probing one owned by another thread propagates the JDK's
     * {@link WrongThreadException} before that check. An unexpected ordinary install failure
     * restores earlier exact storage
     * associations and retains any rollback failures as suppressed exceptions. Allocations
     * already made in the supplied Arena cannot be reclaimed individually. Neither this method
     * nor a subsequent session closes the source or destination Arena. The caller keeps the
     * destination Arena alive until after any session using the copied values has closed.</p>
     *
     * @param module non-null, fully initialized Module tree whose parameter wrappers, Tensor
     *     bindings, and bytes are exclusively available to this call
     * @param arena non-null live, shareable caller-owned destination Arena; the caller closes it
     *     after all sessions using the copied storage have closed
     * @throws NullPointerException if {@code module} or {@code arena} is null
     * @throws IllegalStateException if discovery is empty or incomplete, source storage is absent,
     *     dead, or inaccessible to this thread, or the destination Arena is dead
     * @throws IllegalArgumentException if Tensor identity repeats, source regions overlap,
     *     parameter geometry/type/size/value is unsupported, source storage has wrong capacity,
     *     or an Arena accessible to this thread is not shareable
     * @throws WrongThreadException if the destination Arena is confined to a different thread
     *     and its zero-byte scope probe rejects access
     * @throws RuntimeException if allocation, copying, or an ordinary installation fails; no
     *     Tensor association changes on allocation/copy failure, and earlier installs are rolled
     *     back on installation failure
     */
    public static void copyToSharedNative(Module module, Arena arena) {
        Module sourceModule = Objects.requireNonNull(module, "module");
        Arena destinationArena = Objects.requireNonNull(arena, "arena");
        Map<String, Parameter> discovered = sourceModule.parametersRecursively();
        if (discovered.isEmpty()) {
            throw new IllegalStateException("training module has no parameters");
        }
        if (!destinationArena.scope().isAlive()) {
            throw new IllegalStateException("destination Arena is not alive");
        }
        Thread accessProbe = Thread.ofVirtual().unstarted(() -> { });
        if (!destinationArena.allocate(0, 1).isAccessibleBy(accessProbe)) {
            throw new IllegalArgumentException("destination Arena must be shareable");
        }

        List<Copy> copies = new ArrayList<>(discovered.size());
        IdentityHashMap<Tensor, String> tensors = new IdentityHashMap<>();
        for (Map.Entry<String, Parameter> entry : discovered.entrySet()) {
            String path = entry.getKey();
            Tensor tensor = Objects.requireNonNull(entry.getValue().value(), "parameter.value()");
            DataType type = tensor.descriptor().dataType();
            if (type != DataType.FLOAT64 && type != DataType.FLOAT32
                    && type != DataType.BFLOAT16 && type != DataType.FLOAT16) {
                throw new IllegalArgumentException("unsupported training parameter type at path "
                        + path + ": " + type);
            }
            if (!tensor.descriptor().requiresGrad()
                    || !tensor.descriptor().shape().isFullyStatic()) {
                throw new IllegalArgumentException(
                        "training parameter requires gradients and a static shape at path " + path);
            }
            var layout = tensor.descriptor().layout().orElseThrow(() ->
                    new IllegalArgumentException("unresolved parameter layout at path " + path));
            if (layout.kind() != LayoutKind.DENSE_CONTIGUOUS
                    || layout.storageOffset() != 0 || layout.isView()) {
                throw new IllegalArgumentException(
                        "parameter must be dense, offset-zero and non-view at path " + path);
            }
            long count = tensor.descriptor().shape().knownElementCount().orElseThrow();
            if (count <= 0 || count > Integer.MAX_VALUE || layout.referencedElementSpan() != count) {
                throw new IllegalArgumentException("invalid parameter element span at path " + path);
            }
            TrainingParameter.validateSnapshotByteSizes(path, type, count);
            HostTensorStorage source = tensor.hostStorage().orElseThrow(() ->
                    new IllegalStateException("training parameter has no host storage at path " + path));
            validateSource(path, source, type, count);
            TrainingParameter.validateFiniteStorage(path, source.segment(), type, (int) count);
            String firstPath = tensors.putIfAbsent(tensor, path);
            if (firstPath != null) {
                throw new IllegalArgumentException("parameter Tensor identity is repeated at paths "
                        + firstPath + " and " + path);
            }
            for (Copy previous : copies) {
                if (source.segment().asOverlappingSlice(previous.source.segment()).isPresent()) {
                    throw new IllegalArgumentException("parameter storages overlap at paths "
                            + previous.path + " and " + path);
                }
            }
            copies.add(new Copy(path, tensor, source, type, count));
        }

        for (Copy copy : copies) {
            long bytes = copy.source.byteSize();
            MemorySegment segment = destinationArena.allocate(bytes, copy.type.byteWidth());
            copy.destination = new MemorySegmentStorage(copy.type, copy.count, segment);
            TrainingParameter.validateStorage(
                    copy.path, copy.destination, copy.type, copy.count, true, true);
            if (!segment.isAccessibleBy(accessProbe)) {
                throw new IllegalArgumentException(
                        "destination storage must be shareable at path " + copy.path);
            }
            MemorySegment.copy(copy.source.segment(), 0, segment, 0, bytes);
        }
        int installed = 0;
        try {
            for (Copy copy : copies) {
                copy.tensor.replaceHostStorage(copy.destination);
                installed++;
            }
        } catch (RuntimeException failure) {
            for (int index = installed - 1; index >= 0; index--) {
                Copy copy = copies.get(index);
                try {
                    copy.tensor.replaceHostStorage(copy.source);
                } catch (RuntimeException rollbackFailure) {
                    failure.addSuppressed(rollbackFailure);
                }
            }
            throw failure;
        }
    }

    /**
     * Checks exact readable source capacity without imposing destination alignment on heap arrays.
     *
     * @param path non-null diagnostic parameter path
     * @param source non-null candidate borrowed source storage
     * @param type non-null declared parameter type
     * @param count positive static parameter element count
     * @throws IllegalArgumentException if source type or physical capacity differs
     * @throws IllegalStateException if source storage is dead or inaccessible to this thread
     */
    private static void validateSource(
            String path, HostTensorStorage source, DataType type, long count) {
        if (source.dataType() != type || source.elementCapacity() != count
                || source.byteSize() != Math.multiplyExact(count, type.byteWidth())
                || source.segment().byteSize() != source.byteSize()) {
            throw new IllegalArgumentException("parameter source storage schema mismatch at path " + path);
        }
        if (!source.isAlive()) {
            throw new IllegalStateException("parameter source storage is not alive at path " + path);
        }
        if (!source.segment().isAccessibleBy(Thread.currentThread())) {
            throw new IllegalStateException("parameter source storage is not accessible at path " + path);
        }
    }

    private static final class Copy {
        private final String path;
        private final Tensor tensor;
        private final HostTensorStorage source;
        private final DataType type;
        private final long count;
        private HostTensorStorage destination;

        private Copy(String path, Tensor tensor, HostTensorStorage source, DataType type, long count) {
            this.path = path;
            this.tensor = tensor;
            this.source = source;
            this.type = type;
            this.count = count;
        }
    }
}
