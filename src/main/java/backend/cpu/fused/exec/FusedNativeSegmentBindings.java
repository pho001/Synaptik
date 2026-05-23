package backend.cpu.fused.exec;

import tensor.storage.NativeTensorStorage;

import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Objects;

/**
 * Per-dispatch native segment bindings for one fused CPU output.
 */
public record FusedNativeSegmentBindings(
        List<NativeTensorStorage> inputs,
        NativeTensorStorage output
) {
    public FusedNativeSegmentBindings {
        inputs = List.copyOf(inputs == null ? List.of() : inputs);
        output = Objects.requireNonNull(output, "output cannot be null");
    }

    public NativeTensorStorage inputStorage(int index) {
        if (index < 0 || index >= inputs.size()) {
            throw new IndexOutOfBoundsException("Fused native input index out of range: " + index);
        }
        NativeTensorStorage storage = inputs.get(index);
        storage.ensureOpen();
        return storage;
    }

    public MemorySegment inputSegment(int index) {
        return inputStorage(index).segment();
    }

    public MemorySegment outputSegment() {
        output.ensureOpen();
        return output.segment();
    }
}
