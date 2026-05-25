package backend.cpu.storage;

import java.util.List;
import java.util.Objects;

public record CpuStorageBindings(
        List<CpuStorageView> inputs,
        CpuStorageView output
) {
    public CpuStorageBindings {
        inputs = List.copyOf(Objects.requireNonNull(inputs, "inputs cannot be null"));
        for (CpuStorageView input : inputs) {
            Objects.requireNonNull(input, "inputs cannot contain null views");
        }
        output = Objects.requireNonNull(output, "output cannot be null");
    }

    public CpuStorageView input(int index) {
        return inputs.get(index);
    }
}
