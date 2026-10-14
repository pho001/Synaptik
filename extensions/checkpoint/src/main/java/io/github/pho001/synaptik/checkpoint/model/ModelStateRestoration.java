package io.github.pho001.synaptik.checkpoint.model;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.module.StateDictionary;
import io.github.pho001.synaptik.nn.module.StateEntry;
import io.github.pho001.synaptik.nn.module.StateKind;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Validates a complete bound target against a detached v1 candidate and reconstructs one heap
 * leaf per target Tensor-identity group. It neither installs state nor retains the target.
 */
final class ModelStateRestoration {
    private static final ValueLayout.OfLong NATIVE_LONG =
            ValueLayout.JAVA_LONG.withOrder(ByteOrder.nativeOrder());
    private static final ValueLayout.OfInt NATIVE_INT =
            ValueLayout.JAVA_INT.withOrder(ByteOrder.nativeOrder());
    private static final ValueLayout.OfShort NATIVE_SHORT =
            ValueLayout.JAVA_SHORT.withOrder(ByteOrder.nativeOrder());

    private ModelStateRestoration() { }

    /**
     * Validates all paths, descriptors, and alias bytes before allocating any restored Tensor.
     * A parameter/buffer mixed alias is gradient-eligible because its parameter path requires it.
     *
     * @param candidate fully decoded detached model artifact
     * @param target complete bound NN state snapshot, under caller-owned quiescence
     * @return complete candidate dictionary in target order, with one leaf per alias group
     * @throws IllegalArgumentException if paths, roles, descriptors, or alias payloads mismatch
     * @throws RuntimeException if compatible host reconstruction fails before installation
     */
    static StateDictionary reconstruct(ModelArtifactCandidate candidate, StateDictionary target) {
        List<ModelArtifactCodec.State> states = candidate.states();
        var payloads = candidate.envelope().entries();
        Map<String, Integer> byPath = new HashMap<>();
        for (int i = 0; i < states.size(); i++) byPath.put(states.get(i).path(), i);
        if (target.entries().size() != states.size())
            throw new IllegalArgumentException("Model checkpoint path count mismatch");
        IdentityHashMap<Tensor, Group> byTensor = new IdentityHashMap<>();
        List<Group> groups = new ArrayList<>();
        List<Group> assigned = new ArrayList<>(target.entries().size());
        for (StateEntry entry : target.entries()) {
            Integer index = byPath.get(entry.path());
            if (index == null) throw new IllegalArgumentException("Missing model checkpoint path: " + entry.path());
            ModelArtifactCodec.State state = states.get(index);
            var descriptor = entry.value().descriptor();
            if (entry.kind() != state.kind() || descriptor.dataType() != state.type()
                    || !descriptor.shape().equals(state.shape())
                    || entry.kind() == StateKind.PARAMETER && !descriptor.requiresGrad())
                throw new IllegalArgumentException("Model checkpoint descriptor mismatch: " + entry.path());
            ByteBuffer bytes = payloads.get(index + 1).payload();
            Group group = byTensor.get(entry.value());
            if (group == null) {
                group = new Group(state, bytes, entry.kind() == StateKind.PARAMETER);
                byTensor.put(entry.value(), group);
                groups.add(group);
            } else {
                if (!group.bytes.equals(bytes))
                    throw new IllegalArgumentException("Model checkpoint alias values differ: " + entry.path());
                group.parameter |= entry.kind() == StateKind.PARAMETER;
            }
            assigned.add(group);
        }
        // No installation has occurred; allocation or bit-copy failure discards this candidate.
        for (Group group : groups) group.restored = restore(group.state, group.bytes, group.parameter);
        List<StateEntry> entries = new ArrayList<>(assigned.size());
        for (int i = 0; i < assigned.size(); i++) {
            StateEntry current = target.entries().get(i);
            entries.add(new StateEntry(current.path(), current.kind(), assigned.get(i).restored));
        }
        return new StateDictionary(entries);
    }

    private static Tensor restore(ModelArtifactCodec.State state, ByteBuffer canonical,
                                  boolean requiresGrad) {
        TensorDescriptor descriptor = new TensorDescriptor(state.type(), state.shape(),
                Optional.of(LayoutDescriptor.contiguous(state.shape())), requiresGrad);
        Tensor tensor = TensorFactory.allocate(descriptor);
        MemorySegment nativeBytes = tensor.hostStorage().orElseThrow().segment();
        ByteBuffer input = canonical.duplicate().order(ByteOrder.BIG_ENDIAN);
        long offset = 0;
        switch (state.type()) {
            case FLOAT64, INT64 -> {
                while (input.hasRemaining()) {
                    nativeBytes.set(NATIVE_LONG, offset, input.getLong());
                    offset += 8;
                }
            }
            case FLOAT32, INT32 -> {
                while (input.hasRemaining()) {
                    nativeBytes.set(NATIVE_INT, offset, input.getInt());
                    offset += 4;
                }
            }
            case BFLOAT16, FLOAT16 -> {
                while (input.hasRemaining()) {
                    nativeBytes.set(NATIVE_SHORT, offset, input.getShort());
                    offset += 2;
                }
            }
            case BOOL -> {
                while (input.hasRemaining()) nativeBytes.set(ValueLayout.JAVA_BYTE, offset++, input.get());
            }
        }
        return tensor;
    }

    private static final class Group {
        final ModelArtifactCodec.State state;
        final ByteBuffer bytes;
        boolean parameter;
        Tensor restored;

        Group(ModelArtifactCodec.State state, ByteBuffer bytes, boolean parameter) {
            this.state = state;
            this.bytes = bytes;
            this.parameter = parameter;
        }
    }
}
