package io.github.pho001.synaptik.checkpoint.model;

import io.github.pho001.synaptik.checkpoint.artifact.ArtifactEnvelope;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactLimits;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.nn.module.Module;
import io.github.pho001.synaptik.nn.module.StateEntry;
import java.nio.ByteBuffer;
import java.util.AbstractList;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Objects;

/**
 * Synchronous NN-to-Engine handoff for a complete bound state snapshot. Caller quiescence covers
 * NN bindings and selected leaf storage throughout capture. No partial candidate is returned on
 * failure, and this helper never installs state or retains a Tensor beyond the call.
 */
final class ModelArtifactCapture {
    private ModelArtifactCapture() {
    }

    /**
     * Preflights the complete artifact without entry-proportional Checkpoint storage. Only then
     * does it index aliases for one ordered bounded Engine compute and copy detached canonical
     * values into an immutable candidate. The limit bounds encoded canonical payload bytes, not
     * Engine workspace, source values, defensive copies, or peak heap.
     *
     * @param module complete bound NN tree, quiescent for this synchronous call
     * @param engine open public Engine facade; unused for empty state
     * @param configDigest exact 32-byte SHA-256 of caller-defined versioned configuration
     * @param inputDigest optional exact 32-byte SHA-256 of caller-defined input artifact
     * @param limits artifact limits including aggregate schema and every aliased state payload
     * @return detached immutable model candidate, never null
     * @throws NullPointerException if a required argument or digest is null
     * @throws IllegalArgumentException if metadata or any preflight bound is invalid
     * @throws ArithmeticException if checked state or artifact size arithmetic overflows
     * @throws IllegalStateException if NN export is incomplete, Engine fails its lifecycle, or
     *     returned materialization does not match the preflighted descriptors
     * @throws RuntimeException if Engine computation or copying reports another unchecked failure
     */
    static ModelArtifactCandidate capture(Module module, Engine engine, byte[] configDigest,
                                          byte[] inputDigest, ArtifactLimits limits) {
        Objects.requireNonNull(module, "module");
        Objects.requireNonNull(engine, "engine");
        Objects.requireNonNull(limits, "limits");
        // NN owns this shallow snapshot and its completeness check. It is intentionally called once.
        List<StateEntry> entries = module.stateDictionary().entries();
        if (entries.size() > 100_000_000L || entries.size() + 1L > limits.maxEntries())
            throw new IllegalArgumentException("Model entry count exceeds limits");
        List<ModelArtifactCodec.State> descriptors = new AbstractList<>() {
            @Override public ModelArtifactCodec.State get(int index) {
                StateEntry entry = entries.get(index);
                var descriptor = entry.value().descriptor();
                long count = descriptor.shape().knownElementCount().orElseThrow(
                        () -> new IllegalArgumentException("Dynamic state shape"));
                long bytes = Math.multiplyExact(count, descriptor.dataType().byteWidth());
                if (bytes > Integer.MAX_VALUE - 8 || bytes > limits.maxPayloadBytes())
                    throw new IllegalArgumentException("State payload exceeds limits");
                return new ModelArtifactCodec.State(entry.path(), entry.kind(),
                        descriptor.dataType(), descriptor.shape(), (int) bytes);
            }

            @Override public int size() { return entries.size(); }
        };
        ModelArtifactCodec.Metadata metadata = ModelArtifactCodec.encode(
                descriptors, configDigest, inputDigest, limits);
        List<ModelArtifactCodec.State> states = metadata.states();
        var unique = new ArrayList<Tensor>();
        var positions = new IdentityHashMap<Tensor, Integer>();
        var ordinals = new int[entries.size()];
        long uniqueBytes = 0;
        for (int i = 0; i < entries.size(); i++) {
            Tensor tensor = entries.get(i).value();
            Integer position = positions.get(tensor);
            if (position == null) {
                position = unique.size();
                unique.add(tensor);
                positions.put(tensor, position);
                uniqueBytes = Math.addExact(uniqueBytes, states.get(i).length());
            }
            ordinals[i] = position;
        }
        List<HostTensorValue> values = unique.isEmpty() ? List.of()
                : engine.compute(unique, uniqueBytes);
        if (values.size() != unique.size()) throw new IllegalStateException("Engine result count mismatch");
        // Verify the complete result before constructing any artifact payload entry.
        for (int i = 0; i < entries.size(); i++) {
            HostTensorValue value = Objects.requireNonNull(values.get(ordinals[i]), "Engine value");
            ModelArtifactCodec.State state = states.get(i);
            if (value.dataType() != state.type() || !value.shape().equals(state.shape())
                    || value.byteSize() != state.length())
                throw new IllegalStateException("Engine value differs from state descriptor");
        }
        var payloads = new ArrayList<ArtifactEnvelope.Entry>(entries.size() + 1);
        payloads.add(new ArtifactEnvelope.Entry("schema", metadata.bytes()));
        for (int i = 0; i < entries.size(); i++) {
            ByteBuffer source = values.get(ordinals[i]).bytes();
            byte[] bytes = new byte[states.get(i).length()];
            source.get(bytes);
            payloads.add(new ArtifactEnvelope.Entry(ModelArtifactCodec.ordinal(i), bytes));
        }
        return new ModelArtifactCandidate(metadata, new ArtifactEnvelope("model.v1", payloads));
    }
}
