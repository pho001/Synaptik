package io.github.pho001.synaptik.backend.cpu.internal.ir;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Immutable ordinary exact SUM/MEAN composition, not a second arithmetic vocabulary.
 * Producer sites execute in the contribution domain; epilogue sites execute only after the
 * exact result has rounded. Bindings are positional and retain every baked address fact.
 * No private intermediate has a carrier, and callbacks used to emit sites are generation-only.
 *
 * @param aggregate existing exact static SUM/MEAN meaning and domain-sized state
 * @param producer optional single-use same-type pointwise chain
 * @param producerInputs producer input ordinals mapped to physical boundary positions
 * @param producerAccesses normalized producer reads followed by its removed output
 * @param aggregateInput physical read position when producer is absent, otherwise minus one
 * @param epilogue optional ADD/RELU or ADD then RELU chain
 * @param epilogueInputs epilogue inputs mapped to boundaries, with minus one for rounded aggregate
 * @param epilogueAccesses normalized epilogue reads followed by final output
 * @param boundaries unique physical reads followed by the sole injective final write
 * @param inputExtents static aggregate contribution shape, copied
 */
public record CpuAggregateFusionIr(CpuAggregateIr aggregate, Optional<CpuKernelIr> producer,
        List<Integer> producerInputs, List<CpuAccessPlan.Binding> producerAccesses,
        int aggregateInput, Optional<CpuKernelIr> epilogue, List<Integer> epilogueInputs,
        List<CpuAccessPlan.Binding> epilogueAccesses, List<CpuAccessPlan.Binding> boundaries,
        List<Long> inputExtents) implements CpuPortableKernelIr {
    /**
     * Snapshots and validates the typed composition and complete positional read bindings.
     * @param aggregate existing exact SUM/MEAN IR, nonnull
     * @param producer optional producer chain, nonnull
     * @param producerInputs immutable-after-copy producer boundary mapping
     * @param producerAccesses immutable-after-copy producer access geometry
     * @param aggregateInput direct input boundary, or minus one with producer
     * @param epilogue optional completed-result tail, nonnull
     * @param epilogueInputs immutable-after-copy tail mapping; minus one names aggregate local
     * @param epilogueAccesses immutable-after-copy tail access geometry
     * @param boundaries immutable-after-copy unique reads and final write
     * @param inputExtents immutable-after-copy nonnegative contribution extents
     * @throws NullPointerException if a reference or collection element is null
     * @throws IllegalArgumentException if type, topology, domain or binding roles disagree
     */
    public CpuAggregateFusionIr {
        Objects.requireNonNull(aggregate, "aggregate");
        Objects.requireNonNull(producer, "producer"); Objects.requireNonNull(epilogue, "epilogue");
        producerInputs = List.copyOf(producerInputs); producerAccesses = List.copyOf(producerAccesses);
        epilogueInputs = List.copyOf(epilogueInputs); epilogueAccesses = List.copyOf(epilogueAccesses);
        boundaries = List.copyOf(boundaries); inputExtents = List.copyOf(inputExtents);
        if (aggregate.kind() != CpuAggregateIr.Kind.SUM && aggregate.kind() != CpuAggregateIr.Kind.MEAN
                || aggregate.dataType() != DataType.FLOAT32 && aggregate.dataType() != DataType.FLOAT64
                || aggregate.form() == CpuAggregateIr.Form.SUM_TO_SHAPE
                || producer.isEmpty() && epilogue.isEmpty() || boundaries.size() < 2
                || inputExtents.size() != aggregate.inputAccess().iterationRank()
                || inputExtents.stream().anyMatch(e -> e < 0)
                || producer.isPresent() != (aggregateInput == -1)
                || producer.isEmpty() && aggregateInput < 0
                || aggregateInput >= boundaries.size() - 1
                || 1+producer.map(p -> p.instructions().size()).orElse(0)
                    +epilogue.map(p -> p.instructions().size()).orElse(0) > 8
                || boundaries.getLast().plan().accessKind() != CpuAccessPlan.AccessKind.WRITE
                || boundaries.subList(0, boundaries.size()-1).stream().anyMatch(b ->
                    b.plan().accessKind() != CpuAccessPlan.AccessKind.READ))
            throw new IllegalArgumentException("ordinary aggregate fusion facts disagree");
        validateSites(producer, producerInputs, producerAccesses, boundaries.size(), aggregate.dataType(), false);
        validateSites(epilogue, epilogueInputs, epilogueAccesses, boundaries.size(), aggregate.dataType(), true);
        if (producer.isPresent() && !producerAccesses.getLast().extents().equals(inputExtents))
            throw new IllegalArgumentException("producer contribution domain disagrees");
        if (epilogue.isPresent() && !epilogueAccesses.getLast().equals(boundaries.getLast()))
            throw new IllegalArgumentException("epilogue output boundary disagrees");
    }

    /**
     * Checks one optional typed domain's complete positional binding, not its arithmetic.
     * @param sites immutable pointwise domain, nonnull
     * @param inputs physical positions in input-value ordinal order
     * @param accesses normalized reads followed by the one output
     * @param boundaries physical boundary count including final write
     * @param type homogeneous represented dtype
     * @param tail whether minus one names the rounded aggregate and only ADD/RELU are allowed
     * @throws IllegalArgumentException if optionality, cardinality, type or positions disagree
     */
    private static void validateSites(Optional<CpuKernelIr> sites, List<Integer> inputs,
            List<CpuAccessPlan.Binding> accesses, int boundaries, DataType type, boolean tail) {
        if (sites.isEmpty()) {
            if (!inputs.isEmpty() || !accesses.isEmpty()) throw new IllegalArgumentException("absent sites have bindings");
            return;
        }
        var ir = sites.orElseThrow();
        if (ir.stores().size() != 1 || inputs.size() + 1 != accesses.size()
                || inputs.size() != ir.values().stream().filter(v -> v.kind() == CpuKernelIr.Value.Kind.INPUT).count()
                || ir.values().stream().anyMatch(v -> v.dataType() != type)
                || ir.instructions().stream().anyMatch(i -> i.opcode() == CpuPointwiseOpcode.CAST)
                || inputs.stream().anyMatch(i -> i < (tail ? -1 : 0) || i >= boundaries-1)
                || tail && inputs.stream().filter(i -> i == -1).count() != 1)
            throw new IllegalArgumentException("pointwise site bindings disagree");
        if (tail) {
            var instructions = ir.instructions();
            if (instructions.isEmpty() || instructions.size() > 2
                    || instructions.getFirst().opcode() != CpuPointwiseOpcode.ADD
                        && instructions.getFirst().opcode() != CpuPointwiseOpcode.RELU
                    || instructions.size() == 2 && (instructions.getFirst().opcode() != CpuPointwiseOpcode.ADD
                        || instructions.getLast().opcode() != CpuPointwiseOpcode.RELU))
                throw new IllegalArgumentException("unsupported aggregate epilogue");
        }
    }

    /**
     * Encodes a new selective ordinary family without changing any existing family projection.
     * Typed sites, existing aggregate axis/domain/state exceptions, topology and structural
     * access roles participate. New extent/stride magnitudes belong only to the separate baked
     * specialization payload, preserving the canonical versus emitted-geometry separation.
     * @return new instruction-free physical-boundary envelope; never null
     */
    public CpuKernelIr encodedKernelIr() {
        var values = new ArrayList<CpuKernelIr.Value>();
        for (int i = 0; i < boundaries.size(); i++) values.add(new CpuKernelIr.Value(i,
                aggregate.dataType(), i+1 == boundaries.size() ? CpuKernelIr.Value.Kind.OUTPUT
                    : CpuKernelIr.Value.Kind.INPUT, boundaries.get(i).plan()));
        return new CpuKernelIr(values, List.of(), new CpuKernelIr.Loop("start", "end"),
                List.of(new CpuKernelIr.Store(values.size()-1, 0)), "ordinary-aggregate-fusion-v1:"
                    + aggregate.structuralKey() + ":producer=" + producer.map(CpuKernelIr::structuralKey)
                    + ":pi=" + producerInputs + ":direct=" + aggregateInput
                    + ":tail=" + epilogue.map(CpuKernelIr::structuralKey) + ":ti=" + epilogueInputs
                    );
    }

    /**
     * Binds only this family's baked static geometry in both specialization identity streams.
     * Extents, offsets, strides and spans determine direct cursor/address shape and int-width
     * proofs. Run bases, carrier instances, slots, subranges and mutable state are never included.
     * @return deterministic nonnull ordinary-fusion-v1 geometry payload
     */
    public String bakedSpecializationIdentity() {
        return structuralKey()+":shape="+inputExtents+":producer="+bakedAccesses(producerAccesses)
                +":tail="+bakedAccesses(epilogueAccesses)+":boundaries="+bakedAccesses(boundaries);
    }

    /**
     * Projects only static facts consulted by emission or its integer-address proof.
     * @param accesses immutable normalized domain accesses, nonnull
     * @return deterministic nonnull ordered static access geometry, without range cursor state
     */
    private static List<String> bakedAccesses(List<CpuAccessPlan.Binding> accesses) {
        return accesses.stream().map(a -> a.extents()+":"+a.baseElementOffset()+":"+a.effectiveStrides()
                +":span="+a.referencedElementSpan()).toList();
    }

    /** Returns the graph/slot/carrier-free selective identity.
     * @return nonnull lowercase hexadecimal fingerprint */
    @Override public String structuralKey() { return encodedKernelIr().structuralKey(); }
}
