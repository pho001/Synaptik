package io.github.pho001.synaptik.backend.cpu.internal.ir;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Fixed ordinary softmax derivative composition with no intermediate tensor carriers.
 * The exact SUM consumes represented products (softmax) or cotangents (log-softmax).
 * Its rounded result feeds the existing typed primitive sites in a second full-domain pass.
 * This physical form does not define a derivative or recompute the stored forward output.
 * The admitted full domain is static rank one or two with its last axis reduced, homogeneous
 * FLOAT32 or FLOAT64, canonical saved/output mappings and a canonical or scalar/column-broadcast
 * cotangent. Bindings are immutable and int-bounded; actual carriers, nonoverlap and invocation
 * bases are checked during binding. Preparation separately checks the aggregate geometry and
 * exact-state declaration against this recipe before finalization.
 *
 * @param logarithmic whether the captured pattern is log-softmax's EXP/MUL/SUB
 * @param aggregate unchanged kept single-trailing-axis exact SUM
 * @param boundaries nonnull unique full-domain reads followed by the injective full-domain
 *     write; copied, with nonnegative offsets and byte spans bounded by Integer.MAX_VALUE
 * @param savedInput position of the exact stored forward output
 * @param gradientInput position of the incoming cotangent; may equal savedInput
 */
public record CpuSoftmaxBackwardIr(boolean logarithmic, CpuAggregateIr aggregate,
        List<CpuAccessPlan.Binding> boundaries, int savedInput, int gradientInput)
        implements CpuPortableKernelIr {
    /**
     * Snapshots the complete baked geometry and rejects unsupported physical declarations.
     * @param logarithmic true for log-softmax, false for softmax
     * @param aggregate nonnull original exact SUM with domain-sized limbs
     * @param boundaries nonnull ordered physical bindings, copied
     * @param savedInput read position holding the saved output
     * @param gradientInput read position holding the cotangent
     * @throws NullPointerException if a reference or binding is null
     * @throws IllegalArgumentException if domains, roles, types or integer bounds disagree
     * @throws ArithmeticException if checked span arithmetic overflows
     */
    public CpuSoftmaxBackwardIr {
        Objects.requireNonNull(aggregate, "aggregate");
        boundaries = List.copyOf(boundaries);
        int reads = boundaries.size()-1;
        if (reads < 1 || reads > 2 || savedInput < 0 || savedInput >= reads
                || gradientInput < 0 || gradientInput >= reads
                || reads == 2 && savedInput == gradientInput
                || aggregate.kind() != CpuAggregateIr.Kind.SUM
                || aggregate.form() != CpuAggregateIr.Form.SINGLE_AXIS || !aggregate.keepDimensions()
                || aggregate.dataType() != DataType.FLOAT32 && aggregate.dataType() != DataType.FLOAT64)
            throw new IllegalArgumentException("unsupported softmax backward physical form");
        var shape = boundaries.getLast().extents();
        int rank = shape.size();
        if (rank < 1 || rank > 2 || !java.util.Arrays.equals(aggregate.selectedAxes(),new int[]{rank-1})
                || aggregate.domainCount() != shape.getLast()
                || aggregate.inputAccess().iterationRank() != rank)
            throw new IllegalArgumentException("softmax backward requires a trailing slice");
        for (int i = 0; i < boundaries.size(); i++) {
            var b = boundaries.get(i);
            boolean write = i == reads;
            if (!b.extents().equals(shape) || b.baseElementOffset() < 0
                    || Math.multiplyExact(b.baseElementOffset(),aggregate.dataType().byteWidth()) > Integer.MAX_VALUE
                    || b.plan().accessKind() != (write ? CpuAccessPlan.AccessKind.WRITE : CpuAccessPlan.AccessKind.READ)
                    || b.elementCount() > Integer.MAX_VALUE
                    || Math.multiplyExact(b.referencedElementSpan(),aggregate.dataType().byteWidth()) > Integer.MAX_VALUE
                    || b.extents().stream().anyMatch(e -> e > Integer.MAX_VALUE)
                    || b.effectiveStrides().stream().anyMatch(s -> s < 0 || s > Integer.MAX_VALUE))
                throw new IllegalArgumentException("softmax backward binding is not int-bounded");
            long row = rank == 1 ? 0 : b.effectiveStrides().getFirst();
            long feature = b.effectiveStrides().getLast();
            boolean canonical = feature == 1 && (rank == 1 || row == shape.getLast());
            boolean broadcast = feature == 0 && (rank == 1 || row == 0 || row == 1);
            if (!canonical && (write || i == savedInput || !broadcast))
                throw new IllegalArgumentException("softmax backward requires canonical saved/write and proved cotangent broadcast");
        }
    }

    /**
     * Reuses the existing typed instruction vocabulary at every captured primitive site.
     * Ordinals zero/one/two are saved output, cotangent and rounded SUM, respectively.
     * The softmax MUL in pass one rounds before exact accumulation; pass two performs SUB then
     * MUL. Log-softmax pass two performs EXP, MUL then SUB after summing stored cotangents.
     * @return a new immutable typed site description; the first softmax site belongs only
     *     to pass one, and the rounded SUM is a local input rather than a tensor carrier
     */
    public CpuKernelIr sites() {
        var values = new ArrayList<CpuKernelIr.Value>();
        for (int i = 0; i < 6; i++) values.add(new CpuKernelIr.Value(i,aggregate.dataType(),
                i < 3 ? CpuKernelIr.Value.Kind.INPUT : i == 5 ? CpuKernelIr.Value.Kind.OUTPUT
                    : CpuKernelIr.Value.Kind.VIRTUAL,
                i == 0 ? boundaries.get(savedInput).plan() : i == 1 ? boundaries.get(gradientInput).plan()
                    : i == 2 ? aggregate.outputAccess() : boundaries.getLast().plan()));
        var instructions = logarithmic ? List.of(
                new CpuKernelIr.Instruction(CpuPointwiseOpcode.EXP,List.of(0),3),
                new CpuKernelIr.Instruction(CpuPointwiseOpcode.MUL,List.of(3,2),4),
                new CpuKernelIr.Instruction(CpuPointwiseOpcode.SUB,List.of(1,4),5)) : List.of(
                new CpuKernelIr.Instruction(CpuPointwiseOpcode.MUL,List.of(1,0),3),
                new CpuKernelIr.Instruction(CpuPointwiseOpcode.SUB,List.of(1,2),4),
                new CpuKernelIr.Instruction(CpuPointwiseOpcode.MUL,List.of(0,4),5));
        return new CpuKernelIr(values,instructions,new CpuKernelIr.Loop("start","end"),List.of(new CpuKernelIr.Store(5,0)));
    }

    /**
     * Encodes only this new family's typed topology and structural access facts.
     * @return instruction-free physical envelope, without graph IDs, extents or run bindings
     */
    public CpuKernelIr encodedKernelIr() {
        var values = new ArrayList<CpuKernelIr.Value>();
        for (int i = 0; i < boundaries.size(); i++) values.add(new CpuKernelIr.Value(i,aggregate.dataType(),
                i == boundaries.size()-1 ? CpuKernelIr.Value.Kind.OUTPUT : CpuKernelIr.Value.Kind.INPUT,
                boundaries.get(i).plan()));
        return new CpuKernelIr(values,List.of(),new CpuKernelIr.Loop("start","end"),
                List.of(new CpuKernelIr.Store(values.size()-1,0)),"ordinary-softmax-backward-v1:"
                    +logarithmic+":"+aggregate.structuralKey()+":"+sites().structuralKey()
                    +":saved="+savedInput+":g="+gradientInput);
    }

    /**
     * Identifies every baked extent, offset, stride, span and exact-state fact.
     * @return nonnull deterministic payload, excluding slots, objects and invocation bases
     */
    public String bakedSpecializationIdentity() {
        return structuralKey()+":limbs="+aggregate.stateLimbCount()+":scratch="+aggregate.scratchSliceBytes()
                +":domain="+aggregate.domainCount()+":access="+boundaries.stream().map(b ->
                    b.extents()+":"+b.baseElementOffset()+":"+b.effectiveStrides()+":"+b.referencedElementSpan()).toList();
    }

    /** Returns this family's structural identity.
     * @return nonnull lowercase hexadecimal fingerprint */
    @Override public String structuralKey() { return encodedKernelIr().structuralKey(); }
}
