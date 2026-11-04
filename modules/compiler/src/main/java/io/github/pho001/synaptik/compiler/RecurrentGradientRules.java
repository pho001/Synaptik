package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentDirection;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentScanKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.OptionalInt;

/** Fixed-cell BPTT through the ordinary formula owners and the shared length-aware Scan rule. */
final class RecurrentGradientRules {
    private RecurrentGradientRules() {}

    /**
     * Builds one closed transition recipe, then differentiates it jointly for the actual seeds.
     * No time unrolling or independent cell derivative exists. Positional identity views keep tied
     * actual roles separate until the enclosing contribution owner performs its ordered merge.
     * BFLOAT16, FLOAT16 and OFP8 storage have F32 projections/gates, explicit state narrowing,
     * and a narrowed LSTM cell read before hidden publication, exactly as the fixed forward sites.
     * @param producer selected fixed RNN, GRU or LSTM occurrence
     * @param seeds actual cotangents aligned with the occurrence's canonical output ports
     * @param selected input-position mask of requested differentiable roles
     * @param constants enclosing request's scalar-splat and activity owner
     * @return input-position-aligned contributions, with null in unselected positions
     */
    static Tensor[] apply(TensorProducer producer, List<Tensor> seeds, boolean[] selected,
            FirstOrderAutograd.DerivativeConstants constants) {
        var kind = (RecurrentScanKind) producer.operation().kind();
        boolean lstm = kind == RecurrentScanKind.LSTM;
        int states = lstm ? 2 : 1;
        int weight = lstm ? 4 : 3;
        DataType storage = producer.inputs().getFirst().descriptor().dataType();
        DataType work = storage == DataType.FLOAT64 ? DataType.FLOAT64 : DataType.FLOAT32;
        var actuals = new ArrayList<Tensor>(producer.inputs().size());
        for (int i = 0; i < producer.inputs().size(); i++) {
            Tensor input = producer.inputs().get(i);
            actuals.add(i == 1 ? input : input.reshape(input.descriptor().shape()));
        }
        var carries = new ArrayList<Tensor>(states);
        var initial = new ArrayList<Tensor>(states);
        for (int i = 0; i < states; i++) {
            initial.add(actuals.get(2 + i));
            carries.add(formal(initial.get(i).descriptor()));
        }
        Shape inputShape = actuals.getFirst().descriptor().shape();
        Tensor step = formal(new TensorDescriptor(storage,
                Shape.ofDimensions(inputShape.dimension(1), inputShape.dimension(2)),
                Optional.empty(), actuals.getFirst().descriptor().requiresGrad()));
        var invariantActuals = new ArrayList<Tensor>();
        var invariants = new ArrayList<Tensor>();
        for (int i = weight; i < actuals.size(); i++) {
            invariantActuals.add(actuals.get(i));
            invariants.add(formal(actuals.get(i).descriptor()));
        }
        Tensor previous = working(carries.getFirst(), work);
        Tensor inputProjection = working(step, work).linear(working(invariants.get(0), work));
        Tensor hiddenProjection = previous.linear(working(invariants.get(1), work));
        if (invariants.size() == 3) inputProjection = inputProjection.add(working(invariants.get(2), work));
        long hidden = ((io.github.pho001.synaptik.model.shape.StaticDimension)
                carries.getFirst().descriptor().shape().dimension(1)).size();
        Tensor nextHidden;
        var next = new ArrayList<Tensor>(states);
        if (kind == RecurrentScanKind.RNN_TANH) {
            nextHidden = working(inputProjection.add(hiddenProjection).tanh(), storage);
        } else if (kind == RecurrentScanKind.GRU_RESET_AFTER) {
            Tensor reset = gate(inputProjection, hiddenProjection, 0, hidden).sigmoid();
            Tensor update = gate(inputProjection, hiddenProjection, 1, hidden).sigmoid();
            Tensor candidate = part(inputProjection, 2, hidden)
                    .add(reset.mul(part(hiddenProjection, 2, hidden))).tanh();
            nextHidden = working(candidate.add(update.mul(previous.sub(candidate))), storage);
        } else {
            Tensor input = gate(inputProjection, hiddenProjection, 0, hidden).sigmoid();
            Tensor forget = gate(inputProjection, hiddenProjection, 1, hidden).sigmoid();
            Tensor candidate = gate(inputProjection, hiddenProjection, 2, hidden).tanh();
            Tensor output = gate(inputProjection, hiddenProjection, 3, hidden).sigmoid();
            Tensor cell = working(forget.mul(working(carries.get(1), work))
                    .add(input.mul(candidate)), storage);
            nextHidden = working(output.mul(working(cell, work).tanh()), storage);
            next.add(cell);
        }
        next.addFirst(nextHidden);
        var body = new ScanBody(carries, List.of(step), invariants, next, List.of(nextHidden));
        var scan = Scan.apply(initial, List.of(actuals.getFirst()), invariantActuals, actuals.get(1),
                states, List.of(OptionalInt.of(0)), body,
                producer.operation().attrs() == RecurrentDirection.REVERSE);
        var ports = new ArrayList<Tensor>();
        ports.add(scan.outputs().getFirst());
        ports.addAll(scan.finalCarry());
        var roots = new ArrayList<Tensor>();
        var rootSeeds = new ArrayList<Tensor>();
        for (int i = 0; i < seeds.size(); i++) if (seeds.get(i) != null) {
            roots.add(ports.get(i));
            rootSeeds.add(seeds.get(i));
        }
        var targets = new ArrayList<Tensor>();
        for (int i = 0; i < selected.length; i++) if (selected[i]) targets.add(actuals.get(i));
        var plan = AutogradPreflight.preflightRecipe(roots, targets);
        List<Tensor> gradients = FirstOrderAutograd.expandRecipe(plan, rootSeeds, constants);
        Tensor[] result = new Tensor[selected.length];
        for (int i = 0, target = 0; i < selected.length; i++) if (selected[i]) result[i] = gradients.get(target++);
        return result;
    }

    private static Tensor formal(TensorDescriptor descriptor) {
        return TensorFactory.create(new TensorDescriptor(descriptor.dataType(), descriptor.shape(),
                Optional.of(LayoutDescriptor.contiguous(descriptor.shape())), descriptor.requiresGrad()));
    }

    private static Tensor working(Tensor value, DataType type) {
        return value.descriptor().dataType() == type ? value : value.cast(type);
    }

    private static Tensor gate(Tensor input, Tensor hidden, int gate, long width) {
        return part(input, gate, width).add(part(hidden, gate, width));
    }

    private static Tensor part(Tensor value, int gate, long width) {
        return value.slice(new long[] {gate * width}, new long[] {(gate + 1) * width},
                new int[] {1}, new long[] {1});
    }
}
