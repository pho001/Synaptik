package io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn;

import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.List;

/**
 * Versioned, length-delimited native assignment identity. Hashes typed fields directly with one
 * reusable eight-byte buffer, never materializing descriptor strings or an aggregate payload.
 * The opaque result is always 87 ASCII bytes, so 6,560 alternatives contribute at most 570,720
 * fingerprint bytes. Source encoding is independently capped at one MiB before each write.
 */
final class CpuOneDnnCandidateIdentity {
    private static final int MAX_SOURCE_BYTES = 1_048_576;
    private static final String PREFIX = "onednn-finite-1:sha256:";
    private final MessageDigest digest;
    private final byte[] word = new byte[Long.BYTES];
    private int remaining = MAX_SOURCE_BYTES;

    private CpuOneDnnCandidateIdentity() {
        try {
            digest = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException impossible) {
            throw new AssertionError(impossible);
        }
    }

    /**
     * Hashes the integration policy and exact native declarations without retaining their objects.
     *
     * @param policy non-null configured AUTO/PRIMITIVE/GRAPH policy
     * @param route non-null immutable native assignment
     * @return fixed 87-character ASCII versioned SHA-256 identity
     * @throws IllegalArgumentException if the canonical source encoding exceeds one MiB
     */
    static String fingerprint(CpuOneDnnRoutePlan.Api policy, CpuOneDnnRoutePlan route) {
        var identity = new CpuOneDnnCandidateIdentity();
        identity.text(PREFIX);
        identity.text(policy.name());
        identity.text(route.sessionIdentity());
        identity.number(route.units().size());
        for (var unit : route.units()) {
            identity.number(unit.unitPosition());
            var projection = unit.projection();
            identity.spec(projection.spec());
            identity.longs(projection.byteOffsets());
            identity.ints(projection.inputBoundaryPositions());
            identity.number(projection.outputBoundaryPosition());
            identity.text(projection.logicalOutputType().name());
            identity.text(unit.api().name());
            identity.number(unit.scratchpadBytes());
            identity.text(unit.implementation());
            identity.number(unit.inputCopies().size());
            for (var copy : unit.inputCopies()) {
                identity.ints(copy.inputPositions());
                identity.workspace(copy.workspace());
                identity.number(copy.refresh() ? 1 : 0);
            }
            identity.workspace(unit.invocationWorkspace());
            identity.number(unit.outputStagingWorkspace().isPresent() ? 1 : 0);
            unit.outputStagingWorkspace().ifPresent(identity::workspace);
        }
        return PREFIX + HexFormat.of().formatHex(identity.digest.digest());
    }

    private void spec(CpuOneDnnKernelSpec spec) {
        // Explicit kind tags are independent of class names and record-generated toString formats.
        switch (spec) {
            case CpuOneDnnKernelSpec.Contraction contraction -> {
                number(1);
                text(contraction.operation().name());
                tensors(contraction.inputs());
                tensor(contraction.output());
                number(contraction.groups());
                longs(contraction.strides());
                longs(contraction.dilations());
                longs(contraction.paddingLeft());
                longs(contraction.paddingRight());
                text(contraction.bias().name());
                text(contraction.activation().name());
            }
            case CpuOneDnnKernelSpec.Unary unary -> {
                number(2);
                text(unary.algorithm().name());
                tensor(unary.input());
                tensor(unary.output());
            }
            case CpuOneDnnKernelSpec.Binary binary -> {
                number(3);
                text(binary.algorithm().name());
                tensor(binary.left());
                tensor(binary.right());
                tensor(binary.output());
            }
            case CpuOneDnnKernelSpec.Reduction reduction -> {
                number(4);
                text(reduction.algorithm().name());
                tensor(reduction.input());
                tensor(reduction.output());
            }
            case CpuOneDnnKernelSpec.Reorder reorder -> {
                number(5);
                tensor(reorder.input());
                tensor(reorder.output());
            }
            case CpuOneDnnKernelSpec.Pooling pooling -> {
                number(6);
                tensor(pooling.input());
                tensor(pooling.output());
                longs(pooling.kernel());
                longs(pooling.strides());
                longs(pooling.dilations());
                longs(pooling.paddingLeft());
                longs(pooling.paddingRight());
            }
            case CpuOneDnnKernelSpec.Pipeline pipeline -> {
                number(7);
                tensors(pipeline.inputs());
                tensor(pipeline.output());
                tensors(pipeline.intermediates());
                number(pipeline.constants().size());
                for (var constant : pipeline.constants()) {
                    tensor(constant.tensor());
                    ints(constant.rawWords());
                }
                number(pipeline.steps().size());
                for (var step : pipeline.steps()) {
                    spec(step.spec());
                    ints(step.inputSlots());
                    number(step.outputSlot());
                }
            }
        }
    }

    private void workspace(PreparationResourceRequirement.Workspace workspace) {
        number(workspace.requirementId());
        number(workspace.byteSize());
        number(workspace.byteAlignment());
    }

    private void tensors(List<CpuOneDnnKernelSpec.Tensor> tensors) {
        number(tensors.size());
        for (var tensor : tensors) tensor(tensor);
    }

    private void tensor(CpuOneDnnKernelSpec.Tensor tensor) {
        text(tensor.dataType().name());
        longs(tensor.dimensions());
        longs(tensor.strides());
    }

    private void longs(List<Long> values) {
        number(values.size());
        for (long value : values) number(value);
    }

    private void ints(List<Integer> values) {
        number(values.size());
        for (int value : values) number(value);
    }

    private void text(String text) {
        // Encode Java UTF-16 code units explicitly, including any unpaired surrogate, without an
        // intermediate UTF-8 byte array. Length and fixed-width code units make this unambiguous.
        if (text.length() > (remaining - Long.BYTES) / Long.BYTES) tooLarge();
        number(text.length());
        for (int position = 0; position < text.length(); position++) number(text.charAt(position));
    }

    private void number(long value) {
        if (remaining < Long.BYTES) tooLarge();
        remaining -= Long.BYTES;
        for (int position = Long.BYTES - 1; position >= 0; position--) {
            word[position] = (byte) value;
            value >>>= Byte.SIZE;
        }
        digest.update(word);
    }

    private static void tooLarge() {
        throw new IllegalArgumentException("oneDNN candidate identity source exceeds one MiB");
    }
}
