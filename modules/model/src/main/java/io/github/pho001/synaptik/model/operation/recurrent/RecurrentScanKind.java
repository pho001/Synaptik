package io.github.pho001.synaptik.model.operation.recurrent;

import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Identifies the closed fixed recurrent-scan semantic family.
 *
 * <p>Each kind describes one flat, identity-distinct, multi-output Model occurrence with a
 * {@link RecurrentDirection} attribute. The structural signatures admit only the explicit
 * bias-free and biased input counts. They do not describe graph regions, execution support,
 * backend capabilities, or gradient rules.</p>
 *
 * <p>For row vectors {@code x} and prior hidden state {@code h}, input weight {@code W}, hidden
 * weight {@code U}, and optional input-side bias {@code b}, RNN uses
 * {@code h' = tanh((x @ W^T + b?) + (h @ U^T))}. GRU packs reset {@code r}, update {@code z},
 * and candidate {@code n} gates and fixes:</p>
 * <pre>{@code
 * P_x = x @ W^T + b?; P_h = h @ U^T
 * r = sigmoid(P_x[r] + P_h[r])
 * z = sigmoid(P_x[z] + P_h[z])
 * n = tanh(P_x[n] + r * P_h[n])
 * h' = n + z * (h - n)
 * }</pre>
 * <p>LSTM additionally carries prior cell state {@code c}, packs input {@code i}, forget
 * {@code f}, candidate {@code g}, and output {@code o} gates, and fixes:</p>
 * <pre>{@code
 * P_x = x @ W^T + b?; P_h = h @ U^T
 * i = sigmoid(P_x[i] + P_h[i]); f = sigmoid(P_x[f] + P_h[f])
 * g = tanh(P_x[g] + P_h[g]); o = sigmoid(P_x[o] + P_h[o])
 * c' = f * c + i * g; h' = o * tanh(c')
 * }</pre>
 * <p>The equations define semantic association and packing, not decomposition, accumulator
 * widening, fusion, or another execution algorithm. There is no recurrent-side bias.</p>
 */
public enum RecurrentScanKind implements OperationKind {
    /**
     * Tanh recurrence with packed gate count one and outputs
     * {@code [outputs, finalHidden]}.
     */
    RNN_TANH(5, 6, 2),

    /**
     * Reset-after GRU recurrence with reset/update/candidate packing and outputs
     * {@code [outputs, finalHidden]}.
     */
    GRU_RESET_AFTER(5, 6, 2),

    /**
     * LSTM recurrence with input/forget/candidate/output packing and outputs
     * {@code [outputs, finalHidden, finalCell]}.
     */
    LSTM(6, 7, 3);

    private final List<OperationSignature> signatures;

    RecurrentScanKind(int minimumInputs, int maximumInputs, int outputCount) {
        signatures = List.of(OperationSignature.inputRange(
                RecurrentDirection.class, minimumInputs, maximumInputs, outputCount));
    }

    /**
     * Returns this kind's sole immutable structural signature.
     *
     * @return the stable immutable singleton list requiring exact
     *     {@link RecurrentDirection} attributes and this kind's fixed input/output cardinalities
     */
    @Override
    public List<OperationSignature> signatures() {
        return signatures;
    }
}
