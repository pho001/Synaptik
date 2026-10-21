package io.github.pho001.synaptik.text;

import io.github.pho001.synaptik.data.LongSequenceBatcher;
import io.github.pho001.synaptik.data.PaddedSequenceBatch;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Immutable, safely shareable text encoding configuration that delegates all padding to Data.
 *
 * <p>The fixed tokenizer defines ID meaning; one configured {@link LongSequenceBatcher} owns
 * numeric allocation, valid counts and padding. Each call produces fresh heap-backed INT64
 * values. Output metadata is immutable, not the Tensor's contents/storage association; ordinary
 * Model access obligations remain. No native owner, execution or close protocol is introduced.
 * PAD remains an ordinary numeric ID, not an instruction to mask or zero an Embedding row.</p>
 */
public final class TextSequenceBatcher {
    private final WhitespaceTokenizer tokenizer;
    private final LongSequenceBatcher batcher;

    /**
     * Retains complete immutable encoding and numeric batching configurations.
     * @param tokenizer non-null frozen tokenizer
     * @param batcher non-null immutable Data batcher configured with the tokenizer's padding ID
     */
    private TextSequenceBatcher(WhitespaceTokenizer tokenizer, LongSequenceBatcher batcher) {
        this.tokenizer = tokenizer;
        this.batcher = batcher;
    }

    /**
     * Starts explicit side/axis configuration for one fixed tokenizer.
     * @param tokenizer non-null frozen tokenizer retained exactly
     * @return a fresh non-null builder with neither side nor axis order selected
     * @throws NullPointerException if tokenizer is null, with message tokenizer
     */
    public static Builder builder(WhitespaceTokenizer tokenizer) {
        return new Builder(Objects.requireNonNull(tokenizer, "tokenizer"));
    }

    /**
     * Encodes each row once in sequential order, then delegates numeric batching exactly once.
     *
     * <p>Empty outer input yields B=0,T=0; all-empty encodings yield T=0. No input list or text
     * is retained. Data supplies fresh mutable heap values and immutable host counts; these
     * counts are not runtime graph inputs. LEFT does not change scans' right-valid-prefix rule.</p>
     *
     * @param texts non-null ordered list of non-null strings; coordinate list mutation throughout
     *     this synchronous call; no transactional snapshot is promised
     * @return a fresh non-null padded batch using the configured side/order and reserved PAD ID
     * @throws NullPointerException if texts is null (texts), or a row is null (texts row and index)
     * @throws IllegalArgumentException if Data's checked geometry exceeds its allocation limits
     * @throws IllegalStateException if Model tensor identifiers are exhausted
     * @throws OutOfMemoryError if encoding or delegated heap allocation fails
     */
    public PaddedSequenceBatch batch(List<String> texts) {
        Objects.requireNonNull(texts, "texts");
        List<long[]> encoded = new ArrayList<>();
        int row = 0;
        for (String text : texts) {
            encoded.add(tokenizer.encode(Objects.requireNonNull(text, "texts row " + row)));
            row++;
        }
        return batcher.batch(encoded);
    }

    /**
     * Reusable non-thread-safe configuration for one fixed tokenizer. Side and axis order have
     * no defaults; setters return this builder and the last selection wins. Each successful
     * build freezes an independent configuration, unaffected by subsequent builder use.
     */
    public static final class Builder {
        private final WhitespaceTokenizer tokenizer;
        private final LongSequenceBatcher.Builder batcher;

        /**
         * Fixes ID meaning and Data's padding value without selecting side or axis order.
         * @param tokenizer non-null factory-checked frozen tokenizer
         */
        private Builder(WhitespaceTokenizer tokenizer) {
            this.tokenizer = tokenizer;
            batcher = LongSequenceBatcher.builder().paddingValue(tokenizer.paddingId());
        }

        /**
         * Selects padding before each valid suffix.
         * @return this exact builder, replacing the previous side
         */
        public Builder leftPadded() {
            batcher.leftPadded();
            return this;
        }

        /**
         * Selects padding after each valid prefix.
         * @return this exact builder, replacing the previous side
         */
        public Builder rightPadded() {
            batcher.rightPadded();
            return this;
        }

        /**
         * Selects output shape {@code [T,B]}.
         * @return this exact builder, replacing the previous axis order
         */
        public Builder timeMajor() {
            batcher.timeMajor();
            return this;
        }

        /**
         * Selects output shape {@code [B,T]}.
         * @return this exact builder, replacing the previous axis order
         */
        public Builder batchMajor() {
            batcher.batchMajor();
            return this;
        }

        /**
         * Freezes the current side/order through Data's configuration validator.
         * @return a fresh non-null immutable configuration retaining this tokenizer
         * @throws IllegalStateException if side or axis order has not been selected
         */
        public TextSequenceBatcher build() {
            return new TextSequenceBatcher(tokenizer, batcher.build());
        }
    }
}
