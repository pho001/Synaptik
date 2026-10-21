package io.github.pho001.synaptik.text;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Frozen, safely shareable mapping from exact whitespace-delimited strings to numeric IDs.
 *
 * <p>PAD is always 0, UNK is always 1, and ordinary tokens receive IDs from 2 in supplied order.
 * No spelling denotes a reserved ID. Runs are separated only by
 * {@link Character#isWhitespace(int)} code points. Case, punctuation and non-normalized UTF-16,
 * including unpaired surrogates, remain exact. There is no training, decoding, normalization,
 * persistence, I/O or resource lifetime. Each encoding belongs exclusively to its caller.</p>
 */
public final class WhitespaceTokenizer {
    private final Map<String, Long> ids;

    /**
     * Freezes the factory's validated, ordered ID assignments without retaining caller data.
     * @param ids non-null private mapping of ordinary tokens to IDs starting at 2
     */
    private WhitespaceTokenizer(Map<String, Long> ids) {
        this.ids = Map.copyOf(ids);
    }

    /**
     * Copies ordinary tokens in sequential list order and freezes their ID assignments.
     *
     * @param ordinaryTokens non-null ordered list of distinct, nonempty, non-null strings with
     *     no whitespace code point; empty is valid; coordinate list mutation during this call
     * @return a non-null immutable tokenizer independent of subsequent list changes
     * @throws NullPointerException if the list is null (ordinaryTokens) or a token is null
     *     (ordinaryTokens row followed by its zero-based index)
     * @throws IllegalArgumentException if size plus two exceeds Integer.MAX_VALUE, or a token
     *     is empty, contains whitespace, or duplicates an earlier token
     * @throws OutOfMemoryError if vocabulary allocation fails
     */
    public static WhitespaceTokenizer of(List<String> ordinaryTokens) {
        Objects.requireNonNull(ordinaryTokens, "ordinaryTokens");
        if (ordinaryTokens.size() > Integer.MAX_VALUE - 2) {
            throw new IllegalArgumentException("Vocabulary size including PAD and UNK exceeds int range");
        }
        Map<String, Long> ids = new HashMap<>();
        int row = 0;
        for (String token : ordinaryTokens) {
            Objects.requireNonNull(token, "ordinaryTokens row " + row);
            if (token.isEmpty() || token.codePoints().anyMatch(Character::isWhitespace)) {
                throw new IllegalArgumentException("Ordinary token at row " + row
                        + " must be nonempty and contain no whitespace");
            }
            if (ids.putIfAbsent(token, (long) row + 2) != null) {
                throw new IllegalArgumentException("Duplicate ordinary token at row " + row);
            }
            row++;
        }
        return new WhitespaceTokenizer(ids);
    }

    /**
     * Reports the complete ID space, including both reserved IDs.
     * @return ordinary token count plus two, always at least two
     */
    public int vocabularySize() {
        return ids.size() + 2;
    }

    /**
     * Identifies the ID used by text batching outside valid intervals, not an ordinary spelling.
     * @return reserved padding ID 0
     */
    public long paddingId() {
        return 0;
    }

    /**
     * Identifies the ID emitted once for each absent complete run.
     * @return reserved unknown ID 1
     */
    public long unknownId() {
        return 1;
    }

    /**
     * Encodes maximal nonempty non-whitespace code-point runs in original order.
     *
     * <p>Lookup uses exact source substrings, without case folding, character replacement or
     * Unicode normalization. Unknown runs each yield one UNK, regardless of their length.
     * Empty and whitespace-only strings yield fresh empty arrays; no BOS/EOS is inserted.</p>
     *
     * @param text non-null immutable source text; never retained or changed
     * @return a non-null fresh caller-owned array of IDs, independent of all other encodings
     * @throws NullPointerException if text is null, with message text
     * @throws OutOfMemoryError if encoding allocation fails
     */
    public long[] encode(String text) {
        Objects.requireNonNull(text, "text");
        int count = 0;
        boolean inRun = false;
        for (int offset = 0; offset < text.length();) {
            int codePoint = text.codePointAt(offset);
            boolean ordinary = !Character.isWhitespace(codePoint);
            if (ordinary && !inRun) count++;
            inRun = ordinary;
            offset += Character.charCount(codePoint);
        }
        long[] encoded = new long[count];
        int start = -1;
        int index = 0;
        for (int offset = 0; offset < text.length();) {
            int codePoint = text.codePointAt(offset);
            if (Character.isWhitespace(codePoint)) {
                if (start >= 0) {
                    encoded[index++] = ids.getOrDefault(text.substring(start, offset), unknownId());
                    start = -1;
                }
            } else if (start < 0) {
                start = offset;
            }
            offset += Character.charCount(codePoint);
        }
        if (start >= 0) encoded[index] = ids.getOrDefault(text.substring(start), unknownId());
        return encoded;
    }
}
