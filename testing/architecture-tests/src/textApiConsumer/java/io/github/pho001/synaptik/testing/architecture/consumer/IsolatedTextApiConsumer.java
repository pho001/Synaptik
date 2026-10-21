package io.github.pho001.synaptik.testing.architecture.consumer;

import io.github.pho001.synaptik.data.PaddedSequenceBatch;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.text.TextSequenceBatcher;
import io.github.pho001.synaptik.text.WhitespaceTokenizer;
import java.util.List;

/** Compiles Text, Data and Model declarations with only a Text project dependency. */
final class IsolatedTextApiConsumer {
    /**
     * Exercises transitive API exposure without an NN or Engine classpath.
     * @return fresh heap-backed INT64 values for two left-padded text rows
     */
    static Tensor values() {
        WhitespaceTokenizer tokenizer = WhitespaceTokenizer.of(List.of("Hello", "world"));
        PaddedSequenceBatch batch = TextSequenceBatcher.builder(tokenizer)
                .leftPadded().batchMajor().build().batch(List.of("Hello world", "missing"));
        return batch.values();
    }
}
