# Text tokenization and batching API

## Purpose and mental model

Use Text to turn strings into frozen token IDs, then Data to pad those IDs into a numeric
batch. The caller composes the batch with NN layers and Engine execution; Text provides no
execution facade. Gradle `:extensions:text` exports Data, which exports Model transitively.
The two final concrete types live in `io.github.pho001.synaptik.text`.

## Frozen token IDs

`WhitespaceTokenizer.of(List<String> ordinaryTokens)` copies the supplied order and never grows
its vocabulary. Padding (PAD) is ID `0`, unknown (UNK) is ID `1`, and ordinary IDs start at `2`.
For `WhitespaceTokenizer.of(List.of("Hello", "world"))`, `vocabularySize()` is `4`,
`paddingId()` is `0`, and `unknownId()` is `1`.

`encode(String)` returns a fresh caller-owned `long[]`, even for empty input. It emits one ID
per maximal nonempty run separated by `Character.isWhitespace(int)` code points. With that
vocabulary, `encode(" Hello\tworld missing ")` gives `[2,3,1]`; `encode("hello Hello,")`
gives `[1,1]`. Each unknown run produces one UNK, not one ID per character.

Case, punctuation and exact UTF-16 are preserved, including supplementary characters and unpaired
surrogates. There is no Unicode normalization: composed and decomposed spellings differ.
Java whitespace includes tabs and U+2003 EM SPACE, but not U+00A0 NO-BREAK SPACE,
U+2007 FIGURE SPACE, U+202F NARROW NO-BREAK SPACE or U+200B ZERO WIDTH SPACE.
Those non-separators remain part of a token. Empty or whitespace-only input encodes to `[]`.
Listed spellings such as `"PAD"` and `"UNK"` receive ordinary IDs, not reserved IDs.

An empty vocabulary is valid and has size `2`. Null lists, tokens or text throw
`NullPointerException`, identifying the parameter or token row. Empty ordinary tokens,
whitespace-containing tokens, duplicates, or a vocabulary size including reserved IDs above
`Integer.MAX_VALUE` throw `IllegalArgumentException`.

## Explicit batch configuration

`TextSequenceBatcher.builder(tokenizer)` fixes a non-null tokenizer. Select both a padding side
and an axis order before `build()`: `leftPadded()` or `rightPadded()`, and `timeMajor()` or
`batchMajor()`. There are no defaults; missing choices throw `IllegalStateException`.
Setters return the same builder, the last selection wins, and each build freezes an independent
configuration. Builders are reusable but not thread-safe; built configurations and tokenizers
are immutable and safely shareable.

`batch(List<String>)` encodes each row once in order and delegates once to Data's
`LongSequenceBatcher`, using `paddingId()`. Data owns padding, allocation and valid counts.
The returned `PaddedSequenceBatch` exposes `values()`, `validLengths()`, `axisOrder()` and
`paddingSide()`. Values are fresh contiguous heap-backed, non-gradient INT64 tensors.
Counts are immutable host metadata, not runtime graph inputs. Output tensor contents and storage
associations remain subject to ordinary Model access obligations; Text adds no close protocol.

For rows `["Hello world", "missing", ""]`, right/batch-major values are
`[[2,3],[1,0],[0,0]]`, shape `[3,2]`, and valid counts are `[2,1,0]`.
Left padding instead gives `[[2,3],[0,1],[0,0]]` with the same counts.
Batch-major shape is `[B,T]`; time-major shape is `[T,B]`, where B is row count and T is
maximum encoded length. Empty outer input has B=T=0; all-empty rows have T=0.

Null tokenizer, list or row throws `NullPointerException` with the parameter or row identified.
Coordinate input-list mutation throughout construction/batching; no transactional snapshot is
promised. Inputs are not retained afterward and repeated calls produce independent outputs.
Data rejects overflowing geometry or more than `Integer.MAX_VALUE` elements with
`IllegalArgumentException`; allocation failures and Model identifier exhaustion propagate.
There is no truncation or sequence-limit policy. See the [Data API](data-api.md) for allocation
and metadata contracts.

## Complete example

This runnable Java program gathers deterministic two-component embeddings for known, unknown
and padded positions. Use the repository JDK and an application classpath containing Text, NN
and Engine with its standard backend configuration. NN and Engine dependencies belong to the
application, not Text. The weight construction follows the existing numerical integration test.

```java
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.layers.Embedding;
import io.github.pho001.synaptik.text.TextSequenceBatcher;
import io.github.pho001.synaptik.text.WhitespaceTokenizer;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;

public class TextEmbeddingExample {
    public static void main(String[] args) {
        var tokenizer = WhitespaceTokenizer.of(List.of("Hello", "world"));
        var batcher = TextSequenceBatcher.builder(tokenizer)
                .rightPadded().batchMajor().build();
        var batch = batcher.batch(List.of("Hello world", "missing", ""));
        float[] table = {11.5f, -7.25f, 21f, 22f, 31f, 32f, 41f, 42f};
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            Shape shape = Shape.of(tokenizer.vocabularySize(), 2);
            MemorySegment segment = arena.allocate((long) table.length * Float.BYTES, Float.BYTES);
            segment.copyFrom(MemorySegment.ofArray(table));
            var weight = TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, shape,
                    Optional.of(LayoutDescriptor.contiguous(shape)), true), Optional.empty(),
                    Optional.of(new MemorySegmentStorage(DataType.FLOAT32, table.length, segment)));
            var embedding = new Embedding(weight);
            var output = engine.compute(embedding.forward(batch.values()));
            float[] values = new float[12];
            output.bytes().asFloatBuffer().get(values);
            System.out.println(Arrays.toString(batch.validLengths().toLongArray()));
            System.out.println(Arrays.toString(values));
        }
    }
}
```

The table has shape `[4,2]`, one row per ID including PAD and UNK. Execution appends the
embedding axis to the input, producing FLOAT32 shape `[3,2,2]`. The program prints:

```text
[2, 1, 0]
[31.0, 32.0, 41.0, 42.0, 21.0, 22.0, 11.5, -7.25, 11.5, -7.25, 11.5, -7.25]
```

IDs 2 and 3 select the known rows, ID 1 selects UNK, and ID 0 gathers the ordinary nonzero
PAD row. Padding does not zero, freeze or mask Embedding weights or gradients. The caller keeps
weight storage alive through compute and reads the result before Engine and then Arena close.

## Boundaries and related contracts

There is no tokenizer interface, byte-pair encoding (BPE), vocabulary trainer, decode,
persistence, normalization or automatic beginning/end token insertion. No recurrent neural
network (RNN) binding is added by Text. A returned RIGHT/TIME_MAJOR Data batch can explicitly
export a fresh INT64 length Tensor through
[`toScanLengthsTensor()`](data-api.md#explicit-runtime-scan-lengths); Data still does not bind it
or convert token IDs to floating features. The current fluent ModelBuilder accepts only floating
inputs and has no Embedding recipe, so this export does not enable Text IDs → Embedding → scan
in one builder chain. The [numeric inference example](data-api.md#complete-numeric-inference-example)
starts with FLOAT32 features, not integer IDs. Left padding still does not extend recurrent scans
beyond their right-valid-prefix contract, and no backpropagation through time (BPTT) support is added.

The [owning Text contract](../architecture/contracts/extensions-training.md#extensionstext)
and [owning Data contract](../architecture/contracts/extensions-training.md#extensionsdata)
define the boundaries. [ADR 0043](../design/decisions/0043-whitespace-text-and-padded-integral-batches.md)
explains the decision; the [glossary](../glossary.md#frozen-whitespace-tokenizer) defines the terms.
