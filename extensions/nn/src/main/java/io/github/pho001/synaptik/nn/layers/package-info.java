/**
 * Provides stateful neural-network layers that own named parameters and compose Model Tensor
 * expressions.
 *
 * <p>A layer in this package is a {@link io.github.pho001.synaptik.nn.module.Module}: it owns the
 * parameter wrappers used by later forward construction, while generic Tensor operation meaning
 * remains in {@code modules/model}. Calling a forward method builds storage-free expression
 * metadata. It does not evaluate values, compile a graph, select a backend, or execute work.</p>
 *
 * <p>{@link io.github.pho001.synaptik.nn.layers.Linear},
 * {@link io.github.pho001.synaptik.nn.layers.LayerNorm}, and
 * {@link io.github.pho001.synaptik.nn.layers.Embedding} are
 * {@link io.github.pho001.synaptik.nn.module.UnaryTensorModule} instances and can be owned by
 * {@link io.github.pho001.synaptik.nn.module.Sequential}.
 * {@link io.github.pho001.synaptik.nn.layers.BatchNorm} and
 * {@link io.github.pho001.synaptik.nn.layers.Dropout} remain direct {@code Module} subclasses
 * because their complete forward contracts require explicit context or graph random state and a
 * result carrier. {@link io.github.pho001.synaptik.nn.layers.RnnCell},
 * {@link io.github.pho001.synaptik.nn.layers.GruCell}, and
 * {@link io.github.pho001.synaptik.nn.layers.LstmCell} are also direct {@code Module} subclasses:
 * their complete one-step contracts require an input plus explicit caller-threaded recurrent
 * state, so they are intentionally excluded from {@code Sequential}. LSTM returns both next
 * states through {@link io.github.pho001.synaptik.nn.layers.LstmCellForwardResult}.</p>
 *
 * <p>{@link io.github.pho001.synaptik.nn.layers.RnnSequence},
 * {@link io.github.pho001.synaptik.nn.layers.GruSequence}, and
 * {@link io.github.pho001.synaptik.nn.layers.LstmSequence} each own their matching concrete cell
 * and statically unroll a time-major input from static Java lengths. Callers may supply exact
 * state and lengths, omit lengths to make every row valid for the complete time extent, or omit
 * state to derive fresh non-gradient typed zero state for the current call. Every represented
 * step gathers only its stable active batch: the original rows whose validated and, for a
 * represented traversal, copied lengths exceed that step, kept in original order. Callers
 * coordinate writes through validation and any snapshot. The RNN and GRU result types expose
 * exact compact next-hidden outputs and restore final hidden rows. The LSTM result exposes the same kind of
 * compact hidden outputs while carrying compact cell state internally and restoring both final
 * hidden and final cell rows. A zero-length row selects its corresponding initial-state row; an
 * all-zero request returns the corresponding exact initial-state references, invokes no cell,
 * and therefore does not bind an automatic cell. Derived RNN/GRU state is one fresh eager leaf;
 * derived LSTM hidden and cell states are two distinct fresh eager leaves. These state leaves use
 * the cell parameter type, have Shape {@code [batch, hiddenSize]}, require no gradient, have no
 * name, and are never retained.
 * Each sequence can own one caller-supplied cell or construct one automatic matching cell from
 * explicit hidden width, bias, parameter type, initialization policy, and seed.</p>
 *
 * <p>This sequence packing is different from packed gate parameters: it omits padded logical
 * rows from cell expressions, while gate packing places several trainable gate matrices in one
 * parameter Tensor. It is also different from {@code Sequential}, which composes unary modules,
 * and from a runtime recurrent scan, which current Model contracts do not yet provide. Only the
 * validated and defensively copied explicit Java lengths determine padding; numeric zero remains
 * ordinary data. Every unroll reuses one cell and the same exact parameter leaf identities while
 * creating fresh operation producers per represented step and retaining temporal state ancestry.
 * The Compiler's existing exact-identity fan-out contract can therefore combine repeated
 * parameter contributions. The sequence classes themselves construct Model expressions and make
 * no numerical-gradient, public-training-loop, backend, scheduling, or execution guarantee.</p>
 *
 * <p>{@link io.github.pho001.synaptik.nn.layers.Linear} uses the conventional
 * {@code [outFeatures, inFeatures]} weight orientation and delegates each forward call to
 * {@link io.github.pho001.synaptik.model.tensor.Tensor#linear(
 * io.github.pho001.synaptik.model.tensor.Tensor)} or its biased overload. Its input-width-
 * inferring constructor reserves state names at construction and initializes the complete
 * parameter set at the start of the first compatible forward before constructing that call's
 * ordinary expression. Existing supplied and eager constructors remain immediately initialized.
 * Only the final input-feature extent is inferred; output width, bias, type, policy, factory, and
 * seed remain explicit. Sampling policies create the documented standard generator; zero/one
 * policies never invoke the retained factory. Every form behaves identically in training and
 * evaluation mode, and none numerically executes its constructed expression.</p>
 *
 * <p>The three recurrent cells also provide automatic constructors that retain an explicit
 * hidden width, bias choice, floating parameter type, closed initialization policy, and seed.
 * They infer only the positive static input width at the first represented forward call, then
 * publish the complete input-weight, hidden-weight, and optional typed-zero-bias group atomically.
 * Random policies use one fresh standard {@code L64X128MixRandom} stream per attempt, initializing
 * the complete input matrix before the complete hidden matrix; each Shape independently supplies
 * fan values. Zero and one policies create no generator, bias never draws, and gate order remains
 * layer-owned. Strict state loading can bind every reservation without initialization. Failed
 * attempts bind no partial group, retry from the retained seed, and serialize cell-locally;
 * accessors remain unavailable until forward or strict load publishes the group.</p>
 *
 * <p>{@link io.github.pho001.synaptik.nn.layers.LayerNorm} owns mandatory equal-Shape scale and
 * bias parameters over one positive-rank fully static normalized Shape, plus one exact
 * parameter-typed positive epsilon. It delegates to the affine
 * {@link io.github.pho001.synaptik.model.tensor.Tensor#layerNorm(
 * io.github.pho001.synaptik.model.shape.Shape,
 * io.github.pho001.synaptik.model.tensor.Tensor,
 * io.github.pho001.synaptik.model.tensor.Tensor,
 * io.github.pho001.synaptik.model.datatype.ScalarValue)} expression and is also mode-insensitive.
 * It exposes no default epsilon, partial affine state, or second configuration-introspection
 * surface. Compatible parameter replacement affects later forward calls; already constructed
 * Tensor expressions keep their earlier exact inputs.</p>
 *
 * <p>{@link io.github.pho001.synaptik.nn.layers.Embedding} owns one caller-supplied positive
 * rank-two floating table in {@code [vocabularySize, embeddingSize]} orientation. It delegates
 * each forward call to the current table's
 * {@link io.github.pho001.synaptik.model.tensor.Tensor#embedding(
 * io.github.pho001.synaptik.model.tensor.Tensor)} convenience and is mode-insensitive. Model owns
 * accepted index types, result metadata, ordinary Gather failures, and provenance. The layer adds
 * no table initializer, padding-row policy, numerical lookup, or execution behavior. Compatible
 * table replacement affects later calls, while already constructed expressions retain their
 * earlier exact table.</p>
 *
 * <p>{@link io.github.pho001.synaptik.nn.layers.BatchNorm} owns mandatory rank-one
 * {@code scale}, {@code bias}, {@code runningMean}, and {@code runningVariance} state for one
 * explicit logical channel axis. Its immutable
 * {@link io.github.pho001.synaptik.nn.module.ForwardContext} selects exact Model inference or
 * training composition. Evaluation preserves both buffers; successful training installs the
 * pure producer's next mean and then next variance expressions into their stable wrappers. This
 * symbolic NN binding transition performs no eager value mutation, compiler/runtime publication,
 * backend work, or numerical execution.</p>
 *
 * <p>{@link io.github.pho001.synaptik.nn.layers.RnnCell} owns positive fully static
 * {@code inputWeight [hiddenSize, inputSize]}, {@code hiddenWeight [hiddenSize, hiddenSize]}, and
 * optional shared {@code bias [hiddenSize]} parameters. Each call constructs exactly
 * {@code tanh((input @ inputWeight^T + bias?) + (hidden @ hiddenWeight^T))} through existing
 * Model expressions and returns that one Tensor as both output and next hidden state. Leading
 * Dimensions use ordinary right-aligned batch broadcasting; the cell performs no sequence loop,
 * time traversal, hidden-state retention, numerical evaluation, or execution.</p>
 *
 * <p>{@link io.github.pho001.synaptik.nn.layers.GruCell} owns reset/update/candidate-packed
 * {@code inputWeight [3 * hiddenSize, inputSize]},
 * {@code hiddenWeight [3 * hiddenSize, hiddenSize]}, and optional input-side
 * {@code bias [3 * hiddenSize]}. It constructs independent final-axis gate slices and the fixed
 * reset-after equations
 * {@code r = sigmoid(x_r + h_r)}, {@code z = sigmoid(x_z + h_z)},
 * {@code n = tanh(x_n + r * h_n)}, and {@code n + z * (hidden - n)}. The returned Tensor is both
 * output and next hidden state; update one retains the old hidden value. The cell retains no
 * hidden value, time axis, sequence traversal, numerical evaluation, or execution behavior.</p>
 *
 * <p>{@link io.github.pho001.synaptik.nn.layers.LstmCell} owns input/forget/candidate/output-
 * packed {@code inputWeight [4 * hiddenSize, inputSize]},
 * {@code hiddenWeight [4 * hiddenSize, hiddenSize]}, and optional input-side
 * {@code bias [4 * hiddenSize]}. It constructs independent projection slices and the fixed
 * equations {@code i = sigmoid(x_i + h_i)}, {@code f = sigmoid(x_f + h_f)},
 * {@code g = tanh(x_g + h_g)}, {@code o = sigmoid(x_o + h_o)},
 * {@code nextCell = f * cell + i * g}, and
 * {@code nextHidden = o * tanh(nextCell)}. Initialized bias is zero across every gate, including
 * forget, and there is no hidden-side bias. This gate order, bias association, and zero-bias
 * default form the Synaptik checkpoint schema rather than a framework-compatibility promise. The
 * caller supplies and threads both states explicitly through the exact references in
 * {@link io.github.pho001.synaptik.nn.layers.LstmCellForwardResult}; the cell adds no time axis,
 * sequence traversal, retained recurrent value, numerical evaluation, or execution behavior.</p>
 *
 * <p>{@link io.github.pho001.synaptik.nn.layers.Dropout} declares no module state and receives an
 * explicit {@link io.github.pho001.synaptik.model.tensor.GraphRngState} on every forward call. Its
 * immutable {@link io.github.pho001.synaptik.nn.module.ForwardContext} selects one Model training
 * dropout occurrence or an evaluation bypass. The bypass returns the exact input and incoming
 * state references without allocating a Tensor or advancing state; training returns the exact
 * public Model output and next-state references through a fresh NN-owned
 * {@link io.github.pho001.synaptik.nn.layers.DropoutForwardResult}. The layer owns no hidden
 * generator, seed, counter, parameter, buffer, execution, or backend behavior.</p>
 */
package io.github.pho001.synaptik.nn.layers;
