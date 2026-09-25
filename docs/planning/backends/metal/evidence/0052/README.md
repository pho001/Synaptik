# Task 0052 Gate-1 Evidence

This directory is the permanent, non-production review copy of the exact custom-source and proof
package used by Task 0052. The files were recovered byte-for-byte from the original execution
session's recorded file writes and successful anchored edits. Their recomputed SHA-256 identities
match the identities recorded before the sole device invocation:

| Artifact | Role | SHA-256 |
|---|---|---|
| [`oracle.metal`](oracle.metal) | 15 custom Metal entry points and their indexing/numerical primitives | `ff15f63c9d2e54d62fa2157e1c424cc00acb90d5355101b18a07175e789a40c0` |
| [`oracle.mm`](oracle.mm) | single-command-buffer staging harness, corpus, metadata, and guards | `6db1c52113482753fcc96d3c14ffcee0479d9ca7b616e3a69f603d7ccb4c9683` |
| [`validate.py`](validate.py) | exact offline corpus and numerical validator | `94fd86bb6bf16b00195f990979ff892e22c8fac6cfe024173bae0bdf27f83e08` |
| [`audit.py`](audit.py) | pre-run source/corpus/control audit | `0c522be1e4f93f82576ad674a07d411c0c68a1c181ea88ac0724f56cfefc1d14` |
| [`run-once.py`](run-once.py) | audited one-process capture wrapper | `ec73cd5f6cef831f10fbd2300ca3bf6d3e0796920aa58e402dd56ac0b6481046` |
| [`domain-proof.txt`](domain-proof.txt) | exact pre-run Gate-1/Gate-1B proof record | `589cb804a00c35ea4589818eeedafcc238d4b91c2ec07961b561eabda9a69153` |

The compiled executable, raw output, and temporary workspace were intentionally not restored. Their
recorded identities and the unedited validator verdict remain in the task result. Recovery and hash
validation did **not** execute the oracle, create a Metal device, compile source, or consume another
measurement.

These files are evidence, not production code. They grant no capability and are not a reusable
probe invitation. A future authorized implementation must re-establish all preflight and lifecycle
obligations against its then-current production source.

## Complete-domain assumptions

The proof is for the exact Task-0052 domain: rank `1..16`, positive static extents, checked element
and byte products, dense zero-offset row-major tensors, normalized axes, exact right-aligned
broadcast metadata, and resource-feasible buffers and dispatches. Metadata reaches a kernel only
after those checks. In particular, every dimension and stride multiplication used below must fit
`uint64`; every byte count must fit the host allocation type; and every derived last offset must be
inside its corresponding allocation. The recovered harness makes the rank, positive-dimension,
product-overflow, axis, and dispatch-domain checks explicit. Production authorization would also
require the task's complete Java/native topology and buffer-preflight checks.

## Launch and flattened-index invariant

For grid extents `W`, `H`, `D` and thread coordinate `(x,y,z)`, every kernel computes

```text
linear(x,y,z) = x + W * (y + H * z).
```

The host chooses

```text
W = min(N, UINT32_MAX)
Q = ceil(N / W)
H = min(Q, UINT32_MAX)
D = ceil(Q / H), with D <= UINT32_MAX.
```

Thus `W*H*D >= N`. The kernel rejects `linear >= N`. On the admitted checked/resource-feasible
FLOAT32 domain, the products fit `uint64`. The mapping is one-to-one: equality of two linear values
first gives equal `x` after reduction modulo `W`, then equal `y` after division by `W` and reduction
modulo `H`, then equal `z`. Consequently every `linear` in `[0,N)` has exactly one writer, padding
threads have none, and no 32-bit flattening truncation occurs.

## Broadcast, comparison, extrema, and clamp invariant

Let output dimensions be `d[0..r-1]`, `N = product(d[a])`, and let `i` be a flattened output index.
The kernel decodes coordinates from the last axis to the first:

```text
remaining = i
for a = r-1 .. 0:
    coordinate[a] = remaining % d[a]
    remaining     = remaining / d[a]
leftOffset  = sum(coordinate[a] * leftStride[a])
rightOffset = sum(coordinate[a] * rightStride[a])
```

Right-aligned broadcast preflight supplies row-major input strides on matching dimensions and stride
zero for a broadcast or absent leading dimension. Mixed-radix decomposition is unique for
`0 <= i < N`; each coordinate lies in `[0,d[a])`; and the validated strides therefore put both
offsets inside their inputs. Scalar extrema and fused clamp use `i` directly on the same checked
flat domain.

`exact_compare` classifies raw binary32 words. NaNs make equality and all ordered predicates false;
`NOT_EQUAL` is the complement of equality; the two signed zeros compare equal; and all remaining
words compare through the monotone key `negative ? ~word : word ^ 0x80000000`. Every comparison
writes exactly byte `0` or `1`.

`exact_extreme(left,right,maximum)` returns a source word: a NaN source first, otherwise the fixed
zero winner (`left & right` sign for MAX and `left | right` sign for MIN), otherwise the source
selected by the exact ordering key. NaN payload/sign/quiet state is intentionally abstract in the
Model result. Fused clamp is exactly
`exact_extreme(exact_extreme(x,lower,MAX),upper,MIN)`. The composed custom clamp materializes the
first source-word result in a guarded FLOAT32 buffer before the second custom primitive; it contains
no opaque numerical dependency.

## Reduction mapping, bounds, and contributor invariant

For input dimensions `d[0..r-1]`, define row-major strides

```text
s[a] = product(d[b], b = a+1 .. r-1).
```

Let normalized reduced axes be the set `A`. Define

```text
O = product(d[a], a not in A)   # output count
R = product(d[a], a in A)       # contributors per output
```

Both start from one, so the empty-axis form has `O = N`, `R = 1`. For each output index
`o in [0,O)` and contributor index `q in [0,R)`, `reduction_offset` walks axes from `r-1` to `0`:

```text
outputRemaining  = o
reducedRemaining = q
offset = 0
for a = r-1 .. 0:
    if a in A:
        coordinate[a] = reducedRemaining % d[a]
        reducedRemaining = reducedRemaining / d[a]
    else:
        coordinate[a] = outputRemaining % d[a]
        outputRemaining = outputRemaining / d[a]
    offset += coordinate[a] * s[a]
```

The two mixed-radix decoders are bijections over the reduced and non-reduced Cartesian products.
Their coordinates jointly form exactly one input coordinate tuple. Hence, for fixed `o`, the
`q=0..R-1` offsets are distinct and contain every contributor exactly once; across `o=0..O-1`, the
partitions are disjoint and cover all `N=O*R` input elements. Since every coordinate is less than
its dimension, `0 <= offset < N`; checked stride/products exclude arithmetic overflow.

The fold loads contributor zero unchanged, then visits contributors `1..R-1` in increasing `q` and
applies `exact_extreme`. Positive dimensions guarantee `R>=1`. The empty-axis form therefore copies
one raw input word per output. Full, single-axis, and multi-axis forms use the same mapping. Keep- and
drop-dimensions alter only the public descriptor shape, not flattened output order. The ordered axis
list `[2,0]` normalizes to the same membership mask; for non-NaN extrema the source-word winner and
fixed zero sign are order-independent, while any NaN source is accepted as the abstract NaN class.
No contributor is dropped, duplicated, invented, or read out of bounds.

## Scan mapping, direction, placement, and bounds

For scan axis `k`, define

```text
L         = d[k]
inner     = product(d[b], b = k+1 .. r-1)
lineCount = N / L
```

A line index `ell in [0,lineCount)` selects

```text
outer      = ell / inner
innerIndex = ell % inner
base       = outer * L * inner + innerIndex
logical(t) = t                 # forward
logical(t) = L - 1 - t         # reverse
offset(t)  = base + logical(t) * inner,  0 <= t < L.
```

`ell -> (outer,innerIndex)` is a mixed-radix bijection over all axis lines. `logical` is a
permutation of `[0,L)`, so each line visits every axis position exactly once in the requested
direction. Positive extents give `L>0`; `t<L` makes reverse subtraction nonnegative; and
`outer`, `innerIndex`, and `logical` bounds give `0 <= offset(t) < N`. Distinct lines cannot share an
input/output coordinate, so the scan has exactly one writer for every output element.

The loop's contributor order is exactly `input[offset(0)], ..., input[offset(L-1)]`. Inclusive mode
writes after consuming the current contributor. Exclusive mode writes before consumption and emits
the exact `+0` SUM identity or `+1` PRODUCT identity for the empty prefix. The first nonempty prefix
copies its contributor's raw word without an arithmetic site. Each later step performs one safe
FLOAT32 operation on the loop-carried accumulator and current contributor, yielding the legal
left-associated, order-preserving contiguous-prefix tree.

## Ordered-prefix numerical invariant

The independent validator models every legal ordered tree, not merely the kernel's left fold. For a
nonempty ordered sequence `X`, it computes

```text
T([x]) = { raw(x) }, with every NaN raw word represented by NAN
T(X)   = union over split p=1..len(X)-1 of
         { primitive(l,r) | l in T(X[:p]), r in T(X[p:]) }.
```

Only contiguous splits are used, so leaf order and exact contributor membership are invariant.
Reverse mode reverses the traversal sequence before prefixes are formed. Inclusive mode validates
the prefix including the current leaf; exclusive mode validates the preceding prefix, with the
empty prefix restricted to the exact identity.

At every primitive arithmetic site:

1. **DAZ:** each binary32 subnormal operand independently remains its raw value or becomes the
   same-signed zero;
2. **RNE:** finite operands are converted to exact rational values, added or multiplied exactly,
   then rounded once to binary32 ties-to-even, including exact overflow and signed-zero rules;
3. **FTZ:** a nonzero subnormal result independently remains exact or becomes either signed zero,
   exactly matching the frozen Model allowance; and
4. **NaN:** a NaN operand, opposite infinities in SUM, or zero-times-infinity in PRODUCT produces the
   single abstract `NAN` token; any later operation consuming that token remains `NAN`. Validation
   accepts any binary32 NaN word for that token and no non-NaN word.

Therefore every accepted non-NaN output belongs to the complete DAZ/RNE/FTZ result set for the exact
ordered prefix, and every accepted NaN occurs only at a prefix with a NaN-reachable arithmetic path.
The bounded corpus remains regression evidence; this source-and-invariant review is the separate
complete-domain basis for the custom candidates' `DOMAIN-PASS` records.
