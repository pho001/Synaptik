namespace Task0069

/-- A structured presentation of every binary32 raw word. -/
structure Word where
  sign : Bool
  exponent : Fin 256
  fraction : Fin 8388608
  deriving DecidableEq, Repr

/-- A checked two-sided correspondence, kept independent of library-specific finite-set APIs. -/
structure Bijection (α β : Type) where
  toFun : α → β
  invFun : β → α
  leftInverse : ∀ value, invFun (toFun value) = value
  rightInverse : ∀ value, toFun (invFun value) = value

/-- The carrier of every unsigned 32-bit raw bit pattern. -/
abbrev Raw32 := Fin (2 ^ 32)

/-- Encode sign/exponent/fraction into the corresponding IEEE-754 raw bit pattern. -/
def rawWord (word : Word) : Raw32 :=
  ⟨(if word.sign then 2147483648 else 0) +
      word.exponent.val * 8388608 + word.fraction.val, by
    have exponentBound := word.exponent.isLt
    have fractionBound := word.fraction.isLt
    cases word.sign <;> simp <;> omega⟩

/-- Decode every unsigned 32-bit raw bit pattern into sign/exponent/fraction. -/
def wordOfRaw (raw : Raw32) : Word :=
  { sign := if raw.val < 2147483648 then false else true
    exponent := ⟨(raw.val / 8388608) % 256, Nat.mod_lt _ (by decide)⟩
    fraction := ⟨raw.val % 8388608, Nat.mod_lt _ (by decide)⟩ }

theorem wordOfRaw_rawWord (word : Word) : wordOfRaw (rawWord word) = word := by
  cases word with
  | mk sign exponent fraction =>
    have exponentBound := exponent.isLt
    have fractionBound := fraction.isLt
    cases sign
    case false =>
      have packed :
          exponent.val * 8388608 + fraction.val =
            8388608 * exponent.val + fraction.val := by omega
      have belowSignBit :
          8388608 * exponent.val + fraction.val < 2147483648 := by omega
      simp [wordOfRaw, rawWord, packed, belowSignBit, Nat.mul_add_div,
        Nat.div_eq_of_lt fractionBound, Nat.mod_eq_of_lt fractionBound,
        Nat.mod_eq_of_lt exponentBound]
    case true =>
      have packed :
          2147483648 + exponent.val * 8388608 + fraction.val =
            8388608 * (256 + exponent.val) + fraction.val := by omega
      have atOrAboveSignBit :
          2147483648 ≤ 8388608 * (256 + exponent.val) + fraction.val := by omega
      simp [wordOfRaw, rawWord, packed, atOrAboveSignBit, Nat.mul_add_div,
        Nat.div_eq_of_lt fractionBound, Nat.mod_eq_of_lt fractionBound,
        Nat.mod_eq_of_lt exponentBound]

theorem rawWord_wordOfRaw (raw : Raw32) : rawWord (wordOfRaw raw) = raw := by
  apply Fin.ext
  have rawBound := raw.isLt
  have lowDecomposition := Nat.div_add_mod raw.val 8388608
  have highDecomposition := Nat.div_add_mod (raw.val / 8388608) 256
  simp only [rawWord, wordOfRaw]
  split <;> simp_all <;> omega

/-- Checked bijection between structured words and all 2^32 unsigned raw words. -/
def wordRawBijection : Bijection Word Raw32 :=
  { toFun := rawWord
    invFun := wordOfRaw
    leftInverse := wordOfRaw_rawWord
    rightInverse := rawWord_wordOfRaw }

theorem raw_word_count : 2 * 256 * 8388608 = 2 ^ 32 := by decide

inductive RawClass where
  | zero
  | subnormal
  | normal
  | infinity
  | nan
  deriving DecidableEq, Repr

/-- Total raw-word classification; the sign never changes the class. -/
def rawClass (word : Word) : RawClass :=
  if word.exponent.val = 0 then
    if word.fraction.val = 0 then .zero else .subnormal
  else if word.exponent.val = 255 then
    if word.fraction.val = 0 then .infinity else .nan
  else
    .normal

/-- The kernel's integer sign-bit clear, represented structurally. -/
def absWord (word : Word) : Word :=
  { word with sign := false }

/-- Same-sign zero used by the admitted DAZ and FTZ alternatives. -/
def signedZero (sign : Bool) : Word :=
  { sign := sign, exponent := 0, fraction := 0 }

/-- A load may retain a subnormal or present it to the add site as same-sign zero. -/
def daz (source presented : Word) : Prop :=
  presented = source ∨
    (rawClass source = .subnormal ∧ presented = signedZero source.sign)

/-- A rounded subnormal may remain exact or publish as same-sign zero. -/
def ftz (rounded published : Word) : Prop :=
  published = rounded ∨
    (rawClass rounded = .subnormal ∧ published = signedZero rounded.sign)

/--
One safe binary32 addition site: each operand has only the exact-or-DAZ choices, `rne`
performs the single binary32 rounding step, and the result has only exact-or-FTZ choices.
-/
def addSite (rne : Word → Word → Word) (left right published : Word) : Prop :=
  ∃ presentedLeft presentedRight rounded,
    daz left presentedLeft ∧
    daz right presentedRight ∧
    rounded = rne presentedLeft presentedRight ∧
    ftz rounded published
/--
An operand reachable at an L1 addition site: NaNs may carry either sign, while every non-NaN
operand is sign-clear.
-/
def L1Reachable (word : Word) : Prop :=
  rawClass word = .nan ∨ word.sign = false


/-- Explicit class obligations for primitive RNE addition on reachable L1 operands. -/
structure RneSpecialClassContract (rne : Word → Word → Word) : Prop where
  nanLeft : ∀ left right,
    rawClass left = .nan → L1Reachable right → rawClass (rne left right) = .nan
  nanRight : ∀ left right,
    rawClass right = .nan → L1Reachable left → rawClass (rne left right) = .nan
  positiveInfinityLeft : ∀ left right,
    left.sign = false → rawClass left = .infinity →
    right.sign = false → rawClass right ≠ .nan →
    (rne left right).sign = false ∧ rawClass (rne left right) = .infinity
  positiveInfinityRight : ∀ left right,
    right.sign = false → rawClass right = .infinity →
    left.sign = false → rawClass left ≠ .nan →
    (rne left right).sign = false ∧ rawClass (rne left right) = .infinity
  nonNanClosure : ∀ left right,
    left.sign = false → right.sign = false →
    rawClass left ≠ .nan → rawClass right ≠ .nan →
    rawClass (rne left right) ≠ .nan
  nonnegativeClosure : ∀ left right,
    left.sign = false → right.sign = false →
    rawClass left ≠ .nan → rawClass right ≠ .nan →
    (rne left right).sign = false

@[simp] theorem absWord_sign (word : Word) : (absWord word).sign = false := rfl

@[simp] theorem rawClass_absWord (word : Word) : rawClass (absWord word) = rawClass word := rfl

@[simp] theorem rawClass_signedZero (sign : Bool) : rawClass (signedZero sign) = .zero := rfl

theorem daz_complete (source presented : Word) :
    daz source presented ↔
      presented = source ∨
        (rawClass source = .subnormal ∧ presented = signedZero source.sign) := Iff.rfl

theorem ftz_complete (rounded published : Word) :
    ftz rounded published ↔
      published = rounded ∨
        (rawClass rounded = .subnormal ∧ published = signedZero rounded.sign) := Iff.rfl

theorem daz_preserves_or_zero {source presented : Word} (site : daz source presented) :
    presented = source ∨ rawClass presented = .zero := by
  rcases site with exact | ⟨_, zero⟩
  · exact Or.inl exact
  · exact Or.inr (zero ▸ rawClass_signedZero source.sign)

theorem ftz_preserves_or_zero {rounded published : Word} (site : ftz rounded published) :
    published = rounded ∨ rawClass published = .zero := by
  rcases site with exact | ⟨_, zero⟩
  · exact Or.inl exact
  · exact Or.inr (zero ▸ rawClass_signedZero rounded.sign)

/-- The exact typed positive-one word needed by the later literal-site slices. -/
def typedOne : Word :=
  { sign := false, exponent := ⟨127, by decide⟩, fraction := ⟨0, by decide⟩ }

@[simp] theorem typedOne_sign : typedOne.sign = false := rfl

@[simp] theorem typedOne_class : rawClass typedOne = .normal := rfl

/-- Nonnegative finite words are precisely the finite inputs reached by L1 before overflow. -/
def nonnegativeFinite (word : Word) : Prop :=
  word.sign = false ∧ rawClass word ≠ .nan ∧ rawClass word ≠ .infinity

/--
Every finite binary32 magnitude as one integer count of the least positive subnormal, `2^-149`.
The definition is total, but the exact-RNE relation uses it only under `nonnegativeFinite`.
-/
def finiteMagnitudeScaled (word : Word) : Nat :=
  if word.exponent.val = 0 then
    word.fraction.val
  else
    (2 ^ 23 + word.fraction.val) * 2 ^ (word.exponent.val - 1)

def natDistance (left right : Nat) : Nat :=
  if left ≤ right then right - left else left - right

/-- Midpoint between maximum finite binary32 and overflow under round-to-nearest-even. -/
def overflowThresholdScaled : Nat :=
  2 ^ 277 - 2 ^ 252

/--
Bit-level round-to-nearest-even for the nonnegative finite additions used by L1. Below the
overflow midpoint the result is a closest finite binary32 word; an equal-distance alternative
requires the retained result significand to be even. At or above the midpoint the result is
positive infinity.
-/
def exactRneNonnegative (left right result : Word) : Prop :=
  let exact := finiteMagnitudeScaled left + finiteMagnitudeScaled right
  if overflowThresholdScaled ≤ exact then
    result.sign = false ∧ rawClass result = .infinity
  else
    nonnegativeFinite result ∧
      ∀ candidate, nonnegativeFinite candidate →
        let resultDistance := natDistance exact (finiteMagnitudeScaled result)
        let candidateDistance := natDistance exact (finiteMagnitudeScaled candidate)
        resultDistance < candidateDistance ∨
          (resultDistance = candidateDistance ∧
            (candidate = result ∨ result.fraction.val % 2 = 0))

/--
The primitive premise is no longer an arbitrary `Word → Word → Word`: it must obey the complete
special-class contract and the explicit bit-level nearest-even relation on every nonnegative
finite operand pair reached by L1.
-/
structure Binary32RneContract (rne : Word → Word → Word) : Prop
    extends RneSpecialClassContract rne where
  finiteRounding : ∀ left right,
    nonnegativeFinite left →
    nonnegativeFinite right →
    exactRneNonnegative left right (rne left right)

/-- Every non-NaN/non-infinity word, including both signed zeros and all subnormals. -/
def finiteWord (word : Word) : Prop :=
  rawClass word ≠ .nan ∧ rawClass word ≠ .infinity

/-- Exact signed value as an integer count of the least positive binary32 subnormal. -/
def finiteSignedScaled (word : Word) : Int :=
  if word.sign then
    -Int.ofNat (finiteMagnitudeScaled word)
  else
    Int.ofNat (finiteMagnitudeScaled word)

def intDistance (left right : Int) : Nat :=
  (left - right).natAbs

/--
Set-valued exact round-to-nearest-even relation for arbitrary finite binary32 operands. It covers
signed cancellation and signed zero, every normal/subnormal pair, and both overflow directions.
-/
def exactRneFiniteSigned (left right result : Word) : Prop :=
  let exact := finiteSignedScaled left + finiteSignedScaled right
  if exact = 0 then
    rawClass result = .zero ∧ result.sign = (left.sign && right.sign)
  else if overflowThresholdScaled ≤ exact.natAbs then
    rawClass result = .infinity ∧ result.sign = decide (exact < 0)
  else
    finiteWord result ∧
      result.sign = decide (exact < 0) ∧
      ∀ candidate, finiteWord candidate →
        let resultDistance := intDistance exact (finiteSignedScaled result)
        let candidateDistance := intDistance exact (finiteSignedScaled candidate)
        resultDistance < candidateDistance ∨
          (resultDistance = candidateDistance ∧
            (candidate = result ∨ result.fraction.val % 2 = 0))

/--
Total set-valued binary32 RNE relation. NaN inputs produce NaN; opposite infinities produce NaN;
one or equal-sign infinities preserve their sign and class; every remaining finite pair uses the
signed exact-value nearest-even relation above.
-/
def exactRneGeneral (left right result : Word) : Prop :=
  if rawClass left = .nan ∨ rawClass right = .nan then
    rawClass result = .nan
  else if rawClass left = .infinity then
    if rawClass right = .infinity ∧ left.sign ≠ right.sign then
      rawClass result = .nan
    else
      rawClass result = .infinity ∧ result.sign = left.sign
  else if rawClass right = .infinity then
    rawClass result = .infinity ∧ result.sign = right.sign
  else
    exactRneFiniteSigned left right result

/--
The Scatter primitive is constrained for every possible raw operand pair, not merely the
nonnegative L1 subset. `exactRneGeneral` is set-valued only where binary32 permits multiple NaN
payloads or the source's explicit DAZ/FTZ alternatives surround the primitive result.
-/
structure GeneralBinary32RneContract (rne : Word → Word → Word) : Prop where
  exactRounding : ∀ left right, exactRneGeneral left right (rne left right)

def generalBinary32AddSite
    (rne : Word → Word → Word)
    (_contract : GeneralBinary32RneContract rne)
    (left right published : Word) : Prop :=
  addSite rne left right published

def binary32AddSite
    (rne : Word → Word → Word)
    (_contract : Binary32RneContract rne)
    (left right published : Word) : Prop :=
  addSite rne left right published

theorem daz_exact_of_not_subnormal
    {source presented : Word}
    (site : daz source presented)
    (notSubnormal : rawClass source ≠ .subnormal) :
    presented = source := by
  rcases site with retained | ⟨subnormal, _⟩
  · exact retained
  · exact False.elim (notSubnormal subnormal)

theorem ftz_exact_of_not_subnormal
    {rounded published : Word}
    (site : ftz rounded published)
    (notSubnormal : rawClass rounded ≠ .subnormal) :
    published = rounded := by
  rcases site with retained | ⟨subnormal, _⟩
  · exact retained
  · exact False.elim (notSubnormal subnormal)

theorem daz_preserves_false_sign
    {source presented : Word}
    (site : daz source presented)
    (sourceSign : source.sign = false) :
    presented.sign = false := by
  rcases site with retained | ⟨_, zero⟩
  · simpa [retained] using sourceSign
  · simp [zero, signedZero, sourceSign]

theorem ftz_preserves_false_sign
    {rounded published : Word}
    (site : ftz rounded published)
    (roundedSign : rounded.sign = false) :
    published.sign = false := by
  rcases site with retained | ⟨_, zero⟩
  · simpa [retained] using roundedSign
  · simp [zero, signedZero, roundedSign]

theorem daz_preserves_not_nan
    {source presented : Word}
    (site : daz source presented)
    (sourceNotNan : rawClass source ≠ .nan) :
    rawClass presented ≠ .nan := by
  rcases site with retained | ⟨_, zero⟩
  · simpa [retained] using sourceNotNan
  · simp [zero]

theorem ftz_preserves_not_nan
    {rounded published : Word}
    (site : ftz rounded published)
    (roundedNotNan : rawClass rounded ≠ .nan) :
    rawClass published ≠ .nan := by
  rcases site with retained | ⟨_, zero⟩
  · simpa [retained] using roundedNotNan
  · simp [zero]
theorem daz_preserves_l1_reachable
    {source presented : Word}
    (site : daz source presented)
    (sourceReachable : L1Reachable source) :
    L1Reachable presented := by
  rcases site with retained | ⟨sourceSubnormal, zero⟩
  · simpa [retained] using sourceReachable
  · apply Or.inr
    have sourceNotNan : rawClass source ≠ .nan := by
      intro sourceNan
      have impossible : RawClass.subnormal = RawClass.nan :=
        sourceSubnormal.symm.trans sourceNan
      cases impossible
    have sourceSign : source.sign = false :=
      sourceReachable.resolve_left sourceNotNan
    simpa [zero, signedZero] using sourceSign


theorem addSite_nan_left
    {rne : Word → Word → Word}
    (contract : RneSpecialClassContract rne)
    {left right published : Word}
    (site : addSite rne left right published)
    (leftNan : rawClass left = .nan)
    (rightReachable : L1Reachable right) :
    rawClass published = .nan := by
  rcases site with ⟨presentedLeft, presentedRight, rounded,
    leftDaz, rightDaz, roundedEq, publishedFtz⟩
  have leftExact : presentedLeft = left := daz_exact_of_not_subnormal leftDaz (by
    intro subnormal
    have impossible : RawClass.nan = RawClass.subnormal := leftNan.symm.trans subnormal
    cases impossible)
  subst presentedLeft
  have roundedNan : rawClass rounded = .nan := by
    rw [roundedEq]
    exact contract.nanLeft left presentedRight leftNan
      (daz_preserves_l1_reachable rightDaz rightReachable)
  have publishedExact : published = rounded := ftz_exact_of_not_subnormal publishedFtz (by
    intro subnormal
    have impossible : RawClass.nan = RawClass.subnormal := roundedNan.symm.trans subnormal
    cases impossible)
  simpa [publishedExact] using roundedNan

theorem addSite_nan_right
    {rne : Word → Word → Word}
    (contract : RneSpecialClassContract rne)
    {left right published : Word}
    (site : addSite rne left right published)
    (rightNan : rawClass right = .nan)
    (leftReachable : L1Reachable left) :
    rawClass published = .nan := by
  rcases site with ⟨presentedLeft, presentedRight, rounded,
    leftDaz, rightDaz, roundedEq, publishedFtz⟩
  have rightExact : presentedRight = right := daz_exact_of_not_subnormal rightDaz (by
    intro subnormal
    have impossible : RawClass.nan = RawClass.subnormal := rightNan.symm.trans subnormal
    cases impossible)
  subst presentedRight
  have roundedNan : rawClass rounded = .nan := by
    rw [roundedEq]
    exact contract.nanRight presentedLeft right rightNan
      (daz_preserves_l1_reachable leftDaz leftReachable)
  have publishedExact : published = rounded := ftz_exact_of_not_subnormal publishedFtz (by
    intro subnormal
    have impossible : RawClass.nan = RawClass.subnormal := roundedNan.symm.trans subnormal
    cases impossible)
  simpa [publishedExact] using roundedNan

theorem addSite_nonnegative
    {rne : Word → Word → Word}
    (contract : RneSpecialClassContract rne)
    {left right published : Word}
    (site : addSite rne left right published)
    (leftSign : left.sign = false)
    (rightSign : right.sign = false)
    (leftNotNan : rawClass left ≠ .nan)
    (rightNotNan : rawClass right ≠ .nan) :
    published.sign = false := by
  rcases site with ⟨presentedLeft, presentedRight, rounded,
    leftDaz, rightDaz, roundedEq, publishedFtz⟩
  have roundedSign : rounded.sign = false := by
    rw [roundedEq]
    exact contract.nonnegativeClosure presentedLeft presentedRight
      (daz_preserves_false_sign leftDaz leftSign)
      (daz_preserves_false_sign rightDaz rightSign)
      (daz_preserves_not_nan leftDaz leftNotNan)
      (daz_preserves_not_nan rightDaz rightNotNan)
  exact ftz_preserves_false_sign publishedFtz roundedSign

theorem addSite_non_nan
    {rne : Word → Word → Word}
    (contract : RneSpecialClassContract rne)
    {left right published : Word}
    (site : addSite rne left right published)
    (leftSign : left.sign = false)
    (rightSign : right.sign = false)
    (leftNotNan : rawClass left ≠ .nan)
    (rightNotNan : rawClass right ≠ .nan) :
    rawClass published ≠ .nan := by
  rcases site with ⟨presentedLeft, presentedRight, rounded,
    leftDaz, rightDaz, roundedEq, publishedFtz⟩
  have roundedNotNan : rawClass rounded ≠ .nan := by
    rw [roundedEq]
    exact contract.nonNanClosure presentedLeft presentedRight
      (daz_preserves_false_sign leftDaz leftSign)
      (daz_preserves_false_sign rightDaz rightSign)
      (daz_preserves_not_nan leftDaz leftNotNan)
      (daz_preserves_not_nan rightDaz rightNotNan)
  exact ftz_preserves_not_nan publishedFtz roundedNotNan

theorem addSite_positive_infinity_left
    {rne : Word → Word → Word}
    (contract : RneSpecialClassContract rne)
    {left right published : Word}
    (site : addSite rne left right published)
    (leftInfinity : left.sign = false ∧ rawClass left = .infinity)
    (rightValid : right.sign = false ∧ rawClass right ≠ .nan) :
    published.sign = false ∧ rawClass published = .infinity := by
  rcases site with ⟨presentedLeft, presentedRight, rounded,
    leftDaz, rightDaz, roundedEq, publishedFtz⟩
  have leftExact : presentedLeft = left := daz_exact_of_not_subnormal leftDaz (by
    intro subnormal
    have impossible : RawClass.infinity = RawClass.subnormal :=
      leftInfinity.2.symm.trans subnormal
    cases impossible)
  subst presentedLeft
  have roundedInfinity : rounded.sign = false ∧ rawClass rounded = .infinity := by
    rw [roundedEq]
    exact contract.positiveInfinityLeft left presentedRight
      leftInfinity.1 leftInfinity.2
      (daz_preserves_false_sign rightDaz rightValid.1)
      (daz_preserves_not_nan rightDaz rightValid.2)
  have publishedExact : published = rounded := ftz_exact_of_not_subnormal publishedFtz (by
    intro subnormal
    have impossible : RawClass.infinity = RawClass.subnormal :=
      roundedInfinity.2.symm.trans subnormal
    cases impossible)
  simpa [publishedExact] using roundedInfinity

theorem addSite_positive_infinity_right
    {rne : Word → Word → Word}
    (contract : RneSpecialClassContract rne)
    {left right published : Word}
    (site : addSite rne left right published)
    (leftValid : left.sign = false ∧ rawClass left ≠ .nan)
    (rightInfinity : right.sign = false ∧ rawClass right = .infinity) :
    published.sign = false ∧ rawClass published = .infinity := by
  rcases site with ⟨presentedLeft, presentedRight, rounded,
    leftDaz, rightDaz, roundedEq, publishedFtz⟩
  have rightExact : presentedRight = right := daz_exact_of_not_subnormal rightDaz (by
    intro subnormal
    have impossible : RawClass.infinity = RawClass.subnormal :=
      rightInfinity.2.symm.trans subnormal
    cases impossible)
  subst presentedRight
  have roundedInfinity : rounded.sign = false ∧ rawClass rounded = .infinity := by
    rw [roundedEq]
    exact contract.positiveInfinityRight presentedLeft right
      rightInfinity.1 rightInfinity.2
      (daz_preserves_false_sign leftDaz leftValid.1)
      (daz_preserves_not_nan leftDaz leftValid.2)
  have publishedExact : published = rounded := ftz_exact_of_not_subnormal publishedFtz (by
    intro subnormal
    have impossible : RawClass.infinity = RawClass.subnormal :=
      roundedInfinity.2.symm.trans subnormal
    cases impossible)
  simpa [publishedExact] using roundedInfinity

theorem addSite_nonempty (rne : Word → Word → Word) (left right : Word) :
    ∃ published, addSite rne left right published := by
  refine ⟨rne left right, left, right, rne left right, ?_⟩
  exact ⟨Or.inl rfl, Or.inl rfl, rfl, Or.inl rfl⟩

/-- The raw ABS leaf transform is total over zero, subnormal, normal, infinity, and NaN. -/
theorem absWord_class_contract (word : Word) :
    (absWord word).sign = false ∧ rawClass (absWord word) = rawClass word := by
  exact ⟨rfl, rfl⟩


/-- The unique finite singleton-variance result required by both output forms. -/
def positiveZero : Word := signedZero false

/--
A primitive site is a typed binary32 relation surrounded by the only admitted operand DAZ and
result FTZ presentations. The relation, rather than an unconstrained function, names the exact
arithmetic and the one RNE publication for DIV, SUB, or MUL.
-/
def binary32PrimitiveSite
    (primitive : Word → Word → Word → Prop)
    (left right published : Word) : Prop :=
  ∃ presentedLeft presentedRight rounded,
    daz left presentedLeft ∧
    daz right presentedRight ∧
    primitive presentedLeft presentedRight rounded ∧
    ftz rounded published

def SmallFinite (word : Word) : Prop :=
  rawClass word = .zero ∨ rawClass word = .subnormal
instance smallFiniteDecidable (word : Word) : Decidable (SmallFinite word) := by
  unfold SmallFinite
  infer_instance

/--
The exact finite subtraction fragment reachable by the singleton kernel. Its integer is measured in
`2^-149` units: exact cancellation publishes `+0`, while every reachable nonzero magnitude is
strictly below `2^23` and is therefore an exactly representable subnormal. This deliberately has
no catch-all branch for finite SUB inputs outside the Task-0069 trace.
-/
def task0069ExactSubValue (exact : Int) (result : Word) : Prop :=
  if exact = 0 then
    result = positiveZero
  else
    exact.natAbs < 2 ^ 23 ∧
      rawClass result = .subnormal ∧ finiteSignedScaled result = exact

/--
The exact finite multiplication fragment reachable by the singleton kernel. The numerator is
measured in `2^-298` units. A same-sign square whose magnitude is strictly below the `2^148`
halfway point rounds to `+0` under nearest-even. There is intentionally no general-product
fallthrough.
-/
def task0069RneProductBelowHalf
    (exactNumerator : Int) (zeroSign : Bool) (result : Word) : Prop :=
  exactNumerator.natAbs < 2 ^ 148 ∧
    zeroSign = false ∧ result = positiveZero


/--
Typed binary32 DIV used by the literal kernel. The divisor is explicitly the typed `+1`. The
finite equation is the exact arithmetic identity `x / 1 = x`; infinity preserves both class and
sign, and NaN remains NaN. This is a primitive result relation, not a requested VARIANCE result.
-/
def exactDivRne (left divisor rounded : Word) : Prop :=
  divisor = typedOne ∧
    match rawClass left with
    | .nan => rawClass rounded = .nan
    | .infinity =>
        rawClass rounded = .infinity ∧ rounded.sign = left.sign
    | .zero | .subnormal | .normal =>
        finiteWord rounded ∧
        finiteSignedScaled rounded = finiteSignedScaled left ∧
        rounded.sign = left.sign ∧
        rawClass rounded = rawClass left

/--
Typed binary32 SUB restricted exactly to values reachable at the singleton kernel's subtraction
site. NaN and infinity behavior is complete. A finite pair must have the same sign and either both
operands must be zero/subnormal or their exact signed values must cancel; the result then follows
the exact integer-in-`2^-149` relation above. The relation makes no assertion about unreachable
general finite subtraction.
-/
def exactSubRne (left right rounded : Word) : Prop :=
  if rawClass left = .nan ∨ rawClass right = .nan then
    rawClass rounded = .nan
  else if rawClass left = .infinity ∧ rawClass right = .infinity then
    if left.sign = right.sign then
      rawClass rounded = .nan
    else
      rawClass rounded = .infinity ∧ rounded.sign = left.sign
  else if rawClass left = .infinity then
    rawClass rounded = .infinity ∧ rounded.sign = left.sign
  else if rawClass right = .infinity then
    rawClass rounded = .infinity ∧ rounded.sign = !right.sign
  else
    let exact := finiteSignedScaled left - finiteSignedScaled right
    if left.sign = right.sign ∧
        ((SmallFinite left ∧ SmallFinite right) ∨ exact = 0) then
      task0069ExactSubValue exact rounded
    else
      False

/--
Typed binary32 MUL restricted exactly to values reachable at the singleton kernel's square site.
NaN/infinity behavior is complete. A finite pair must be same-sign zero/subnormal values, whose
exact product numerator is below the positive-zero halfway bound. The relation intentionally makes
no assertion about unreachable general finite products.
-/
def exactMulRne (left right rounded : Word) : Prop :=
  if rawClass left = .nan ∨ rawClass right = .nan then
    rawClass rounded = .nan
  else if
      (rawClass left = .infinity ∧ rawClass right = .zero) ∨
      (rawClass left = .zero ∧ rawClass right = .infinity) then
    rawClass rounded = .nan
  else if rawClass left = .infinity ∨ rawClass right = .infinity then
    rawClass rounded = .infinity ∧ rounded.sign = xor left.sign right.sign
  else if SmallFinite left ∧ SmallFinite right ∧ left.sign = right.sign then
    let exactNumerator := finiteSignedScaled left * finiteSignedScaled right
    task0069RneProductBelowHalf exactNumerator
      (xor left.sign right.sign) rounded
  else
    False

theorem daz_preserves_sign
    {source presented : Word}
    (site : daz source presented) :
    presented.sign = source.sign := by
  rcases site with exact | ⟨_, zero⟩
  · simp [exact]
  · simp [zero, signedZero]

theorem ftz_preserves_sign
    {rounded published : Word}
    (site : ftz rounded published) :
    published.sign = rounded.sign := by
  rcases site with exact | ⟨_, zero⟩
  · simp [exact]
  · simp [zero, signedZero]

theorem daz_preserves_small
    {source presented : Word}
    (site : daz source presented)
    (sourceSmall : SmallFinite source) :
    SmallFinite presented := by
  rcases daz_preserves_or_zero site with exact | zero
  · simpa [exact] using sourceSmall
  · exact Or.inl zero

theorem ftz_preserves_small
    {rounded published : Word}
    (site : ftz rounded published)
    (roundedSmall : SmallFinite rounded) :
    SmallFinite published := by
  rcases ftz_preserves_or_zero site with exact | zero
  · simpa [exact] using roundedSmall
  · exact Or.inl zero

theorem smallFinite_scaled_bound
    {word : Word}
    (small : SmallFinite word) :
    (finiteSignedScaled word).natAbs < 2 ^ 23 := by
  by_cases exp : word.exponent.val = 0
  · cases sign : word.sign <;>
      simp [finiteSignedScaled, finiteMagnitudeScaled, exp, sign]
  · by_cases max : word.exponent.val = 255
    · by_cases frac : word.fraction.val = 0
      · rcases small with zero | subnormal
        · simp [rawClass, max, frac] at zero
        · simp [rawClass, max, frac] at subnormal
      · rcases small with zero | subnormal
        · simp [rawClass, max, frac] at zero
        · simp [rawClass, max, frac] at subnormal
    · rcases small with zero | subnormal
      · simp [rawClass, exp, max] at zero
      · simp [rawClass, exp, max] at subnormal

theorem natSignedDifference_bound
    {left right bound : Nat}
    (leftBound : left < bound)
    (rightBound : right < bound) :
    (Int.ofNat left - Int.ofNat right).natAbs < bound := by
  by_cases order : left ≤ right
  · have cast :
        Int.ofNat (right - left) = Int.ofNat right - Int.ofNat left :=
      Int.ofNat_sub order
    have difference :
        Int.ofNat left - Int.ofNat right = -Int.ofNat (right - left) := by
      omega
    rw [difference]
    simp
    exact Nat.sub_lt_of_lt rightBound
  · have reverse : right ≤ left :=
      Nat.le_of_lt (Nat.lt_of_not_ge order)
    have difference :
        Int.ofNat left - Int.ofNat right = Int.ofNat (left - right) :=
      (Int.ofNat_sub reverse).symm
    rw [difference]
    simp
    exact Nat.sub_lt_of_lt leftBound

theorem smallFinite_difference_bound
    {left right : Word}
    (leftSmall : SmallFinite left)
    (rightSmall : SmallFinite right)
    (sameSign : left.sign = right.sign) :
    (finiteSignedScaled left - finiteSignedScaled right).natAbs < 2 ^ 23 := by
  have leftBound := smallFinite_scaled_bound leftSmall
  have rightBound := smallFinite_scaled_bound rightSmall
  cases leftSign : left.sign
  · have rightSign : right.sign = false := sameSign.symm.trans leftSign
    simp [finiteSignedScaled, leftSign, rightSign] at leftBound rightBound ⊢
    exact natSignedDifference_bound leftBound rightBound
  · have rightSign : right.sign = true := sameSign.symm.trans leftSign
    simp [finiteSignedScaled, leftSign, rightSign] at leftBound rightBound ⊢
    have reordered :
        -(finiteMagnitudeScaled left : Int) +
            (finiteMagnitudeScaled right : Int) =
          (finiteMagnitudeScaled right : Int) -
            (finiteMagnitudeScaled left : Int) := by
      omega
    rw [reordered]
    exact natSignedDifference_bound rightBound leftBound

theorem smallFinite_product_below_half
    {left right : Word}
    (leftSmall : SmallFinite left)
    (rightSmall : SmallFinite right) :
    (finiteSignedScaled left * finiteSignedScaled right).natAbs < 2 ^ 148 := by
  rw [Int.natAbs_mul]
  have leftBound := smallFinite_scaled_bound leftSmall
  have rightBound := smallFinite_scaled_bound rightSmall
  rcases Nat.eq_zero_or_pos (finiteSignedScaled right).natAbs with
    rightZero | rightPositive
  · simp [rightZero]
  · have first :
        (finiteSignedScaled left).natAbs * (finiteSignedScaled right).natAbs <
          2 ^ 23 * (finiteSignedScaled right).natAbs :=
        Nat.mul_lt_mul_of_pos_right leftBound rightPositive
    have second :
        2 ^ 23 * (finiteSignedScaled right).natAbs ≤ 2 ^ 23 * 2 ^ 23 :=
      Nat.mul_le_mul_left (2 ^ 23) (Nat.le_of_lt rightBound)
    calc
      (finiteSignedScaled left).natAbs * (finiteSignedScaled right).natAbs <
          2 ^ 23 * 2 ^ 23 := Nat.lt_of_lt_of_le first second
      _ = 2 ^ 46 := by rw [← Nat.pow_add]
      _ < 2 ^ 148 := by decide

theorem zeroWord_eq_signedZero
    {word : Word}
    (wordZero : rawClass word = .zero) :
    word = signedZero word.sign := by
  unfold rawClass at wordZero
  split at wordZero
  next exponentZero =>
    split at wordZero
    next fractionZero =>
      have exponentExact : word.exponent = 0 := Fin.eq_of_val_eq exponentZero
      have fractionExact : word.fraction = 0 := Fin.eq_of_val_eq fractionZero
      cases word
      simp [signedZero] at exponentExact fractionExact ⊢
      exact ⟨exponentExact, fractionExact⟩
    next => contradiction
  next =>
    split at wordZero
    next => split at wordZero <;> contradiction
    next => contradiction

theorem divSite_nan
    {source result : Word}
    (site : binary32PrimitiveSite exactDivRne source typedOne result)
    (sourceNan : rawClass source = .nan) :
    rawClass result = .nan := by
  rcases site with
    ⟨presentedSource, presentedOne, rounded,
      sourceDaz, oneDaz, primitive, resultFtz⟩
  have sourceExact : presentedSource = source :=
    daz_exact_of_not_subnormal sourceDaz (by simp [sourceNan])
  have oneExact : presentedOne = typedOne :=
    daz_exact_of_not_subnormal oneDaz (by simp)
  subst presentedSource
  subst presentedOne
  have roundedNan : rawClass rounded = .nan := by
    simpa [exactDivRne, sourceNan] using primitive
  have resultExact : result = rounded :=
    ftz_exact_of_not_subnormal resultFtz (by simp [roundedNan])
  simpa [resultExact] using roundedNan

theorem divSite_infinity
    {source result : Word}
    (site : binary32PrimitiveSite exactDivRne source typedOne result)
    (sourceInfinity : rawClass source = .infinity) :
    rawClass result = .infinity ∧ result.sign = source.sign := by
  rcases site with
    ⟨presentedSource, presentedOne, rounded,
      sourceDaz, oneDaz, primitive, resultFtz⟩
  have sourceExact : presentedSource = source :=
    daz_exact_of_not_subnormal sourceDaz (by simp [sourceInfinity])
  have oneExact : presentedOne = typedOne :=
    daz_exact_of_not_subnormal oneDaz (by simp)
  subst presentedSource
  subst presentedOne
  have roundedFacts :
      rawClass rounded = .infinity ∧ rounded.sign = source.sign := by
    simpa [exactDivRne, sourceInfinity] using primitive
  have resultExact : result = rounded :=
    ftz_exact_of_not_subnormal resultFtz (by simp [roundedFacts.1])
  simpa [resultExact] using roundedFacts

theorem divSite_finite_exact
    {source result : Word}
    (site : binary32PrimitiveSite exactDivRne source typedOne result)
    (sourceClass : rawClass source = .zero ∨ rawClass source = .normal) :
    finiteSignedScaled result = finiteSignedScaled source ∧
      result.sign = source.sign ∧ rawClass result = rawClass source := by
  rcases site with
    ⟨presentedSource, presentedOne, rounded,
      sourceDaz, oneDaz, primitive, resultFtz⟩
  have sourceNotSubnormal : rawClass source ≠ .subnormal := by
    rcases sourceClass with zero | normal
    · simp [zero]
    · simp [normal]
  have sourceExact : presentedSource = source :=
    daz_exact_of_not_subnormal sourceDaz sourceNotSubnormal
  have oneExact : presentedOne = typedOne :=
    daz_exact_of_not_subnormal oneDaz (by simp)
  subst presentedSource
  subst presentedOne
  rcases sourceClass with sourceZero | sourceNormal
  · have roundedFacts :
        finiteWord rounded ∧
        finiteSignedScaled rounded = finiteSignedScaled source ∧
        rounded.sign = source.sign ∧ rawClass rounded = rawClass source := by
      simpa [exactDivRne, sourceZero] using primitive
    have resultExact : result = rounded :=
      ftz_exact_of_not_subnormal resultFtz
        (by simp [roundedFacts.2.2.2, sourceZero])
    simpa [resultExact] using roundedFacts.2
  · have roundedFacts :
        finiteWord rounded ∧
        finiteSignedScaled rounded = finiteSignedScaled source ∧
        rounded.sign = source.sign ∧ rawClass rounded = rawClass source := by
      simpa [exactDivRne, sourceNormal] using primitive
    have resultExact : result = rounded :=
      ftz_exact_of_not_subnormal resultFtz
        (by simp [roundedFacts.2.2.2, sourceNormal])
    simpa [resultExact] using roundedFacts.2

theorem divSite_small
    {source result : Word}
    (site : binary32PrimitiveSite exactDivRne source typedOne result)
    (sourceSmall : SmallFinite source) :
    SmallFinite result ∧ result.sign = source.sign := by
  rcases site with
    ⟨presentedSource, presentedOne, rounded,
      sourceDaz, oneDaz, primitive, resultFtz⟩
  have oneExact : presentedOne = typedOne :=
    daz_exact_of_not_subnormal oneDaz (by simp)
  subst presentedOne
  have presentedSmall := daz_preserves_small sourceDaz sourceSmall
  have presentedSign := daz_preserves_sign sourceDaz
  rcases presentedSmall with presentedZero | presentedSubnormal
  · have roundedFacts :
        finiteWord rounded ∧
        finiteSignedScaled rounded = finiteSignedScaled presentedSource ∧
        rounded.sign = presentedSource.sign ∧
        rawClass rounded = rawClass presentedSource := by
      simpa [exactDivRne, presentedZero] using primitive
    exact ⟨ftz_preserves_small resultFtz
        (Or.inl (by simpa [presentedZero] using roundedFacts.2.2.2)),
      (ftz_preserves_sign resultFtz).trans
        (roundedFacts.2.2.1.trans presentedSign)⟩
  · have roundedFacts :
        finiteWord rounded ∧
        finiteSignedScaled rounded = finiteSignedScaled presentedSource ∧
        rounded.sign = presentedSource.sign ∧
        rawClass rounded = rawClass presentedSource := by
      simpa [exactDivRne, presentedSubnormal] using primitive
    exact ⟨ftz_preserves_small resultFtz
        (Or.inr (by simpa [presentedSubnormal] using roundedFacts.2.2.2)),
      (ftz_preserves_sign resultFtz).trans
        (roundedFacts.2.2.1.trans presentedSign)⟩

theorem subSite_nan_left
    {left right result : Word}
    (site : binary32PrimitiveSite exactSubRne left right result)
    (leftNan : rawClass left = .nan) :
    rawClass result = .nan := by
  rcases site with
    ⟨presentedLeft, presentedRight, rounded,
      leftDaz, rightDaz, primitive, resultFtz⟩
  have leftExact : presentedLeft = left :=
    daz_exact_of_not_subnormal leftDaz (by simp [leftNan])
  subst presentedLeft
  have roundedNan : rawClass rounded = .nan := by
    simpa [exactSubRne, leftNan] using primitive
  have resultExact : result = rounded :=
    ftz_exact_of_not_subnormal resultFtz (by simp [roundedNan])
  simpa [resultExact] using roundedNan

theorem subSite_infinity_self
    {left right result : Word}
    (site : binary32PrimitiveSite exactSubRne left right result)
    (leftInfinity : rawClass left = .infinity)
    (rightInfinity : rawClass right = .infinity)
    (sameSign : left.sign = right.sign) :
    rawClass result = .nan := by
  rcases site with
    ⟨presentedLeft, presentedRight, rounded,
      leftDaz, rightDaz, primitive, resultFtz⟩
  have leftExact : presentedLeft = left :=
    daz_exact_of_not_subnormal leftDaz (by simp [leftInfinity])
  have rightExact : presentedRight = right :=
    daz_exact_of_not_subnormal rightDaz (by simp [rightInfinity])
  subst presentedLeft
  subst presentedRight
  have roundedNan : rawClass rounded = .nan := by
    simpa [exactSubRne, leftInfinity, rightInfinity, sameSign] using primitive
  have resultExact : result = rounded :=
    ftz_exact_of_not_subnormal resultFtz (by simp [roundedNan])
  simpa [resultExact] using roundedNan

theorem subSite_exact_zero
    {left right result : Word}
    (site : binary32PrimitiveSite exactSubRne left right result)
    (leftNotSubnormal : rawClass left ≠ .subnormal)
    (rightNotSubnormal : rawClass right ≠ .subnormal)
    (leftFinite : finiteWord left)
    (rightFinite : finiteWord right)
    (sameSign : left.sign = right.sign)
    (sameScaled : finiteSignedScaled left = finiteSignedScaled right) :
    result = positiveZero := by
  rcases site with
    ⟨presentedLeft, presentedRight, rounded,
      leftDaz, rightDaz, primitive, resultFtz⟩
  have leftExact : presentedLeft = left :=
    daz_exact_of_not_subnormal leftDaz leftNotSubnormal
  have rightExact : presentedRight = right :=
    daz_exact_of_not_subnormal rightDaz rightNotSubnormal
  subst presentedLeft
  subst presentedRight
  have exactZero :
      finiteSignedScaled left - finiteSignedScaled right = 0 := by
    simp [sameScaled]
  have roundedZero : rounded = positiveZero := by
    simpa [exactSubRne, task0069ExactSubValue, finiteWord, leftFinite.1,
      leftFinite.2, rightFinite.1, rightFinite.2, sameSign, exactZero] using primitive
  have resultExact : result = rounded :=
    ftz_exact_of_not_subnormal resultFtz
      (by simp [roundedZero, positiveZero])
  exact resultExact.trans roundedZero

theorem subSite_small_same_sign
    {left right result : Word}
    (site : binary32PrimitiveSite exactSubRne left right result)
    (leftSmall : SmallFinite left)
    (rightSmall : SmallFinite right)
    (sameSign : left.sign = right.sign) :
    SmallFinite result := by
  rcases site with
    ⟨presentedLeft, presentedRight, rounded,
      leftDaz, rightDaz, primitive, resultFtz⟩
  have presentedLeftSmall := daz_preserves_small leftDaz leftSmall
  have presentedRightSmall := daz_preserves_small rightDaz rightSmall
  have presentedSameSign :
      presentedLeft.sign = presentedRight.sign := by
    rw [daz_preserves_sign leftDaz, daz_preserves_sign rightDaz, sameSign]
  have leftNotNan : rawClass presentedLeft ≠ .nan := by
    rcases presentedLeftSmall with zero | subnormal
    · simp [zero]
    · simp [subnormal]
  have rightNotNan : rawClass presentedRight ≠ .nan := by
    rcases presentedRightSmall with zero | subnormal
    · simp [zero]
    · simp [subnormal]
  have leftNotInfinity : rawClass presentedLeft ≠ .infinity := by
    rcases presentedLeftSmall with zero | subnormal
    · simp [zero]
    · simp [subnormal]
  have rightNotInfinity : rawClass presentedRight ≠ .infinity := by
    rcases presentedRightSmall with zero | subnormal
    · simp [zero]
    · simp [subnormal]
  have exactBound :
      (finiteSignedScaled presentedLeft - finiteSignedScaled presentedRight).natAbs <
        2 ^ 23 :=
    smallFinite_difference_bound presentedLeftSmall presentedRightSmall
      presentedSameSign
  by_cases exactZero :
      finiteSignedScaled presentedLeft - finiteSignedScaled presentedRight = 0
  · have roundedZero : rounded = positiveZero := by
      simpa [exactSubRne, task0069ExactSubValue, leftNotNan, rightNotNan,
        leftNotInfinity, rightNotInfinity, presentedLeftSmall,
        presentedRightSmall, presentedSameSign, exactZero] using primitive
    exact ftz_preserves_small resultFtz
      (Or.inl (by simp [roundedZero, positiveZero]))
  · have roundedSubnormal : rawClass rounded = .subnormal := by
      have facts :
          rawClass rounded = .subnormal ∧
            finiteSignedScaled rounded =
              finiteSignedScaled presentedLeft -
                finiteSignedScaled presentedRight := by
        simpa [exactSubRne, task0069ExactSubValue, leftNotNan, rightNotNan,
          leftNotInfinity, rightNotInfinity, presentedLeftSmall,
          presentedRightSmall, presentedSameSign, exactZero, exactBound] using primitive
      exact facts.1
    exact ftz_preserves_small resultFtz (Or.inr roundedSubnormal)

theorem mulSite_nan_left
    {left right result : Word}
    (site : binary32PrimitiveSite exactMulRne left right result)
    (leftNan : rawClass left = .nan) :
    rawClass result = .nan := by
  rcases site with
    ⟨presentedLeft, presentedRight, rounded,
      leftDaz, rightDaz, primitive, resultFtz⟩
  have leftExact : presentedLeft = left :=
    daz_exact_of_not_subnormal leftDaz (by simp [leftNan])
  subst presentedLeft
  have roundedNan : rawClass rounded = .nan := by
    simpa [exactMulRne, leftNan] using primitive
  have resultExact : result = rounded :=
    ftz_exact_of_not_subnormal resultFtz (by simp [roundedNan])
  simpa [resultExact] using roundedNan

theorem mulSite_small
    {left right result : Word}
    (site : binary32PrimitiveSite exactMulRne left right result)
    (leftSmall : SmallFinite left)
    (rightSmall : SmallFinite right)
    (sameSign : left.sign = right.sign) :
    result = positiveZero := by
  rcases site with
    ⟨presentedLeft, presentedRight, rounded,
      leftDaz, rightDaz, primitive, resultFtz⟩
  have presentedLeftSmall := daz_preserves_small leftDaz leftSmall
  have presentedRightSmall := daz_preserves_small rightDaz rightSmall
  have presentedSameSign :
      presentedLeft.sign = presentedRight.sign := by
    rw [daz_preserves_sign leftDaz, daz_preserves_sign rightDaz, sameSign]
  have leftNotNan : rawClass presentedLeft ≠ .nan := by
    rcases presentedLeftSmall with zero | subnormal
    · simp [zero]
    · simp [subnormal]
  have rightNotNan : rawClass presentedRight ≠ .nan := by
    rcases presentedRightSmall with zero | subnormal
    · simp [zero]
    · simp [subnormal]
  have leftNotInfinity : rawClass presentedLeft ≠ .infinity := by
    rcases presentedLeftSmall with zero | subnormal
    · simp [zero]
    · simp [subnormal]
  have rightNotInfinity : rawClass presentedRight ≠ .infinity := by
    rcases presentedRightSmall with zero | subnormal
    · simp [zero]
    · simp [subnormal]
  have expectedBound :=
    smallFinite_product_below_half presentedLeftSmall presentedRightSmall
  have roundedZero : rounded = positiveZero := by
    simpa [exactMulRne, task0069RneProductBelowHalf, leftNotNan, rightNotNan,
      leftNotInfinity, rightNotInfinity, presentedLeftSmall,
      presentedRightSmall, presentedSameSign, expectedBound] using primitive
  have resultExact : result = rounded :=
    ftz_exact_of_not_subnormal resultFtz
      (by simp [roundedZero, positiveZero])
  exact resultExact.trans roundedZero

/-- The literal four-op source sequence compiled by `variance_f32_0069`. -/
inductive SingletonVarianceLiteral where
  | divBy (divisor : Word)
  | subSourceMean
  | mulDifferenceSelf
  deriving DecidableEq, Repr

def singletonVarianceLiteralSource : List SingletonVarianceLiteral :=
  [.divBy typedOne, .subSourceMean, .mulDifferenceSelf, .divBy typedOne]

theorem singletonVarianceLiteralSource_exact :
    singletonVarianceLiteralSource =
      [.divBy typedOne, .subSourceMean, .mulDifferenceSelf, .divBy typedOne] := rfl

/-- Exact DIV(x,+1), SUB(x,mean), MUL(d,d), DIV(square,+1) primitive-site trace. -/
structure SingletonVarianceTrace (source result : Word) where
  mean : Word
  difference : Word
  square : Word
  meanDivision :
    binary32PrimitiveSite exactDivRne source typedOne mean
  differenceSubtraction :
    binary32PrimitiveSite exactSubRne source mean difference
  squareMultiplication :
    binary32PrimitiveSite exactMulRne difference difference square
  finalDivision :
    binary32PrimitiveSite exactDivRne square typedOne result

/-- NaN and either signed infinity follow the literal primitive trace to a NaN result. -/
theorem singletonVariance_special_is_nan
    {source result : Word}
    (sourceSpecial :
      rawClass source = .nan ∨ rawClass source = .infinity)
    (trace : SingletonVarianceTrace source result) :
    rawClass result = .nan := by
  rcases trace with
    ⟨mean, difference, square, meanSite, differenceSite, squareSite, finalSite⟩
  rcases sourceSpecial with sourceNan | sourceInfinity
  · have differenceNan := subSite_nan_left differenceSite sourceNan
    have squareNan := mulSite_nan_left squareSite differenceNan
    exact divSite_nan finalSite squareNan
  · have meanInfinity := divSite_infinity meanSite sourceInfinity
    have differenceNan := subSite_infinity_self differenceSite
      sourceInfinity meanInfinity.1 meanInfinity.2.symm
    have squareNan := mulSite_nan_left squareSite differenceNan
    exact divSite_nan finalSite squareNan

def binary32Finite (word : Word) : Prop :=
  rawClass word ≠ .nan ∧ rawClass word ≠ .infinity

theorem divSite_positiveZero
    {result : Word}
    (site : binary32PrimitiveSite exactDivRne positiveZero typedOne result) :
    result = positiveZero := by
  have facts := divSite_finite_exact site
    (Or.inl (by simp [positiveZero]))
  have resultZero : rawClass result = .zero := by
    simpa [positiveZero] using facts.2.2
  calc
    result = signedZero result.sign := zeroWord_eq_signedZero resultZero
    _ = positiveZero := by
      rw [facts.2.1]
      rfl

/-- Every finite binary32 singleton, including signed zeros and every DAZ/FTZ choice, yields +0. -/
theorem singletonVariance_finite_is_positiveZero
    {source result : Word}
    (sourceFinite : binary32Finite source)
    (trace : SingletonVarianceTrace source result) :
    result = positiveZero := by
  rcases trace with
    ⟨mean, difference, square, meanSite, differenceSite, squareSite, finalSite⟩
  cases sourceClass : rawClass source with
  | zero =>
      have meanFacts := divSite_finite_exact meanSite (Or.inl sourceClass)
      have differenceZero := subSite_exact_zero differenceSite
        (by simp [sourceClass])
        (by simp [meanFacts.2.2, sourceClass])
        ⟨by simp [sourceClass], by simp [sourceClass]⟩
        ⟨by simp [meanFacts.2.2, sourceClass],
          by simp [meanFacts.2.2, sourceClass]⟩
        meanFacts.2.1.symm
        meanFacts.1.symm
      rw [differenceZero] at squareSite
      have squareZero := mulSite_small squareSite
        (Or.inl (by simp [positiveZero])) (Or.inl (by simp [positiveZero])) rfl
      rw [squareZero] at finalSite
      exact divSite_positiveZero finalSite
  | subnormal =>
      have meanSmall := divSite_small meanSite (Or.inr sourceClass)
      have differenceSmall := subSite_small_same_sign differenceSite
        (Or.inr sourceClass) meanSmall.1 meanSmall.2.symm
      have squareZero :=
        mulSite_small squareSite differenceSmall differenceSmall rfl
      rw [squareZero] at finalSite
      exact divSite_positiveZero finalSite
  | normal =>
      have meanFacts := divSite_finite_exact meanSite (Or.inr sourceClass)
      have differenceZero := subSite_exact_zero differenceSite
        (by simp [sourceClass])
        (by simp [meanFacts.2.2, sourceClass])
        ⟨by simp [sourceClass], by simp [sourceClass]⟩
        ⟨by simp [meanFacts.2.2, sourceClass],
          by simp [meanFacts.2.2, sourceClass]⟩
        meanFacts.2.1.symm
        meanFacts.1.symm
      rw [differenceZero] at squareSite
      have squareZero := mulSite_small squareSite
        (Or.inl (by simp [positiveZero])) (Or.inl (by simp [positiveZero])) rfl
      rw [squareZero] at finalSite
      exact divSite_positiveZero finalSite
  | infinity =>
      exact False.elim (sourceFinite.2 sourceClass)
  | nan =>
      exact False.elim (sourceFinite.1 sourceClass)
end Task0069
