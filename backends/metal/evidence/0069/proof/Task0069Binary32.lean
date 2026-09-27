namespace Task0069

/-- A structured presentation of every binary32 raw word. -/
structure Word where
  sign : Bool
  exponent : Fin 256
  fraction : Fin 8388608
  deriving DecidableEq, Repr

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

/-- Explicit class obligations for the primitive RNE addition function. -/
structure RneSpecialClassContract (rne : Word → Word → Word) : Prop where
  nanLeft : ∀ left right, rawClass left = .nan → rawClass (rne left right) = .nan
  nanRight : ∀ left right, rawClass right = .nan → rawClass (rne left right) = .nan
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

theorem addSite_nan_left
    {rne : Word → Word → Word}
    (contract : RneSpecialClassContract rne)
    {left right published : Word}
    (site : addSite rne left right published)
    (leftNan : rawClass left = .nan) :
    rawClass published = .nan := by
  rcases site with ⟨presentedLeft, presentedRight, rounded,
    leftDaz, _, roundedEq, publishedFtz⟩
  have leftExact : presentedLeft = left := daz_exact_of_not_subnormal leftDaz (by
    intro subnormal
    have impossible : RawClass.nan = RawClass.subnormal := leftNan.symm.trans subnormal
    cases impossible)
  subst presentedLeft
  have roundedNan : rawClass rounded = .nan := by
    rw [roundedEq]
    exact contract.nanLeft left presentedRight leftNan
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
    (rightNan : rawClass right = .nan) :
    rawClass published = .nan := by
  rcases site with ⟨presentedLeft, presentedRight, rounded,
    _, rightDaz, roundedEq, publishedFtz⟩
  have rightExact : presentedRight = right := daz_exact_of_not_subnormal rightDaz (by
    intro subnormal
    have impossible : RawClass.nan = RawClass.subnormal := rightNan.symm.trans subnormal
    cases impossible)
  subst presentedRight
  have roundedNan : rawClass rounded = .nan := by
    rw [roundedEq]
    exact contract.nanRight presentedLeft right rightNan
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

end Task0069
