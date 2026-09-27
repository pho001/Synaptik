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
    rawClass (rne left right) = .infinity
  positiveInfinityRight : ∀ left right,
    right.sign = false → rawClass right = .infinity →
    left.sign = false → rawClass left ≠ .nan →
    rawClass (rne left right) = .infinity
  nonnegativeClosure : ∀ left right,
    left.sign = false → right.sign = false →
    rawClass left ≠ .nan → rawClass right ≠ .nan →
    (rne left right).sign = false ∨ rawClass (rne left right) = .zero

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

theorem addSite_nonempty (rne : Word → Word → Word) (left right : Word) :
    ∃ published, addSite rne left right published := by
  refine ⟨rne left right, left, right, rne left right, ?_⟩
  exact ⟨Or.inl rfl, Or.inl rfl, rfl, Or.inl rfl⟩

/-- The raw ABS leaf transform is total over zero, subnormal, normal, infinity, and NaN. -/
theorem absWord_class_contract (word : Word) :
    (absWord word).sign = false ∧ rawClass (absWord word) = rawClass word := by
  exact ⟨rfl, rfl⟩

end Task0069
