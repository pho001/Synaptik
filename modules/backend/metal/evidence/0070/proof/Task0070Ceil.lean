import Task0069Binary32

namespace Task0070Ceil

open Task0069

def rawBits (word : Word) : Nat := (Task0069.rawWord word).val

def ceilIdentityClass (word : Word) : Bool :=
  word.exponent.val == 255 ||
    word.exponent.val * 8388608 + word.fraction.val == 0

/-- Literal raw-u32 semantics of the frozen MSL `ceil_bits` helper. -/
def ceilBitsSource (word : Word) : Nat :=
  if ceilIdentityClass word then rawBits word
  else if word.exponent.val < 127 then
    if word.sign then 0x80000000 else 0x3f800000
  else if 150 ≤ word.exponent.val then rawBits word
  else
    let unit := 2 ^ (150 - word.exponent.val)
    let truncated :=
      (if word.sign then 0x80000000 else 0) +
      word.exponent.val * 8388608 +
      (word.fraction.val / unit) * unit
    if word.fraction.val % unit = 0 then rawBits word
    else if word.sign then truncated else truncated + unit

/-- Result-set model for mathematical ceil on every binary32 raw class. -/
def CeilModel (word : Word) (result : Nat) : Prop :=
  if ceilIdentityClass word then result = rawBits word
  else if word.exponent.val < 127 then
    result = if word.sign then 0x80000000 else 0x3f800000
  else if 150 ≤ word.exponent.val then result = rawBits word
  else
    let quantum := 2 ^ (150 - word.exponent.val)
    let integral :=
      (if word.sign then 0x80000000 else 0) +
      word.exponent.val * 8388608 +
      (word.fraction.val / quantum) * quantum
    if word.fraction.val % quantum = 0 then result = rawBits word
    else result = if word.sign then integral else integral + quantum

theorem ceil_universal_raw_certificate (word : Word) :
    CeilModel word (ceilBitsSource word) := by
  unfold CeilModel ceilBitsSource
  dsimp
  by_cases identity : ceilIdentityClass word = true
  · simp [identity]
  · by_cases subunit : word.exponent.val < 127
    · simp [identity, subunit]
    · by_cases integralRange : 150 ≤ word.exponent.val
      · simp [identity, subunit, integralRange]
      · by_cases alreadyIntegral :
          word.fraction.val % 2 ^ (150 - word.exponent.val) = 0
        · simp [identity, subunit, integralRange, alreadyIntegral]
        · simp [identity, subunit, integralRange, alreadyIntegral]

theorem ceil_permitted_iff (word : Word) (result : Nat) :
    CeilModel word result ↔ result = ceilBitsSource word := by
  constructor
  · intro allowed
    unfold CeilModel at allowed
    unfold ceilBitsSource
    dsimp at allowed ⊢
    by_cases identity : ceilIdentityClass word = true
    · simp [identity] at allowed ⊢
      exact allowed
    · by_cases subunit : word.exponent.val < 127
      · simp [identity, subunit] at allowed ⊢
        exact allowed
      · by_cases integralRange : 150 ≤ word.exponent.val
        · simp [identity, subunit, integralRange] at allowed ⊢
          exact allowed
        · by_cases alreadyIntegral :
            word.fraction.val % 2 ^ (150 - word.exponent.val) = 0
          · simp [identity, subunit, integralRange, alreadyIntegral] at allowed ⊢
            exact allowed
          · simp [identity, subunit, integralRange, alreadyIntegral] at allowed ⊢
            exact allowed
  · intro equality
    rw [equality]
    exact ceil_universal_raw_certificate word

theorem ceil_all_raw32_certificate (raw : Raw32) :
    CeilModel (wordOfRaw raw) (ceilBitsSource (wordOfRaw raw)) :=
  ceil_universal_raw_certificate (wordOfRaw raw)

theorem ceil_special_raw_identity (word : Word)
    (special : word.exponent.val = 255) :
    ceilBitsSource word = rawBits word := by
  simp [ceilBitsSource, ceilIdentityClass, special]

theorem ceil_zero_raw_identity (word : Word)
    (zero : word.exponent.val = 0)
    (fraction : word.fraction.val = 0) :
    ceilBitsSource word = rawBits word := by
  simp [ceilBitsSource, ceilIdentityClass, zero, fraction]

end Task0070Ceil
