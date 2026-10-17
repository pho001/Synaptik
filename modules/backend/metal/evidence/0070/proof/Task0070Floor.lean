import Task0069Binary32

namespace Task0070Floor

open Task0069

def rawBits (word : Word) : Nat := (Task0069.rawWord word).val

def floorIdentityClass (word : Word) : Bool :=
  word.exponent.val == 255 ||
    word.exponent.val * 8388608 + word.fraction.val == 0

/-- Literal raw-u32 semantics of the frozen MSL `floor_bits` helper. -/
def floorBitsSource (word : Word) : Nat :=
  if floorIdentityClass word then rawBits word
  else if word.exponent.val < 127 then
    if word.sign then 0xbf800000 else 0
  else if 150 ≤ word.exponent.val then rawBits word
  else
    let unit := 2 ^ (150 - word.exponent.val)
    let truncated :=
      (if word.sign then 0x80000000 else 0) +
      word.exponent.val * 8388608 +
      (word.fraction.val / unit) * unit
    if word.fraction.val % unit = 0 then rawBits word
    else if word.sign then truncated + unit else truncated

/-- Result-set model for mathematical floor on every binary32 raw class. -/
def FloorModel (word : Word) (result : Nat) : Prop :=
  if floorIdentityClass word then result = rawBits word
  else if word.exponent.val < 127 then
    result = if word.sign then 0xbf800000 else 0
  else if 150 ≤ word.exponent.val then result = rawBits word
  else
    let quantum := 2 ^ (150 - word.exponent.val)
    let integral :=
      (if word.sign then 0x80000000 else 0) +
      word.exponent.val * 8388608 +
      (word.fraction.val / quantum) * quantum
    if word.fraction.val % quantum = 0 then result = rawBits word
    else result = if word.sign then integral + quantum else integral

/-- The exact helper result belongs to the model for every structured word. -/
theorem floor_universal_raw_certificate (word : Word) :
    FloorModel word (floorBitsSource word) := by
  unfold FloorModel floorBitsSource
  dsimp
  by_cases identity : floorIdentityClass word = true
  · simp [identity]
  · by_cases subunit : word.exponent.val < 127
    · simp [identity, subunit]
    · by_cases integralRange : 150 ≤ word.exponent.val
      · simp [identity, subunit, integralRange]
      · by_cases alreadyIntegral :
          word.fraction.val % 2 ^ (150 - word.exponent.val) = 0
        · simp [identity, subunit, integralRange, alreadyIntegral]
        · simp [identity, subunit, integralRange, alreadyIntegral]

/-- The permitted set is exactly the singleton containing the helper result. -/
theorem floor_permitted_iff (word : Word) (result : Nat) :
    FloorModel word result ↔ result = floorBitsSource word := by
  constructor
  · intro allowed
    unfold FloorModel at allowed
    unfold floorBitsSource
    dsimp at allowed ⊢
    by_cases identity : floorIdentityClass word = true
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
    exact floor_universal_raw_certificate word

/-- Quantification over `Raw32` plus the shared bijection covers all 2^32 raw words. -/
theorem floor_all_raw32_certificate (raw : Raw32) :
    FloorModel (wordOfRaw raw) (floorBitsSource (wordOfRaw raw)) :=
  floor_universal_raw_certificate (wordOfRaw raw)

/-- Every NaN payload/sign and both infinities are returned bit-for-bit. -/
theorem floor_special_raw_identity (word : Word)
    (special : word.exponent.val = 255) :
    floorBitsSource word = rawBits word := by
  simp [floorBitsSource, floorIdentityClass, special]

/-- Both signed zeros are returned bit-for-bit. -/
theorem floor_zero_raw_identity (word : Word)
    (zero : word.exponent.val = 0)
    (fraction : word.fraction.val = 0) :
    floorBitsSource word = rawBits word := by
  simp [floorBitsSource, floorIdentityClass, zero, fraction]

end Task0070Floor
