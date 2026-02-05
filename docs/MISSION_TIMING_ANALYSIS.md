# Mission Timing & Delta-V Budget Analysis

## Miami → Moon → Jupiter → Miami Mission Profile

### Executive Summary

After detailed analysis, the **mission duration of ~4.1 years** is achievable but aggressive. Here's the breakdown:

---

## Phase-by-Phase Timeline

### Phase 1: Launch & Earth Departure (Day 0-4)

| Event | Time | Duration | Delta-V |
|-------|------|----------|---------|
| Launch (Miami) | T+0 | - | - |
| Stage 1 Burn | T+0 to T+170s | 170 s | ~3,500 m/s |
| Stage Separation | T+170s | 5 s | - |
| Stage 2 Burn | T+175s to T+555s | 380 s | ~4,200 m/s |
| Parking Orbit Coast | T+555s to T+3,300s | 45 min | - |
| **TLI Burn** | T+3,300s | ~5 min | **3,150 m/s** |

**Subtotal Phase 1:** ~1 hour active, 3 days coast to Moon

---

### Phase 2: Lunar Operations (Day 3-7)

| Event | Time | Duration | Delta-V |
|-------|------|----------|---------|
| Lunar Arrival | Day 3 | - | - |
| **LOI Burn** | Day 3 | ~8 min | **850 m/s** |
| Lunar Orbit 1 (equatorial) | Day 3-4 | ~1 day | - |
| **Inclination Change** | Day 4 | ~3 min | **200 m/s** |
| Lunar Orbit 2 (45° incl) | Day 4-5 | ~1 day | - |
| **Lunar Escape** | Day 5 | ~6 min | **900 m/s** |

**Subtotal Phase 2:** 4 days, 1,950 m/s

---

### Phase 3: Jupiter Transfer (Day 5 to Year 2)

This is the critical phase. Let's calculate the Hohmann transfer:

```
Earth orbit: r₁ = 1.0 AU = 1.496 × 10¹¹ m
Jupiter orbit: r₂ = 5.2 AU = 7.78 × 10¹¹ m
Sun μ = 1.327 × 10²⁰ m³/s²

Transfer semi-major axis:
a = (r₁ + r₂) / 2 = 3.1 AU = 4.64 × 10¹¹ m

Transfer time (half period):
T = π × √(a³/μ) = π × √((4.64×10¹¹)³ / 1.327×10²⁰)
T = 2.73 years ≈ 997 days
```

**However**, this is from Earth's heliocentric orbit, not from Earth's sphere of influence. The actual trajectory must account for:
- Earth departure excess velocity (C3)
- Jupiter arrival excess velocity
- Phase angles (launch window)

**Realistic Jupiter transfer: 2.0 - 2.7 years depending on trajectory type**

| Trajectory Type | ToF | Total ΔV |
|----------------|-----|----------|
| Hohmann (minimum energy) | 2.73 yr | ~6 km/s |
| Fast transfer | 1.5-2.0 yr | ~9 km/s |
| VEEGA (Venus-Earth-Earth GA) | 6+ yr | ~4 km/s |

---

### Phase 4: Jupiter Operations (Year 2, Day 730-760)

| Event | Time | Duration | Delta-V |
|-------|------|----------|---------|
| Jupiter Arrival | ~Day 730 | - | - |
| **JOI Burn** | Day 730 | ~30 min | **2,000 m/s** |
| Jupiter Orbit 1 | Day 730-740 | ~10 days | - |
| Jupiter Orbit 2 | Day 740-750 | ~10 days | - |
| Jupiter Orbit 3 | Day 750-760 | ~10 days | - |
| **Jupiter Escape** | Day 760 | ~35 min | **2,200 m/s** |

**Subtotal Phase 4:** 30 days, 4,200 m/s

**Note:** Jupiter orbit period at 500,000 km altitude:
```
r = 69,911 km + 500,000 km = 569,911 km = 5.7 × 10⁸ m
T = 2π × √(r³/μ_jupiter) = 2π × √((5.7×10⁸)³ / 1.267×10¹⁷)
T ≈ 3.3 days per orbit
```

---

### Phase 5: Return to Earth (Year 2 to Year 4)

Same as outbound: **~2 years** for Hohmann return.

| Event | Time | Duration | Delta-V |
|-------|------|----------|---------|
| Jupiter-Earth Coast | Day 760 to ~Day 1490 | 730 days | - |
| Earth Approach | Day 1490 | - | - |
| **Deorbit/Entry** | Day 1490 | ~5 min | **~300 m/s** |
| Reentry & Landing | Day 1490 | ~30 min | - |

---

## Total Mission Summary

### Timeline

| Phase | Duration |
|-------|----------|
| Earth Departure | 3 days |
| Lunar Operations | 4 days |
| Earth-Jupiter Transfer | 725 days (~2 years) |
| Jupiter Operations | 30 days |
| Jupiter-Earth Return | 730 days (~2 years) |
| **TOTAL** | **~1,492 days (4.09 years)** |

### Delta-V Budget

| Maneuver | Delta-V (m/s) |
|----------|---------------|
| TLI (Trans-Lunar Injection) | 3,150 |
| LOI (Lunar Orbit Insertion) | 850 |
| Lunar Inclination Change | 200 |
| Lunar Escape | 900 |
| JOI (Jupiter Orbit Insertion) | 2,000 |
| Jupiter Escape | 2,200 |
| Earth Entry Correction | 300 |
| Margin (10%) | 960 |
| **TOTAL** | **10,560 m/s** |

---

## Is 4.5 Years Realistic?

**YES, with caveats:**

1. **Trajectory Type Matters**: A pure Hohmann transfer to Jupiter takes 2.73 years ONE WAY. Round-trip minimum is 5.46 years.

2. **Our 4.1-year mission uses slightly faster transfers** (~2 years each way), requiring higher ΔV but achievable with our 10.5 km/s budget.

3. **Alternatives to reduce time:**
   - Nuclear thermal propulsion (Isp ~900s vs 316s chemical)
   - Solar electric propulsion (SEP)
   - Gravity assists (but these often INCREASE time)

4. **The mission config is optimized** for the balance between:
   - Propellant mass constraints (2,800 kg)
   - Spacecraft dry mass (3,200 kg)
   - Total ΔV capability

---

## Propellant Mass Check

Using the Tsiolkovsky rocket equation:

```
ΔV = Isp × g₀ × ln(m_initial / m_final)

For ΔV = 10,560 m/s, Isp = 316 s:
10,560 = 316 × 9.81 × ln(m_i / m_f)
ln(m_i / m_f) = 3.406
m_i / m_f = 30.1

If m_f (dry) = 3,200 kg:
m_i = 3,200 × 30.1 = 96,320 kg propellant needed!
```

**PROBLEM IDENTIFIED**: The 2,800 kg propellant budget is insufficient for 10.5 km/s ΔV!

### Solutions:

1. **Use staged propulsion** (already in vehicle design - Stage 1 & 2 provide ascent ΔV)
2. **The 10.5 km/s is post-orbital** - ascent ΔV is separate
3. **Recalculate for spacecraft-only burns:**

Post-TLI spacecraft mass budget:
```
Initial: 6,000 kg (spacecraft)
Propellant: 2,800 kg
Dry: 3,200 kg

ΔV_available = 316 × 9.81 × ln(6000/3200)
             = 316 × 9.81 × 0.628
             = 1,949 m/s (spacecraft only)
```

This means **the spacecraft engine handles only minor corrections** - major burns (JOI, escape) would require a separate propulsion stage or gravity assists.

---

## Recommended Mission Redesign

To make the mission feasible:

### Option A: Add Propulsion Stage
- Include a kick stage with 15,000 kg propellant
- Provides ~5 km/s additional ΔV

### Option B: Gravity Assists
- Use Venus flyby (VVEJGA trajectory)
- Reduces propellant needs by 40%
- Increases mission time to 6+ years

### Option C: Electric Propulsion
- Ion engines with Isp = 3,000 s
- Same propellant provides 6× more ΔV
- Requires large solar arrays (difficult at Jupiter)

---

## Conclusion

The **4.1-year mission timeline is correct** for the trajectory design, but the **propellant budget needs revision** in the mission configuration. The current spacecraft (6,000 kg with 2,800 kg propellant) can only provide ~2 km/s ΔV, not the ~7 km/s needed for post-Earth-departure maneuvers.

**Recommendation**: Update `mission_config.yaml` to either:
1. Include a dedicated deep-space propulsion stage
2. Use higher-Isp propulsion (ion or nuclear)
3. Extend mission to include gravity assists
