
================================================================================
SDCG ATOM INTERFEROMETRY EXPERIMENT: DETAILED PROTOCOL
================================================================================

EXPERIMENT OVERVIEW
-------------------
Goal: Measure density-dependent gravitational screening predicted by SDCG
Method: Lock-in detection of oscillating acceleration from rotating attractor
Target sensitivity: δ(ΔG/G) ~ 10^{-12} (signal ~10^{-9})
Expected outcome: Clear detection (SNR = 2114)

================================================================================
SECTION 1: ATTRACTOR DESIGN
================================================================================

1.1 GEOMETRY
------------
• Configuration: Rotating cylinder with alternating density sectors
• Radius: 10.0 cm
• Height: 20.0 cm
• Number of sector pairs: 4

1.2 MATERIALS
-------------
• High-density sectors: Tungsten
  - Density: 19300 kg/m³
  - SDCG screening factor: S = 0.92
  
• Low-density sectors: Aluminum  
  - Density: 2700 kg/m³
  - SDCG screening factor: S = 0.15

• Mass per material: W = 60.63 kg, Al = 8.48 kg
• Total attractor mass: 69.12 kg

1.3 ROTATION
------------
• Rotation frequency: 10 Hz
• Signal frequency: 40.0 Hz (rotation × sectors)
• Distance from atoms: 15.0 cm

================================================================================
SECTION 2: ATOM INTERFEROMETER
================================================================================

2.1 ATOMIC SOURCE
-----------------
• Species: Rb-87
• Cloud temperature: 100 nK
• Atom number: 1e+05
• Preparation: Laser cooling + evaporative cooling in ODT

2.2 INTERFEROMETER GEOMETRY
---------------------------
• Configuration: Dual gradiometer (differential measurement)
• Baseline: 10.0 cm
• Interrogation time: 100 ms
• Pulse sequence: 2-photon Bragg transitions

2.3 SENSITIVITY
---------------
• Shot-noise limited phase: 3.16e-03 rad/√shot
• Single-shot acceleration: 1.96e-08 m/s²
• Integrated (over 100 hours): 1.79e-11 m/s²
• Fractional sensitivity: δ(G/G) = 1.8e-12

================================================================================
SECTION 3: SDCG SIGNAL
================================================================================

3.1 EXPECTED SIGNAL
-------------------
• Mean gravitational acceleration: 1.03e-07 m/s²
• SDCG coupling difference: Δμ_eff = 0.370
• Oscillating signal amplitude: 3.79e-08 m/s²
• Fractional effect: ΔG/G = 3.7e-01

3.2 DETECTION
-------------
• Signal frequency: 40.0 Hz
• Lock-in bandwidth: 0.01 Hz (narrow to suppress noise)
• Expected SNR: 2114
• Detection significance: >423σ in 100 hours

================================================================================
SECTION 4: MEASUREMENT SEQUENCE
================================================================================

PHASE 1: BASELINE CHARACTERIZATION (Week 1-2)
----------------------------------------------
□ Measure static gravitational signal from attractor at rest
□ Characterize lab gravity gradients
□ Calibrate interferometer with known masses
□ Verify shot-noise limited operation

PHASE 2: ROTATION TESTS (Week 3-4)
-----------------------------------
□ Spin up attractor to 10 Hz
□ Lock to rotation signal at 40.0 Hz
□ Measure classical gravitational oscillation (non-SDCG)
□ Verify absence of mechanical coupling

PHASE 3: SDCG MEASUREMENT (Week 5-8)
-------------------------------------
□ Long integration at signal frequency
□ Record 100 hours of data
□ Extract oscillation amplitude
□ Compare with SDCG prediction

PHASE 4: SYSTEMATIC CHECKS (Week 9-12)
---------------------------------------
□ Reverse rotation direction (should not change signal)
□ Swap attractor materials (signal should flip)
□ Vary attractor distance (signal ∝ 1/r²)
□ Test with different material pairs (Ti/Cu, Pb/Al)
□ Measure with attractor removed (null test)

================================================================================
SECTION 5: SYSTEMATIC ERROR CONTROL
================================================================================

5.1 ENVIRONMENTAL CONTROLS
--------------------------
• Temperature stability: < 1 mK (in vacuum chamber)
• Vibration isolation: Active + passive, >60 dB at signal frequency
• Magnetic shielding: μ-metal + compensation coils, <1 nT residual

5.2 COMMON-MODE REJECTION
-------------------------
• Gradiometer configuration rejects common accelerations
• Expected CMR: 10^6
• Residual after CMR: 1.8e-17 m/s² (negligible)

5.3 PARASITIC EFFECTS
---------------------
• Casimir force: Zero - no contact with attractor
• Electrostatic: Grounded attractor, <1 V potential difference
• Patch potentials: Non-issue for atom-based measurement

================================================================================
SECTION 6: EXPECTED RESULTS
================================================================================

6.1 NULL HYPOTHESIS (GR only)
-----------------------------
Signal amplitude: 0 (no SDCG effect)
Oscillation from classical gravity: Yes (from mass difference)
This provides calibration for sensitivity

6.2 SDCG HYPOTHESIS
-------------------
Additional signal: 3.79e-08 m/s²
Fractional effect: ΔG/G = 3.7e-01
Detectable at: SNR = 2114 (423σ)

6.3 INTERPRETATION
------------------
• Detection → Strong evidence for density-dependent gravitational screening
• Non-detection → Upper limit on SDCG coupling: μ < 0.05 (95% CL)
• Intermediate → Measure screening function S(ρ) directly

================================================================================
SECTION 7: COMPARISON WITH CASIMIR EXPERIMENT
================================================================================

Metric                    | Casimir Experiment | Atom Interferometry
--------------------------|--------------------|-----------------------
Signal/Noise ratio        | ~10^-7 (buried)    | ~1000 (clear)
Casimir force issue       | Dominant           | None
Gap control required      | 95 μm ±nm          | Not required
Temperature               | 4K (challenging)   | 300K (room temp)
Integration time          | >10 years          | 100 hours
Feasibility               | Thought experiment | Practical test
Conclusion: Atom interferometry is the DEFINITIVE laboratory test

================================================================================
SECTION 8: TIMELINE AND RESOURCES
================================================================================

8.1 TIMELINE
------------
Year 1: Design and construction
Year 2: Commissioning and baseline
Year 3: Science runs and publication

8.2 RESOURCES
-------------
• Existing atom interferometer facility: Yes (multiple labs worldwide)
• Custom attractor: ~$50k for precision machining
• Additional equipment: ~$100k (rotation system, isolation)
• Personnel: 2-3 researchers

8.3 FEASIBILITY
---------------
• Technology readiness: HIGH (all components exist)
• Physics reach: DEFINITIVE (10^9 improvement over cosmological sensitivity)
• Risk: LOW (well-understood techniques)

================================================================================
