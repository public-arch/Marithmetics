# MARI Referee Close-Out Audit — Readout

**Prereg SHA256:** `485d338c92064cc8f65382771e5af29a193b293096512557a18fce308f27cce4`

## Rosetta

- Rosetta Φ_α equality b∈{7,10,16}: **PASS**
- CRT injectivity (2,3,5): **PASS**
- Φ_α values: **2.000000, 2.000000, 2.000000**

## Stationarity (Fejér mean-free Rayleigh)

- Rayleigh(mean): **0.064775**
- Rayleigh(std): **0.029212**
- Note: **Mean-free Rayleigh ratio for T=Π S Π (indicative stationary multiplier).**

## Fejér spectrum (head)

- k,  ̂F_r(k): (0,1.000000), (1,0.950986), (2,0.815283), (3,0.623436), (4,0.415986), (5,0.231765), (6,0.097637)

## SCFP++

- C1→C6 counts: **4→4→3→1→1→1**
- Survivor: **P6_pair_2_5**
- PASS: **PASS**

## β-law

- β-signs (U1 up / SU down): **PASS**
- β RMSE (toy ladder): **2.343e-03**

## Sharp +1 demo

- Boundary delta (sharp − smoothed): **0.444444**
- Sweep r (delta vs theory (r-1)/(2r)):
  - r=5: delta=0.400000, theory=0.400000
  - r=7: delta=0.428571, theory=0.428571
  - r=9: delta=0.444444, theory=0.444444
  - r=11: delta=0.454545, theory=0.454545
  - r=13: delta=0.461538, theory=0.461538

## UFET alias-tail ladder (proxy r/M)

- M=97, r=3, r/M=3.092784e-02
- M=193, r=3, r/M=1.554404e-02
- M=389, r=3, r/M=7.712082e-03
- M=769, r=4, r/M=5.201560e-03

## Designed FAIL demos

- Non-coprime CRT collisions (D=810): 720 (expected) 
- Digits-in-Φ drift: 1.800e-09 (expected FAIL). Values → 1.999999999400, 2.000000000000, 2.000000001200
