# DART — Evolutionary Rocket AI

Rachete 2D care învață singure, prin evoluție, să navigheze printr-un câmp de obstacole
până la o bază — cu o rețea neuronală scrisă de mână în NumPy. Fără TensorFlow, fără
PyTorch, fără biblioteci de RL: doar numpy, pygame și multă selecție naturală.

Antrenat pe o singură hartă. Testat pe 20 de hărți nevăzute: **20/20 (100% zero-shot).**

## Cum funcționează

- **Creier:** rețea 4 → 32 → 1 (ReLU + tanh), inițializată aleator, fără backpropagation —
  învățarea e pur evolutivă.
- **Senzori (inputuri):** 3 raze de distanță (stânga / centru / dreapta, rază 50px) +
  unghiul relativ spre bază.
- **Ieșire:** puterea de virare (±3 rad/s). Viteza e constantă — AI-ul controlează doar volanul.
- **Evoluție:** populație de 20; la fiecare generație top-3 creiere (elita) trec nemodificate,
  restul sunt mutații ale elitei (σ=0.05). Elita se salvează pe disc doar la record istoric.
- **Determinism:** pas de timp fix (1/60s) — același creier face exact același traseu,
  indiferent de lag. Fără asta, campionii mureau din fluctuații de frame-rate.

## Războiul cu exploiturile (partea interesantă)

Fiecare funcție de fitness a fost "citită" de evoluție mai atent decât am scris-o eu:

1. **Camparea.** Fitness = distanța finală față de bază → strategia optimă descoperită:
   stai pe loc lângă spawn și încasezi punctele garantate. *Fix:* recompensă pe progresul
   maxim atins + taxă pe timp + penalizare de timeout.
2. **Spinning-ul.** Cu virare de ±10 rad/s, rachetele au descoperit că învârtitul pe loc
   evită orice coliziune. *Fix:* virare limitată la ±3 + bonus dublu pe progres real.
3. **Dinastiile fragile.** Campionii treceau la 1px de obstacole; o fluctuație de dt îi
   omora și o generație proastă suprascria elita salvată. *Fix:* dt fix + salvare pe disc
   doar la record istoric.
4. **Loteria spawn-ului** (prins la design, înainte de implementare): spawn aleatoriu fără
   normalizarea fitness-ului la distanța de start ar fi premiat norocul, nu priceperea.

Lecția: agentul optimizează ce măsori, nu ce vrei. Orice contract are o gaură,
iar evoluția e un avocat cu răbdare infinită.

## Rezultate

Benchmark zero-shot: elita antrenată pe harta 41, evaluată pe 20 de hărți noi
(seed 100–119), o singură generație per hartă, fără antrenament:

    20/20 hărți atinse — 100%
    fitness între 10606 și 10848 (atingerea bazei = 10000+)

## Rulare

    pip install pygame numpy
    python dart_ai_evolutiv.py

- `MOD_TEST = False` → antrenament (jurnal în `istoric_antrenament.txt`)
- `MOD_TEST = True` → benchmark automat pe 20 de hărți (raport în `rezultate_test.txt`)
- Creierul elitei se salvează în `creier_salvat.npz`; șterge-l pentru antrenament de la zero.

## Ce am învățat

- Reward shaping e mai greu decât rețeaua în sine — am rescris fitness-ul de 3 ori.
- Determinismul nu e detaliu: fizica dependentă de frame-rate a ucis mai mulți campioni
  decât obstacolele.
- Elitismul fără arhivă protejată pierde progresul; cu ea, progresul devine ireversibil.
- Benchmark-ul automat bate impresia vizuală: de două ori am crezut că "nu merge"
  când jurnalul arăta progres, și o dată că merge când nu mergea.
- Memorare ≠ generalizare — și singurul mod să afli diferența e să testezi pe date nevăzute.
