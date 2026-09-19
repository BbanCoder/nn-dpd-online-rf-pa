# Adaptation en ligne d'une prédistorsion numérique neuronale

Banc de simulation reproductible accompagnant le mémoire de Master
**« Adaptation en ligne d'une prédistorsion numérique neuronale pour un
amplificateur RF en environnement dynamique »**.

MATLAB R2023a · 70 fichiers `.m` · 11 modules · 9 scénarios · 6 méthodes comparées

---

## Ce que fait ce banc

Une chaîne d'émission est simulée de bout en bout :

```
x[n] ──► NN-DPD ──► z[n] ──► PA dynamique ──► y[n] ──► métriques
             ▲                                            │
             └──────── boucle d'adaptation ◄──────────────┘
                    (surveillance → déclencheur → mise à jour)
```

Un amplificateur de puissance travaille près de sa saturation pour préserver son
rendement, et y devient non linéaire. La prédistorsion numérique corrige cette
distorsion en appliquant au signal la transformation inverse. Mais un
amplificateur ne reste pas identique à lui-même : température, puissance
d'émission, vieillissement et impédance de charge le font dériver, si bien
qu'une prédistorsion calibrée une seule fois cesse progressivement de lui
correspondre.

Ce banc met à jour le prédistorteur neuronal **pendant le fonctionnement**, puis
mesure ce que cette adaptation apporte réellement.

La boucle ne se referme que sur la NN-DPD adaptative. Les quatre autres méthodes
comparées, à savoir sans DPD, MP, GMP et NN-DPD statique, parcourent exactement
la même chaîne sans elle. C'est ce qui garantit que la comparaison porte sur ce
seul mécanisme.

---

## Résultat principal

Face à un réseau statique **correctement entraîné**, l'adaptation en ligne
n'apporte **aucun gain mesurable**, et ne coûte rien non plus. Sur neuf
scénarios rejoués chacun sur dix graines :

| | |
|---|---|
| Réseau statique face au MP-DPD | devant sur S0 à S5 (**0,5 à 1,3 dB** de NMSE), à égalité en S6, derrière de 0,7 dB sur la porteuse 5G NR (S7) ; mais **2,2 à 2,4 dBc d'ACPR** de moins bien partout |
| Adaptative face à statique | écart compris entre **−0,07 et +0,14 dB** sur S0 à S7, aucun significatif, malgré jusqu'à 7 mises à jour |
| Politique apprise (S8) | seul écart reproductible, **−0,07 dB** sur dix graines sur dix, obtenu par des mises à jour faites avant la dérive puis une abstention |
| Désadaptation de charge (S5) | aucune méthode ne descend sous ≈ −11 dB : c'est un problème d'égalisation, pas de prédistorsion |

L'apport de l'adaptation **diminue à mesure que le modèle hors ligne
s'améliore, jusqu'à s'annuler**. Le facteur limitant n'est pas la détection de
la dérive mais **la qualité statistique des mises à jour faites sur peu de
données** : les échantillons de forte amplitude sont rares dans un signal à fort
PAPR, donc sous-représentés dans chaque mini-lot, là où la prédistorsion doit
être la plus juste.

Une première version de ce banc concluait autrement (gain de 0,29 dB sous
stress 5G NR). Quatre défauts l'expliquaient, tous corrigés ici : un
entraînement hors ligne sans validation ni arrêt anticipé, un garde-fou qui
mesurait la perte sur les données de la mise à jour, et deux artefacts de
découpage par blocs (ligne à retards du réseau, puis mémoire du PA) qui
biaisaient le bras adaptatif. Les résultats de cette première version sont
conservés dans `results/*/archive_*`.

---

## Démarrage

```matlab
validate_project                                  % vérifie l'installation
run_all                                           % exécution de référence
run_montecarlo(1:10, {'S0','S1','S2','S3','S4','S5','S6','S7','S8'})   % analyse statistique, ~9 h
```

`run_train_dqn` réentraîne l'agent de renforcement, en environ 90 minutes. Cette
étape est facultative : l'agent entraîné est fourni dans
`rl_policy/dqn_agent.mat`. Son absence déclencherait le repli heuristique et les
résultats ne seraient plus reproductibles à l'identique.

### Prérequis

MATLAB R2023a. Toutes les toolboxes ci-dessous sont facultatives, et **tout
repli est signalé bruyamment en console** plutôt que silencieusement :

| Toolbox | Sans elle |
|---|---|
| 5G Toolbox | générateur OFDM/QAM interne au lieu de la porteuse 5G NR |
| Deep Learning | les méthodes neuronales sont ignorées, les baselines continuent |
| Reinforcement Learning | le scénario S8 bascule sur la politique heuristique |
| Communications, Signal Processing | fonctions complémentaires |

---

## Organisation du code

| Module | Fich. | Rôle |
|---|---|---|
| *(racine)* | 4 | points d'entrée : exécution, Monte-Carlo, entraînement RL, validation |
| `config/` | 3 | configuration par défaut, spécialisation par scénario, détection des toolboxes |
| `waveforms/` | 4 | génération 5G NR et OFDM/QAM, normalisation, calcul du PAPR |
| `pa_models/` | 6 | modèles d'amplificateur et génération des profils de dérive |
| `dpd_classical/` | 7 | prédistorteurs MP et GMP, moindres carrés régularisés |
| `dpd_neural/` | 10 | réseau RVTDNN : création, caractéristiques, entraînement, mise à jour en ligne, *replay buffer* |
| `adaptation/` | 5 | planificateur, détecteurs de dérive, politiques de mise à jour |
| `rl_policy/` | 5 | environnement RL, observation, exécuteur d'actions, agent DQN |
| `metrics/` | 7 | NMSE, EVM, ACPR, BER, PAE, largeur de bande occupée |
| `scenarios/` | 10 | les neuf scénarios S0 à S8 et le moteur d'exécution commun |
| `plots/` | 8 | production des figures |

---

## Les neuf scénarios

| | Scénario | Mécanisme de dérive |
|---|---|---|
| S0 | Nominal | aucune, référence de calibration |
| S1 | Thermique | rampe sur gain, phase et saturation |
| S2 | Saut de puissance | échelon sur l'IBO |
| S3 | Mémoire variable | oscillation des coefficients de retard |
| S4 | Vieillissement | dégradation lente monotone |
| S5 | Charge | rotation gain/phase, effet VSWR |
| S6 | Combinée | superposition des précédents |
| S7 | Stress 5G NR | 256-QAM, régime sévère |
| S8 | Politique apprise | dérive de S6, décision par agent DQN |

Les six profils se distinguent par leur **forme temporelle**, non par leur
amplitude. C'est ce qui permet de savoir à quel *type* de dérive l'adaptation
réagit, et pas seulement si elle réagit.

---

## Le mécanisme d'adaptation

### Quand déclencher

`adaptation/adaptationScheduler.m` compare le NMSE du bloc courant à la
**référence saine** observée depuis le début, et non à un seuil absolu. Le seuil
relatif vaut `max(2 dB, 4·σ_sain)`, où `σ_sain` est l'écart-type du NMSE sur
les six premiers blocs : un seuil fixe de 2 dB, adapté à un réseau à −18,8 dB,
réagissait au bruit inter-blocs d'un réseau à −21 dB. `onlineFineTunePolicy.m`
dose ensuite l'action :

| Dégradation | Action |
|---|---|
| sous le seuil | abstention |
| 2 à 4 dB | fine-tuning léger, 2 époques |
| > 4 dB | fine-tuning fort, 5 époques |

La recalibration complète reste dans l'espace d'action mais l'heuristique ne
l'emploie jamais : déclenchée à l'aveugle sur un seuil, elle détruisait plus
qu'elle ne réparait dans tous les cas testés. Elle est laissée à l'agent RL, qui
peut apprendre quand elle paie.

### Sur quelles données

`dpd_neural/updateOnlineNNDPD.m` reçoit les **paires ILA strictes** `(y, z)`,
sortie du PA vers entrée du PA, et **jamais** `(y, x)`. Une fois la chaîne
linéarisée, `y ≈ K·x` : les paires `(y, x)` dégénèrent vers l'identité et
effaceraient progressivement la prédistorsion.

`dpd_neural/replayBufferUpdate.m` compose le jeu d'adaptation à partir de trois
sources :

| Source | Taille | Risque couvert |
|---|---|---|
| Bloc courant | 4 096 | porte la dérive, mais biaisé par cette réalisation |
| Rejeu uniforme | 4 096 | oubli catastrophique |
| **Rejeu de crêtes** | 8 × 128 | **sous-représentation des fortes amplitudes** |

Le troisième est le plus spécifique, et répond directement au facteur limitant
identifié plus haut : les huit fenêtres de plus forte amplitude vues depuis le
début sont réinjectées de force à chaque mise à jour.

### Les garde-fous

1. **Pénalité proximale** (μ = 0,3) — la mise à jour reste proche des poids
   antérieurs. Cas particulier isotrope d'EWC, sans matrice de Fisher.
2. **Tronc gelé** — seuls les 66 poids de la couche de sortie sont adaptés, sur
   les 2 642 du réseau. Une adaptation à faible dimension est bien moins
   gourmande en données.
3. **Normalisation figée** — les vecteurs μ et σ sont hérités de l'entraînement
   hors ligne et ne sont jamais recalculés. Les recalculer déplacerait le repère
   d'entrée et rendrait les nouveaux poids incohérents avec un tronc gelé.
4. **Garde-fou de bout en bout** — le modèle conserve un segment d'ancrage de
   32 768 échantillons tiré de son jeu de validation, jamais employé pour une
   mise à jour. Avant et après chaque mise à jour, le NMSE réel (prédistorteur
   puis PA nominal) y est mesuré ; si la dégradation dépasse 0,1 dB, la mise à
   jour est annulée et le modèle antérieur restauré. Un garde-fou sur la perte
   du post-inverse, employé d'abord, ne voyait pas une dégradation concentrée
   dans les crêtes : une mise à jour pouvait ne la déplacer que de 0,4 % et
   coûter près d'un décibel de NMSE.
5. **Continuité entre blocs** — le bras adaptatif traite le signal par blocs de
   4 096 ; chaque bloc est préfixé des derniers échantillons du précédent, pour
   la ligne à retards du réseau (`nn.memoryDepth`) comme pour la mémoire du
   modèle de PA (`pa.memoryDepth`). Sans cela, les premiers échantillons de
   chaque bloc étaient faux et le bras adaptatif partait avec un handicap,
   même sans aucune mise à jour.

---

## Résultats fournis

| Chemin | Contenu |
|---|---|
| `results/figures/` | figures PNG de l'exécution de référence et synthèses |
| `results/tables/summary_all_scenarios.csv` | exécution de référence, 45 lignes |
| `results/montecarlo/montecarlo_raw.csv` | 450 lignes, une par graine × scénario × méthode (9 scénarios × 10 graines) |
| `results/sweep_bw_mod/sweep_bande_modulation.csv` | 540 lignes, plan croisé 3 bandes × 4 modulations (banc initial, non rejoué) |
| `results/*/archive_*` | résultats des versions antérieures du banc, conservés pour traçabilité |

Les états MATLAB intermédiaires (`results/**/*.mat`, environ 290 Mo) ne sont pas
versionnés : ils sont entièrement reconstruits par `run_all.m`. Seul
`rl_policy/dqn_agent.mat` est conservé, car il conditionne la reproductibilité
du scénario S8.

**Graines** — `rng(42)` pour l'exécution de référence, graines 1 à 10 pour le
Monte-Carlo. Elles sont appliquées à l'identique aux deux bras comparés : à
graine égale, le bras statique et le bras adaptatif voient le même signal, le
même bruit et la même dérive. Seule la méthode diffère, ce qui autorise un test
apparié sur les écarts.

---

## Deux points de méthode à connaître

**NMSE et EVM ne sont pas indépendants.** Les deux sont calculés sur le même
signal, après le même alignement en gain complexe, sur le même rapport d'énergie
d'erreur. Il en résulte l'identité exacte

```
EVM(%) = 100 × 10^(NMSE_dB / 20)
```

vérifiée sur les 135 points du tableau de synthèse avec un écart relatif
inférieur à 10⁻¹⁴. Le banc ne produit donc que **deux** métriques indépendantes :
l'erreur dans la bande et la régénération spectrale.

**L'ACPR est comparatif, pas normatif.** La forme d'onde, dépourvue de fenêtrage
et de filtre de mise en forme, impose un plancher de −20 dBc avant même le
passage par le PA. La fenêtre de mesure est en outre volontairement décalée de
0,25 largeur de bande hors de la position normative, pour disposer d'une
dynamique exploitable. Les valeurs sont comparables entre elles, jamais
confrontables à la limite de −45 dBc de TS 38.104.

---

## Paramètres principaux

Définis dans `config/defaultConfig.m`. Reproduire ces valeurs suffit à
reproduire les résultats.

| Signal | | Adaptation en ligne | |
|---|---|---|---|
| Fréquence d'échantillonnage | 122,88 MHz | Seuil relatif de NMSE | max(2 dB, 4σ) |
| Largeur de bande | 40 MHz | Taille de bloc | 4 096 |
| Modulation | 64-QAM | Ratio de rejeu | 1,0 |
| Sous-porteuses actives | 334 | Pénalité proximale μ | 0,3 |

| PA et réseau | | Apprentissage | |
|---|---|---|---|
| Modèle de PA | MP, ordres 1, 3, 5, 7 | Époques hors ligne | ≤ 300, arrêt anticipé (patience 8), validation 20 % |
| Profondeur mémoire | 3 | Époques en ligne | 3 |
| Architecture | RVTDNN 20–48–32–2 | Pas hors ligne | 5·10⁻⁴ |
| Poids adaptés en ligne | 66 / 2 642 | Pas en ligne | 5·10⁻⁵ |

---

## Limites

Ce banc est **logiciel**. Le modèle de PA est comportemental, non mesuré sur
composant réel, et le PA simulé étant lui-même un polynôme à mémoire, les
prédistorteurs MP et GMP y bénéficient d'un avantage structurel qu'ils
n'auraient pas sur du matériel. Les conclusions portent donc sur des
**mécanismes**, transposables ; les valeurs absolues, elles, ne le sont pas.

L'agent DQN a été entraîné sur un seul type de dérive et une seule fonction de
récompense : sa parcimonie est en partie un choix de conception, la récompense
pénalisant explicitement le coût des actions, et non une propriété découverte.
