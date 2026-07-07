# Résultats attendus

Ce document définit, pour chaque graphique produit par `research/generate_graphs.py` (équivalent exécutable de `graphs.ipynb`, décrit dans [`architecture.md`, section 12](architecture.md#12-couche-dorchestration-de-bout-en-bout-researchrun_pipeline-researchgenerate_graphs)), la forme
statistique attendue du résultat sous une implémentation correcte des règles décrites dans [`rules.md`](rules.md#table-des-matières). Il sert de référence pour construire
des tests de validation (assertions sur bornes, signes de corrélation, ordres de grandeur) plutôt qu'une simple inspection visuelle. Chaque section ci-dessous renvoie, lorsque pertinent, à la section de `rules.md` qui définit la mécanique testée.

## 1. Distribution des VP par rang de sortie (`vp_by_rank_violin.png`)

Attendu : décroissance monotone (au sens large) du VP moyen avec le rang de sortie $k$, quel que soit le barème (`LEGACY_STEPPED`, `LINEAR`,
`SYMMETRICAL`, définis dans [`rules.md`, section 4](rules.md#4-rôles-et-points-de-victoire-victorypoints)). En mode `SYMMETRICAL` avec $N$ joueurs, la moyenne empirique de chaque rang doit converger vers $VP(k) = (N-1)/2 - k$ à mesure
que le nombre de manches augmente, avec une variance non nulle uniquement due à la composition de rôles pour les rangs neutres si $N > 4$.
Test de validation : `|moyenne_empirique(k) - VP(k)| < tolérance` décroissante avec le nombre de manches (loi des grands nombres).

## 2. Taux de victoire par joueur (`win_rate_by_player_ci.png`)

Attendu : sous des agents identiques et un `first_trick_opener_id` fixe, une légère avantage structurel peut apparaître pour le joueur
ouvrant le premier pli de la partie, mais sur un grand nombre de manches indépendantes (rôles et graines de distribution variant à chaque
manche), le taux de victoire par joueur doit converger vers $1/N$ à un intervalle de confiance près. Test : chevauchement des intervalles de
confiance à 95 % entre joueurs sur un échantillon suffisamment grand (loi binomiale, $p = 1/N$).

## 3. Taux de passe sous-optimal (`suboptimal_pass_rate_ci.png`)

Attendu : proche de zéro pour tout agent déterministe committing uniquement des actions légales (`GreedyBot`, `RuleBasedBot`, `MCTSBot`
utilisant des options légales). Une valeur significativement non nulle signale soit un agent mal implémenté (propose des combinaisons hors de
sa main ou de puissance insuffisante), soit une régression dans `core.rules_engine.generate_uniform_plays`/`generate_sequence_plays` qui omet
des options réellement légales. Test : `sub_optimal_pass_rate == 0.0` pour `GreedyBot`, `RuleBasedBot`, `MCTSBot`.

## 4. Matrice de transition des rôles (`role_transition_heatmap.png`)

Attendu : la diagonale (`ROLE_PRESIDENT → ROLE_PRESIDENT`, `ROLE_SCUM → ROLE_SCUM`) doit présenter une probabilité strictement supérieure à
$1/5$ (persistance de rôle, due à l'avantage structurel de l'échange de cartes décrit en [`rules.md`, section 5.2](rules.md#52-échange-exchange_phase)), sous réserve que `putsch_enabled` soit désactivé ou peu
invoqué (voir [`rules.md`, section 6.1](rules.md#61-modificateurs-de-pré-manche-pre_round_modifiers)). Chaque ligne doit sommer à 1 (loi de probabilité totale). Test : `abs(matrix.sum(axis=1) - 1.0) < 1e-9` pour chaque ligne non vide.

## 5. Indice de Gini de la puissance de main initiale (`gini_hand_power_histogram.png`)

Attendu : une distribution centrée et concentrée autour d'une valeur faible à modérée (typiquement $[0.05, 0.25]$ pour une distribution
aléatoire uniforme des cartes, cf. [`rules.md`, section 5.1](rules.md#51-phase-initiale)), jamais nulle (l'inégalité résiduelle entre mains est structurelle à toute distribution finie), jamais proche
de 1 (qui signalerait une distribution des cartes gravement biaisée). Test : `0 < gini_moyen < 0.35` sur un grand échantillon de manches en
distribution non truquée (`strict_remainder_allocation=False`, voir [`rules.md`, section 6.1](rules.md#61-modificateurs-de-pré-manche-pre_round_modifiers)).

## 6. Facteur de branchement moyen vs nombre de joueurs (`branching_factor_vs_player_count.png`)

Attendu : croissance du facteur de branchement moyen avec le nombre de joueurs (plus de joueurs → plus de paquets → plus de cartes par
rang → plus de tailles de combinaison légales par rang). Ordre attendu entre profils : `random_bot` et `mcts_bot` (qui n'excluent aucune
option) partagent le même facteur de branchement que le nombre d'options légales généré par le moteur ; seul `rule_based_bot` peut réduire
artificiellement son propre facteur de branchement s'il calcule `legal_action_count` après filtrage (ce qui n'est pas le cas ici, puisque
`EventActionRequest.legal_action_count` est calculé avant la décision de l'agent, cf. section 4.3 de l'architecture). Le facteur de
branchement doit donc être quasi identique entre profils pour un même état de jeu. Test : écart-type inter-profils faible relativement à la
moyenne, à nombre de joueurs fixé.

## 7. Distribution des tailles de combinaison (`combo_size_distribution.png`)

Attendu : décroissance approximative en fréquence des tailles 1 à 4 (plus une combinaison est grande, plus elle est rare à composer), avec un
pic résiduel sur les tailles associées aux seuils de Révolution (4) et de Double Révolution (8) si ces règles sont actives, car les agents
heuristiques ne évitent pas nécessairement ces tailles. Test : `fréquence(taille=1) > fréquence(taille=4)` pour un paquet standard sans
biais de configuration favorisant les grandes combinaisons.

## 8. Longueur des plis / actions par manche (`actions_per_round_violin.png`)

Attendu : la longueur moyenne d'un pli est bornée par $N$ (au plus $N$ actions avant clôture par épuisement d'éligibilité sous
`pass_type=HARD_ONLY`), et peut dépasser $N$ sous `pass_type=ALLOW_SOFT` (un joueur peut resurenchérir). Test :
`moyenne(actions_par_pli) <= N` strictement sous `HARD_ONLY`.

## 9. Volatilité de la Révolution vs nombre de joueurs (`e_rev_volatility_vs_players.png`)

Attendu : corrélation positive entre le nombre de joueurs et la volatilité (plus de joueurs → plus de paquets → plus de cartes du même rang
disponibles → plus de combinaisons de taille $\ge 4$ jouables, déclencheur défini en [`rules.md`, section 6.5](rules.md#65-révolution-et-double-révolution-revolution_enabled-double_revolution_enabled)). Test : coefficient de régression positif et statistiquement significatif sur
un grand échantillon.

## 10. Corrélation position d'ouverture / rang de sortie (`opening_position_rank_regression.png`)

Attendu : corrélation faible et non significativement différente de zéro sous des agents de force égale sur toutes les manches d'une
campagne longue (l'identifiant de siège n'a pas de causalité intrinsèque sur le rang de sortie, hormis l'avantage transitoire du tout premier
pli de la toute première manche). Test : `|r| < 0.1` sur un grand nombre de manches indépendantes.

## 11. Fréquence des règles déclenchées (`rule_trigger_counts.png`)

Attendu : présence uniquement des règles effectivement activées par `GameConfig` pour la campagne considérée ; absence totale d'occurrences
d'une règle désactivée (ex : aucun `SKIP_TURN` si `skip_turn_enabled=False`, règle définie en [`rules.md`, section 6.7](rules.md#67-saut-de-tour-skip_turn_enabled)). Les noms de règles possibles sont énumérés dans la [matrice de compatibilité (section 7)](rules.md#7-matrice-de-compatibilité-et-résolution-des-conflits-truth-table). Test : `set(rule_triggered_df.rule_name.unique()) ⊆
règles_activées_par_config`.

## 12. Courbes d'apprentissage (`training_learning_curves.png`)

Attendu : tendance croissante (ou non décroissante) du VP moyen glissant au fil des manches d'entraînement pour `RLAgent`/`TorchRLAgent`
contre des adversaires fixes, la politique nulle initiale ne pouvant statistiquement pas être optimale. Test : VP moyen des 10 % dernières
manches strictement supérieur au VP moyen des 10 % premières manches, à epsilon d'exploration décroissant.

## 13. Efficacité du Putsch (`putsch_efficiency_ci.png`)

Attendu : sous la condition mathématique $P_{putsch}$ (main favorable, définie en [`rules.md`, section 5.1](rules.md#51-phase-initiale), sous-section 5.1.2), le taux de victoire du rôle `ROLE_SCUM` sollicité doit être
supérieur ou égal lorsque le Putsch est invoqué que lorsqu'il ne l'est pas, puisque la condition n'est vérifiée avant sollicitation que
lorsque la main est déjà favorable ; l'invocation ajoute la conservation de cette main favorable en évitant l'échange désavantageux (voir [`rules.md`, section 6.1](rules.md#61-modificateurs-de-pré-manche-pre_round_modifiers)). Test :
`taux_victoire(invoqué) >= taux_victoire(non_invoqué)` à tolérance statistique près.

## 14. Corrélation poids de la taxe d'échange / VP du destinataire (`tax_weight_vp_regression.html`)

Attendu : corrélation négative (recevoir des cartes de poids élevé, donc de puissance élevée en amont, désavantage le destinataire déjà
`ROLE_SCUM`/`ROLE_VICE_SCUM` sous cette mécanique redistributive), en pratique la corrélation attendue dépend du rôle du destinataire
(`ROLE_PRESIDENT` reçoit des cartes fortes du `ROLE_SCUM`, ce qui est plutôt favorable). Test : corrélation de signe cohérent avec le rôle
receveur majoritaire de l'échantillon (à documenter au cas par cas plutôt qu'un signe universel).

## 15. Taux d'interception manquée (`missed_interception_rate_ci.png`)

Attendu : nul pour tout agent implémentant `on_interception_opportunity` de façon gloutonne (`GreedyBot`), strictement positif ou nul pour
les agents à décision partielle (`RuleBasedBot`, qui refuse en fin de manche). Test : `missed_interception_rate(GreedyBot) == 0.0`.

## 16. Distribution de la magnitude du Saut de Tour (`skip_turn_magnitude_histogram.png`)

Attendu : concentrée sur les tailles de combinaison réellement composées du rang `skip_turn_rank`, bornée par $N-1$ (nombre maximal de
joueurs sautables, règle définie en [`rules.md`, section 6.7](rules.md#67-saut-de-tour-skip_turn_enabled), résolution [F] et [H] de la [matrice de compatibilité (section 7)](rules.md#7-matrice-de-compatibilité-et-résolution-des-conflits-truth-table)). Test : `max(magnitude) <= N - 1`.

## 17. Évaluation comparative des profils (`evaluation_vp_violin.png`, `evaluation_president_rate_ci.png`)

Attendu, sous la hiérarchie de complexité décrite dans l'architecture (`RandomBot < GreedyBot < RuleBasedBot < MCTSBot`) : ordre croissant du
VP cumulé moyen et du taux de manches `ROLE_PRESIDENT` dans cet ordre, face à des adversaires fixes identiques. Test : ordre total strict des
moyennes de VP cumulé entre les quatre profils sur un échantillon suffisamment grand pour être statistiquement significatif (test de
Mann-Whitney ou intervalles de confiance disjoints).

## 18. Performance finale d'entraînement vs taux d'apprentissage (`learning_rate_final_performance_boxplot.png`)

Attendu : une relation en cloche (un taux d'apprentissage trop faible converge lentement, un taux trop élevé déstabilise l'apprentissage),
non monotone. Test : existence d'un taux d'apprentissage intermédiaire dont la performance finale médiane dépasse celle des extrêmes testés.

## 19. Taux de victoire et taux de validité par profil d'agent réel (`win_rate_by_profile_ci.png`, `action_validity_by_profile_ci.png`, `suboptimal_pass_rate_ci.png`)

Attendu : ces trois graphiques regroupent désormais les mesures par identité réelle d'agent (`profile`, reconstruite depuis les colonnes
JSON `player_profiles` des résumés de simulation, cf. [`usage.md`, section 9](usage.md#9-simulations-de-masse-parallélisées-researchrun_simulation)) plutôt que par identifiant de siège brut, celui-ci changeant d'occupant d'une partie à l'autre du fait du mélange
aléatoire des sièges. Ordre attendu, cohérent avec la hiérarchie de complexité algorithmique documentée dans `architecture.md`,
section 5.3 : `random_bot` présente le taux de victoire le plus faible et `probabilistic_bot`/`scoring_bot`/`adaptive_bot`/`lookahead_bot`
dominent `greedy_bot`/`rule_based_bot` de façon stable sur un échantillon suffisamment grand. Tout agent déterministe committing
uniquement des actions légales doit présenter un taux de passe sous-optimal proche de zéro et un taux de validité des actions proche de un.

## 20. Distribution normalisée des tailles de combinaison par profil (`combo_size_distribution_normalized.png`)

Attendu : contrairement à un histogramme de comptages bruts (biaisé par le nombre inégal de poses observées par profil selon le volume de
parties simulées pour chacun), la version normalisée additionne à 1 par profil et permet une comparaison directe de la propension
relative de chaque profil à composer de grandes combinaisons. Test : `sum(proportion) ≈ 1.0` pour chaque profil.

## 21. Distribution des rôles obtenus par profil (`role_distribution_by_profile_heatmap.png`)

Attendu : les profils les plus performants (mesurés par le taux de victoire de la section 19) doivent présenter une proportion de
`ROLE_PRESIDENT` supérieure à $1/5$ et une proportion de `ROLE_SCUM` inférieure à $1/5$, symétriquement aux profils les moins performants.
Chaque ligne de la heatmap doit sommer à 1 (loi de probabilité totale par profil).

## 22. Persistance de rôle (`role_persistence_summary.png`)

Attendu : la probabilité de conserver `ROLE_PRESIDENT` ou `ROLE_SCUM` d'une manche à l'autre doit dépasser le seuil uniforme $1/5$,
matérialisé par la ligne de référence du graphique, tant que `putsch_enabled` reste désactivé ou peu invoqué (mécanisme redistributif
décrit en [`rules.md`, section 6.1](rules.md#61-modificateurs-de-pré-manche-pre_round_modifiers)).

## 23. Vérification de convergence des points de victoire (`vp_convergence_check.png`)

Attendu : sous le mode `SYMMETRICAL`, le VP moyen empirique par rang de sortie doit converger vers la droite théorique
$VP(k) = (N-1)/2 - k$ à mesure que le volume de manches simulées augmente ; l'écart résiduel par rang doit décroître avec le nombre de
manches accumulées au fil des lancements successifs du pipeline.

## 24. Balayage systématique de paramètres (`sweep_<paramètre>_vs_<métrique>.png`) et heatmap d'impact (`parameter_impact_heatmap.png`)

Attendu : un graphique distinct est produit automatiquement pour chaque couple (paramètre de `GameConfig` exploré par
`research.run_pipeline`, métrique macro suivie) disposant d'au moins deux valeurs observées de chacun, sans qu'aucune combinaison ne
doive être ajoutée manuellement à `research.generate_graphs` lors de l'extension future de la grille de règles. La heatmap d'impact
synthétise en une seule vue le signe et l'amplitude de l'effet marginal (`True` moins `False`) de chaque paramètre booléen sur chaque
métrique, permettant d'identifier rapidement les règles les plus influentes sur la complexité combinatoire (`branching_factor_average`,
`action_space_entropy`) et sur la dynamique de la Révolution (`e_rev_volatility`) sans consulter individuellement chaque graphique de
balayage.

## 25. Tournoi linéaire contre neuronal (`tournament_results_ci.png`)

Attendu : à volume d'entraînement croissant sur des lancements successifs du pipeline, l'écart de VP cumulé moyen entre `rl_agent` et
`torch_rl_agent` doit se réduire ou s'inverser en faveur de `torch_rl_agent` (capacité de représentation supérieure du réseau à deux
couches cachées face à la politique strictement linéaire), sans qu'aucune relation monotone ne soit garantie sur un unique lancement à
faible volume d'entraînement.
