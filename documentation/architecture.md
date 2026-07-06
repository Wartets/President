# DOCUMENT D'ARCHITECTURE

Ce document décrit la conception interne du projet : principes de conception retenus, structure des données, contrats entre modules, machines à états, invariants garantis, et raisons des choix techniques. Il complète `usage.md`, qui couvre exclusivement la mise en œuvre opérationnelle (commandes, exemples d'exécution, exploitation des résultats) ; aucun contenu de ce document n'y est répété.

## 1. Principes de conception globaux

### 1.1. Séparation stricte entre calcul pur et orchestration mutable

Le projet distingue systématiquement deux catégories de code :

* des **fonctions et objets purs**, sans effet de bord, immuables (`core.config.GameConfig`, `core.models.Card`/`Hand`/`Action`, l'intégralité de `core.math_utils` et `core.rules_engine`) ;
* une **couche d'orchestration mutable**, isolée dans `engine` (`GameState`, `TrickState`, `run_round`), qui est la seule autorisée à muter un état de jeu.

Cette séparation permet à `core.rules_engine` d'être testé, vérifié et réutilisé indépendamment de toute notion de tour de jeu ou de séquencement (il est d'ailleurs consommé aussi bien par le moteur événementiel `engine.round` que par le moteur vectorisé `training.fast_path`, qui n'ont aucune dépendance croisée l'un envers l'autre). Toute fonction de `core.rules_engine` est explicitement documentée comme sans effet de bord dans son docstring, ce qui constitue un contrat de conception vérifiable à la lecture du code : aucune fonction de ce module n'accède à un état global ni ne mute ses arguments.

### 1.2. Immutabilité par construction

Toutes les structures de données représentant un fait acquis (une carte, une main à un instant donné, une action décidée, une configuration de partie, un événement) sont des `@dataclass(frozen=True)`. Seules deux structures sont volontairement mutables : `engine.state.GameState`/`TrickState` (la vue de travail du moteur, qui doit être mutée en place pour des raisons de performance) et `training.fast_path.FastPathState` (des tableaux `numpy` mutés en place pour la vectorisation). Cette dichotomie immuable/mutable est le critère qui sépare la couche `core`/`events` de la couche `engine`/`training.fast_path`.

Un corollaire de cette immutabilité est que `Hand.without`/`Hand.with_added` retournent systématiquement une nouvelle instance plutôt que de muter `self` : toute opération sur une main est donc, à ce niveau, une allocation. Le coût est jugé acceptable au regard de la taille des mains (quelques dizaines de cartes maximum) et de la garantie de non-aliasing qu'il procure entre les événements journalisés (qui capturent des références à des `Hand` passées) et l'état courant qui continue d'évoluer.

### 1.3. Double implémentation volontaire des règles : moteur événementiel et moteur vectorisé

Le projet contient deux implémentations indépendantes du cœur des règles de jeu :

* le moteur événementiel complet (`engine.round.run_round`), qui délègue à `core.rules_engine` pour toute décision de légalité et implémente l'intégralité des règles avancées de [`rules.md`](rules.md) (Putsch, Taxe Aveugle, Suites, Interception, Saut de Tour, pénalités étendues, remainder strict) ;
* le moteur vectorisé (`training.fast_path.FastPathEngine`), qui réimplémente en `numpy` un sous-ensemble volontairement restreint des règles (combinaisons uniformes uniquement, Jokers, Révolution, Double Révolution, clôture magique généralisée, passes durs et souples), sans Suites, sans Interception, sans Putsch, sans Taxe Aveugle, sans pénalité de sortie étendue, et sans représentation des couleurs.

Ce choix n'est pas un oubli mais une conséquence directe de la contrainte de vectorisation par lot : les règles omises (Suites avec comblement de Joker, Interception nécessitant une résolution hors-tour asynchrone, Putsch conditionnant l'annulation d'une phase entière) sont difficilement exprimables comme des opérations tensorielles denses sur un lot de parties hétérogènes sans dégrader fortement le débit qui justifie l'existence même de ce second moteur. Le moteur vectorisé est donc un instrument d'entraînement à haut débit sur un sous-ensemble de règles, et non un substitut fonctionnellement complet du moteur événementiel ; `research.run_simulation` et `play_game.py` reposent exclusivement sur le moteur événementiel complet, tandis que `Game.vectorized_run` et `training.fast_path` sont réservés à l'exploration algorithmique et à l'entraînement massif.

### 1.4. Event-sourcing comme mécanisme d'observabilité, pas comme mécanisme de mutation

Le bus d'événements (`engine.event_bus.EventBus`) ne participe à aucune décision du moteur : `engine.round.run_round` mute directement `GameState` puis publie, après coup, un événement décrivant la transition qui vient de se produire. Les événements ne sont donc jamais rejoués pour reconstruire un état (contrairement à un event-sourcing classique où l'état est dérivé du flux d'événements) ; ils constituent un journal d'observation a posteriori, corrélé à l'état par `state_hash` (voir section 4.2). Ce choix privilégie la simplicité et la performance du chemin critique de simulation au prix de la perte de la capacité à reconstruire un `GameState` strictement à partir du seul flux d'événements sans REJOUER la logique de `run_round` elle-même ; en pratique, la reconstruction d'informations dérivées (mains reconstituées, rôles à un instant donné) se fait par relecture ciblée de sous-ensembles d'événements structurels et transactionnels (voir `analytics.metrics_calc.missed_interception_rate` pour un exemple de reconstruction incrémentale de mains à partir de `EventRoundStart`, `EventExchange` et `EventActionPlayed`).

## 2. Couche de données pures (`core`)

### 2.1. `core.models` : représentation d'une carte et calcul différé de la valeur

`Card` ne stocke ni puissance ni valeur de points sur l'instance : ces grandeurs sont recalculées à la demande par `core.math_utils.f_power`/`f_points` à partir du seul `rank` et, pour la puissance, de l'état de révolution courant `e_rev`. Ce choix évite toute désynchronisation entre une carte et l'état de révolution global au moment où sa puissance est interrogée (une carte reste un objet purement descriptif, jamais porteur d'un état contextuel qui pourrait devenir périmé). Le corollaire est que toute fonction manipulant des puissances doit systématiquement recevoir `e_rev` en paramètre explicite ; ce paramètre traverse la quasi-totalité des signatures de `core.rules_engine` et `core.math_utils`.

`Card.instance_id` porte `compare=False` dans sa définition de champ : deux cartes de même rang et de même couleur sont égales au sens de `==` indépendamment de leur exemplaire d'origine (pertinent en particulier pour les Jokers, dont l'unique moyen de distinction est cet identifiant, utilisé uniquement pour le départage déterministe des égalités, cf. `core.rules_engine._tiebreak_key`).

`Hand` est un tuple immuable ; `without`/`with_added` opèrent par reconstruction de liste puis re-tuplage. `without` retire les cartes une à une par `list.remove`, ce qui suppose implicitement que les cartes retirées appartiennent bien à la main (contrat non vérifié à ce niveau, c'est `core.rules_engine.is_action_valid` qui a la responsabilité de garantir cette précondition avant tout appel à `without` dans `engine.round._apply_play`).

`RANK_ORDER`/`RANK_INDEX` définissent l'unique source de vérité de l'ordre total des rangs faciaux, y compris `JOKER` en position finale ; `core.math_utils.rank_facial_index` et le remappage de la carte magique en révolution (`core.rules_engine._remapped_magic_rank`) reposent tous deux sur cette même table, garantissant la cohérence de l'ordre utilisé pour le symétrique de révolution et pour le calcul du rang magique effectif.

### 2.2. `core.config.GameConfig` : configuration comme unique source de vérité, validée à la construction

`GameConfig` centralise l'intégralité des paramètres de partie sous une unique dataclass immuable. La validation des contraintes croisées entre champs (`__post_init__`) est volontairement placée à la construction plutôt qu'au moment de l'utilisation de chaque règle individuelle : ceci garantit qu'une configuration invalide ne peut jamais atteindre `engine.round.run_round`, éliminant toute une classe de vérifications défensives qui auraient dû sinon être répétées à chaque point d'usage d'un champ contraint (`double_revolution_enabled`, `interception_enabled`, rangs magiques/saut de tour, etc.).

Deux méthodes dérivées, `effective_magic_card_enabled()` et `effective_magic_card_rank()`, encapsulent la logique de généralisation de `magic_two` vers `magic_card_enabled` : ce sont les seuls points d'accès utilisés par `core.rules_engine` et `training.fast_path`, ce qui garantit qu'aucun code consommateur n'a besoin de connaître la relation historique entre les deux booléens `magic_two`/`magic_card_enabled`, un renommage futur de cette relation n'affecterait que ces deux méthodes.

### 2.3. `core.math_utils` : fonctions de valeur et théorème de l'inversion

La puissance en révolution n'est pas stockée sous forme de table statique symétrique à celle de la puissance standard ; elle est dérivée algébriquement par la relation $f_{power}(c, E_{rev}=\text{True}) = 18 - f_{std}(c)$ (`f_power`), tirant parti du fait que la somme de la puissance standard et de la puissance inversée d'une carte non-Joker vaut invariablement 18 (démontré et nommé « Théorème de l'Inversion » dans [`rules.md`](rules.md) §3.2.B). Ce choix algébrique, par opposition à une table de correspondance explicite par rang, élimine tout risque de désynchronisation entre les deux moitiés de la table advenant une modification future de l'ordre des rangs, puisque la seule quantité qui doit rester correcte est la constante `_REVOLUTION_SUM = 18`.

`compute_vp` et `role_for_rank` implémentent chacun une fonction totale sur l'espace `(k, n)`, avec des cas particuliers structurels explicitement traités pour $n=3$ (absence de `ROLE_VICE_PRESIDENT`/`ROLE_VICE_SCUM`, l'unique rang intermédiaire recevant `ROLE_NEUTRAL`) et $n=4$ (absence de `ROLE_NEUTRAL`). Ces deux fonctions sont volontairement séparées de la logique de manche (`engine.round`), qui ne fait qu'itérer sur `state.finish_order` et leur déléguer le calcul, la logique de rôle et de VP reste ainsi testable indépendamment de toute exécution de manche.

### 2.4. `core.rules_engine` : moteur de légalité et matrice de compatibilité

Ce module concentre la totalité de la logique de légalité et de déclenchement des règles avancées, matérialisant la « Matrice de Compatibilité » de [`rules.md`](rules.md) §7 sous forme de fonctions dédiées (`triggers_revolution`, `triggers_double_revolution`, `triggers_magic_closure`, `triggers_skip_turn`, `can_intercept`), chacune recevant les paramètres contextuels strictement nécessaires à sa résolution [A]–[H] plutôt qu'un état global. Par exemple, `triggers_revolution` reçoit `is_sequence` en paramètre explicite pour appliquer la résolution [B] (une Suite ne déclenche jamais de révolution), sans avoir besoin de connaître la nature exacte de la combinaison au-delà de ce booléen.

**Numba comme optimisation ciblée, pas systémique** : seules trois fonctions purement numériques et appelées à haute fréquence sont compilées via `@numba.njit(cache=True)` : `_is_uniform_power_array` (test d'uniformité d'un tableau de puissances), `_is_consecutive_run` (test de suite consécutive), et, dans `analytics.metrics_calc`, `_gini_from_sorted` et `_pearson_correlation_jit`. Ce choix ciblé reflète le fait que ces fonctions sont invoquées à l'intérieur de boucles de génération de combinaisons potentiellement combinatoires (`generate_uniform_plays`, `generate_sequence_plays`), alors que le reste du module reste en Python pur, où le gain de compilation serait négligeable au regard du coût de compilation à froid et de la perte de lisibilité.

**Génération d'options comme énumération exhaustive plutôt que recherche heuristique** : `generate_uniform_plays` et `generate_sequence_plays` énumèrent la totalité des combinaisons légales distinctes (par couple `(power, size)` pour les uniformes, par fenêtre glissante de puissances consécutives pour les suites), y compris les substitutions par Joker à chaque position possible. Cette exhaustivité est ce qui permet à `RandomBot` de tirer uniformément parmi toutes les options réellement légales, à `EventActionRequest.legal_action_count` de servir de mesure fiable du facteur de branchement (`analytics.metrics_calc.branching_factor_average`), et à `core.action_masking.build_action_mask_batch` de projeter fidèlement cet ensemble sur un espace d'action discret sans perte de combinaisons légales.

**Validation stricte en aval de la génération** : `is_action_valid` est un second filtre indépendant, appliqué non pas aux options générées par l'agent lui-même mais à l'action *retournée* par un agent quelconque, avant application à l'état. Cette redondance volontaire (l'agent pourrait en théorie s'appuyer sur `generate_uniform_plays`/`generate_sequence_plays` pour ne proposer que des options déjà valides, mais rien ne l'y oblige) constitue la seule barrière de sécurité du moteur contre un agent buggé ou malveillant : aucune action n'atteint `_apply_play` sans être repassée par `is_action_valid` dans `engine.round.run_round`.

### 2.5. `core.action_masking` : projection sur un espace d'action discret indépendant du moteur événementiel

`build_action_space_index` construit un espace `(power, size)` dense et fixe, indépendant de toute main ou configuration particulière. Ce découplage permet à un réseau de neurones de sortie fixe (`agents.torch_rl_agent.PolicyNet`, dont l'espace de sortie est en réalité construit différemment via un score par option plutôt qu'un espace fixe, voir section 8) ou à tout autre consommateur de disposer d'un espace d'action stable entre deux configurations différentes tant que `max_power`/`max_combo_size` restent identiques, sans avoir à réénumérer dynamiquement les options à chaque appel pour connaître la dimension de sortie attendue.

## 3. Le moteur événementiel (`engine`)

### 3.1. `engine.state` : la vue matérialisée comme unique état mutable

`GameState` et `TrickState` sont les deux seules structures mutables de tout le projet en dehors de `training.fast_path.FastPathState`. `GameState` a un constructeur `__init__` explicite plutôt que de s'appuyer sur celui généré par `@dataclass`, précisément pour permettre l'initialisation par valeurs par défaut mutables (`hands or {}`, etc.) sans exposer les pièges classiques des arguments par défaut mutables de Python (`field(default_factory=...)` reste néanmoins déclaré en parallèle pour la cohérence de la dataclass, mais l'initialisation effective passe par `__init__`).

`snapshot_key()` construit une représentation canonique et totalement ordonnée de l'état (tri des dictionnaires par clé, tri des cartes de chaque main par leur `repr` pour un ordre déterministe indépendant de l'ordre d'insertion) : c'est cette représentation, et uniquement elle, qui alimente `events.base.compute_state_hash`. La canonicalisation par tri est ce qui garantit que deux mains identiques en contenu mais insérées dans un ordre différent produisent le même `state_hash`.

### 3.2. `engine.round.run_round` : orchestration séquentielle d'une manche

La fonction implémente une machine à états à trois phases strictement séquentielles (distribution, échange conditionnel, plis), suivies d'une clôture de manche, correspondant terme à terme à l'algorithme décrit dans [`rules.md`](rules.md) §5. Quelques décisions d'implémentation notables :

**Le compteur `tick`** capturé par fermeture (`tick = [0]`, `next_tick()`) fournit l'horodatage logique croissant de chaque événement de la manche, strictement local à l'appel de `run_round`, deux manches distinctes recommencent chacune à 1, l'ordre global n'étant reconstitué qu'à travers le triplet `(game_id, round_id, timestamp)`.

**`emit` comme point d'unification de la publication** : toute la fonction `run_round` publie exclusivement via la fermeture locale `emit(event_cls, **kwargs)`, qui injecte systématiquement `timestamp`, `game_id`, `round_id` et `state_hash` (calculé à partir de `state.snapshot_key()` au moment de l'émission, donc après la mutation qui vient de se produire). Ce point d'unification garantit qu'aucun événement ne peut être publié sans empreinte d'état associée, et centralise en un seul endroit la politique d'horodatage.

**Garde de sécurité `_MAX_ACTIONS_PER_TRICK`** : fixée à `max(64, n * 8)`, cette borne détecte un pli qui ne se clôture jamais (typiquement une incohérence entre `pass_type` et `skip_on_equal`, documentée comme risque connu dans [`rules.md`](rules.md) §6.3) et lève une `RuntimeError` explicite plutôt que de boucler indéfiniment, un choix de robustesse défensive spécifiquement motivé par le contexte d'exécution en simulation massive (`research.run_simulation`), où une manche bloquée bloquerait silencieusement tout un acteur Ray sans ce garde-fou.

**Séparation `_apply_play`/`_finish_player`** : l'application d'une pose (retrait des cartes, mise à jour de la puissance du pli, déclenchement des règles avancées, résolution de l'interception, du saut de tour) est isolée dans `_apply_play`, qui retourne optionnellement l'action d'interception appliquée lorsque celle-ci vide elle-même la main de l'intercepteur, ce retour permet à l'appelant de déclencher un second traitement de sortie de joueur (`_finish_player`) pour l'intercepteur, sans dupliquer la logique de sortie entre le joueur ayant posé et un éventuel intercepteur ayant lui-même terminé sa main par la carte interceptée.

**`_finish_player` comme unique point de calcul de la pénalité de sortie** : la fonction encapsule à la fois la détection de la pénalité (`matches_finish_penalty`), son application différentielle selon `finish_penalty_type` (report de cartes en main pour `PENALTY_DRAW_CARDS`, qui *annule* la sortie en retournant avant de marquer `is_finished`, contre rétrogradation immédiate pour `PENALTY_INSTANT_SCUM`, qui laisse la sortie se produire mais l'enregistre dans `instant_scum_players` pour substitution du rôle et du VP en fin de manche) et la mise à jour de `forced_scum_ref`, une liste à un élément passée par référence pour simuler une variable de sortie mutable partagée entre les multiples appels de `_finish_player` au sein d'une même manche (au plus un joueur peut être forcé au rôle `ROLE_SCUM` de cette manière par manche, le dernier appel l'emportant si plusieurs conditions se produisent).

**Ordre d'évaluation Interception avant Saut de Tour** dans `_apply_play`, reflétant explicitement la résolution [H] de la matrice de compatibilité de [`rules.md`](rules.md) §7 : le bloc d'interception est évalué et peut retourner avant que le bloc de saut de tour ne soit atteint, garantissant qu'un intercepteur dispose de la priorité sur l'application du saut.

### 3.3. `engine.event_bus.EventBus` : dispatcher minimal sans garantie de robustesse inter-abonnés

Le bus est délibérément minimal : une liste ordonnée de fonctions, appelées séquentiellement sans isolation d'erreur (une exception levée par un abonné se propage et interrompt la diffusion aux abonnés suivants). Ce choix reflète le fait que le bus est utilisé exclusivement en configuration mono-processus et synchrone au sein d'un unique appel à `run_round` ; aucune garantie de livraison, de asynchronicité ou de tolérance aux pannes n'est nécessaire, contrairement au tampon `RedisReplayBuffer` de la section 8, dont le rôle est précisément de fournir ces garanties dans un contexte distribué.

### 3.4. `engine.game_runner.Game` : orchestration multi-manches et double interface d'exécution

`Game` maintient `cumulative_vp` et `roles` comme seul état persistant entre manches successives, `round_index` s'incrémentant à chaque appel à `play_round`. La classe publie elle-même `EventGameConfig` et `EventGameStart` à la construction (avant toute manche), ce qui garantit que tout abonné du bus dispose de la configuration complète de la partie avant le premier événement de manche, sans avoir à l'extraire après coup d'un état de manche.

`Game` expose une **double interface d'exécution** strictement indépendante : `play_round`/`play_rounds`/`play_rounds_vectorized` délèguent à `engine.round.run_round` (moteur événementiel complet, avec publication d'événements), tandis que `vectorized_run` délègue à `training.fast_path.vectorized_run` (moteur tensoriel, sans aucune publication d'événement). Ces deux chemins ne partagent aucun état : `vectorized_run` ne met à jour ni `cumulative_vp` ni `roles`, et n'a aucune interaction avec `event_bus`. Ce cloisonnement explicite est ce qui permet à `Game` de rester l'unique point d'entrée haut niveau de la partie sans pour autant contraindre le moteur vectorisé aux mêmes garanties d'observabilité que le moteur événementiel.

## 4. Le système d'événements comme mécanisme d'event-sourcing (`events`)

### 4.1. Hiérarchie de types et séparation structurel/transactionnel

`events.base.Event` est la racine commune, portant les quatre champs universels (`timestamp`, `game_id`, `round_id`, `state_hash`). La distinction entre `events.structural` (le déroulement macroscopique : configuration, démarrage, distribution, ouverture/clôture de pli, sortie, fin de manche) et `events.transactional` (les décisions individuelles : échange, Putsch, action jouée, interception, déclenchement de règle) reflète deux granularités d'analyse différentes : les événements structurels bornent des intervalles temporels exploitables pour des agrégations par manche ou par pli (voir `analytics.metrics_calc.trick_length_average`, qui délimite les plis par comptage d'`EventActionPlayed` entre deux `EventTrickStart`), tandis que les événements transactionnels portent le détail exploitable pour des métriques par décision individuelle.

### 4.2. `state_hash` comme mécanisme de corrélation et de détection de divergence

`compute_state_hash` calcule un SHA-256 de la représentation textuelle canonique (`repr`) de `state.snapshot_key()`. Ce hash n'est pas destiné à un usage cryptographique mais à deux usages d'ingénierie : (1) permettre de vérifier, lors d'un rejeu ou d'une comparaison entre deux exécutions supposées identiques (même graine, même configuration), que l'état exact au moment de chaque événement correspondant est bien identique, sans devoir sérialiser et comparer l'état complet ; (2) fournir une empreinte compacte et stable pour l'indexation ou la déduplication d'événements dans un contexte de stockage à grande échelle (Parquet). Le choix de `repr()` plutôt qu'une sérialisation JSON stricte pour le hachage tire parti du fait que `snapshot_key()` retourne déjà une structure canonique et totalement triée, rendant `repr()` suffisant pour garantir le déterminisme sans dépendance supplémentaire.

### 4.3. `EventActionRequest.legal_action_count` comme métrique de complexité capturée au plus près de la décision

Ce champ est calculé par `engine.round._count_legal_plays` au moment précis de la sollicitation de l'agent (avant que celui-ci ne choisisse), et non recalculé après coup à partir de l'action effectivement choisie. Ce choix garantit que la métrique de branchement (section 6.4) mesure fidèlement l'espace de décision réel offert à l'agent à cet instant, indépendamment de la stratégie qu'il adopte ensuite, une propriété qui serait perdue si le nombre d'options légales était dérivé rétrospectivement de l'action jouée.

### 4.4. `EventActionPlayed.was_suboptimal` comme unique canal de détection d'invalidité

Ce booléen est le seul mécanisme par lequel une action initialement proposée par un agent, mais rejetée par `core.rules_engine.is_action_valid` (ou un passe alors qu'une option légale existait), devient observable après coup. Le choix de porter cette information sur l'événement de l'action *effectivement appliquée* (le passe de substitution), plutôt que de publier un événement distinct de rejet, simplifie le modèle d'événements au prix de la perte de l'action originale erronée, celle-ci n'est jamais journalisée, seul le fait qu'une substitution a eu lieu l'est.

## 5. La couche d'agents et le contrat polymorphe (`agents`)

### 5.1. `AbstractBaseAgent` : quatre points de sollicitation et un point d'optimisation optionnel

Le moteur ne connaît des agents que les quatre méthodes abstraites (`choose_action`, `choose_exchange_cards`, `ask_putsch`, `on_interception_opportunity`) et ne fait aucune supposition sur leur implémentation interne au-delà du type de retour, validé après coup par le moteur (section 2.4). `get_batch_action` est fourni avec une implémentation par défaut purement séquentielle (`[self.choose_action(state) for state in game_states]`), qui n'est jamais utilisée par le moteur événementiel lui-même (celui-ci ne traite qu'un état à la fois) : ce point d'extension existe exclusivement à l'intention de code d'entraînement personnalisé souhaitant exploiter une inférence par lot sur GPU (voir section 8), et n'est override que par `RLAgent` et `TorchRLAgent`.

### 5.2. Duplication volontaire de `_legal_options` entre agents

Chaque agent automatisé (`RandomBot`, `GreedyBot`, `RuleBasedBot`, `MCTSBot`, `HumanAgent`, `RLAgent`, `TorchRLAgent`) réimplémente une méthode privée `_legal_options` de structure identique (assemblage des options uniformes et, si `straights_enabled`, des options de suite, avec réduction de la carte de Joker à sa puissance déclarée minimale). Cette duplication n'est pas accidentelle : chaque agent est conçu comme une unité autonome, sans dépendance envers les autres implémentations d'agent, de sorte qu'un agent puisse être copié, modifié ou supprimé sans effet de bord sur les autres profils. Le coût de cette duplication (sept implémentations quasi identiques) est jugé acceptable au regard de la stabilité qu'elle procure : toute modification de la logique de génération d'options reste centralisée dans `core.rules_engine.generate_uniform_plays`/`generate_sequence_plays`, la duplication ne portant que sur l'assemblage, pas sur le calcul de légalité lui-même.

### 5.3. Gradation algorithmique des profils fournis

Les quatre profils automatisés forment une hiérarchie de complexité croissante partageant une même famille de règles de repli, révélatrice de la conception incrémentale du projet :

* `RandomBot` : tirage uniforme sur l'espace complet des options légales, sans aucune heuristique, sert de ligne de base de comparaison statistique pour toute métrique de performance relative.
* `GreedyBot` : minimise localement la puissance résultante de la combinaison posée (`resulting_power`), sans aucune anticipation au-delà du coup courant.
* `RuleBasedBot` : ajoute deux couches de filtrage successif au-dessus de la logique gloutonne, exclusion des options déclenchant la pénalité de sortie étendue quand une alternative existe (`finish_penalty_extended`), puis exclusion des combinaisons de taille ≥ 4 (réserve de puissance, `_RESERVE_COMBINATION_SIZE`) tant que la main compte plus de 4 cartes (`_ENDGAME_HAND_SIZE`), avant de retomber sur la même minimisation gloutonne parmi les options restantes. Cette structure en filtres successifs (`candidates` réduit progressivement) illustre un principe général de composition d'heuristiques par restriction de l'espace de candidats plutôt que par pondération d'un score composite.
* `MCTSBot` : remplace entièrement l'heuristique déterministe par une estimation Monte-Carlo, simulant `rollout_count` fins de manche par option candidate à l'aide de copies profondes (`copy.deepcopy`) de l'état courant, jouées par des `GreedyBot` de référence pour les autres joueurs. Le choix de `GreedyBot` plutôt que `RandomBot` comme politique de rollout reflète un compromis délibéré entre fidélité de simulation (des adversaires purement aléatoires produiraient des estimations peu représentatives d'un jeu réel) et coût de calcul (une politique plus sophistiquée que gloutonne, comme `RuleBasedBot`, alourdirait chaque rollout sans nécessairement améliorer la qualité de l'estimation, `GreedyBot` étant jugé suffisant comme approximation de second ordre).

### 5.4. `MCTSBot._simulate_rollout` : simulation isolée sur copie profonde, jamais sur l'état réel

Le choix de `copy.deepcopy(initial_state)` en tête de chaque rollout, plutôt qu'une structure de retour en arrière (undo/redo) sur l'état partagé, privilégie la simplicité d'implémentation et l'absence totale de risque de fuite d'état entre rollouts au prix d'un surcoût mémoire et CPU proportionnel au nombre de rollouts et à la taille de l'état. Ce choix est cohérent avec le fait que `MCTSBot` est explicitement documenté comme le profil le plus coûteux et réservé aux parties courtes ou à l'analyse ponctuelle plutôt qu'à la simulation de masse.

### 5.5. `agents.rl_agent`/`agents.torch_rl_agent` : deux politiques entraînables partageant un même vecteur de caractéristiques

`FEATURE_DIM = 5` et `_option_features` (défini une seule fois dans `agents.rl_agent`, réimporté par `agents.torch_rl_agent`) constituent le point de couplage volontaire entre les deux voies d'entraînement (linéaire mono-processus et neuronale distribuée) : les deux politiques scorent le même espace de caractéristiques par option (puissance normalisée, ratio de taille répété deux fois, redondance délibérée renforçant le poids de ce facteur dans une politique purement linéaire, indicateur de Joker, biais constant), ce qui garantit que des transitions collectées sous l'une des deux architectures restent structurellement compatibles avec l'autre, et que `training.rollout_worker` peut réutiliser directement `_option_features` de `agents.rl_agent` sans dupliquer la définition du vecteur de caractéristiques.

`PolicyNet` (`agents.torch_rl_agent`) remplace la simple projection linéaire `features @ weights` par un perceptron à deux couches cachées de 32 neurones (`_HIDDEN_DIM`) avec activations ReLU, conservant la même interface d'entrée/sortie (un score scalaire par vecteur de caractéristiques) : ce remplacement est donc strictement local à l'intérieur de `choose_action`/`get_batch_action`, sans qu'aucune autre partie du système n'ait à distinguer les deux architectures de politique.

## 6. La couche analytique (`analytics`)

### 6.1. `EventLogger` : accumulation en mémoire et double stratégie d'export Parquet

`EventLogger.__call__` fait de l'instance elle-même une fonction abonnable directement au bus, évitant d'exposer une méthode `on_event` distincte que l'appelant devrait explicitement relier. Deux stratégies d'export Parquet coexistent délibérément dans la même classe, pour deux besoins distincts : `to_parquet` construit une unique table à schéma large (une colonne par champ de tout type d'événement rencontré), adaptée à une analyse colonnaire ponctuelle après une exécution unique ; `flush_to_parquet`/`close`, utilisées par `research.run_simulation`, écrivent au contraire dans un schéma fixe et étroit (`_STREAM_SCHEMA` : `event_type`, `timestamp`, `game_id`, `round_id`, `state_hash`, `payload` en JSON), permettant une écriture incrémentale par lots bornés (`parquet_buffer_size`) sans jamais matérialiser l'ensemble des événements d'une campagne massive en mémoire simultanément, un compromis explicite entre facilité d'exploitation (schéma large, colonnes directement typées) et scalabilité (schéma étroit, écriture en flux).

`_serialize_value` traite récursivement quatre cas : primitifs (identité), tuples (conversion en listes, nécessaire car Parquet/JSON ne portent pas de type tuple), dataclasses (aplatissement récursif champ par champ, appliqué en particulier aux `Card` imbriquées dans des `Tuple[Card, ...]`), et énumérations (réduction à leur `.value`) ; tout objet ne correspondant à aucun de ces cas retombe sur `repr()` comme filet de sécurité générique.

### 6.2. `analytics.metrics_calc` : reconstruction d'état par relecture séquentielle plutôt que requête indexée

Plusieurs métriques comportementales (`missed_interception_rate`, `card_ttl`, `capture_efficiency_ratio`, `trick_dominance_factor`, `revolution_counter_attack_rate`) opèrent par balayage séquentiel unique du journal (`for event in logger.events`), maintenant un état local minimal reconstruit au fil de la lecture (mains courantes, pli courant, manche courante). Ce choix, plutôt qu'une indexation préalable de type base de données en mémoire, reflète la nature intrinsèquement chronologique du journal d'événements : la plupart des métriques nécessitent de connaître un contexte qui n'existe que par accumulation progressive (par exemple, `missed_interception_rate` doit reconstruire l'état exact des mains à l'instant précis de chaque diffusion d'interception, ce qui n'est possible qu'en rejouant `EventRoundStart`→`EventExchange`→`EventActionPlayed` dans l'ordre chronologique réel).

### 6.3. Compilation Numba ciblée sur les noyaux numériques réutilisables

`_gini_from_sorted` et `_pearson_correlation_jit` sont les deux seules fonctions de `metrics_calc` compilées via Numba, toutes deux étant des noyaux purement numériques opérant sur des tableaux `numpy` déjà construits, invocables potentiellement des milliers de fois lors de l'agrégation de métriques sur des campagnes massives (par exemple pour chaque paire de séries dans une analyse de corrélations multiples) ; les fonctions englobantes (`gini_initial_hand_power`, `_pearson_correlation`, `tax_weight_vp_correlation`, `opening_position_rank_correlation`) restent en Python pur, la conversion `numpy`/tri restant négligeable face au coût du parcours du journal qui les alimente.

### 6.4. `LiveMonitor` : découplage total entre collecte de métriques et affichage

`LiveMonitor` ne collecte rien par lui-même : `record_games(count, rewards)` reçoit passivement les données à afficher, sans jamais interroger directement un `EventLogger`, un `Game` ou un acteur Ray. Ce découplage permet à `research.run_simulation.launch_research` de piloter `LiveMonitor` depuis sa propre boucle d'attente `ray.wait`, sans que `LiveMonitor` n'ait besoin de connaître l'existence de Ray, des acteurs distribués, ou de la nature de la campagne en cours ; `_reward_samples` est borné (`max_reward_samples`) pour garantir une empreinte mémoire constante indépendamment de la durée de la campagne suivie, au prix de la perte des échantillons les plus anciens (fenêtre glissante plutôt qu'historique complet).

## 7. Le moteur vectorisé et la double implémentation des règles (`training.fast_path`, `core.action_masking`)

### 7.1. Représentation tensorielle de la main : comptage par rang plutôt qu'ensemble de cartes

`FastPathState.hands` est un tenseur `(B, N, 14)` de comptages par rang (13 rangs standard + colonne Joker), et non une collection d'objets `Card`. Ce choix est ce qui rend le moteur vectorisable : toute opération de légalité ou de retrait de cartes devient une opération arithmétique sur des colonnes entières du tenseur plutôt qu'une manipulation d'ensembles hétérogènes par partie du lot. La contrepartie assumée est la perte totale de l'information de couleur, incompatible avec la représentation par comptage, d'où l'absence structurelle de l'Interception (qui nécessite un appariement rang+couleur exact) et du départage d'égalité par couleur dans ce moteur.

### 7.2. Encodage de l'espace d'action comme produit cartésien rang × taille

`action_space_size()` vaut `_HIDDEN_COLUMNS * max_combo_size` (14 rangs possibles, dont le Joker pur, multipliés par les tailles de 1 à `max_combo_size`), et `_decode_action` retrouve le couple `(rank_index, size)` par division/modulo entiers. Cet encodage dense et fixe, indépendant du contenu réel de la main à un instant donné, est ce qui permet à `legal_action_mask()` de produire un masque booléen de forme stable `(B, action_space_size())` exploitable directement comme sortie d'un réseau de neurones à tête de classification fixe, sans device de renumérotation dynamique de l'espace d'action d'un pas de temps à l'autre.

### 7.3. `step` : boucle Python explicite sur les lignes actives plutôt que vectorisation complète

Bien que l'état soit tensoriel, `FastPathEngine.step` itère explicitement en Python sur les indices de lignes concernées par une pose (`for row in np.nonzero(is_play)[0]`) plutôt que d'exprimer l'intégralité de la transition comme des opérations `numpy` pleinement vectorisées. Ce choix reflète une limite pragmatique assumée : la richesse des règles conditionnelles à appliquer par ligne (déclenchement de révolution, verrouillage, clôture magique, calcul du prochain joueur actif en tenant compte des joueurs déjà sortis) rendrait une vectorisation complète disproportionnellement complexe à exprimer et à maintenir en `numpy` pur, pour un gain de performance incertain étant donné que le nombre de lignes réellement actives à un pas donné (`is_play.sum()`) reste généralement une fraction modeste de `batch_size`. Le gain de vectorisation du moteur provient donc principalement de la construction du masque de légalité (`legal_action_mask`, qui est, elle, pleinement vectorisée sur l'ensemble du lot et de l'espace d'action) plutôt que de l'application des transitions elles-mêmes.

### 7.4. `core.action_masking` comme pont indépendant entre options énumérées et espace fixe

Contrairement à `FastPathEngine`, qui construit son propre espace d'action interne, `core.action_masking.build_action_space_index`/`build_action_mask_batch`/`legal_option_for_action` opèrent en aval des options réellement énumérées par `core.rules_engine.generate_uniform_plays`/`generate_sequence_plays` (donc compatibles avec le moteur événementiel complet, Suites comprises), projetées sur un espace `(power, size)` fixe plutôt que sur l'espace `(rank, size)` du moteur vectorisé. Cette différence d'espace (puissance résolue plutôt que rang facial brut) reflète le fait que ce module est pensé pour être combiné au moteur événementiel complet, où la notion pertinente pour un réseau de neurones est la puissance effective de la combinaison (qui dépend de l'état de révolution), et non le rang facial brut de la carte, à la différence du moteur vectorisé, qui encode le rang brut car il gère lui-même la conversion `rank → power` en interne à chaque pas (`_power_for_rank`).

## 8. L'architecture d'entraînement distribué (`training.*`)

### 8.1. Séparation Rollout Worker / Trainer comme pattern acteur-apprenant classique

L'architecture distribuée sépare strictement la production de transitions (`RolloutWorker`, acteurs Ray stateless du point de vue du tampon partagé, chacun maintenant néanmoins un état local `round_index`/`roles` pour la continuité des rôles d'une manche à l'autre au sein de son propre flux de parties) et la consommation/apprentissage (`Trainer`, processus unique). Ce découplage permet de faire varier indépendamment le nombre de Rollout Workers (borné par les cœurs CPU disponibles pour la simulation) et la présence ou non d'un GPU pour le `Trainer` (le calcul de gradient bénéficiant d'une accélération matérielle que la simulation de règles de jeu, intrinsèquement séquentielle et peu parallélisable au niveau de l'instruction, ne peut pas exploiter de la même manière).

### 8.2. Synchronisation asynchrone et faiblement couplée via Redis

Les Rollout Workers ne reçoivent jamais directement les poids mis à jour par le `Trainer` par un canal de communication directe (RPC, mémoire partagée) : ils les récupèrent en interrogeant périodiquement la clé Redis `president:policy_weights` au début de chaque nouvelle manche simulée (`RolloutWorker._load_latest_weights`, appelée à chaque itération de `run_rounds`). Ce couplage faible signifie qu'un Rollout Worker peut simuler plusieurs manches avec une version légèrement périmée de la politique avant de récupérer la version suivante, un choix délibéré de cohérence relâchée (« eventually consistent ») privilégiant le débit de simulation (aucune synchronisation bloquante entre workers et Trainer à chaque étape) au prix d'un léger décalage entre la politique utilisée pour explorer et la politique la plus récente.

### 8.3. Le tampon de rejeu comme file bornée et non comme structure d'échantillonnage priorisé

`RedisReplayBuffer` implémente un simple FIFO tronqué (`LPUSH` + `LTRIM` à chaque insertion), et un échantillonnage strictement uniforme avec remise (`sample`, tirage d'index aléatoires suivi de lectures individuelles `LINDEX`). Aucune priorisation d'échantillonnage (par exemple par magnitude d'avantage, comme dans un tampon à priorité) n'est implémentée : ce choix reflète la simplicité recherchée pour ce composant, dont le rôle premier est de découpler dans le temps la production et la consommation de transitions plutôt que d'optimiser la vitesse de convergence de l'apprentissage par un schéma d'échantillonnage sophistiqué.

### 8.4. Instrumentation par substitution de méthode plutôt que par sous-classement

`training.train_rl._run_training_round` et `training.rollout_worker.RolloutWorker.run_rounds` instrumentent tous deux la collecte de transitions par la même technique : remplacement temporaire de `trainee.choose_action` par une fermeture instrumentée (`_instrumented_choose`) qui appelle la méthode originale puis enregistre les caractéristiques de la décision prise, avant de restaurer la méthode originale dans un bloc `finally`. Ce choix, plutôt que de faire hériter une classe d'entraînement spécifique de `RLAgent`/`TorchRLAgent` qui surchargerait `choose_action`, permet de réutiliser telles quelles les classes d'agent définies dans `agents.rl_agent`/`agents.torch_rl_agent` (utilisables aussi bien en entraînement qu'en évaluation ou en partie jouée) sans avoir à maintenir une variante « entraînable » distincte de chaque agent.

### 8.5. Précision mixte conditionnée au type de périphérique, jamais forcée

`Trainer.__init__` détermine `use_amp = self.device.type == "cuda"` et instancie `torch.cuda.amp.GradScaler(enabled=self.use_amp)` en conséquence ; `TorchRLAgent` applique la même logique pour son propre `use_amp`. La précision mixte n'est donc jamais activée sur CPU (où `torch.autocast` n'apporterait aucun bénéfice et pourrait même dégrader la précision numérique sans accélération correspondante), ce qui évite d'avoir à exposer un paramètre de configuration supplémentaire pour ce comportement : il se déduit entièrement et automatiquement du périphérique résolu par `resolve_device`.

### 8.6. `training.launch_distributed` : vérification préalable comme principe de défaillance rapide

`launch` vérifie la joignabilité de Redis (`RedisReplayBuffer.ping()`) avant tout appel à `ray.init` ou démarrage de thread. Ce choix de séquencement, vérifier la dépendance la plus fragile et la plus susceptible d'être mal configurée (un serveur Redis externe, potentiellement non démarré) avant d'engager des ressources plus coûteuses à initialiser et à nettoyer (un cluster Ray local, un thread de Trainer), illustre un principe général de défaillance rapide (« fail fast ») retenu dans toute la chaîne de lancement distribué, en évitant qu'une erreur de configuration ne se manifeste tardivement sous une forme obscure (blocage silencieux d'un thread, exception profondément imbriquée dans un acteur Ray).

## 9. Diagrammes de séquence

### 9.1. Séquence d'une manche complète (moteur événementiel)

```
Game.play_round()
  -> engine.round.run_round(config, agents, event_bus, round_index, roles, game_id)
       -> build_deck(config) ; deal_hands(config, deck, round_index, ...)
       -> GameState(...)
       -> emit(EventRoundStart, initial_hands=...)
       -> [si Putsch actif] emit(EventAskPutsch) -> agents[scum].ask_putsch(hand)
            -> [si invoqué] emit(EventPutschInvoked)
       -> [si échange actif et Putsch non invoqué]
            -> max_power_cards / random_cards -> emit(EventExchange) x2
            -> agents[giver].choose_exchange_cards(...) -> emit(EventExchangeIntent) -> emit(EventExchange) x2
       -> boucle plis :
            emit(EventTrickStart, opener_id, trick_index)
            boucle actions :
                emit(EventActionRequest, legal_action_count=_count_legal_plays(...))
                action = agents[pid].choose_action(state)
                is_action_valid(...) -> [invalide] substitution par passe, was_suboptimal=True
                [ACTION_PLAY] _apply_play(...) -> emit(EventActionPlayed)
                    -> déclenchements : EventRuleTriggered(REVOLUTION/DOUBLE_REVOLUTION/MAGIC_CLOSURE/SKIP_TURN)
                    -> [interception] emit(EventInterceptionBroadcast) -> agents[k].on_interception_opportunity
                         -> emit(EventInterceptionResolved) -> [succès] emit(EventRuleTriggered INTERCEPTION), emit(EventActionPlayed)
                    -> [main vide] _finish_player(...) -> emit(EventHandEmpty), emit(EventPlayerFinished)
                [PASS] emit(EventActionPlayed, cards_played=())
            emit(EventTrickClosed, winner_id, trick_size)
       -> [dernier joueur restant] emit(EventPlayerFinished)
       -> compute_vp / role_for_rank pour chaque rang de sortie
       -> emit(EventRoundEnd, vp_by_player, roles_by_player)
       <- retour (roles_by_player, vp_by_player, finish_order)
  <- mise à jour de Game.cumulative_vp, Game.roles, Game.round_index
```

### 9.2. Séquence d'une étape d'entraînement distribué

```
training.launch_distributed.launch(...)
  -> RedisReplayBuffer(...).ping() -> [échec] RuntimeError immédiate
  -> ray.init(...)
  -> Trainer(...) démarré dans un thread (Trainer.run(batch_size, total_steps))
       -> attente active : buffer.size() >= batch_size
       -> boucle total_steps :
            train_step(batch_size) -> buffer.sample(batch_size)
                -> features/returns -> avantage = return - baseline
                -> policy(features) -> loss = -(scores * advantages).mean()
                -> scaler.scale(loss).backward() ; scaler.step(optimizer) ; scaler.update()
            [tous les _PUBLISH_INTERVAL pas] _publish_weights() -> Redis[president:policy_weights]
  -> boucle principale : tant que le thread Trainer est vivant
       -> pour chaque RolloutWorker : run_rounds.remote(opponent_pool, rounds_per_worker_batch)
            -> pour chaque manche :
                 _load_latest_weights(trainee) <- Redis[president:policy_weights]
                 run_round(config, agents={0: trainee, ...}, EventBus(), round_index, roles, game_id)
                      (trainee.choose_action instrumenté -> collecte de transitions)
                 return_value = vp_by_player[trainee.player_id]
                 buffer.push_batch(transitions avec return_value renseigné)
       -> ray.get(futures)  # attente de la fin du lot avant redistribution
```

## 10. Invariants garantis et contrats inter-modules

* **Aucune fonction de `core.rules_engine` ne mute ses arguments.** Toute combinaison retournée par `generate_uniform_plays`/`generate_sequence_plays` est un nouveau tuple ; la main source (`Hand`) n'est jamais modifiée par ces fonctions, seule `engine.round._apply_play` appelle `Hand.without` pour produire une nouvelle main assignée à `state.hands[pid]`.
* **Toute action appliquée à `GameState` a préalablement passé `is_action_valid`.** C'est la seule porte de validation du moteur événementiel ; aucune autre fonction de `engine.round` ne revalide la légalité d'une combinaison déjà acceptée.
* **`state_hash` est toujours calculé après la mutation qui vient de se produire, jamais avant.** Chaque appel à `emit` dans `run_round` intervient après que la ligne de code précédente a fini de muter `state` pour cette transition, garantissant que le hash associé à un événement reflète fidèlement l'état résultant de cet événement et non l'état qui le précédait.
* **Un joueur marqué `is_finished=True` ne redevient jamais éligible ni actif pour la manche courante.** `_advance_player` et `active_players` filtrent systématiquement sur ce booléen ; aucune règle avancée (y compris la pénalité `PENALTY_DRAW_CARDS`, qui réintègre des cartes en main) ne réinitialise `is_finished` à `False` une fois positionné à `True`, la réintégration de cartes via `PENALTY_DRAW_CARDS` intervient avant même que `is_finished` ne soit positionné (`_finish_player` retourne avant cette affectation dans ce cas précis), ce qui évite toute incohérence entre l'état de sortie et la présence de cartes en main.
* **`random_seed` ne garantit la reproductibilité stricte que pour les composants qui en dérivent explicitement leur propre graine locale.** `deal_hands` dérive `f"{random_seed}:{round_index}"`, les agents fournis dérivent `f"{random_seed}:{player_id}[...]"` ; tout agent personnalisé ou toute source d'aléa additionnelle non dérivée de cette convention rompt la garantie de reproductibilité globale d'une exécution.
* **Le moteur vectorisé et le moteur événementiel ne partagent aucun état ni aucune instance.** Un `GameConfig` peut être transmis aux deux, mais `FastPathState` et `GameState` sont des types disjoints, jamais convertis l'un vers l'autre ; toute divergence de comportement entre les deux moteurs sur le sous-ensemble de règles qu'ils ont en commun constituerait un défaut à corriger indépendamment dans chacune des deux implémentations.

## 11. Limites connues et points d'extension

* Le moteur vectorisé ne modélise ni les couleurs, ni les Suites, ni l'Interception, ni le Putsch, ni la Taxe Aveugle, ni la pénalité de sortie étendue (section 1.3) : toute recherche nécessitant l'entraînement sous ces règles doit s'appuyer sur le moteur événementiel complet, au prix d'un débit de simulation très inférieur.
* Le bus d'événements ne fournit aucune garantie transactionnelle : une exception levée par un abonné interrompt la diffusion aux abonnés suivants pour l'événement courant (section 3.3). Un futur mécanisme d'isolation par abonné (capture d'exception individuelle, files par abonné) constituerait une extension naturelle si des abonnés non fiables devaient être ajoutés au bus en production.
* Le tampon Redis ne priorise pas l'échantillonnage (section 8.3) ; l'introduction d'un tampon à priorité (par exemple pondéré par la magnitude de l'avantage observé) est un point d'extension direct de `RedisReplayBuffer.sample`, sans impact sur l'interface `push`/`push_batch`/`size` consommée par les Rollout Workers.
* `core.action_masking` et `training.fast_path` maintiennent chacun leur propre encodage d'espace d'action (`(power, size)` contre `(rank_index, size)`) sans passerelle directe entre les deux : un agent entraîné sur l'un des deux espaces ne peut pas être directement transposé vers l'autre sans réécrire une couche de traduction dédiée.
* L'ajout d'un nouvel agent au registre de `play_game.py` ou de `research.run_simulation` nécessite une modification manuelle du dictionnaire `_AGENT_REGISTRY` correspondant dans chacun de ces deux points d'entrée séparément, ces registres n'étant pas mutualisés entre les deux scripts.

## 12. Couche d'orchestration de bout en bout (`research.run_pipeline`, `research.generate_graphs`)

`research.run_pipeline.run_pipeline` est le point d'entrée unique orchestrant, sans intervention humaine, l'ensemble du cycle de recherche :
entraînement de l'agent linéaire (`training.train_rl`), tentative d'entraînement distribué de l'agent neuronal (`training.launch_distributed`,
ignorée proprement si Redis n'est pas joignable plutôt que de bloquer le pipeline), simulations de référence par profil heuristique
(`research.run_simulation`), évaluation comparative (`research.evaluate_agent`), génération de graphiques (`research.generate_graphs`) et
rédaction d'un rapport de synthèse. Chaque étape est journalisée dans `data/pipeline_state.json` avec un statut `"done"`/`"failed"` ; une
étape déjà `"done"` n'est jamais réexécutée, ce qui permet de relancer le pipeline après une interruption (arrêt du processus, panne) sans
perdre le travail déjà accompli. Une étape en échec ne bloque pas les étapes suivantes et est retentée au prochain lancement.

`research.generate_graphs.generate_all` est la contrepartie non interactive du carnet d'analyse historique : chaque fonction de tracé y est
isolée et protégée par une garde d'absence de données, de sorte que l'indisponibilité d'une source (par exemple aucun événement
`EventInterceptionBroadcast` si l'Interception n'a jamais été activée) n'empêche pas la génération des autres graphiques de la même
exécution.

Un modèle scientifique des résultats statistiquement attendus pour chaque graphique produit par ce module, destiné à servir de base à des
tests de validation de l'implémentation plutôt qu'à une simple inspection visuelle, est disponible séparément.