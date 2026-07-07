# RÈGLES

## Table des matières

1. [Définitions Fondamentales et Vocabulaire](#1-définitions-fondamentales-et-vocabulaire)
2. [Configuration (`GameConfig`)](#2-configuration-gameconfig)
3. [Matériel, Dimensionnement et Hiérarchie Mathématique](#3-matériel-dimensionnement-et-hiérarchie-mathématique)
   - 3.1. [Le Paquet (`Deck`)](#31-le-paquet-deck)
   - 3.2. [Fonctions de Valeur](#32-fonctions-de-valeur)
4. [Rôles et Points de Victoire (`VictoryPoints`)](#4-rôles-et-points-de-victoire-victorypoints)
5. [Algorithme d'une Manche (`Round`)](#5-algorithme-dune-manche-round)
   - 5.1. [Phase Initiale](#51-phase-initiale)
   - 5.2. [Échange (`exchange_phase`)](#52-échange-exchange_phase)
   - 5.3. [Phase de Jeu (`trick_phase`)](#53-phase-de-jeu-trick_phase)
   - 5.4. [Fin de Manche (`round_closing`)](#54-fin-de-manche-round_closing)
6. [Règles Supplémentaires et Événements de Jeu](#6-règles-supplémentaires-et-événements-de-jeu)
   - 6.1. [Modificateurs de Pré-Manche (`pre_round_modifiers`)](#61-modificateurs-de-pré-manche-pre_round_modifiers)
   - 6.2. [Clôture Magique Alternative (`magic_card`)](#62-clôture-magique-alternative-magic_card)
   - 6.3. [Forçage par Égalité (`skip_on_equal`)](#63-forçage-par-égalité-skip_on_equal)
   - 6.4. [Substitution Joker (`use_jokers`)](#64-substitution-joker-use_jokers)
   - 6.5. [Révolution et Double Révolution](#65-révolution-et-double-révolution-revolution_enabled-double_revolution_enabled)
   - 6.6. [Les Suites / Escaliers (`straights_enabled`)](#66-les-suites--escaliers-straights_enabled)
   - 6.7. [Saut de Tour (`skip_turn_enabled`)](#67-saut-de-tour-skip_turn_enabled)
   - 6.8. [L'Interception / Fermeture à la volée (`interception_enabled`)](#68-linterception--fermeture-à-la-volée-interception_enabled)
   - 6.9. [Extension des Pénalités de Clôture (`finish_penalty_extended`)](#69-extension-des-pénalités-de-clôture-finish_penalty_extended)
7. [Matrice de Compatibilité et Résolution des Conflits (Truth Table)](#7-matrice-de-compatibilité-et-résolution-des-conflits-truth-table)

Ce document est la référence normative citée par [`architecture.md`](architecture.md#1-principes-de-conception-globaux) (rationale d'implémentation) et par [`expected_results.md`](expected_results.md) (résultats statistiques attendus par mécanique).

## 1. Définitions Fondamentales et Vocabulaire

*   **`Player`** : Élément $p_i \in P = \{p_0, p_1, ..., p_{N-1}\}$ où $N$ est le nombre total de joueurs ($N \ge 3$). Chaque joueur est une instance d'un agent cognitif (Humain ou Bot) possédant un identifiant unique `player_id`.
*   **`Game`** : L'instance d'exécution globale. Constituée d'une séquence de $M$ objets `Round`.
*   **`Round`** : Une manche de jeu. Débute par la distribution (`deal_cards`), passe par l'échange (`exchange_cards`) et se termine par la complétion (`round_closing`) lorsque $N-1$ joueurs ont l'attribut `hand_size == 0`.
*   **`Trick`** : Un pli. Séquence d'actions (`Action`) débutant par une table vide et se terminant par un état de clôture (`is_closed == True`).
*   **`Turn`** : L'opportunité d'action d'un `Player` $p_i$ à l'instant $t$.
*   **`Action`** : Vecteur de décision émis par un agent. Énumération d'états d'action (`ACTION_PLAY`, `ACTION_SOFT_PASS`, `ACTION_HARD_PASS`), assorti d'un sous-ensemble de cartes $C$ et, lorsque nécessaire, d'une **intention de valeur** `declared_power` (obligatoire pour la résolution des Jokers, cf. section 6.4 et Document 3 section 2.B).
*   **`Card`** : Objet défini par un rang facial (`rank`), une couleur (`suit`), une valeur de puissance dynamique $f_{power}(c, E_{rev})$ et une valeur de points statique $f_{points}(c)$.
*   **`Combination`** : Sous-ensemble $C \subset \text{Hand}$ de taille $X = |C|$.
    *   Toutes les cartes $c \in C$ doivent satisfaire $f_{power}(c_a) == f_{power}(c_b)$ (hors utilisation de Jokers en tant que `Wildcard`, ou activation de la règle des Suites, cf. section 6.6).

## 2. Configuration (`GameConfig`)

Objet de paramètres immuable passé à l'initialisation de la `Game`. Union exhaustive des paramètres retenus :

**Système Core**
*   `random_seed` (int) : Graine de génération aléatoire pour la reproductibilité.
*   `player_count` (int) : $N$.
*   `first_trick_opener_id` (int) : Identifiant $p_i$ du joueur ouvrant le $R_0$.

**Topologie du Deck**
*   `deck_scaling_auto` (bool) : Si `True`, le nombre de paquets $N_D$ est calculé dynamiquement.

**Règles de Flux**
*   `pass_type` (str) : `'HARD_ONLY'` (un passe exclut définitivement le joueur du pli en cours) ou `'ALLOW_SOFT'` (un passe autorise le joueur à ressurenchérir plus tard dans le même pli si le tour lui revient). Détermine laquelle des deux sémantiques d'`Action` (`ACTION_SOFT_PASS` / `ACTION_HARD_PASS`) est légale/forcée durant la partie (voir section 5.3.2 et section 6.3).
*   `vp_distribution_type` (str) : `'SYMMETRICAL'` (par défaut, Z-score centré sur 0) ou `'LINEAR'` (voir section 4).

**Modificateurs de Jeu**
*   `use_jokers` (bool) : Activation des Jokers.
*   `magic_two` (bool) : Activation de la règle de clôture par le 2 (cas historique particulier).
*   `magic_two_single_clears_all` (bool) : Un seul 2 clôture un pli de taille $X \ge 1$.
*   `magic_card_enabled` (bool) : Généralisation de `magic_two`, activation d'une clôture magique sur un rang paramétrable (`magic_two` ⟺ `magic_card_enabled` avec `magic_card_rank = 2`).
*   `magic_card_rank` : Rang défini comme magique (par défaut 2, peut être 10, etc.).
*   `magic_single_clears_all` (bool) : Généralisation de `magic_two_single_clears_all` pour `magic_card_rank`.
*   `skip_on_equal` (bool) : Activation de l'obligation de jouer suite à une égalité.
*   `revolution_enabled` (bool) : Activation du système de Révolution.
*   `double_revolution_enabled` (bool) : Activation de la Double Révolution (nécessite $N_D \ge 2$).
*   `straights_enabled` (bool) : Activation des Suites/Escaliers.
*   `skip_turn_enabled` (bool) : Activation du Saut de Tour.
*   `skip_turn_rank` : Rang d'évitement $R_{skip}$ (généralement 8).
*   `interception_enabled` (bool) : Activation de l'Interception (nécessite $N_D \ge 2$).

**Modificateurs d'État / Pénalité**
*   `putsch_enabled` (bool) : Activation du Putsch.
*   `blind_tax_enabled` (bool) : Activation de la Taxe Aveugle.
*   `strict_remainder_allocation` (bool) : Activation de l'attribution stricte du reste de la distribution.
*   `strict_remainder_role` : Rôle ciblé par `strict_remainder_allocation` (ex : `ROLE_SCUM`).
*   `finish_penalty_enabled` (bool) : Activation de la pénalité de sortie.
*   `finish_penalty_type` (str) : `PENALTY_INSTANT_SCUM` ou `PENALTY_DRAW_CARDS`.
*   `finish_penalty_draw_count` (int) : Nombre de cartes $K$ à piocher si `PENALTY_DRAW_CARDS`.
*   `finish_penalty_extended` (bool) : Extension des conditions de pénalité (cf. section 6.9), incluant les sous-conditions `no_finish_on_joker` et `no_finish_on_revolution`.

## 3. Matériel, Dimensionnement et Hiérarchie Mathématique

### 3.1. Le Paquet (`Deck`)
Le nombre de paquets de 52 cartes $N_D$ dépend de $N$ (si `deck_scaling_auto == True`) :
*   $N_D = \max(1, \lfloor \frac{N - 1}{4} \rfloor + 1)$
*   Taille totale du paquet $S_{deck} = N_D \times 52$ (ou $N_D \times 54$ si `use_jokers == True`), soit formellement $S_{deck} = N_D \times (52 + 2 \times \text{int}(\text{use\_jokers}))$.
*   Une `Combination` peut donc théoriquement atteindre une taille $X_{max} = 4 \times N_D$.

### 3.2. Fonctions de Valeur

Soit $R$ l'ensemble ordonné des rangs faciaux : $\{3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A, 2, Joker\}$.

**A. Valeur de Points de Fin de Partie : $f_{points}(c)$** (Constante absolue, identique à $f_{std}(c)$)
*   $c \in \{3..10\} \rightarrow f_{points}(c) = \text{valeur faciale}$
*   $c = J \rightarrow 11$, $c = Q \rightarrow 12$, $c = K \rightarrow 13$, $c = A \rightarrow 14$, $c = 2 \rightarrow 15$
*   $c = Joker \rightarrow 16$

**B. Valeur de Puissance de Jeu : $f_{power}(c, E_{rev})$** (Dynamique)

Soit l'état booléen de la Révolution $E_{rev} \in \{True, False\}$.

*Formulation détaillée (référence) :*
*   Si $E_{rev} == False$ (Standard) :
    *   $f_{power}(3..10) = 3..10$
    *   $f_{power}(J) = 11$, $f_{power}(Q) = 12$, $f_{power}(K) = 13$, $f_{power}(A) = 14$, $f_{power}(2) = 15$
*   Si $E_{rev} == True$ (Inversé) :
    *   $f_{power}(2) = 3$
    *   $f_{power}(A) = 4$, $f_{power}(K) = 5$, $f_{power}(Q) = 6$, $f_{power}(J) = 7$
    *   $f_{power}(10..3) = 8..15$
*   Dans tous les cas, le Joker n'est pas affecté par $E_{rev}$ : $f_{power}(Joker) = 16$.

*Formulation algébrique canonique (Théorème de l'Inversion)*, équivalente à la formulation détaillée ci-dessus, et à privilégier pour toute implémentation :

Pour toute carte $c \neq Joker$, la puissance standard $f_{std}(c)$ est identique à $f_{points}(c)$, et sa puissance inversée est le symétrique par rapport à la médiane (9). La somme de la puissance standard et de la puissance inversée d'une carte vaut toujours **18** (ex : le 2 vaut $15$, inversé il vaut $3$ ; $15+3=18$. Le 3 vaut $3$, inversé il vaut $15$ ; $3+15=18$).

$$f_{power}(c, E_{rev}) = \begin{cases} 16 & \text{si } c = Joker \\ f_{std}(c) & \text{si } E_{rev} = False \\ 18 - f_{std}(c) & \text{si } E_{rev} = True \end{cases}$$

Le mécanisme qui fait basculer $E_{rev}$ en cours de partie est détaillé en [section 6.5](#65-révolution-et-double-révolution-revolution_enabled-double_revolution_enabled).

## 4. Rôles et Points de Victoire (`VictoryPoints`)

À la clôture de $R_m$, on obtient la liste de sortie $O = [p_{o_0}, p_{o_1}, ..., p_{o_{N-1}}]$, où l'index $k$ correspond à l'ordre de sortie ($k=0$ étant le premier joueur à finir). Cette liste $O$ est construite progressivement au fil de la manche et finalisée en [section 5.4](#54-fin-de-manche-round_closing).

**Attribution des rôles (indépendante du mode de calcul des VP) :**
*   $k = 0$ : `ROLE_PRESIDENT`
*   $k = 1$ : `ROLE_VICE_PRESIDENT`
*   $2 \le k \le N-3$ : `ROLE_NEUTRAL`
*   $k = N-2$ : `ROLE_VICE_SCUM`
*   $k = N-1$ : `ROLE_SCUM`

*Remarque* : même lorsque plusieurs joueurs `ROLE_NEUTRAL` obtiennent un $VP$ identique (mode `SYMMETRICAL`), il est impératif de conserver l'ordre exact de sortie au sein des neutres : un neutre terminant plus tôt qu'un autre reste, pour l'analyse, une performance supérieure.

*Cas particuliers structurels de $N$* :
*   Si $N = 3$ : $O[0]$=`ROLE_PRESIDENT`, $O[1]$=`ROLE_NEUTRAL`, $O[2]$=`ROLE_SCUM`.
*   Si $N = 4$ : Les rôles `ROLE_NEUTRAL` n'existent pas.

**Mode `vp_distribution_type = 'LEGACY_STEPPED'` (barème historique)** :
*   $k = 0$ : $VP = N$
*   $k = 1$ : $VP = N - 1$
*   $2 \le k \le N-3$ : $VP = N - k$
*   $k = N-2$ : $VP = 0$
*   $k = N-1$ : $VP = -1$

**Mode `vp_distribution_type = 'LINEAR'`**, récompense linéaire, sans rupture de pente :
$$VP_{linear}(k) = N - 1 - k$$
(le premier reçoit $N-1$, le dernier reçoit $0$.)

**Mode `vp_distribution_type = 'SYMMETRICAL'`** (par défaut recommandé), récompense centrée sur 0, de type Z-score, plus lisible pour l'évaluation de la *fitness* d'un agent :
$$VP(k) = \frac{N-1}{2} - k$$

*Exemple pour $N=5$ :*
*   $k = 0$ (`PRESIDENT`) $\rightarrow VP = 2.0$
*   $k = 1$ (`VICE_PRESIDENT`) $\rightarrow VP = 1.0$
*   $k = 2$ (`NEUTRAL`) $\rightarrow VP = 0.0$
*   $k = 3$ (`VICE_SCUM`) $\rightarrow VP = -1.0$
*   $k = 4$ (`SCUM`) $\rightarrow VP = -2.0$

*Justification du remplacement du barème historique* : dans le barème `LEGACY_STEPPED`, pour $N=5$, la séquence $(5, 4, 3, 0, -1)$ présente une rupture brutale entre le `NEUTRAL` (3) et le `VICE_SCUM` (0). Les modes `LINEAR` et `SYMMETRICAL` corrigent cette discontinuité pour les besoins d'entraînement de modèles et de comparaison de profils.

## 5. Algorithme d'une Manche (`Round`)

### 5.1. Phase Initiale

**5.1.1. Distribution (`deal_phase`)**
Vecteur initial des mains $H = [H_0, H_1, ..., H_{N-1}]$. Le paquet global de taille $S_{deck}$ est mélangé via `random_seed`. Les cartes sont distribuées modulo $N$. Pour tout joueur $p_i$, $|H_i| \in \{\lfloor \frac{S_{deck}}{N} \rfloor, \lceil \frac{S_{deck}}{N} \rceil\}$.

*Modificateur, Attribution stricte du reste (`strict_remainder_allocation`)* : Si $R_{cards} = S_{deck} \pmod N \neq 0$ et que `strict_remainder_allocation == True`, les cartes restantes ne sont pas distribuées modulo $N$. Elles sont allouées d'un bloc à la main $H_k$ du joueur possédant le rôle ciblé par `strict_remainder_role` (ex : `ROLE_SCUM`).

**5.1.2. Vérification du Putsch (`pre_exchange_check`)**
Évalué strictement entre `deal_phase` et `exchange_phase`, uniquement si `putsch_enabled == True`. Le moteur interroge l'agent `ROLE_SCUM`. Si $H_{scum}$ valide une condition $P_{putsch}$ (ex : $\exists C \subset H_{scum}, |C| \ge 4$ ou $\max_{power}(H_{scum}, 1) \le 10$) et que l'agent choisit de l'invoquer, la `exchange_phase` entière est annulée pour $R_m$.

### 5.2. Échange (`exchange_phase`)

Condition : Uniquement si $m > 0$ (où $m$ est l'index de la `Round` actuelle) **et** Putsch non invoqué.

Soit $\max_{power}(H, n)$ la fonction retournant les $n$ cartes ayant le $f_{power}$ maximal.

1.  `ROLE_SCUM` transfère $\max_{power}(H_{scum}, 2)$ au `ROLE_PRESIDENT` (ou une sélection aléatoire uniforme $rand(H_{scum}, 2)$ si `blind_tax_enabled == True`, cf. section 6.1).
2.  `ROLE_PRESIDENT` transfère $2$ cartes au choix de $H_{president}$ au `ROLE_SCUM`.
3.  `ROLE_VICE_SCUM` transfère $\max_{power}(H_{vice\_scum}, 1)$ au `ROLE_VICE_PRESIDENT`.
4.  `ROLE_VICE_PRESIDENT` transfère $1$ carte au choix de $H_{vice\_president}$ au `ROLE_VICE_SCUM`.

*Cas d'égalité* : Si $\max_{power}$ trouve plusieurs cartes de puissance égale, la couleur (`suit`) détermine l'ordre arbitraire défini dans l'implémentation pour garantir le déterminisme.

### 5.3. Phase de Jeu (`trick_phase`)

**5.3.1. Ouverture (`trick_opening`)**
L'identifiant du joueur ouvrant $p_{open}$ est déterminé par :
*   Si $m = 0 \land t_{trick} = 0$ : $p_{open} = $ `first_trick_opener_id`.
*   Si $m > 0 \land t_{trick} = 0$ : $p_{open} = $ identifiant du `ROLE_SCUM` de $R_{m-1}$.
*   Si $t_{trick} > 0$ : $p_{open} = $ identifiant du joueur ayant remporté le pli précédent. Si $H_{open} = \emptyset$, $p_{open} = (p_{open} + 1) \mod N$ en ignorant les joueurs dont l'état est `is_finished == True`.

**5.3.2. Action de Tour (`play_turn`)**
Pour un `Trick` actif avec une taille $X$ et une puissance courante $P_{current}$ à l'instant $t$, le moteur émet une **Requête d'Action** vers l'agent $p_i$, qui doit retourner un objet `Action` :
1.  `ACTION_PLAY` : Poser une `Combination` $C$ de taille $X$ telle que $f_{power}(C) > P_{current}$. Si $C$ contient un Joker, l'`Action` doit obligatoirement spécifier `declared_power` (cf. section 6.4).
2.  `ACTION_SOFT_PASS` : Ne joue pas, mais `is_eligible = True` pour la suite du `Trick`. Légal uniquement si `pass_type == 'ALLOW_SOFT'`.
3.  `ACTION_HARD_PASS` : Ne joue pas, et `is_eligible = False` jusqu'au prochain `trick_opening`. Sémantique forcée si `pass_type == 'HARD_ONLY'`.

*Résolution des passes selon `GameConfig`* :
*   Si `pass_type == 'HARD_ONLY'`, tout passe bascule `is_eligible = False` jusqu'au prochain pli.
*   Si `pass_type == 'ALLOW_SOFT'`, un passe peut maintenir l'éligibilité (`ACTION_SOFT_PASS`), permettant au joueur de resurenchérir si le tour lui revient dans le même `Trick`.

**Broadcast d'Interception** (si `interception_enabled == True`) : voir section 6.8, le moteur suspend le flux séquentiel avant de valider $A_{t+1}$ pour interroger les autres joueurs.

**5.3.3. Clôture du Pli (`trick_closing`)**
Un `Trick` passe à `is_closed = True` si :
1.  Tous les joueurs $p_k$ où $k \neq$ l'index du dernier joueur ayant fait `ACTION_PLAY` ont le statut `is_eligible == False` (par `ACTION_HARD_PASS` ou par `ACTION_SOFT_PASS` consécutifs), de manière équivalente : $N-1$ joueurs éligibles ont passé.
2.  Un événement de règles additionnelles provoque une clôture immédiate (voir Section 6).

Le vainqueur du pli (dernier joueur ayant validé `ACTION_PLAY`) ouvre le `Trick` suivant.

### 5.4. Fin de Manche (`round_closing`)
*   Dès que $|H_i| == 0$, $p_i$ est inséré à la fin de la liste $O$, et son état devient `is_finished = True`.
*   Si $|O| == N - 1$, le joueur restant $p_{last}$ est inséré à $O[N-1]$. La `Round` se termine et déclenche la distribution des `VictoryPoints`.
*   *Interaction avec la Pénalité Étendue (`finish_penalty_extended`)* : un joueur qui termine sa main en remplissant l'une des conditions du section 6.9 (carte suprême, sortie sur Joker, sortie déclenchant une Révolution) se voit attribuer automatiquement le rôle `ROLE_SCUM` pour la manche suivante $R_{m+1}$, indépendamment de son index de sortie $k$ réel dans $O$.

## 6. Règles Supplémentaires et Événements de Jeu

Ces événements s'évaluent de manière conditionnelle à différents instants $t$ de la `Game` si le booléen ou la variable `GameConfig` correspondant est actif.

### 6.1. Modificateurs de Pré-Manche (`pre_round_modifiers`)
Interviennent lors des phases 5.1 et 5.2.
*   **Attribution stricte du reste (`strict_remainder_allocation`)** : cf. section 5.1.1.
*   **Taxe Aveugle (`blind_tax_enabled`)** : Lors de `exchange_phase`, la fonction de transfert du `ROLE_SCUM` vers le `ROLE_PRESIDENT` est redéfinie. L'opérateur $\max_{power}(H_{scum}, 2)$ est remplacé par une fonction de sélection aléatoire uniforme $rand(H_{scum}, 2)$.
*   **Le Putsch (`putsch_enabled`)** : cf. section 5.1.2. Ce n'est pas un effet automatique mais un **droit** que l'agent `ROLE_SCUM` peut choisir d'invoquer. Conséquence : la `exchange_phase` entière est annulée pour $R_m$.

### 6.2. Clôture Magique Alternative (`magic_card`)
Généralisation de la règle du 2. Soit $Card_{magic}$ le rang défini par `magic_card_rank` (par défaut 2, mais peut être 10, etc.).
*   Si `ACTION_PLAY` contient une carte $c$ où $rank(c) == Card_{magic}$, `is_closed` devient **immédiatement** `True`. Le joueur posant la combinaison remporte le `Trick`.
*   *Interaction avec $X$* : Si `magic_single_clears_all == True`, un joueur peut répondre à un `Trick` de taille $X$ par $C$ où $|C| = 1$ et $rank(c \in C) == Card_{magic}$.
*   *Interaction avec Révolution* : Si $E_{rev} == True$, $Card_{magic}$ devient dynamiquement le rang possédant la valeur $f_{power}$ symétriquement opposée (ex : si le 2 est magique, en Révolution le 3 devient magique, sauf paramétrage contraire en valeur absolue). Voir [section 6.5](#65-révolution-et-double-révolution-revolution_enabled-double_revolution_enabled) pour le déclenchement de la Révolution elle-même, et la résolution [C] de la [matrice de compatibilité (section 7)](#7-matrice-de-compatibilité-et-résolution-des-conflits-truth-table) pour la formalisation complète de ce remappage.

### 6.3. Forçage par Égalité (`skip_on_equal`)
*   Si $A_{t+1}$ est `ACTION_PLAY` tel que $f_{power}(C_{t+1}) == P_{current}$, l'état global passe à `is_equal_forced = True`.
*   Conséquence à $t+2$ : $p_{t+2}$ **doit** fournir $C_{t+2}$ telle que $f_{power}(C_{t+2}) == f_{power}(C_{t+1})$. S'il ne le peut pas, il est forcé de jouer `ACTION_SOFT_PASS` ou `ACTION_HARD_PASS`.
*   L'état `is_equal_forced = False` est restauré dès qu'une action autre que `ACTION_PLAY` est jouée.
*   *Note de cohérence* : dans les règles traditionnelles du jeu, un passe exclut en principe le joueur du pli en cours (`ACTION_HARD_PASS`), le `ACTION_SOFT_PASS` (passer maintenant mais pouvoir resurenchérir plus tard dans le même pli) modifie fortement la combinatoire. Le paramètre `pass_type` de `GameConfig` doit trancher strictement entre les deux sémantiques pour toute la partie, faute de quoi un `Trick` peut ne jamais se clôturer (état fantôme).

### 6.4. Substitution Joker (`use_jokers`)
*   Les Jokers agissent comme `Wildcard`. Dans une `Combination` $C$ ($X \ge 2$), un Joker $j$ prend la valeur $f_{power}$ de $c_{std} \in C$, précisée par le champ `declared_power` de l'`Action`.
* Restriction : Un Joker $j \in C$ invalide mathématiquement la possibilité pour $C$ de déclencher la règle [section 6.5](#65-révolution-et-double-révolution-revolution_enabled-double_revolution_enabled) (`revolution_enabled`), voir résolution [A] de la [matrice de compatibilité (section 7)](#7-matrice-de-compatibilité-et-résolution-des-conflits-truth-table).

### 6.5. Révolution et Double Révolution (`revolution_enabled`, `double_revolution_enabled`)
Soit la variable d'état de verrouillage $L_{rev} \in \{True, False\}$, réinitialisée à `False` au début de chaque `Round`.
*   **Révolution standard** : Si $L_{rev} == False$ et que $A_t$ est `ACTION_PLAY` avec $C$ telle que $|C| \ge 4$. L'état de puissance globale est inversé : $E_{rev} = \neg E_{rev}$.
*   **Double Révolution** : Si `double_revolution_enabled == True` et que $A_t$ contient $C$ telle que $|C| \ge 8$ (nécessite $N_D \ge 2$). Effet : $E_{rev} = \neg E_{rev}$ **ET** $L_{rev} = True$. L'état $E_{rev}$ est définitivement figé jusqu'au `round_closing`.

### 6.6. Les Suites / Escaliers (`straights_enabled`)
Introduction d'un nouveau sous-ensemble valide $C_{seq} \subset \text{Hand}$.
*   Définition : $C_{seq}$ est valide si $|C_{seq}| = X \ge 3$ et si $\forall i \in [0, X-2], f_{power}(c_{i+1}) == f_{power}(c_i) + 1$.
*   Conditions de jeu : Pour répondre à $C_{seq}$ à l'instant $t$, l'action $A_{t+1}$ doit proposer $C'_{seq}$ telle que $|C'_{seq}| == X$ et $\min(f_{power}(C'_{seq})) > \min(f_{power}(C_{seq}))$.
*   *Exclusion mathématique* : Un $C_{seq}$ de taille $X \ge 4$ ne déclenche **pas** de Révolution (réservé aux combinaisons de même $f_{power}$).

### 6.7. Saut de Tour (`skip_turn_enabled`)
Soit le rang d'évitement $R_{skip}$ défini par `skip_turn_rank` (généralement 8).
*   Déclencheur : Si $A_t$ est `ACTION_PLAY` avec $C$ telle que $rank(c \in C) == R_{skip}$.
*   Effet : Le système attribue obligatoirement `ACTION_HARD_PASS` (ou `is_eligible = False` temporairement) aux $|C|$ joueurs suivants dans l'ordre du tour. L'index de tour fait un bond $t \rightarrow t + |C| + 1$.

### 6.8. L'Interception / Fermeture à la volée (`interception_enabled`)
Mécanique d'interruption hors-tour (nécessite $N_D \ge 2$).
*   Soit $C_t$ la dernière combinaison posée (où $|C_t| == 1$). À tout instant avant que le joueur $p_{t+1}$ ne valide son `Action`, n'importe quel joueur $p_k$ ($k \neq t+1$) possédant $c_{intercept} \in H_k$ peut déclencher l'interception.
*   Condition absolue : $rank(c_{intercept}) == rank(C_t[0])$ **ET** $suit(c_{intercept}) == suit(C_t[0])$.
*   Effet de résolution : $p_k$ valide `ACTION_PLAY` instantanément ($t_{new} = k$). Le `Trick` passe à `is_closed = True` (ou, selon configuration, l'ordre de jeu reprend simplement à $p_{k+1}$, sautant tous les $p_i$ entre $p_{t+1}$ et $p_{k-1}$).

### 6.9. Extension des Pénalités de Clôture (`finish_penalty_extended`)
Évalué lors du passage de $p_i$ à l'état `is_finished = True`. Soit $C_{final}$ la combinaison ayant permis de vider $H_i$.
Le joueur $p_i$ subit la pénalité de sortie (définie par `finish_penalty_type` dans la Section 2) si au moins **l'une** de ces conditions est vraie :
1.  **Carte Suprême** : $rank(c \in C_{final}) == Card_{sup}$ (où $Card_{sup}$ est 2 si $E_{rev} == False$, ou 3 si $E_{rev} == True$).
2.  **Sortie Joker (`no_finish_on_joker`)** : $\exists j \in C_{final}$ tel que $rank(j) == Joker$.
3.  **Sortie sur Révolution (`no_finish_on_revolution`)** : $C_{final}$ remplit les conditions de la règle 6.5, déclenchant $E_{rev} = \neg E_{rev}$ à l'instant exact de la sortie.

Effet additionnel : si `finish_penalty_extended == True`, le joueur concerné se voit également assigné `ROLE_SCUM` pour $R_{m+1}$ (cf. section 5.4).

## 7. Matrice de Compatibilité et Résolution des Conflits (Truth Table)

Afin d'assurer la stabilité de la machine à états de la `Game`, l'activation croisée des règles avancées obéit à cette matrice de résolution. Les cellules indiquent comment la Règle A (ligne) interagit avec la Règle B (colonne) si les deux valent `True`.

| Règle A \ Règle B | `revolution` | `use_jokers` | `straights` | `skip_turn` (ex : 8) | `magic_card` (ex : 2) | `interception` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`revolution`** | - | **[A]** Joker l'annule | **[B]** Mutuellement exclusif | `True` | **[C]** Inverse la carte | `True` |
| **`use_jokers`** | **[A]** Joker l'annule | - | **[D]** Permis dans la suite | `True` | **[E]** Joker non magique | `True` |
| **`straights`** | **[B]** Mutuellement exclusif | **[D]** Permis dans la suite | - | **[F]** Saut cumulé | **[G]** Casse la suite | `False` |
| **`skip_turn`** | `True` | `True` | **[F]** Saut cumulé | - | `True` | **[H]** Priorité Intercept |
| **`magic_card`** | **[C]** Inverse la carte | **[E]** Joker non magique | **[G]** Casse la suite | `True` | - | `True` |
| **`interception`** | `True` | `True` | `False` | **[H]** Priorité Intercept | `True` | - |

**Spécifications des résolutions mathématiques (A à H) :**
*   **[A]** : Toute inclusion d'un Joker dans $|C| \ge 4$ ou $|C| \ge 8$ empêche de modifier $E_{rev}$ ou $L_{rev}$.
*   **[B]** : Un $C_{seq}$ (Suite) de taille $X \ge 4$ ne modifie pas $E_{rev}$, bien que sa taille soit valide pour une Révolution. Seuls les $C$ de puissance uniforme sont éligibles.
*   **[C]** : L'activation de la Révolution remappe dynamiquement la `magic_card` vers la valeur de puissance inverse ($f_{power}$ miroir), sauf si `magic_card` est paramétrée en valeur absolue.
*   **[D]** : Les Jokers peuvent combler un intervalle dans un $C_{seq}$ (ex : 4 - Joker - 6). Le Joker prend la valeur dynamique requise pour maintenir $\Delta f_{power} = 1$.
*   **[E]** : Un Joker déclaré avec la valeur d'une `magic_card` ne déclenche **pas** la clôture immédiate (`is_closed = True`).
*   **[F]** : Si une Suite contient un ou plusieurs $R_{skip}$, le nombre de joueurs passés est égal au nombre strict de cartes $R_{skip}$ incluses dans $C_{seq}$.
*   **[G]** : Si un joueur pose un $C_{seq}$ contenant une `magic_card` (sans `magic_single_clears_all == True`), la propriété magique prend le pas : le pli passe à `is_closed = True` immédiatement.
*   **[H]** : L'évaluation de l'événement `interception` s'exécute **avant** l'évaluation de l'événement `skip_turn`. Si un 8 est posé, un joueur peut intercepter avec le 8 identique avant que l'ordre de jeu ne saute le joueur suivant.
