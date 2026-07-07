"use strict";

const state = {
    gameId: null,
    humanSeats: [],
    botLabels: {},
    lastSeq: 0,
    selectedExchangeIndices: {},
    pollHandle: null,
};

function el(id) {
    return document.getElementById(id);
}

function showScreen(name) {
    document.querySelectorAll(".screen").forEach((node) => node.classList.remove("active"));
    el(name).classList.add("active");
}

function suitSymbol(suit) {
    if (suit === "SPADES") return "\u2660";
    if (suit === "HEARTS") return "\u2665";
    if (suit === "DIAMONDS") return "\u2666";
    if (suit === "CLUBS") return "\u2663";
    return "\u2605";
}

function suitColorClass(card) {
    if (card.is_joker) return "joker";
    if (card.suit === "HEARTS" || card.suit === "DIAMONDS") return "red";
    return "";
}

function cardRankLabel(card) {
    return card.is_joker ? "JOK" : card.rank;
}

function buildCardNode(card, options) {
    options = options || {};
    const node = document.createElement("div");
    node.className = "card " + suitColorClass(card) + (options.extraClass ? " " + options.extraClass : "");
    const rank = document.createElement("span");
    rank.className = "rank";
    rank.textContent = cardRankLabel(card);
    const suit = document.createElement("span");
    suit.className = "suit";
    suit.textContent = suitSymbol(card.suit);
    node.appendChild(rank);
    node.appendChild(suit);
    return node;
}

function buildCardRow(cards, options) {
    const row = document.createElement("div");
    row.className = "trick-play-cards";
    cards.forEach((card) => row.appendChild(buildCardNode(card, options)));
    return row;
}

async function loadBots() {
    const res = await fetch("/api/bots");
    const data = await res.json();
    data.bots.forEach((bot) => { state.botLabels[bot.id] = bot.label; });
    return data.bots;
}

function buildSeatOptionsHtml(bots) {
    let html = '<option value="human">Humain (vous)</option>';
    bots.forEach((bot) => {
        html += `<option value="${bot.id}">${bot.label}</option>`;
    });
    return html;
}

function renderSeatRows(bots) {
    const container = el("seats-container");
    const playerCount = parseInt(el("player-count").value, 10) || 4;
    const previousValues = Array.from(container.querySelectorAll("select")).map((s) => s.value);
    container.innerHTML = "";
    for (let seat = 0; seat < playerCount; seat += 1) {
        const row = document.createElement("div");
        row.className = "seat-row";
        const title = document.createElement("div");
        title.className = "seat-title";
        title.textContent = `Siège ${seat}`;
        const select = document.createElement("select");
        select.dataset.seat = String(seat);
        select.innerHTML = buildSeatOptionsHtml(bots);
        const defaultValue = previousValues[seat] || (seat === 0 ? "human" : "rule_based_bot");
        if ([...select.options].some((opt) => opt.value === defaultValue)) {
            select.value = defaultValue;
        }
        row.appendChild(title);
        row.appendChild(select);
        container.appendChild(row);
    }
}

function collectSeats() {
    return Array.from(document.querySelectorAll("#seats-container select")).map((s) => s.value);
}

function collectRules() {
    return {
        straights_enabled: el("rule-straights").checked,
        double_revolution_enabled: el("rule-double-revolution").checked,
        skip_turn_enabled: el("rule-skip-turn").checked,
        interception_enabled: el("rule-interception").checked,
        putsch_enabled: el("rule-putsch").checked,
        blind_tax_enabled: el("rule-blind-tax").checked,
        finish_penalty_enabled: el("rule-finish-penalty").checked,
        finish_penalty_extended: el("rule-finish-penalty").checked,
        no_finish_on_joker: el("rule-finish-penalty").checked,
        no_finish_on_revolution: el("rule-finish-penalty").checked,
        skip_on_equal: el("rule-skip-on-equal").checked,
        pass_type: el("rule-allow-soft-pass").checked ? "ALLOW_SOFT" : "HARD_ONLY",
        vp_distribution_type: el("rule-vp-distribution").value,
    };
}

async function startGame() {
    el("config-error").textContent = "";
    const playerCount = parseInt(el("player-count").value, 10) || 4;
    const rounds = parseInt(el("rounds-count").value, 10) || 10;
    const seats = collectSeats();
    const revealHands = el("reveal-hands").checked;
    const rules = collectRules();

    const res = await fetch("/api/games", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            player_count: playerCount,
            rounds: rounds,
            seats: seats,
            reveal_hands: revealHands,
            rules: rules,
        }),
    });
    const data = await res.json();
    if (!res.ok) {
        el("config-error").textContent = data.error || "Erreur inconnue à la création de la partie.";
        return;
    }

    state.gameId = data.game_id;
    state.humanSeats = data.human_seats;
    state.lastSeq = 0;
    state.selectedExchangeIndices = {};

    showScreen("screen-game");
    el("log-feed").innerHTML = "";
    startPolling();
}

function resetToConfigScreen() {
    stopPolling();
    if (state.gameId) {
        fetch(`/api/games/${state.gameId}`, { method: "DELETE" }).catch(() => {});
    }
    state.gameId = null;
    showScreen("screen-config");
}

function startPolling() {
    stopPolling();
    pollOnce();
    state.pollHandle = setInterval(pollOnce, 900);
}

function stopPolling() {
    if (state.pollHandle) {
        clearInterval(state.pollHandle);
        state.pollHandle = null;
    }
}

async function pollOnce() {
    if (!state.gameId) return;
    try {
        const res = await fetch(`/api/games/${state.gameId}/state`);
        if (!res.ok) return;
        const data = await res.json();
        renderState(data);
    } catch (err) {
        /* connexion momentanément indisponible, on retentera au prochain intervalle */
    }
}

function seatLabel(profile) {
    if (profile === "human") return "Humain";
    return state.botLabels[profile] || profile;
}

function roleLabel(role) {
    const labels = {
        ROLE_PRESIDENT: "Président",
        ROLE_VICE_PRESIDENT: "Vice-Président",
        ROLE_NEUTRAL: "Neutre",
        ROLE_VICE_SCUM: "Vice-Trou du Cul",
        ROLE_SCUM: "Trou du Cul",
    };
    return labels[role] || role;
}

function renderState(data) {
    el("round-indicator").textContent = `Manche ${data.round_index + 1 > 0 ? data.round_index + 1 : "—"} / ${data.rounds_total}`;
    el("trick-indicator").textContent = `Pli ${data.trick_index >= 0 ? data.trick_index + 1 : "—"}`;
    el("revolution-indicator").classList.toggle("hidden", !data.e_rev);
    el("finished-indicator").classList.toggle("hidden", !data.finished);

    renderTable(data);
    renderCurrentTrick(data);
    renderActionPanels(data);
    renderLog(data);
}

function renderTable(data) {
    const table = el("table");
    table.innerHTML = "";
    for (let pid = 0; pid < data.player_count; pid += 1) {
        const seatCard = document.createElement("div");
        seatCard.className = "seat-card";
        if (data.human_seats.includes(pid)) seatCard.classList.add("is-human");
        if (data.finished_players.includes(pid)) seatCard.classList.add("finished");

        const header = document.createElement("div");
        header.className = "seat-card-header";
        const name = document.createElement("span");
        name.className = "seat-name";
        name.textContent = `Siège ${pid} — ${seatLabel(data.seat_profiles[pid])}`;
        const vp = document.createElement("span");
        vp.className = "seat-vp";
        const vpValue = data.cumulative_vp[String(pid)] || 0;
        vp.textContent = `VP ${vpValue >= 0 ? "+" : ""}${vpValue.toFixed(1)}`;
        header.appendChild(name);
        header.appendChild(vp);
        seatCard.appendChild(header);

        const role = data.roles[String(pid)];
        if (role) {
            const roleTag = document.createElement("span");
            roleTag.className = "role-tag role-" + role;
            roleTag.textContent = roleLabel(role);
            seatCard.appendChild(roleTag);
        }

        const handPreview = document.createElement("div");
        handPreview.className = "hand-preview";

        let visibleCards = null;
        if (data.hands && data.hands[String(pid)]) {
            visibleCards = data.hands[String(pid)];
        } else if (data.own_hands && data.own_hands[String(pid)]) {
            visibleCards = data.own_hands[String(pid)];
        }

        if (visibleCards) {
            visibleCards.forEach((card) => handPreview.appendChild(buildCardNode(card)));
        } else {
            const size = data.hand_sizes[String(pid)] || 0;
            for (let i = 0; i < size; i += 1) {
                const back = document.createElement("div");
                back.className = "card-back";
                handPreview.appendChild(back);
            }
        }
        seatCard.appendChild(handPreview);

        const sizeLabel = document.createElement("div");
        sizeLabel.className = "hand-size-label";
        sizeLabel.textContent = `${data.hand_sizes[String(pid)] || 0} carte(s)`;
        seatCard.appendChild(sizeLabel);

        table.appendChild(seatCard);
    }
}

function renderCurrentTrick(data) {
    const container = el("current-trick-cards");
    container.innerHTML = "";
    if (!data.current_trick_plays.length) {
        const empty = document.createElement("div");
        empty.className = "hand-size-label";
        empty.textContent = "Aucune carte posée pour l'instant.";
        container.appendChild(empty);
        return;
    }
    data.current_trick_plays.forEach((play) => {
        const wrapper = document.createElement("div");
        wrapper.className = "trick-play" + (play.cards.length === 0 ? " is-pass" : "");
        if (play.cards.length > 0) {
            wrapper.appendChild(buildCardRow(play.cards));
        }
        const label = document.createElement("div");
        label.className = "trick-play-label";
        label.textContent = play.cards.length > 0
            ? `Joueur ${play.player_id}`
            : `Joueur ${play.player_id} passe`;
        wrapper.appendChild(label);
        container.appendChild(wrapper);
    });
}

function renderActionPanels(data) {
    const container = el("action-panels");
    container.innerHTML = "";
    const entries = Object.entries(data.pending_requests || {});
    entries.forEach(([pidStr, requestPayload]) => {
        const pid = parseInt(pidStr, 10);
        const panel = document.createElement("div");
        panel.className = "action-panel";
        const title = document.createElement("h4");
        title.textContent = `Joueur ${pid} — à vous de jouer`;
        panel.appendChild(title);

        if (requestPayload.type === "action") {
            panel.appendChild(buildActionRequestPanel(pid, requestPayload));
        } else if (requestPayload.type === "exchange") {
            panel.appendChild(buildExchangeRequestPanel(pid, requestPayload));
        } else if (requestPayload.type === "putsch") {
            panel.appendChild(buildPutschRequestPanel(pid, requestPayload));
        } else if (requestPayload.type === "interception") {
            panel.appendChild(buildInterceptionRequestPanel(pid, requestPayload));
        }

        container.appendChild(panel);
    });
}

async function postDecision(pid, endpoint, body) {
    await fetch(`/api/games/${state.gameId}/seats/${pid}/${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    pollOnce();
}

function buildActionRequestPanel(pid, payload) {
    const wrapper = document.createElement("div");

    if (payload.trick_power !== null && payload.trick_power !== undefined) {
        const info = document.createElement("div");
        info.className = "hand-size-label";
        info.textContent = `Puissance à dépasser : ${payload.trick_power}${payload.e_rev ? " (Révolution active)" : ""}`;
        wrapper.appendChild(info);
    }

    const optionList = document.createElement("div");
    optionList.className = "option-list";
    payload.options.forEach((option) => {
        const btn = document.createElement("button");
        btn.className = "option-btn";
        btn.appendChild(buildCardRow(option.cards));
        const label = document.createElement("span");
        label.textContent = option.declared_power !== null && option.declared_power !== undefined
            ? `Joker déclaré à ${option.declared_power}`
            : `Taille ${option.size}`;
        btn.appendChild(label);
        btn.addEventListener("click", () => postDecision(pid, "action", { option_index: option.index }));
        optionList.appendChild(btn);
    });
    wrapper.appendChild(optionList);

    const passBtn = document.createElement("button");
    passBtn.className = "pass-btn";
    passBtn.textContent = "Passer";
    passBtn.addEventListener("click", () => postDecision(pid, "action", { pass: true }));
    wrapper.appendChild(passBtn);

    return wrapper;
}

function buildExchangeRequestPanel(pid, payload) {
    const wrapper = document.createElement("div");

    const countLabel = document.createElement("div");
    countLabel.className = "exchange-count";
    countLabel.textContent = `Choisissez ${payload.count} carte(s) à céder.`;
    wrapper.appendChild(countLabel);

    if (!state.selectedExchangeIndices[pid]) {
        state.selectedExchangeIndices[pid] = [];
    }
    const selected = state.selectedExchangeIndices[pid];

    const handRow = document.createElement("div");
    handRow.className = "exchange-hand";
    payload.hand.forEach((card, index) => {
        const node = buildCardNode(card, { extraClass: "selectable" + (selected.includes(index) ? " selected" : "") });
        node.addEventListener("click", () => {
            const currentIndex = selected.indexOf(index);
            if (currentIndex >= 0) {
                selected.splice(currentIndex, 1);
            } else if (selected.length < payload.count) {
                selected.push(index);
            }
            renderActionPanelsFromCache();
        });
        handRow.appendChild(node);
    });
    wrapper.appendChild(handRow);

    const confirmBtn = document.createElement("button");
    confirmBtn.className = "confirm-btn";
    confirmBtn.textContent = "Confirmer l'échange";
    confirmBtn.disabled = selected.length !== payload.count;
    confirmBtn.addEventListener("click", () => {
        postDecision(pid, "exchange", { card_indices: selected });
        state.selectedExchangeIndices[pid] = [];
    });
    wrapper.appendChild(confirmBtn);

    return wrapper;
}

function buildPutschRequestPanel(pid, payload) {
    const wrapper = document.createElement("div");
    const info = document.createElement("div");
    info.className = "hand-size-label";
    info.textContent = "Voulez-vous invoquer le Putsch et annuler l'échange de cette manche ?";
    wrapper.appendChild(info);

    const yesBtn = document.createElement("button");
    yesBtn.className = "decision-btn";
    yesBtn.textContent = "Invoquer le Putsch";
    yesBtn.addEventListener("click", () => postDecision(pid, "putsch", { invoke: true }));

    const noBtn = document.createElement("button");
    noBtn.className = "decision-btn no";
    noBtn.textContent = "Ne pas invoquer";
    noBtn.addEventListener("click", () => postDecision(pid, "putsch", { invoke: false }));

    wrapper.appendChild(yesBtn);
    wrapper.appendChild(noBtn);
    return wrapper;
}

function buildInterceptionRequestPanel(pid, payload) {
    const wrapper = document.createElement("div");
    const info = document.createElement("div");
    info.className = "hand-size-label";
    info.textContent = "Une carte jumelle est disponible. Voulez-vous intercepter ?";
    wrapper.appendChild(info);

    const cardsRow = buildCardRow(payload.twins);
    wrapper.appendChild(cardsRow);

    const yesBtn = document.createElement("button");
    yesBtn.className = "decision-btn";
    yesBtn.textContent = "Intercepter";
    yesBtn.addEventListener("click", () => postDecision(pid, "interception", { intercept: true, twin_index: 0 }));

    const noBtn = document.createElement("button");
    noBtn.className = "decision-btn no";
    noBtn.textContent = "Ne pas intercepter";
    noBtn.addEventListener("click", () => postDecision(pid, "interception", { intercept: false }));

    wrapper.appendChild(yesBtn);
    wrapper.appendChild(noBtn);
    return wrapper;
}

let _lastRenderedState = null;

function renderActionPanelsFromCache() {
    if (_lastRenderedState) renderActionPanels(_lastRenderedState);
}

function describeEvent(ev) {
    switch (ev.event_type) {
        case "EventRoundStart":
            return `Nouvelle manche (#${ev.round_id + 1}) : distribution des cartes.`;
        case "EventTrickStart":
            return `Ouverture du pli ${ev.trick_index + 1} par le joueur ${ev.opener_id}.`;
        case "EventActionPlayed": {
            if (ev.action_type === "ACTION_PLAY" && ev.cards_played && ev.cards_played.length) {
                const cards = ev.cards_played.map((c) => c.display).join(" ");
                return `Joueur ${ev.player_id} pose : ${cards}`;
            }
            return `Joueur ${ev.player_id} passe.`;
        }
        case "EventTrickClosed":
            return `Pli remporté par le joueur ${ev.winner_id}.`;
        case "EventRuleTriggered":
            return `Règle déclenchée : ${ev.rule_name} (joueur ${ev.triggering_player_id}).`;
        case "EventPlayerFinished": {
            const vp = ev.vp_earned;
            return `Joueur ${ev.player_id} termine au rang ${ev.rank + 1} (VP ${vp >= 0 ? "+" : ""}${vp}).`;
        }
        case "EventRoundEnd":
            return `Fin de la manche ${ev.round_id + 1}.`;
        case "EventExchange": {
            const cards = ev.cards.map((c) => c.display).join(" ");
            return `Échange : joueur ${ev.from_player} → joueur ${ev.to_player} (${cards}).`;
        }
        case "EventPutschInvoked":
            return `Le joueur ${ev.player_id} invoque le Putsch !`;
        case "EventInterceptionResolved":
            return ev.interceptor_id !== null && ev.interceptor_id !== undefined
                ? `Interception réussie par le joueur ${ev.interceptor_id}.`
                : null;
        default:
            return null;
    }
}

function renderLog(data) {
    _lastRenderedState = data;
    const feed = el("log-feed");
    const newEntries = data.log_tail.filter((entry) => entry.seq > state.lastSeq);
    newEntries.forEach((entry) => {
        const description = describeEvent(entry);
        if (!description) return;
        const line = document.createElement("div");
        line.className = "log-entry highlight";
        line.textContent = description;
        feed.appendChild(line);
        state.lastSeq = Math.max(state.lastSeq, entry.seq);
    });
    if (newEntries.length) {
        feed.scrollTop = feed.scrollHeight;
    }
}

async function init() {
    const bots = await loadBots();
    renderSeatRows(bots);
    el("player-count").addEventListener("change", () => renderSeatRows(bots));
    el("start-game-btn").addEventListener("click", startGame);
    el("new-game-btn").addEventListener("click", resetToConfigScreen);
}

init();
