import { app } from "../../../scripts/app.js";

console.log("[zx21] model selector extension loaded");

const TAG = "[zx21]";
const SELECTOR_TYPE = "LLMModelSelector";
const LOADING_PLACEHOLDER = "(loading…)";

function widget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function isSelector(node) {
    return (
        node?.type === SELECTOR_TYPE ||
        node?.comfyClass === SELECTOR_TYPE
    );
}

async function fetchModels(baseUrl, apiKey, force) {
    const res = await fetch("/zx21/models", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            base_url: baseUrl ?? "",
            api_key: apiKey ?? "",
            force: Boolean(force),
        }),
    });
    if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
    }
    return await res.json();
}

function applyModels(node, payload) {
    const combo = widget(node, "model");
    if (!combo) return;

    const models = Array.isArray(payload?.models) ? payload.models : [];
    const current = combo.value;

    if (models.length > 0) {
        combo.options.values = models;
        // Keep a still-valid selection across refreshes.
        if (!models.includes(current)) {
            combo.value = models[0];
        }
        node._llm_model_count = models.length;
    } else {
        // A previously fetched, real selection survives a failed refresh;
        // placeholders are replaced with the reason.
        if (typeof current !== "string" || current.startsWith("(")) {
            const reason = payload?.error ? `: ${payload.error}` : "";
            combo.options.values = [`(no models fetched${reason})`];
            combo.value = combo.options.values[0];
        }
        node._llm_model_count = 0;
    }
    node.setDirtyCanvas?.(true, false);
}

async function refreshModels(node, force = false) {
    const baseUrl = widget(node, "base_url")?.value ?? "";
    const apiKey = widget(node, "api_key")?.value ?? "";
    try {
        const payload = await fetchModels(baseUrl, apiKey, force);
        applyModels(node, payload);
        console.log(
            `${TAG} model selector: ${payload?.models?.length ?? 0} models from ${baseUrl}`,
        );
    } catch (err) {
        applyModels(node, { error: String(err?.message ?? err) });
        console.warn(`${TAG} model selector fetch failed`, err);
    }
}

// Re-fetch shortly after base_url / api_key change.
function hookRefreshTriggers(node) {
    if (node._llm_models_hooked) return;
    node._llm_models_hooked = true;

    let timer = null;
    for (const name of ["base_url", "api_key"]) {
        const w = widget(node, name);
        if (!w) continue;
        const originalCallback = w.callback;
        w.callback = function (...args) {
            const result = originalCallback?.apply(this, args);
            clearTimeout(timer);
            timer = setTimeout(() => refreshModels(node), 400);
            return result;
        };
    }
}

app.registerExtension({
    name: "zx21.modelSelector",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== SELECTOR_TYPE) return;

        // "Refresh model list" in the node context menu (bypasses the
        // server-side cache).
        const originalMenu = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (_, options) {
            const result = originalMenu?.apply(this, arguments);
            options.push({
                content: "🔄 Refresh model list",
                callback: () => refreshModels(this, true),
            });
            return result;
        };

        // Draw the model count below the node, like the token readout.
        const originalDraw = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            originalDraw?.apply(this, arguments);
            if (typeof this._llm_model_count === "number") {
                ctx.save();
                ctx.font = "12px sans-serif";
                ctx.fillStyle = "rgba(160, 160, 160, 0.9)";
                ctx.textAlign = "right";
                ctx.fillText(
                    `${this._llm_model_count} models`,
                    this.size[0] - 10,
                    this.size[1] + 16,
                );
                ctx.restore();
            }
        };
    },

    // Fires for every node instance, including ones restored from a saved
    // workflow; values from the workflow are applied after creation, so
    // afterConfigureGraph below re-fetches with the real base_url.
    async nodeCreated(node) {
        if (!isSelector(node)) return;
        hookRefreshTriggers(node);
        setTimeout(() => refreshModels(node), 0);
    },

    async afterConfigureGraph() {
        for (const node of app.graph._nodes) {
            if (!isSelector(node)) continue;
            hookRefreshTriggers(node);
            refreshModels(node);
        }
    },
});
