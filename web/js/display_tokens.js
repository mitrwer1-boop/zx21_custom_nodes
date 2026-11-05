import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

console.log("[zx21] display extension loaded");

const ENTRY_OPTIONS = [
    "1 (newest)",
    "2",
    "3",
    "4",
    "5 (oldest)",
];

function oneLine(text, width = 120) {
    const line = (text ?? "").split(/\s+/).join(" ").trim();
    return line.length > width ? line.slice(0, width) + "..." : line;
}

function formatHistory(history, entryLabel) {
    if (!Array.isArray(history) || history.length === 0) {
        return "(history is empty)";
    }

    const sidebar = history
        .slice()
        .reverse()
        .map((item, i) => `#${i + 1} [${item.timestamp ?? "?"}] ${oneLine(item.prompt)}`)
        .join("\n");

    let index = ENTRY_OPTIONS.indexOf(entryLabel);
    if (index < 0) index = 0;
    // Newest first, clamped to the oldest entry that actually exists.
    const ordered = history.slice().reverse();
    const item = ordered[Math.min(index, ordered.length - 1)];

    return (
        `${sidebar}\n\n` +
        `── Selected entry ──\n` +
        `Time:     ${item.timestamp ?? "?"}\n` +
        `Prompt:   ${item.prompt ?? ""}\n\n` +
        `Response: ${item.response ?? ""}\n\n` +
        `Reasoning: ${item.reasoning || "(none captured)"}`
    );
}

async function fetchHistory() {
    try {
        const res = await fetch("/zx21/history");
        if (!res.ok) {
            console.warn("[zx21] history endpoint returned", res.status);
            return null;
        }
        return await res.json();
    } catch (err) {
        console.warn("[zx21] could not fetch history", err);
        return null;
    }
}

// Update every viewer widget currently on the graph.
async function refreshViewers() {
    const history = await fetchHistory();
    if (history === null) return;

    for (const node of app.graph._nodes) {
        if (!isViewer(node)) continue;

        const widget = node.widgets?.find((w) => w.name === "history_preview");
        if (!widget) continue;

        const entry = node.widgets?.find((w) => w.name === "entry")?.value;
        widget.value = formatHistory(history, entry);
        node._llm_history_cache = widget.value;
    }
    app.graph.setDirtyCanvas(true, false);
}

// comfyClass is not set yet when nodes are being added, so also match on
// the litegraph type, which always exists on a node instance.
function isViewer(node) {
    return (
        node?.type === "PromptHistoryViewer" ||
        node?.comfyClass === "PromptHistoryViewer"
    );
}

function ensurePreviewWidget(node) {
    if (!isViewer(node)) return null;

    let widget = node.widgets?.find((w) => w.name === "history_preview");
    if (!widget) {
        widget = ComfyWidgets.STRING(
            node,
            "history_preview",
            ["STRING", { multiline: true }],
            app,
        ).widget;
        widget.inputEl.readOnly = true;
        widget.serialize = false; // do not save into the workflow json
        widget.inputEl.style.opacity = "0.8";
        widget.value = node._llm_history_cache ?? "Loading history…";

        // Give the multiline preview room to be readable.
        try {
            node.setSize([
                Math.max(node.size[0], 460),
                Math.max(node.size[1], 420),
            ]);
        } catch { /* setSize is optional on some node shims */ }

        refreshViewers();
    }
    return widget;
}

// Hook the dropdown so changing it re-renders immediately.
function hookEntryWidget(node) {
    const entry = node.widgets?.find((w) => w.name === "entry");
    if (!entry || entry._llm_hooked) return;
    entry._llm_hooked = true;
    const originalCallback = entry.callback;
    entry.callback = function (...args) {
        const result = originalCallback?.apply(this, args);
        refreshViewers();
        return result;
    };
}

app.registerExtension({
    name: "zx21.displayText",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "OpenAICompatibleLoader") {
            // Draw token usage below the LLM node onto the canvas (short text).
            const onDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function (ctx) {
                onDrawForeground?.apply(this, arguments);

                if (this._llm_token_usage) {
                    ctx.save();
                    ctx.font = "12px sans-serif";
                    ctx.fillStyle = "rgba(160, 160, 160, 0.9)";
                    ctx.textAlign = "right";

                    const lines = this._llm_token_usage;
                    let drawY = this.size[1] + 16;

                    for (let i = 0; i < lines.length; i++) {
                        ctx.fillText(lines[i], this.size[0] - 10, drawY + (i * 16));
                    }
                    ctx.restore();
                }
            };

            // Any finished LLM request saved a new history entry: refresh viewers.
            const onExecutedLoader = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecutedLoader?.apply(this, arguments);

                if (message?.text && message.text.length > 0) {
                    const text = Array.isArray(message.text) ? message.text.join("\n") : message.text;
                    this._llm_token_usage = text.split("\n");
                    app.graph.setDirtyCanvas(true, false);
                }
                refreshViewers();
            };
            return;
        }

        if (nodeData.name !== "PromptHistoryViewer") return;

        // Lazy safety net: create the preview widget on the first draw if it
        // is still missing (covers every path that adds nodes to the graph).
        const onDrawForegroundViewer = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            onDrawForegroundViewer?.apply(this, arguments);
            if (!this.widgets?.find((w) => w.name === "history_preview")) {
                ensurePreviewWidget(this);
                hookEntryWidget(this);
            }
        };

        // Text coming back from a real execution of the viewer node.
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);

            const text = Array.isArray(message?.text)
                ? message.text.join("\n")
                : (message?.text ?? "");

            if (!text) return;

            const widget = ensurePreviewWidget(this);
            if (!widget) return;
            widget.value = text;
            this._llm_history_cache = text;
            this.setDirtyCanvas(true, false);
        };
    },

    // Fires for every node added to the graph, including from a saved workflow.
    async nodeAdded(node) {
        if (!isViewer(node)) return;

        // Widgets are not created yet at nodeAdded; defer one tick.
        setTimeout(() => {
            try {
                ensurePreviewWidget(node);
                hookEntryWidget(node);
            } catch (err) {
                console.warn("[zx21] viewer setup failed", err);
            }
        }, 0);
    },
});
