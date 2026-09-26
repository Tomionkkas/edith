/* The page. Three states - cold open, answer, picker - and one POST.
 *
 * Everything shown here comes from the payload infer/web.py builds out of
 * engine.plan(). Nothing is decided in the browser: which record answered,
 * which fields to show and whether to offer a choice were all settled by the
 * same code the terminal runs.
 */

const $ = (id) => document.getElementById(id);

const cold = $("cold");
const app = $("app");
const picker = $("picker");
const transcript = $("transcript");
const right = $("right");

let previous = null;   // the record on screen, threaded like terminal.last_doc
let lastQuestion = "";
let busy = false;

// The session, kept in the browser so a reload does not lose the thread.
// localStorage can throw (private windows, blocked site data), so every
// touch is guarded and the page works without it.
const HISTORY_KEY = "edith.session.v1";
const HISTORY_MAX = 40;
let history = [];

function loadHistory() {
  try {
    history = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  } catch (e) {
    history = [];
  }
  return history;
}

function saveHistory() {
  try {
    localStorage.setItem(HISTORY_KEY,
                         JSON.stringify(history.slice(-HISTORY_MAX)));
  } catch (e) {
    /* a session that cannot be saved is still a session */
  }
}

function clearHistory() {
  history = [];
  try { localStorage.removeItem(HISTORY_KEY); } catch (e) {}
  transcript.replaceChildren();
  right.hidden = true;
  previous = null;
}

/* ------------------------------------------------------------ the art chain
 * own art, then the first appearance's cover, then a plate. The wiki serves
 * some files under a different name than the record gives, so onerror walks
 * to the next rung instead of leaving a hole.
 */
function mountArt(box, urls, name, opts = {}) {
  // The record panel lays its name OVER the art; a picker card carries its
  // name underneath and a provenance badge on top, so neither the caption
  // nor the class may be assumed.
  const { pageTitle = null, reality = null, caption: wantCaption = true,
          artClass = "art", plateClass = "plate", extra = [] } = opts;
  let i = 0;

  const draw = (cls, nodes) => {
    box.className = cls;
    box.replaceChildren(...nodes, ...extra);
  };

  const attempt = () => {
    if (i >= urls.length) {
      const nodes = wantCaption ? [caption(name, pageTitle)] : [];
      if (opts.emptyNote) nodes.push(opts.emptyNote);
      draw(plateClass, nodes);
      return;
    }
    const img = new Image();
    img.referrerPolicy = "no-referrer";
    img.alt = name || "";
    img.onload = () => {
      const nodes = [img, el("div", "shade")];
      if (wantCaption) nodes.push(caption(name, pageTitle));
      if (reality) {
        const tag = el("span", "reality", reality);
        nodes.push(tag);
      }
      draw(artClass, nodes);
    };
    img.onerror = () => { i += 1; attempt(); };
    img.src = urls[i];
  };
  attempt();
}

function caption(name, pageTitle) {
  const wrap = document.createElement("div");
  wrap.className = "caption";
  const big = document.createElement("span");
  big.className = "name";
  big.textContent = (name || "").toUpperCase();
  wrap.appendChild(big);
  if (pageTitle) {
    const sub = document.createElement("span");
    sub.className = "page-title";
    sub.textContent = pageTitle;
    wrap.appendChild(sub);
  }
  return wrap;
}

/* Field values are not uniform: "Public" is one word and Powers can run to
 * 3,000 characters. The terminal solves that by truncating, which is the
 * thing this page exists not to do - so a long value folds into a native
 * <details> instead. No library, no read-more state to keep. */
const LONG = 220;

function valueNode(value) {
  if (value.length <= LONG) return el("span", "value", value);
  const details = el("details", "value long");
  const summary = el("summary", null, value.slice(0, LONG).trimEnd() + "…");
  details.append(summary, el("p", "rest", value));
  return details;
}

function realityOf(pageTitle) {
  const m = /\(([^()]*Earth-[^()]*)\)\s*$/.exec(pageTitle || "");
  return m ? m[1] : "";
}

/* ----------------------------------------------------------- the transcript */

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function addTurn(question, data) {
  if (question) {
    const asked = el("div", "turn");
    asked.append(el("span", "mono label", "YOU"), el("span", "asked", question));
    transcript.appendChild(asked);
  }

  const said = el("div", "turn");
  if (data.corrected) {
    said.appendChild(el("span", "footnote", `reading that as ${data.corrected}`));
  }

  // A narrative ask: the PASSAGE is the answer, and nothing goes above it.
  // The field line there is a creator credit answering a question nobody
  // asked - 69% of quoted turns opened with one before 4.5b measured it on
  // a live run of 260. terminal.py:471 makes the same choice.
  const leads = Boolean(data.quoted && data.quoted_leads);
  if (!leads) said.appendChild(el("p", "said", data.answer));

  // The issue the answer cites, shown rather than named.
  if (data.first_appearance && data.first_appearance.title) {
    const box = el("div", "first-app");
    if (data.first_appearance.image) {
      const cover = new Image();
      cover.referrerPolicy = "no-referrer";
      cover.alt = data.first_appearance.title;
      cover.src = data.first_appearance.image;
      cover.onerror = () => cover.remove();
      box.appendChild(cover);
    }
    const who = el("div", "who");
    who.append(el("span", "footnote", "First appearance"),
               el("span", "title", data.first_appearance.title));
    box.appendChild(who);
    said.appendChild(box);
  }

  if (data.exhausted) {
    said.appendChild(el("span", "footnote",
      "that is everything this record says"));
  }

  if (data.quoted) {
    const quote = el("div", leads ? "quote leads" : "quote");
    quote.appendChild(el("p", null, `“${data.quoted}”`));
    quote.appendChild(el("span", "footnote",
      `quoted verbatim · ${data.page || data.headline || "the record"}`));
    said.appendChild(quote);
  }
  // Clicking any earlier turn brings its record back, which is what a
  // transcript is FOR - otherwise the art of turn two is gone for good.
  if (data.doc_id !== null && data.doc_id !== undefined) {
    said.classList.add("answered");
    const again = el("span", "again");
    again.textContent = `▸ show ${data.headline || "this record"} again`;
    said.appendChild(again);
    said.addEventListener("click", () => {
      showRecord(data);
      previous = data.doc_id;
      for (const el of transcript.querySelectorAll(".current")) {
        el.classList.remove("current");
      }
      said.classList.add("current");
    });
  }

  transcript.appendChild(said);
  transcript.scrollTop = transcript.scrollHeight;
}

/* --------------------------------------------------------- the record panel */

function railNode() {
  // The session, in the empty space under the record - no third column, and
  // the one place on screen that was doing nothing.
  const rail = el("aside", "rail");
  rail.appendChild(el("h2", "rail-head", `This session · ${history.length}`));
  const list = el("div", "rail-list");
  history.forEach((turn, i) => {
    const row = el("button", "rail-row");
    row.type = "button";
    row.append(el("span", null, turn.question),
               el("span", "who", turn.data.headline || ""));
    if (i === history.length - 1) row.classList.add("current");
    row.addEventListener("click", () => {
      showRecord(turn.data);
      previous = turn.data.doc_id;
    });
    list.appendChild(row);
  });
  rail.appendChild(list);
  return rail;
}

function showRecord(data) {
  right.hidden = false;
  right.replaceChildren();

  const art = el("div", "plate");
  right.appendChild(art);
  mountArt(art, data.images || [], data.headline || "",
           { pageTitle: data.page, reality: realityOf(data.page) });

  const body = el("div", "record-body");

  const badges = el("div", "badges");
  badges.appendChild(el("span", "badge identity", "identity"));
  if (data.kind) badges.appendChild(el("span", "badge", data.kind));
  if (data.size) {
    badges.appendChild(el("span", "badge", `${data.size.toLocaleString()} ch`));
  }
  body.appendChild(badges);

  if (data.fields && data.fields.length) {
    const grid = el("div", "fields");
    for (const [key, value] of data.fields) {
      const field = el("div", "field");
      field.append(el("span", "key", key), valueNode(value));
      // A long value is its own row: Wolverine's Powers field is 3,000
      // characters, and half a grid column is not where that goes.
      if (value.length > LONG) field.classList.add("wide");
      grid.appendChild(field);
    }
    body.appendChild(grid);
  }
  right.appendChild(body);
  if (history.length) right.appendChild(railNode());
}

/* --------------------------------------------------------------- the picker */

function cardFor(choice, i) {
  const card = el("button", "card");
  card.type = "button";
  card.style.animationDelay = `${0.06 * i}s`;

  const frame = el("div", "frame");
  card.appendChild(frame);

  // identity = this record IS the name; codename/alias = it is only CALLED
  // it. 4.15's distinction, and what sorts ten identical headlines out.
  const badges = [];
  if (choice.provenance) {
    badges.push(el("span", `tag ${choice.provenance}`, choice.provenance));
    if (choice.provenance !== "identity") card.classList.add("codename");
  }
  mountArt(frame, choice.images || [], choice.headline || "",
           { caption: false, artClass: "frame", plateClass: "frame empty",
             extra: badges,
             // A frame with nothing in it looks like a failure; the wiki
             // simply has no art for this record.
             emptyNote: el("span", "no-art", "no art on the wiki") });

  const meta = el("div", "card-meta");
  meta.append(el("span", "card-name", (choice.headline || "").toUpperCase()),
              el("span", "sub", choice.page || ""),
              el("span", "sub", `${(choice.size || 0).toLocaleString()} ch`));
  card.appendChild(meta);
  card.addEventListener("click", () => pick(choice.doc_id));
  return card;
}

function showPicker(question, data) {
  app.hidden = true;
  picker.hidden = false;
  $("picker-question").textContent = question;

  const n = data.choices.length;
  const total = data.variants_total || n;
  $("picker-title").textContent =
    `${total.toLocaleString()} record${total === 1 ? "" : "s"} answer to that name.`;
  $("picker-note").textContent =
    "The art tells you which is which before the labels do.";
  $("variants-head").textContent = total > n
    ? `The same character · ${n} of ${total.toLocaleString()} shown`
    : `The same character, ${n} realit${n === 1 ? "y" : "ies"}`;

  const cards = $("cards");
  cards.replaceChildren(...data.choices.map(cardFor));

  // The second group is the interesting one: who ELSE has held this name.
  const others = data.others || [];
  $("group-others").hidden = others.length === 0;
  $("others").replaceChildren(...others.map(cardFor));
}

/* ------------------------------------------------------------------ asking */

async function post(body) {
  const res = await fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}

function setBusy(on) {
  busy = on;
  for (const id of ["ask-input", "cold-input"]) $(id).disabled = on;
  for (const form of ["ask-form", "cold-form"]) {
    $(form).querySelector("button").disabled = on;
  }
}

function thinking() {
  const node = el("div", "turn thinking");
  node.append(el("span", "pip"), el("span", null, "thinking"));
  transcript.appendChild(node);
  transcript.scrollTop = transcript.scrollHeight;
  return node;
}

async function ask(question) {
  if (busy) return;
  lastQuestion = question;
  cold.hidden = true;
  picker.hidden = true;
  app.hidden = false;

  const asked = el("div", "turn");
  asked.append(el("span", "mono label", "YOU"), el("span", "asked", question));
  transcript.appendChild(asked);

  const wait = thinking();
  setBusy(true);
  let data;
  try {
    data = await post({ question, previous });
  } catch (err) {
    wait.replaceChildren(el("div", "failed",
      "The engine did not answer. It may still be loading, or it stopped."));
    setBusy(false);
    return;
  }
  wait.remove();
  setBusy(false);
  $("ask-input").focus();

  if (data.choices && data.choices.length > 1) {
    asked.remove();
    showPicker(question, data);
    return;
  }
  if (data.doc_id !== null && data.doc_id !== undefined) previous = data.doc_id;
  addTurn(null, data);
  history.push({ question, data });
  saveHistory();
  if (data.doc_id !== null && data.doc_id !== undefined) {
    showRecord(data);
  } else {
    right.hidden = true;
  }
}

async function pick(docId) {
  if (busy) return;
  picker.hidden = true;
  app.hidden = false;
  const wait = thinking();
  setBusy(true);
  let data;
  try {
    data = await post({ pick: docId });
  } catch (err) {
    wait.replaceChildren(el("div", "failed", "That pick did not come back."));
    setBusy(false);
    return;
  }
  wait.remove();
  setBusy(false);
  previous = data.doc_id;

  const chose = el("div", "turn");
  chose.append(el("span", "mono label", "YOU PICKED"),
               el("span", "asked", data.page || data.headline || ""));
  transcript.appendChild(chose);

  addTurn(null, data);
  showRecord(data);
}

/* -------------------------------------------------------------- the wall */

async function coldOpen() {
  const res = await fetch("/wall");
  if (!res.ok) return;
  const wall = await res.json();
  const grid = $("wall");
  for (const url of wall.covers) {
    const img = new Image();
    img.referrerPolicy = "no-referrer";
    img.alt = "";
    img.src = url;
    img.style.animationDuration = `${9 + Math.random() * 4}s`;
    img.style.animationDelay = `${Math.random()}s`;
    img.onerror = () => img.remove();
    grid.appendChild(img);
  }
  if (wall.hero) $("cold-hero").src = wall.hero;
  if (wall.total) {
    $("record-count").textContent = wall.total.toLocaleString();
    $("bar-meta").textContent = `${wall.total.toLocaleString()} RECORDS · LOCAL`;
  }
}

/* ------------------------------------------------------------------- wiring */

function restore() {
  transcript.replaceChildren();
  for (const turn of history) {
    const asked = el("div", "turn");
    asked.append(el("span", "mono label", "YOU"),
                 el("span", "asked", turn.question));
    transcript.appendChild(asked);
    addTurn(null, turn.data);
  }
  const last = history[history.length - 1];
  if (last && last.data.doc_id !== null && last.data.doc_id !== undefined) {
    previous = last.data.doc_id;
    showRecord(last.data);
  }
}

$("resume").addEventListener("click", () => {
  cold.hidden = true;
  app.hidden = false;
  restore();
  $("ask-input").focus();
});

$("clear-history").addEventListener("click", () => {
  clearHistory();
  $("ask-input").focus();
});

// Typing anywhere goes to the box, Escape leaves the picker, and the up
// arrow brings the last question back to edit - the three things a terminal
// gives you for free and a page has to be told.
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !picker.hidden) {
    picker.hidden = true;
    app.hidden = false;
    $("ask-input").focus();
    return;
  }
  const box = app.hidden ? $("cold-input") : $("ask-input");
  if (document.activeElement === box) {
    if (e.key === "ArrowUp" && !box.value && lastQuestion) {
      box.value = lastQuestion;
      e.preventDefault();
    }
    return;
  }
  if (e.key.length === 1 && !e.ctrlKey && !e.metaKey && !e.altKey) box.focus();
});

$("cold-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const value = $("cold-input").value.trim();
  if (value) ask(value);
});

$("ask-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const value = $("ask-input").value.trim();
  if (!value) return;
  $("ask-input").value = "";
  ask(value);
});

$("picker-back").addEventListener("click", () => {
  picker.hidden = true;
  app.hidden = false;
  $("ask-input").focus();
});

loadHistory();
if (history.length) {
  const n = history.length;
  $("resume").hidden = false;
  $("resume").textContent =
    `↵ ${n} earlier question${n === 1 ? "" : "s"} · pick up where you left off`;
}
$("cold-input").focus();
coldOpen();
