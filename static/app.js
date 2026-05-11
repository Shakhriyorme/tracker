// Attendance Tracker — frontend glue.

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// Chart.js theme to match the dark palette
window.addEventListener("load", () => {
  if (window.Chart) {
    Chart.defaults.color = "#a4a4b8";
    Chart.defaults.borderColor = "rgba(255,255,255,0.06)";
    Chart.defaults.font.family = "'JetBrains Mono', monospace";
    Chart.defaults.font.size = 11;
  }
});

// ---------- Camera picker ---------------------------------------------------

(function initCameraPicker() {
  const sel = $("#cam-source-pick");
  const url = $("#cam-url");
  const apply = $("#cam-apply");
  const flip = $("#cam-flip");
  if (!sel) return;
  const cur = window.__camera_source || "0";
  const isIp = !/^\d+$/.test(cur);
  if (isIp) {
    sel.value = "ipcam";
    url.style.display = "inline-block";
    url.value = cur;
    if (flip) flip.style.display = "inline-block";
  } else {
    sel.value = cur;
  }
  sel.addEventListener("change", () => {
    const ip = sel.value === "ipcam";
    url.style.display = ip ? "inline-block" : "none";
    if (flip) flip.style.display = ip ? "inline-block" : "none";
  });

  // Rotation cycle button
  const rotBtn = $("#cam-rotate");
  const rotCycle = [0, 90, 180, 270];
  let rotIdx = 0;
  if (rotBtn) {
    rotBtn.addEventListener("click", async () => {
      rotIdx = (rotIdx + 1) % rotCycle.length;
      const rot = rotCycle[rotIdx];
      rotBtn.disabled = true;
      try {
        const r = await fetch("/api/camera/rotation", {
          method: "POST", headers: {"Content-Type": "application/json"},
          body: JSON.stringify({rotation: rot}),
        });
        if (r.ok) {
          rotBtn.textContent = rot + "°";
          $$("img[src*='/video_feed']").forEach((img) => {
            const u = new URL(img.src, window.location.href);
            u.searchParams.set("t", Date.now());
            img.src = u.toString();
          });
        }
      } finally { rotBtn.disabled = false; }
    });
  }

  if (flip) {
    fetch("/api/status").then(r => r.json()).then(j => {
      if (j.ffc_front) flip.textContent = "Back cam";
      else flip.textContent = "Selfie";
    }).catch(() => {});

    flip.addEventListener("click", async () => {
      flip.disabled = true;
      try {
        const r = await fetch("/api/camera/toggle_front", {
          method: "POST", headers: {"Content-Type": "application/json"},
          body: JSON.stringify({}),
        });
        const j = await r.json();
        if (!r.ok) { alert(j.error || "Toggle failed"); return; }
        flip.textContent = j.front ? "Back cam" : "Selfie";
        if (rotBtn && j.rotation != null) {
          rotBtn.textContent = j.rotation + "°";
          rotIdx = rotCycle.indexOf(j.rotation);
          if (rotIdx < 0) rotIdx = 0;
        }
        $$("img[src*='/video_feed']").forEach((img) => {
          const u = new URL(img.src, window.location.href);
          u.searchParams.set("t", Date.now());
          img.src = u.toString();
        });
      } finally {
        flip.disabled = false;
      }
    });
  }

  apply.addEventListener("click", async () => {
    const source = sel.value === "ipcam" ? url.value.trim() : sel.value;
    if (!source) return;
    apply.disabled = true; apply.textContent = "...";
    try {
      const r = await fetch("/api/camera", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({source}),
      });
      const j = await r.json();
      if (j.ok) {
        if (rotBtn && j.rotation != null) {
          rotBtn.textContent = j.rotation + "°";
          rotIdx = rotCycle.indexOf(j.rotation);
          if (rotIdx < 0) rotIdx = 0;
        }
        $$("img[src*='/video_feed']").forEach((img) => {
          const u = new URL(img.src, window.location.href);
          u.searchParams.set("t", Date.now());
          img.src = u.toString();
        });
      } else {
        alert("Camera error: " + (j.error || "unknown"));
      }
    } finally {
      apply.disabled = false; apply.textContent = "Apply";
    }
  });
})();

// ---------- Status polling --------------------------------------------------

window.__pollLiveStatus = async function () {
  try {
    const r = await fetch("/api/status");
    const j = await r.json();
    const fps = $("#fps-badge"); if (fps) fps.textContent = `${j.fps || 0} fps`;
    const hf = $("#hud-fps"); if (hf) hf.textContent = j.fps || 0;
    const hi = $("#hud-in"); if (hi) hi.textContent = j.in_frame;
    const hid = $("#hud-id"); if (hid) hid.textContent = j.identified;
    const hu = $("#hud-unk"); if (hu) hu.textContent = j.unknown;

    // Fetch arrivals when on /live
    if ($("#just-arrived") && j.active_session) {
      await refreshArrivals(j.active_session);
    }
  } catch (e) { /* ignore */ }
};
setInterval(window.__pollLiveStatus, 2000);
window.__pollLiveStatus();

let __lastArrivalKey = null;
async function refreshArrivals(sess) {
  if (!sess) return;
  const r = await fetch(`/api/dashboard?session_id=${sess.id}`);
  if (!r.ok) return;
  const data = await r.json();
  const list = $("#just-arrived");
  list.innerHTML = "";
  const arrivals = data.rows.filter(x => x.arrival_ts).sort((a,b)=> b.arrival_ts.localeCompare(a.arrival_ts)).slice(0, 12);
  const avatarColors = ["#3a5bc7","#2e7d5b","#8b5cf6","#b45309","#c03030","#1a8a8a","#6d28d9","#b8860b"];
  for (const a of arrivals) {
    const li = document.createElement("li");
    const t = new Date(a.arrival_ts);
    const ini = (a.name||"?").split(/\s+/).map(w=>w[0]).join("").slice(0,2).toUpperCase();
    const bg = avatarColors[(a.person_id||0) % avatarColors.length];
    li.innerHTML = `<div style="width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:${bg};font-size:10px;font-weight:700;color:#fff;flex-shrink:0">${ini}</div>
      <strong>${escapeHtml(a.name)}</strong>
      <span class="pill ${a.status}">${a.status}</span>
      <span class="when">${t.toLocaleTimeString()}</span>`;
    list.appendChild(li);
  }
}

// ---------- Index: session form / activate / edit / delete ------------------

(function initSessionPage() {
  const f = $("#session-form");
  if (!f) return;

  const submitBtn = $("#form-submit");
  const cancelBtn = $("#cancel-edit");
  const modeLabel = $("#form-mode-label");
  const dateInput = $("#session-date");
  const timeInput = $("#session-start-time");
  const timeWarn = $("#time-warn");

  function todayStr() { return new Date().toISOString().slice(0, 10); }
  function nowTimeStr() {
    const n = new Date();
    return `${String(n.getHours()).padStart(2,"0")}:${String(n.getMinutes()).padStart(2,"0")}`;
  }

  function applyDateConstraints() {
    if (!dateInput) return;
    const editing = !!f.dataset.editId;
    if (!editing) {
      dateInput.min = todayStr();
    } else {
      dateInput.removeAttribute("min");
    }
    enforceTimeMin();
  }

  function enforceTimeMin() {
    if (!dateInput || !timeInput) return;
    const isToday = dateInput.value === todayStr();
    const editing = !!f.dataset.editId;
    if (isToday && !editing) {
      const now = nowTimeStr();
      timeInput.min = now;
      if (timeInput.value < now) {
        timeInput.value = now;
      }
      if (timeWarn) { timeWarn.style.display = "none"; }
    } else {
      timeInput.removeAttribute("min");
      if (timeWarn) timeWarn.style.display = "none";
    }
  }

  if (dateInput) {
    if (!dateInput.value) dateInput.value = todayStr();
    dateInput.addEventListener("change", () => {
      if (!f.dataset.editId && dateInput.value < todayStr()) {
        dateInput.value = todayStr();
      }
      enforceTimeMin();
    });
  }
  if (timeInput) {
    if (timeInput.value === "09:00" && !f.dataset.editId) {
      timeInput.value = nowTimeStr();
    }
    timeInput.addEventListener("change", () => {
      if (!f.dataset.editId && dateInput && dateInput.value === todayStr()) {
        const now = nowTimeStr();
        if (timeInput.value < now) {
          timeInput.value = now;
          if (timeWarn) {
            timeWarn.textContent = "Start time can't be in the past — adjusted to now.";
            timeWarn.style.display = "block";
            setTimeout(() => { timeWarn.style.display = "none"; }, 3000);
          }
        }
      }
    });
  }
  applyDateConstraints();

  // Member picker
  const memberGrid = $("#member-grid");
  const memberFilter = $("#member-filter");
  const memberCount = $("#member-count");
  const memberAll = $("#member-select-all");
  const memberNone = $("#member-select-none");
  const memberBoxes = memberGrid ? Array.from(memberGrid.querySelectorAll("input[type=checkbox]")) : [];

  function updateMemberCount() {
    if (!memberCount) return;
    const n = memberBoxes.filter(cb => cb.checked).length;
    memberCount.textContent = `${n} selected`;
  }
  memberBoxes.forEach(cb => cb.addEventListener("change", updateMemberCount));

  if (memberFilter) {
    memberFilter.addEventListener("input", () => {
      const q = memberFilter.value.toLowerCase();
      memberGrid.querySelectorAll(".member-chip").forEach(chip => {
        const name = chip.dataset.name || "";
        const group = chip.dataset.group || "";
        const eid = chip.dataset.eid || "";
        chip.style.display = (!q || name.includes(q) || group.includes(q) || eid.includes(q)) ? "" : "none";
      });
    });
  }
  if (memberAll) {
    memberAll.addEventListener("click", () => {
      memberBoxes.forEach(cb => {
        if (cb.closest(".member-chip").style.display !== "none") cb.checked = true;
      });
      updateMemberCount();
    });
  }
  if (memberNone) {
    memberNone.addEventListener("click", () => {
      memberBoxes.forEach(cb => cb.checked = false);
      updateMemberCount();
    });
  }

  function getSelectedMemberIds() {
    return memberBoxes.filter(cb => cb.checked).map(cb => parseInt(cb.value));
  }

  function setSelectedMemberIds(ids) {
    const idSet = new Set(ids || []);
    memberBoxes.forEach(cb => { cb.checked = idSet.has(parseInt(cb.value)); });
    updateMemberCount();
  }

  function exitEditMode() {
    f.dataset.editId = "";
    submitBtn.textContent = "Start tracking →";
    cancelBtn.style.display = "none";
    modeLabel.textContent = "";
    f.reset();
    if (dateInput) dateInput.value = todayStr();
    if (timeInput) timeInput.value = nowTimeStr();
    setSelectedMemberIds([]);
    applyDateConstraints();
  }

  function enterEditMode(s) {
    f.dataset.editId = s.id;
    f.elements.name.value = s.name || "";
    f.elements.mode.value = s.mode || "student";
    f.elements.group_name.value = s.group_name || "";
    if (dateInput) dateInput.value = s.date || "";
    if (timeInput) timeInput.value = s.start_time || "09:00";
    f.elements.late_threshold_minutes.value = s.late_threshold_minutes ?? 15;
    if (f.elements.duration_minutes) f.elements.duration_minutes.value = s.duration_minutes || "";
    setSelectedMemberIds(s.member_ids || []);
    submitBtn.textContent = "Save changes →";
    cancelBtn.style.display = "inline-block";
    modeLabel.textContent = `Editing #${s.id}`;
    applyDateConstraints();
    f.scrollIntoView({behavior: "smooth", block: "start"});
  }

  cancelBtn.addEventListener("click", exitEditMode);

  f.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const fd = new FormData(f);
    const body = Object.fromEntries(fd.entries());
    body.member_ids = getSelectedMemberIds();
    delete body.member_id;
    const editId = f.dataset.editId;
    if (!editId) {
      const today = todayStr();
      if (body.session_date && body.session_date < today) {
        alert("Date can't be in the past.");
        return;
      }
      if (body.session_date === today && body.start_time < nowTimeStr()) {
        body.start_time = nowTimeStr();
      }
    }
    const url = editId ? `/api/sessions/${editId}` : "/api/sessions";
    const method = editId ? "PATCH" : "POST";
    const r = await fetch(url, {
      method, headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    });
    if (r.ok) location.reload();
    else alert("Failed: " + (await r.text()));
  });

  $$("[data-activate]").forEach(btn => {
    btn.addEventListener("click", async () => {
      const sid = btn.getAttribute("data-activate");
      await fetch(`/api/sessions/${sid}/activate`, {method: "POST"});
      location.reload();
    });
  });

  $$("[data-edit]").forEach(btn => {
    btn.addEventListener("click", () => {
      const tr = btn.closest("tr");
      try {
        const s = JSON.parse(tr.getAttribute("data-session"));
        enterEditMode(s);
      } catch (e) { alert("Could not load session: " + e.message); }
    });
  });

  $$("[data-delete]").forEach(btn => {
    btn.addEventListener("click", async () => {
      const sid = btn.getAttribute("data-delete");
      if (!confirm("Delete this session? Attendance rows for it will also go.")) return;
      const r = await fetch(`/api/sessions/${sid}`, {method: "DELETE"});
      if (r.ok) location.reload();
      else alert("Delete failed: " + (await r.text()));
    });
  });

  $$("[data-deactivate]").forEach(btn => {
    btn.addEventListener("click", async () => {
      await fetch("/api/sessions/deactivate", {method: "POST"});
      location.reload();
    });
  });
})();

// ---------- Voice assistant -------------------------------------------------

const Voice = (() => {
  const synth = window.speechSynthesis;
  if (!synth) return { speak() { return Promise.resolve(); }, ready: false };
  let _voice = null;
  let _ready = false;

  function pickVoice() {
    const voices = synth.getVoices();
    if (!voices.length) return;
    const prefs = [
      /jenny/i, /zira/i, /aria/i, /samantha/i, /karen/i,
      /hazel/i, /susan/i, /eva/i, /female/i, /woman/i,
    ];
    for (const re of prefs) {
      const v = voices.find(v => re.test(v.name) && /en/i.test(v.lang));
      if (v) { _voice = v; break; }
    }
    if (!_voice) {
      _voice = voices.find(v => /en/i.test(v.lang)) || voices[0];
    }
    _ready = true;
  }

  if (synth.getVoices().length) pickVoice();
  synth.addEventListener("voiceschanged", pickVoice);

  function speak(text) {
    return new Promise((resolve) => {
      if (!synth) { resolve(); return; }
      synth.cancel();
      const u = new SpeechSynthesisUtterance(text);
      if (_voice) u.voice = _voice;
      u.rate = 0.92;
      u.pitch = 1.05;
      u.volume = 1.0;
      u.onend = () => resolve();
      u.onerror = () => resolve();
      synth.speak(u);
      setTimeout(() => { if (synth.speaking) return; resolve(); }, 8000);
    });
  }

  return { speak, get ready() { return _ready; } };
})();

// ---------- Enroll ----------------------------------------------------------

(function initEnroll() {
  const f = $("#enroll-form");
  if (!f) return;

  const poseSelect = $("#pose-select");
  const enrollBtn = $("#enroll-btn");
  const clearBtn = f.querySelector("[type=reset]");
  const allInputs = Array.from(f.querySelectorAll("input, select"));
  const poseOrder = ["front", "left", "right"];

  let captureCount = 0;

  function updateBtnLabel() {
    const pose = poseSelect ? poseSelect.value : "front";
    enrollBtn.textContent = `Capture ${pose}`;
  }
  if (poseSelect) {
    poseSelect.addEventListener("change", updateBtnLabel);
    updateBtnLabel();
  }

  function lockForm(on) {
    enrollBtn.disabled = on;
    if (clearBtn) clearBtn.disabled = on;
    allInputs.forEach(el => { if (el !== poseSelect) el.readOnly = on; });
  }

  f.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const status = $("#enroll-status");
    const curPose = poseSelect ? poseSelect.value : "front";
    status.className = "status";
    status.style.display = "block";
    status.textContent = "Checking...";
    lockForm(true);

    // Pre-check only on the first capture (not mid-sequence)
    if (captureCount === 0) {
      try {
        const pc = await fetch("/api/enroll/precheck", {
          method: "POST", headers: {"Content-Type": "application/json"},
          body: JSON.stringify({}),
        });
        const pj = await pc.json();
        if (pj.recognized && pj.samples >= 3) {
          status.className = "status warn";
          status.textContent = `Already enrolled as ${pj.person.name} (${pj.samples} samples). Clear the form to enroll someone new.`;
          await Voice.speak(`Already enrolled as ${pj.person.name}.`);
          lockForm(false);
          return;
        }
      } catch (_) { /* precheck failed, continue with capture */ }
    }

    status.textContent = "Capturing...";
    await Voice.speak(curPose === "front" ? "Look straight." : `Turn ${curPose}.`);

    const fd = new FormData(f);
    const body = Object.fromEntries(fd.entries());
    try {
      const r = await fetch("/api/enroll", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body),
      });
      const j = await r.json();
      if (r.ok) {
        captureCount++;
        const curIdx = poseOrder.indexOf(curPose);
        const nextIdx = curIdx + 1;
        const samples = j.total_samples || captureCount;

        if (nextIdx < poseOrder.length) {
          const next = poseOrder[nextIdx];
          status.className = "status ok";
          status.textContent = `${captureCount}/3 captured. Now turn ${next}.`;
          await Voice.speak(`Got it. Turn ${next}.`);
          poseSelect.value = next;
        } else {
          status.className = "status ok";
          status.textContent = `3/3 captured. Fully enrolled!`;
          await Voice.speak("All done.");
          captureCount = 0;
          f.reset();
          poseSelect.value = poseOrder[0];
          enrollBtn.textContent = "Capture front";
        }
        updateBtnLabel();
        refreshEnrolledList();
      } else {
        status.className = "status err";
        status.textContent = j.error || "Failed.";
        await Voice.speak("Try again.");
      }
    } catch (e) {
      status.className = "status err";
      status.textContent = e.message;
    } finally {
      lockForm(false);
    }
  });

  async function refreshEnrolledList() {
    try {
      const r = await fetch("/api/persons");
      if (!r.ok) return;
      const people = await r.json();
      const grid = $("#enrolled-list");
      if (!grid) return;
      const heading = grid.closest("section")?.querySelector("h2");
      if (heading) heading.innerHTML = `Already enrolled <span class="muted" style="font-family:'JetBrains Mono',monospace;font-size:14px">(${people.length})</span>`;
      grid.innerHTML = "";
      const avatarColors = ["#3a5bc7","#2e7d5b","#8b5cf6","#b45309","#c03030","#1a8a8a","#6d28d9","#b8860b"];
      if (!people.length) {
        grid.innerHTML = `<p class="muted">No one enrolled yet.</p>`;
        return;
      }
      for (const p of people) {
        const ini = (p.name || "?").slice(0, 2).toUpperCase();
        const bg = avatarColors[p.id % avatarColors.length];
        const card = document.createElement("div");
        card.className = "thumb-card";
        card.dataset.pid = p.id;
        card.innerHTML = `<div class="avatar-initials" style="background:${bg}">${ini}</div>
          <div class="thumb-meta"><strong>${escapeHtml(p.name)}</strong>
          <small>${escapeHtml(p.external_id)} · ${p.role}${p.group_name ? " · " + escapeHtml(p.group_name) : ""}</small>
          <div style="display:flex;gap:8px;margin-top:4px">
            <button class="link delete" data-del="${p.id}">delete</button>
            <a class="link" href="/person/${p.id}">history &rarr;</a>
          </div></div>`;
        grid.appendChild(card);
      }
      bindDeleteButtons();
    } catch (e) { /* ignore */ }
  }

  function bindDeleteButtons() {
    $$("[data-del]").forEach(btn => {
      btn.onclick = async () => {
        if (!confirm("Delete this person?")) return;
        const id = btn.getAttribute("data-del");
        const r = await fetch(`/api/persons/${id}`, {method: "DELETE"});
        if (r.ok) refreshEnrolledList();
        else alert("Delete failed");
      };
    });
  }
  bindDeleteButtons();
})();

// ---------- Dashboard -------------------------------------------------------

let __hourlyChart = null;

window.__loadDashboard = async function (sessionId) {
  if (!sessionId) {
    $("#dash-title").textContent = "No active session — start one on the Home page.";
    return;
  }
  const r = await fetch(`/api/dashboard?session_id=${sessionId}`);
  if (!r.ok) {
    $("#dash-title").textContent = "Dashboard (session not found)";
    return;
  }
  const d = await r.json();
  const s = d.session;
  let title = `${s.name} · ${s.date} · ${s.mode}`;
  if (s.duration_minutes) {
    const startParts = s.start_time.split(":").map(Number);
    const endMs = new Date(`${s.date}T${s.start_time}:00`).getTime() + s.duration_minutes * 60000;
    const remaining = Math.max(0, Math.round((endMs - Date.now()) / 60000));
    if (remaining > 0) title += ` · ${remaining}m left`;
    else title += ` · ended`;
  }
  $("#dash-title").textContent = title;
  $("#kpi-present").textContent = d.counts.present;
  $("#kpi-late").textContent = d.counts.late;
  $("#kpi-absent").textContent = d.counts.absent;
  $("#kpi-total").textContent = d.counts.total;
  $("#kpi-unknown").textContent = d.unknown_count;
  $("#dash-export").href = `/api/dashboard/export?session_id=${sessionId}`;

  // Roster
  const tb = $("#roster-table tbody");
  tb.innerHTML = "";
  const avatarColors = ["#3a5bc7","#2e7d5b","#8b5cf6","#b45309","#c03030","#1a8a8a","#6d28d9","#b8860b"];
  for (const row of d.rows) {
    const tr = document.createElement("tr");
    const initials = (row.name || "?").split(/\s+/).map(w=>w[0]).join("").slice(0,2).toUpperCase();
    const bg = avatarColors[row.person_id % avatarColors.length];
    tr.innerHTML = `
      <td><div style="width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:${bg};font-size:10px;font-weight:700;color:#fff">${initials}</div></td>
      <td><a href="/person/${row.person_id}">${escapeHtml(row.name)}</a></td>
      <td>${escapeHtml(row.external_id)}</td>
      <td><span class="pill ${row.status}">${row.status}</span></td>
      <td>${row.arrival_ts ? new Date(row.arrival_ts).toLocaleTimeString() : "—"}</td>
      <td>${row.hours != null ? row.hours.toFixed(2) : "—"}</td>`;
    tb.appendChild(tr);
  }

  // Hourly chart
  const ctx = document.getElementById("hourlyChart").getContext("2d");
  const labels = d.hourly_labels || Array.from({length: 24}, (_, i) => `${i}:00`);
  if (__hourlyChart) __hourlyChart.destroy();
  __hourlyChart = new Chart(ctx, {
    type: "bar",
    data: { labels, datasets: [{
      label: "Arrivals", data: d.hourly,
      backgroundColor: "rgba(91,141,239,0.6)",
      borderColor: "#5b8def", borderWidth: 1, borderRadius: 3,
    }] },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, ticks: { precision: 0 }, grid: { color: "rgba(255,255,255,0.04)" } },
        x: { grid: { display: false } },
      },
    },
  });

  // Unknowns
  const ur = await fetch("/api/unknowns").then(r => r.json());
  const grid = $("#unknowns-grid");
  grid.innerHTML = "";
  if (!ur.length) {
    grid.innerHTML = `<p class="muted">No unknown sightings yet.</p>`;
  } else {
    grid.innerHTML = `<p class="muted">${ur.length} unknown sighting${ur.length > 1 ? "s" : ""} recorded (embedding vectors only, no images).</p>`;
  }
};

// ---------- Person history --------------------------------------------------

let __pctChart = null;

window.__loadPerson = async function (pid) {
  const r = await fetch(`/api/person/${pid}`);
  if (!r.ok) return;
  const d = await r.json();
  $("#person-pct").textContent = `${d.attendance_pct}%`;

  const cal = $("#cal-grid");
  cal.innerHTML = "";
  for (const cell of d.grid) {
    const div = document.createElement("div");
    div.className = `cal-cell ${cell.status}`;
    div.title = `${cell.date}: ${cell.status}`;
    cal.appendChild(div);
  }

  const tb = $("#history-table tbody");
  tb.innerHTML = "";
  for (const h of d.history) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${h.date}</td><td>${escapeHtml(h.session)}</td>
      <td><span class="pill ${h.status}">${h.status}</span></td>
      <td>${new Date(h.arrival_ts).toLocaleTimeString()}</td>
      <td>${h.departure_ts ? new Date(h.departure_ts).toLocaleTimeString() : "—"}</td>`;
    tb.appendChild(tr);
  }

  const ctx = document.getElementById("pctChart").getContext("2d");
  if (__pctChart) __pctChart.destroy();
  __pctChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Present/late", "Missed"],
      datasets: [{
        data: [d.attendance_pct, Math.max(0, 100 - d.attendance_pct)],
        backgroundColor: ["#34c759", "#1f1f28"],
        borderColor: "#111116", borderWidth: 2,
      }],
    },
    options: {
      cutout: "65%",
      plugins: { legend: { position: "bottom" } },
    },
  });
};

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"
  }[c]));
}
