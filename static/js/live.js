(function () {
  "use strict";

  const emotionUI = {
    happy: { emoji: "\uD83D\uDE0A", color: "#1f9d55" },
    sad: { emoji: "\uD83D\uDE22", color: "#2f6fed" },
    angry: { emoji: "\uD83D\uDE20", color: "#df3d43" },
    fear: { emoji: "\uD83D\uDE28", color: "#805ad5" },
    disgust: { emoji: "\uD83E\uDD22", color: "#0f9f7a" },
    surprise: { emoji: "\uD83D\uDE32", color: "#d98f19" },
    neutral: { emoji: "\uD83D\uDE10", color: "#5c6784" },
  };

  let mediaStream = null;
  let timer = null;
  let inFlight = false;

  function getCookie(name) {
    const value = "; " + document.cookie;
    const parts = value.split("; " + name + "=");
    if (parts.length === 2) return parts.pop().split(";").shift();
    return "";
  }

  function renderLiveState(message, klass) {
    const card = document.getElementById("liveResultCard");
    card.className = "prediction-card " + (klass || "empty");
    card.textContent = "";
    const wrap = document.createElement("div");
    wrap.className = "prediction-empty";
    wrap.textContent = message;
    card.appendChild(wrap);
  }

  function renderLivePrediction(data) {
    const emotionKey = (data.emotion || "neutral").toLowerCase();
    const ui = emotionUI[emotionKey] || emotionUI.neutral;
    const confidence = Math.max(0, Math.min(100, (data.confidence || 0) * 100));
    const scores = data.scores || {};
    const sortedScores = Object.entries(scores).sort(function (a, b) {
      return b[1] - a[1];
    });

    const card = document.getElementById("liveResultCard");
    card.className = "prediction-card";
    card.style.setProperty("--emotion-accent", ui.color);
    card.textContent = "";

    const head = document.createElement("div");
    head.className = "prediction-head";
    head.textContent = "Live Prediction";
    if (data.degraded) {
      const badge = document.createElement("span");
      badge.className = "badge-degraded";
      badge.textContent = "degraded";
      head.appendChild(badge);
    }
    card.appendChild(head);

    const main = document.createElement("div");
    main.className = "prediction-main single-col";
    const col = document.createElement("div");
    col.className = "pred-metrics-col";

    const labelPred = document.createElement("div");
    labelPred.className = "pred-label";
    labelPred.textContent = "Prediction";
    col.appendChild(labelPred);

    const badge = document.createElement("div");
    badge.className = "emotion-badge";
    const emoji = document.createElement("span");
    emoji.className = "emotion-emoji";
    emoji.textContent = ui.emoji;
    const name = document.createElement("span");
    name.className = "emotion-name";
    name.textContent = emotionKey;
    badge.appendChild(emoji);
    badge.appendChild(name);
    col.appendChild(badge);

    const confLabel = document.createElement("div");
    confLabel.className = "pred-label confidence-label";
    confLabel.textContent = "Confidence";
    col.appendChild(confLabel);

    const confRow = document.createElement("div");
    confRow.className = "confidence-row";
    const track = document.createElement("div");
    track.className = "confidence-track";
    const fill = document.createElement("div");
    fill.className = "confidence-fill";
    fill.style.width = confidence.toFixed(2) + "%";
    track.appendChild(fill);
    confRow.appendChild(track);
    const confVal = document.createElement("div");
    confVal.className = "confidence-value";
    confVal.textContent = confidence.toFixed(2) + "%";
    confRow.appendChild(confVal);
    col.appendChild(confRow);

    const topLabel = document.createElement("div");
    topLabel.className = "pred-label";
    topLabel.textContent = "Top Probabilities";
    col.appendChild(topLabel);

    const probWrap = document.createElement("div");
    probWrap.className = "prob-wrap";
    sortedScores.forEach(function (entry) {
      const label = entry[0];
      const value = entry[1];
      const percent = Math.max(0, Math.min(100, value * 100));
      const row = document.createElement("div");
      row.className = "prob-row";
      const lblEl = document.createElement("div");
      lblEl.className = "prob-label";
      lblEl.textContent = label;
      const barTrack = document.createElement("div");
      barTrack.className = "prob-bar-track";
      const barFill = document.createElement("div");
      barFill.className = "prob-bar-fill";
      barFill.style.width = percent.toFixed(2) + "%";
      barTrack.appendChild(barFill);
      const valEl = document.createElement("div");
      valEl.className = "prob-val";
      valEl.textContent = percent.toFixed(2) + "%";
      row.appendChild(lblEl);
      row.appendChild(barTrack);
      row.appendChild(valEl);
      probWrap.appendChild(row);
    });
    col.appendChild(probWrap);
    main.appendChild(col);
    card.appendChild(main);

    const footer = document.createElement("div");
    footer.className = "prediction-footer";
    footer.appendChild(footerCell("Latency", data.latency_ms + " ms"));
    footer.appendChild(footerCell("Model Version", data.model_version));
    footer.appendChild(footerCell("Model Loaded", data.model_loaded ? "yes" : "no"));
    card.appendChild(footer);
  }

  function footerCell(label, value) {
    const cell = document.createElement("div");
    const lbl = document.createElement("span");
    lbl.textContent = label;
    const strong = document.createElement("strong");
    strong.textContent = value;
    cell.appendChild(lbl);
    cell.appendChild(strong);
    return cell;
  }

  async function initCamera() {
    const video = document.getElementById("video");
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });
    video.srcObject = mediaStream;
    await video.play();
  }

  async function pushFrame() {
    if (inFlight) return;
    const canvas = document.getElementById("frameCanvas");
    const video = document.getElementById("video");
    if (!video.videoWidth || !video.videoHeight) return;

    inFlight = true;
    const ctx = canvas.getContext("2d");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.85);
    try {
      const response = await fetch("/api/v1/predict/live-frame/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-CSRFToken": getCookie("csrftoken"),
          "X-Consent": "true",
        },
        body: JSON.stringify({ image: dataUrl, consent: true }),
      });
      const data = await response.json();
      if (!response.ok || data.error) {
        renderLiveState("Prediction failed: " + (data.error || "Unknown error"), "error");
        return;
      }
      renderLivePrediction(data);
    } catch (err) {
      renderLiveState("Prediction failed: " + err.message, "error");
    } finally {
      inFlight = false;
    }
  }

  const consent = document.getElementById("consentCheckbox");
  const startBtn = document.getElementById("startBtn");
  consent.addEventListener("change", function () {
    startBtn.disabled = !consent.checked;
  });

  startBtn.addEventListener("click", async function () {
    if (!consent.checked) {
      renderLiveState("Consent is required before starting live prediction.", "error");
      return;
    }
    try {
      if (!mediaStream) await initCamera();
      if (!timer) {
        renderLiveState("Live prediction started. Analyzing frames...", "loading");
        timer = setInterval(pushFrame, 1200);
      }
    } catch (err) {
      renderLiveState("Camera error: " + err.message, "error");
    }
  });

  document.getElementById("stopBtn").addEventListener("click", function () {
    if (timer) {
      clearInterval(timer);
      timer = null;
    }
    if (mediaStream) {
      mediaStream.getTracks().forEach(function (track) {
        track.stop();
      });
      mediaStream = null;
    }
    renderLiveState("Live prediction stopped.");
  });
})();
