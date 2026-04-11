(function () {
  "use strict";

  const emotionUI = {
    happy: { emoji: "\uD83D\uDE0A", color: "#34e09a" },
    sad: { emoji: "\uD83D\uDE22", color: "#39e4ff" },
    angry: { emoji: "\uD83D\uDE20", color: "#ff4d8d" },
    fear: { emoji: "\uD83D\uDE28", color: "#9d82ff" },
    disgust: { emoji: "\uD83E\uDD22", color: "#34e09a" },
    surprise: { emoji: "\uD83D\uDE32", color: "#ffb547" },
    neutral: { emoji: "\uD83D\uDE10", color: "#b6bcd8" },
  };

  function getCookie(name) {
    const value = "; " + document.cookie;
    const parts = value.split("; " + name + "=");
    if (parts.length === 2) return parts.pop().split(";").shift();
    return "";
  }

  function setEmpty(card, text, cls) {
    card.className = "prediction-card " + (cls || "empty");
    card.style.removeProperty("--emotion-accent");
    card.textContent = "";
    const wrap = document.createElement("div");
    wrap.className = "prediction-empty";
    wrap.textContent = text;
    card.appendChild(wrap);
  }

  function renderError(card, message) {
    card.className = "prediction-card error";
    card.style.removeProperty("--emotion-accent");
    card.textContent = "";
    const wrap = document.createElement("div");
    wrap.className = "prediction-empty";
    const strong = document.createElement("strong");
    strong.textContent = "Prediction failed";
    wrap.appendChild(strong);
    wrap.appendChild(document.createElement("br"));
    wrap.appendChild(document.createTextNode(message || "Unknown error"));
    card.appendChild(wrap);
  }

  function renderPrediction(card, data, sampleImageSrc, sampleLabel) {
    if (data.error) {
      renderError(card, data.error);
      return;
    }

    const emotionKey = (data.emotion || "neutral").toLowerCase();
    const ui = emotionUI[emotionKey] || emotionUI.neutral;
    const confidence = Math.max(0, Math.min(100, (data.confidence || 0) * 100));
    const scores = data.scores || {};
    const sortedScores = Object.entries(scores).sort(function (a, b) {
      return b[1] - a[1];
    });

    card.className = "prediction-card";
    card.style.setProperty("--emotion-accent", ui.color);
    card.textContent = "";

    const head = document.createElement("div");
    head.className = "prediction-head";
    head.textContent = "Emotion Analysis Result";
    if (data.degraded) {
      const badge = document.createElement("span");
      badge.className = "badge-degraded";
      badge.textContent = "degraded";
      head.appendChild(badge);
    }
    card.appendChild(head);

    const main = document.createElement("div");
    main.className = "prediction-main";

    const imageCol = document.createElement("div");
    imageCol.className = "pred-image-col";

    const imgLabel = document.createElement("div");
    imgLabel.className = "pred-label";
    imgLabel.textContent = "Image";
    imageCol.appendChild(imgLabel);

    if (sampleImageSrc) {
      const img = document.createElement("img");
      img.className = "pred-sample-image";
      img.src = sampleImageSrc;
      img.alt = sampleLabel || emotionKey;
      imageCol.appendChild(img);
    }
    const caption = document.createElement("div");
    caption.className = "pred-sample-caption";
    caption.textContent = sampleLabel || "";
    imageCol.appendChild(caption);
    main.appendChild(imageCol);

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
    if (typeof data.model_loaded === "boolean") {
      footer.appendChild(footerCell("Model Loaded", data.model_loaded ? "yes" : "no"));
    }
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

  async function runSamplePrediction(sampleName) {
    const card = document.getElementById("sample-result-card");
    setEmpty(card, "Analyzing sample image...", "loading");
    try {
      const response = await fetch("/api/v1/predict/sample/" + sampleName + "/", {
        method: "POST",
        headers: {
          "X-CSRFToken": getCookie("csrftoken"),
          "X-Consent": "true",
        },
      });
      const data = await response.json();
      const btn = document.querySelector('.sample-card[data-sample="' + sampleName + '"]');
      const imageSrc = btn && btn.querySelector("img") ? btn.querySelector("img").src : "";
      const label = btn && btn.querySelector("span") ? btn.querySelector("span").textContent : sampleName;
      if (!response.ok) {
        renderError(card, data.error || "Server error " + response.status);
        return;
      }
      renderPrediction(card, data, imageSrc, label);
    } catch (err) {
      renderError(card, err.message);
    }
  }

  document.querySelectorAll(".sample-card").forEach(function (card) {
    card.addEventListener("click", function () {
      runSamplePrediction(card.dataset.sample);
    });
  });
})();
