/* Titanic Survival Prediction — Dashboard logic & animations */
(function () {
  "use strict";

  /* ---------- Typing effect for hero title ---------- */
  var words = ["Titanic Survival Prediction", "A Data Science Voyage", "Predicting the Fate of 891"];
  var wordIdx = 0, charIdx = 0, deleting = false;
  var typedEl = document.getElementById("typed");

  function typeLoop() {
    var word = words[wordIdx];
    if (!deleting) {
      charIdx++;
      typedEl.textContent = word.slice(0, charIdx);
      if (charIdx === word.length) {
        deleting = true;
        setTimeout(typeLoop, 1800);
        return;
      }
      setTimeout(typeLoop, 65);
    } else {
      charIdx--;
      typedEl.textContent = word.slice(0, charIdx);
      if (charIdx === 0) {
        deleting = false;
        wordIdx = (wordIdx + 1) % words.length;
      }
      setTimeout(typeLoop, 32);
    }
  }
  typeLoop();

  /* ---------- Scroll reveal ---------- */
  var revealEls = document.querySelectorAll("[data-anim]");
  if ("IntersectionObserver" in window) {
    var io = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting) {
            e.target.style.animationDelay = "0s";
            e.target.style.opacity = "1";
            io.unobserve(e.target);
          }
        });
      },
      { threshold: 0.12 }
    );
    revealEls.forEach(function (el) { io.observe(el); });
  } else {
    revealEls.forEach(function (el) { el.style.opacity = "1"; });
  }

  /* free elements from the entrance animation so 3D tilt transforms can apply */
  revealEls.forEach(function (el) {
    el.addEventListener("animationend", function () {
      el.style.animation = "none";
    });
  });

  /* ---------- 3D tilt on cards ---------- */
  var tiltEls = document.querySelectorAll(".tilt-card");
  var reduceMotion =
    window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!reduceMotion) {
    tiltEls.forEach(function (el) {
      el.addEventListener("mousemove", function (ev) {
        var rect = el.getBoundingClientRect();
        var px = (ev.clientX - rect.left) / rect.width;
        var py = (ev.clientY - rect.top) / rect.height;
        var rx = (0.5 - py) * 10;
        var ry = (px - 0.5) * 12;
        el.style.transform =
          "perspective(900px) rotateX(" + rx + "deg) rotateY(" + ry + "deg) translateY(-4px)";
      });
      el.addEventListener("mouseleave", function () {
        el.style.transform = "perspective(900px) rotateX(0) rotateY(0) translateY(0)";
      });
    });
  }

  /* ---------- Animated counter ---------- */
  function animateCounter(el) {
    var target = parseFloat(el.dataset.target);
    var decimals = parseInt(el.dataset.decimals || "0", 10);
    var suffix = el.dataset.suffix || "";
    var duration = 1600, start = null;

    function step(ts) {
      if (!start) start = ts;
      var progress = Math.min((ts - start) / duration, 1);
      var eased = 1 - Math.pow(1 - progress, 3);
      var value = target * eased;
      el.textContent =
        (decimals ? value.toFixed(decimals) : Math.round(value).toString()) + suffix;
      if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  /* ---------- Load model statistics ---------- */
  function loadStats() {
    fetch("/api/stats")
      .then(function (res) { return res.json(); })
      .then(function (data) {
        var counters = document.querySelectorAll(".counter");
        counters[0].dataset.target = data.total;
        counters[1].dataset.target = Math.round((1 - data.survival_rate) * 100);
        counters[2].dataset.target = (data.accuracy * 100);
        counters[3].dataset.target = (data.cv_score * 100);
        counters.forEach(animateCounter);

        drawDonut(data.survival_rate * 100);
        drawGroupedBars("pclassBars", [
          { label: "1st Class", survived: data.pclass_survival["1"].survived, perished: data.pclass_survival["1"].not_survived },
          { label: "2nd Class", survived: data.pclass_survival["2"].survived, perished: data.pclass_survival["2"].not_survived },
          { label: "3rd Class", survived: data.pclass_survival["3"].survived, perished: data.pclass_survival["3"].not_survived }
        ]);
        drawGroupedBars("ageBars", data.age_bins.map(function (b) {
          return { label: b.label, survived: b.survived, perished: b.not_survived };
        }));
        drawImportance(data.importance);
      })
      .catch(function (err) {
        console.error("Stats failed:", err);
      });
  }

  function drawDonut(percent) {
    var fill = document.getElementById("donutFill");
    var C = 2 * Math.PI * 78; /* circumference for r=78 */
    fill.setAttribute("stroke-dasharray", C);
    setTimeout(function () {
      fill.style.strokeDashoffset = (C * (1 - percent / 100)).toString();
    }, 250);

    var num = document.getElementById("donutNum");
    var start = null;
    function count(ts) {
      if (!start) start = ts;
      var p = Math.min((ts - start) / 1600, 1);
      num.textContent = Math.round(percent * (1 - Math.pow(1 - p, 3))) + "%";
      if (p < 1) requestAnimationFrame(count);
    }
    requestAnimationFrame(count);
  }

  function drawGroupedBars(id, rows) {
    var container = document.getElementById(id);
    if (!container) return;
    var max = Math.max.apply(null, rows.map(function (r) { return r.survived + r.perished; }));
    container.innerHTML = "";
    rows.forEach(function (row) {
      var group = document.createElement("div");
      group.className = "bar-group";

      var label = document.createElement("div");
      label.className = "bar-label";
      label.textContent = row.label;

      var track = document.createElement("div");
      track.className = "bar-track";
      var f1 = document.createElement("div");
      f1.className = "bar-fill survived";
      f1.style.width = (row.survived / max * 100) + "%";
      var f2 = document.createElement("div");
      f2.className = "bar-fill perished";
      f2.style.width = (row.perished / max * 100) + "%";
      track.appendChild(f1);
      track.appendChild(f2);

      var legend = document.createElement("div");
      legend.className = "bar-legend";
      legend.innerHTML =
        '<span class="lg-s">' + row.survived + ' survived</span>' +
        '<span class="lg-p">' + row.perished + ' perished</span>';

      group.appendChild(label);
      group.appendChild(track);
      group.appendChild(legend);
      container.appendChild(group);
    });

    setTimeout(function () {
      container.querySelectorAll(".bar-fill").forEach(function (f) {
        var w = f.style.width;
        f.style.width = "0";
        requestAnimationFrame(function () {
          requestAnimationFrame(function () { f.style.width = w; });
        });
      });
    }, 300);
  }

  function drawImportance(items) {
    var container = document.getElementById("importanceBars");
    if (!container) return;
    var max = Math.max.apply(null, items.map(function (i) { return i.value; }));
    container.innerHTML = "";
    items.forEach(function (item, idx) {
      var row = document.createElement("div");
      row.className = "h-row";
      row.style.animation = "fadeIn 0.6s ease both";
      row.style.animationDelay = idx * 0.12 + "s";

      var label = document.createElement("div");
      label.className = "h-label";
      label.textContent = item.feature;

      var track = document.createElement("div");
      track.className = "h-track";
      var fill = document.createElement("span");
      fill.className = "h-fill";
      var pct = Math.max((item.value / max) * 100, 3);
      fill.style.width = "0%";
      track.appendChild(fill);

      var val = document.createElement("div");
      val.className = "h-val";
      val.textContent = (item.value * 100).toFixed(1) + "%";

      row.appendChild(label);
      row.appendChild(track);
      row.appendChild(val);
      container.appendChild(row);

      setTimeout(function () { fill.style.width = pct + "%"; }, 150 + idx * 120);
    });
  }

  /* ---------- Prediction ---------- */
  var form = document.getElementById("predictForm");
  var btn = document.getElementById("predictBtn");
  var scan = document.getElementById("scanOverlay");
  var empty = document.getElementById("resultEmpty");
  var body = document.getElementById("resultBody");
  var badge = document.getElementById("resultBadge");
  var gaugeFill = document.getElementById("gaugeFill");
  var prob = document.getElementById("resultProb");
  var detail = document.getElementById("resultDetail");
  var chartFigure = document.getElementById("chartFigure");
  var resultChart = document.getElementById("resultChart");

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    btn.classList.add("loading");
    btn.disabled = true;
    scan.classList.add("active");
    body.style.display = "none";
    empty.style.display = "block";

    var payload = {
      name: document.getElementById("name").value,
      sex: document.getElementById("sex").value,
      age: parseFloat(document.getElementById("age").value) || 30,
      pclass: parseInt(document.getElementById("pclass").value, 10),
      fare: parseFloat(document.getElementById("fare").value) || 0,
      sibsp: parseInt(document.getElementById("sibsp").value, 10) || 0,
      parch: parseInt(document.getElementById("parch").value, 10) || 0,
      embarked: document.getElementById("embarked").value,
      cabin_known: document.getElementById("cabin").checked
    };

    fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
      .then(function (res) { return res.json(); })
      .then(function (data) {
        setTimeout(function () { showResult(data); }, 900);
      })
      .catch(function (err) {
        console.error("Prediction failed:", err);
        scan.classList.remove("active");
        btn.classList.remove("loading");
        btn.disabled = false;
      });
  });

  function showResult(data) {
    scan.classList.remove("active");
    empty.style.display = "none";
    body.style.display = "block";
    body.style.animation = "riseIn 0.7s cubic-bezier(0.2,0.8,0.25,1) both";

    badge.className = "result-badge " + (data.prediction === 1 ? "survived" : "perished");
    badge.textContent = data.label;

    var pct = Math.round(data.probability_survived * 100);
    gaugeFill.style.width = "0%";
    prob.textContent = "Confidence: 0%";
    setTimeout(function () { gaugeFill.style.width = pct + "%"; }, 100);
    animateProbText(pct);

    detail.innerHTML =
      "Survival probability: <b>" + pct + "%</b> &middot; " +
      "Model verdict: <b>" + data.label + "</b>";

    if (data.chart) {
      resultChart.src = data.chart;
      chartFigure.hidden = false;
      chartFigure.style.animation = "none";
      void chartFigure.offsetWidth;
      chartFigure.style.animation = "";
    }

    btn.classList.remove("loading");
    btn.disabled = false;
  }

  function animateProbText(target) {
    var start = null;
    function step(ts) {
      if (!start) start = ts;
      var p = Math.min((ts - start) / 1400, 1);
      prob.textContent = "Confidence: " + Math.round(target * (1 - Math.pow(1 - p, 3))) + "%";
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  document.getElementById("againBtn").addEventListener("click", function () {
    body.style.display = "none";
    empty.style.display = "block";
    chartFigure.hidden = true;
    resultChart.src = "";
    form.reset();
    document.getElementById("sex").value = "male";
    document.getElementById("pclass").value = "3";
    document.getElementById("embarked").value = "S";
    document.getElementById("age").value = "28";
    document.getElementById("fare").value = "8.05";
  });

  document.getElementById("scrollBtn").addEventListener("click", function () {
    document.getElementById("stats").scrollIntoView({ behavior: "smooth" });
  });

  /* ---------- Boot ---------- */
  loadStats();
})();
