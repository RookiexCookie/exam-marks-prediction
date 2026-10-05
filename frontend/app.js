// Frontend Controller for Exam Marks Prediction System
document.addEventListener("DOMContentLoaded", () => {
  const API_BASE = "";

  // State
  let modelInfo = null;
  let charts = {};

  // Form input elements
  const inputs = {
    study_hours: { slider: document.getElementById("input-study-hours"), num: document.getElementById("num-study-hours"), disp: document.getElementById("display-study-hours"), unit: " hrs/day" },
    attendance: { slider: document.getElementById("input-attendance"), num: document.getElementById("num-attendance"), disp: document.getElementById("display-attendance"), unit: "%" },
    assignment_score: { slider: document.getElementById("input-assignment"), num: document.getElementById("num-assignment"), disp: document.getElementById("display-assignment"), unit: "%" },
    previous_marks: { slider: document.getElementById("input-prev-exam"), num: document.getElementById("num-prev-exam"), disp: document.getElementById("display-prev-exam"), unit: "" },
    mock_test_score: { slider: document.getElementById("input-mock-test"), num: document.getElementById("num-mock-test"), disp: document.getElementById("display-mock-test"), unit: "" },
  };

  // What-If elements
  const whatif = {
    sliderHours: document.getElementById("whatif-slider-hours"),
    dispHours: document.getElementById("whatif-disp-hours"),
    sliderAtt: document.getElementById("whatif-slider-att"),
    dispAtt: document.getElementById("whatif-disp-att"),
    baseScore: document.getElementById("whatif-base-score"),
    simScore: document.getElementById("whatif-sim-score"),
    deltaBadge: document.getElementById("whatif-delta-badge"),
  };

  // -------------------------------------------------------------
  // 1. Navigation Tab Switching
  // -------------------------------------------------------------
  const tabs = document.querySelectorAll(".nav-btn");
  const tabContents = document.querySelectorAll(".content-tab");
  const pageTitle = document.getElementById("page-title");

  const tabTitles = {
    "tab-dashboard": "Exam Marks Predictor",
    "tab-analysis": "Exploratory Analysis",
    "tab-model": "ML Model Comparison",
    "tab-history": "Prediction History",
    "tab-about": "About ML Pipeline",
  };

  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const targetId = tab.getAttribute("data-tab");
      tabs.forEach(t => t.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));

      tab.classList.add("active");
      const targetContent = document.getElementById(targetId);
      if (targetContent) targetContent.classList.add("active");

      pageTitle.textContent = tabTitles[targetId] || "Dashboard";

      if (targetId === "tab-analysis") renderAnalysisCharts();
      if (targetId === "tab-model") renderModelEvaluation();
      if (targetId === "tab-history") loadPredictionHistory();
    });
  });

  // -------------------------------------------------------------
  // 2. Fetch Model Metadata
  // -------------------------------------------------------------
  async function fetchModelInfo() {
    try {
      const res = await fetch(`${API_BASE}/api/model-info`);
      if (!res.ok) throw new Error("Failed to load model info");
      modelInfo = await res.json();

      document.getElementById("sb-best-model").textContent = modelInfo.best_model_name;
      document.getElementById("sb-best-r2").textContent = modelInfo.r2_score.toFixed(3);
      document.getElementById("sb-best-rmse").textContent = modelInfo.rmse.toFixed(2);
    } catch (e) {
      console.warn("Could not fetch model info:", e);
    }
  }

  // -------------------------------------------------------------
  // 3. Form Input Binding (Sliders <-> Number Inputs)
  // -------------------------------------------------------------
  Object.keys(inputs).forEach(key => {
    const item = inputs[key];

    const syncVal = (val) => {
      item.slider.value = val;
      item.num.value = val;
      item.disp.textContent = `${val}${item.unit}`;
      updateTopKpis();
    };

    item.slider.addEventListener("input", (e) => syncVal(e.target.value));
    item.num.addEventListener("input", (e) => syncVal(e.target.value));
  });

  function updateTopKpis() {
    document.getElementById("kpi-study-hours").innerHTML = `${inputs.study_hours.slider.value} <span class="kpi-unit">hrs/day</span>`;
    document.getElementById("kpi-attendance").innerHTML = `${inputs.attendance.slider.value}<span class="kpi-unit">%</span>`;
    document.getElementById("kpi-assignment").innerHTML = `${inputs.assignment_score.slider.value}<span class="kpi-unit">%</span>`;
    document.getElementById("kpi-prev-exam").innerHTML = `${inputs.previous_marks.slider.value} <span class="kpi-unit">marks</span>`;
  }

  function getFormPayload() {
    return {
      study_hours: parseFloat(inputs.study_hours.slider.value),
      attendance: parseFloat(inputs.attendance.slider.value),
      assignment_score: parseFloat(inputs.assignment_score.slider.value),
      previous_marks: parseFloat(inputs.previous_marks.slider.value),
      mock_test_score: inputs.mock_test_score.slider.value ? parseFloat(inputs.mock_test_score.slider.value) : null
    };
  }

  // -------------------------------------------------------------
  // 4. Run Main Prediction
  // -------------------------------------------------------------
  async function predictMarks() {
    const payload = getFormPayload();
    try {
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (!res.ok) throw new Error("Prediction API call failed");
      const data = await res.json();

      // Major Number & Range
      document.getElementById("res-predicted-marks").textContent = data.predicted_marks.toFixed(1);
      document.getElementById("res-prediction-range").textContent = `${data.prediction_range.min.toFixed(0)}–${data.prediction_range.max.toFixed(0)} expected range`;
      document.getElementById("res-grade-pill").textContent = `Grade ${data.grade}`;
      document.getElementById("res-perf-badge").textContent = data.performance_level;

      // Progress Track
      const pct = Math.min(100, Math.max(0, data.predicted_marks));
      const fillBar = document.getElementById("compact-score-fill");
      if (fillBar) fillBar.style.width = `${pct}%`;

      // Supporting Benchmark
      const diff = data.predicted_marks - 72.4;
      document.getElementById("res-benchmark-note").innerHTML =
        `<strong>${diff >= 0 ? '+' : ''}${diff.toFixed(1)}</strong> vs class average (72.4)`;

      // Features
      renderFeatureImportances(data.factors);

      // What-If
      runWhatIf();
    } catch (e) {
      console.error("Prediction error:", e);
    }
  }

  document.getElementById("btn-predict").addEventListener("click", predictMarks);

  // -------------------------------------------------------------
  // 5. Presets
  // -------------------------------------------------------------
  const presets = {
    topper: { study_hours: 7.5, attendance: 94, assignment_score: 92, previous_marks: 88, mock_test_score: 86 },
    average: { study_hours: 4.5, attendance: 86, assignment_score: 78, previous_marks: 72, mock_test_score: 75 },
    atrisk: { study_hours: 1.5, attendance: 56, assignment_score: 45, previous_marks: 40, mock_test_score: 42 }
  };

  function applyPreset(preset) {
    Object.keys(preset).forEach(k => {
      if (inputs[k]) {
        inputs[k].slider.value = preset[k];
        inputs[k].num.value = preset[k];
        inputs[k].disp.textContent = `${preset[k]}${inputs[k].unit}`;
      }
    });
    updateTopKpis();
    predictMarks();
  }

  document.getElementById("btn-preset-topper").addEventListener("click", () => applyPreset(presets.topper));
  document.getElementById("btn-preset-avg").addEventListener("click", () => applyPreset(presets.average));
  document.getElementById("btn-preset-risk").addEventListener("click", () => applyPreset(presets.atrisk));

  // -------------------------------------------------------------
  // 6. Factors List
  // -------------------------------------------------------------
  function renderFeatureImportances(factors) {
    const list = document.getElementById("factors-list");
    list.innerHTML = "";

    factors.forEach(f => {
      const item = document.createElement("div");
      item.className = "factor-item";
      item.innerHTML = `
        <span class="factor-label">${f.label}</span>
        <div class="factor-bar-track">
          <div class="factor-bar-fill" style="width: ${f.percentage}%;"></div>
        </div>
        <span class="factor-pct">${f.percentage}%</span>
      `;
      list.appendChild(item);
    });
  }

  // -------------------------------------------------------------
  // 7. What-If Analysis
  // -------------------------------------------------------------
  async function runWhatIf() {
    const basePayload = getFormPayload();
    const modPayload = {
      ...basePayload,
      study_hours: parseFloat(whatif.sliderHours.value),
      attendance: parseFloat(whatif.sliderAtt.value)
    };

    whatif.dispHours.textContent = `${basePayload.study_hours} → ${modPayload.study_hours} hrs`;
    whatif.dispAtt.textContent = `${basePayload.attendance}% → ${modPayload.attendance}%`;

    try {
      const res = await fetch(`${API_BASE}/api/what-if`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ baseline: basePayload, modified: modPayload })
      });
      if (!res.ok) throw new Error("What-if failed");
      const data = await res.json();

      whatif.baseScore.textContent = data.baseline_predicted_marks.toFixed(1);
      whatif.simScore.textContent = data.modified_predicted_marks.toFixed(1);

      const diff = data.difference;
      whatif.deltaBadge.textContent = `${diff >= 0 ? '+' : ''}${diff.toFixed(1)} marks`;
      if (diff >= 0) {
        whatif.deltaBadge.style.background = "rgba(16, 185, 129, 0.1)";
        whatif.deltaBadge.style.color = "#10B981";
      } else {
        whatif.deltaBadge.style.background = "rgba(239, 68, 68, 0.1)";
        whatif.deltaBadge.style.color = "#EF4444";
      }
    } catch (e) {
      console.warn("What-if simulation error:", e);
    }
  }

  whatif.sliderHours.addEventListener("input", runWhatIf);
  whatif.sliderAtt.addEventListener("input", runWhatIf);

  // -------------------------------------------------------------
  // 8. Exploratory Data Analysis Charts
  // -------------------------------------------------------------
  async function renderAnalysisCharts() {
    try {
      const res = await fetch(`${API_BASE}/api/statistics`);
      if (!res.ok) throw new Error("Failed to load statistics");
      const data = await res.json();
      const stats = data.summary_statistics;

      const tbody = document.getElementById("stats-table-body");
      tbody.innerHTML = "";
      Object.keys(stats).forEach(feat => {
        const row = stats[feat];
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><strong>${feat.replace("_", " ").toUpperCase()}</strong></td>
          <td>${row.mean}</td>
          <td>${row.std}</td>
          <td>${row.min}</td>
          <td>${row["25%"]}</td>
          <td>${row["50%"]}</td>
          <td>${row["75%"]}</td>
          <td>${row.max}</td>
        `;
        tbody.appendChild(tr);
      });

      const chartDefaults = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { color: "#F1F5F9" } },
          y: { grid: { color: "#F1F5F9" } }
        }
      };

      if (!charts.examDist) {
        const ctx = document.getElementById("chart-exam-dist").getContext("2d");
        charts.examDist = new Chart(ctx, {
          type: "bar",
          data: {
            labels: ["<40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100"],
            datasets: [{
              data: [35, 70, 150, 280, 310, 220, 135],
              backgroundColor: ["#EF4444", "#F59E0B", "#F59E0B", "#2563EB", "#2563EB", "#10B981", "#10B981"],
              borderRadius: 4
            }]
          },
          options: chartDefaults
        });
      }

      if (!charts.studyScatter) {
        const ctx = document.getElementById("chart-study-scatter").getContext("2d");
        const pts = Array.from({ length: 50 }, () => {
          const h = 1 + Math.random() * 10;
          return { x: parseFloat(h.toFixed(1)), y: parseFloat(Math.min(100, Math.max(20, 40 + h * 4.2 + (Math.random() - 0.5) * 12)).toFixed(1)) };
        });
        charts.studyScatter = new Chart(ctx, {
          type: "scatter",
          data: { datasets: [{ data: pts, backgroundColor: "#2563EB", pointRadius: 3 }] },
          options: {
            ...chartDefaults,
            scales: {
              x: { title: { display: true, text: "Study Hours" }, grid: { color: "#F1F5F9" } },
              y: { title: { display: true, text: "Exam Marks" }, grid: { color: "#F1F5F9" } }
            }
          }
        });
      }

      if (!charts.attScatter) {
        const ctx = document.getElementById("chart-att-scatter").getContext("2d");
        const pts = Array.from({ length: 50 }, () => {
          const a = 45 + Math.random() * 55;
          return { x: parseFloat(a.toFixed(1)), y: parseFloat(Math.min(100, Math.max(25, 20 + a * 0.7 + (Math.random() - 0.5) * 14)).toFixed(1)) };
        });
        charts.attScatter = new Chart(ctx, {
          type: "scatter",
          data: { datasets: [{ data: pts, backgroundColor: "#10B981", pointRadius: 3 }] },
          options: {
            ...chartDefaults,
            scales: {
              x: { title: { display: true, text: "Attendance %" }, grid: { color: "#F1F5F9" } },
              y: { title: { display: true, text: "Exam Marks" }, grid: { color: "#F1F5F9" } }
            }
          }
        });
      }

      if (!charts.prevScatter) {
        const ctx = document.getElementById("chart-prev-scatter").getContext("2d");
        const pts = Array.from({ length: 50 }, () => {
          const p = 30 + Math.random() * 70;
          return { x: parseFloat(p.toFixed(1)), y: parseFloat(Math.min(100, Math.max(20, 15 + p * 0.85 + (Math.random() - 0.5) * 10)).toFixed(1)) };
        });
        charts.prevScatter = new Chart(ctx, {
          type: "scatter",
          data: { datasets: [{ data: pts, backgroundColor: "#F59E0B", pointRadius: 3 }] },
          options: {
            ...chartDefaults,
            scales: {
              x: { title: { display: true, text: "Previous Marks" }, grid: { color: "#F1F5F9" } },
              y: { title: { display: true, text: "Final Marks" }, grid: { color: "#F1F5F9" } }
            }
          }
        });
      }
    } catch (e) {
      console.warn("Analysis render error:", e);
    }
  }

  // -------------------------------------------------------------
  // 9. Model Evaluation
  // -------------------------------------------------------------
  async function renderModelEvaluation() {
    try {
      const res = await fetch(`${API_BASE}/api/model-metrics`);
      if (!res.ok) throw new Error("Failed to load metrics");
      const data = await res.json();

      const tbody = document.getElementById("leaderboard-table-body");
      tbody.innerHTML = "";
      data.leaderboard.forEach(item => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><strong>${item.model_name}</strong></td>
          <td>${item.mae.toFixed(2)}</td>
          <td>${item.mse.toFixed(2)}</td>
          <td>${item.rmse.toFixed(2)}</td>
          <td><strong style="color: #10B981;">${item.r2.toFixed(4)}</strong></td>
          <td>
            <span style="font-size: 0.72rem; font-weight: 700; color: ${item.is_best ? '#10B981' : '#64748B'};">
              ${item.is_best ? 'Active Model' : 'Evaluated'}
            </span>
          </td>
        `;
        tbody.appendChild(tr);
      });

      if (!charts.actualPred && data.test_actual_vs_pred) {
        const ctx = document.getElementById("chart-actual-pred").getContext("2d");
        const pts = data.test_actual_vs_pred.map(d => ({ x: d.actual, y: d.predicted }));

        charts.actualPred = new Chart(ctx, {
          type: "scatter",
          data: {
            datasets: [
              { label: "Student", data: pts, backgroundColor: "#2563EB", pointRadius: 3 },
              { label: "y = x", data: [{ x: 20, y: 20 }, { x: 100, y: 100 }], type: "line", borderColor: "#EF4444", borderDash: [4, 4], pointRadius: 0 }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              x: { title: { display: true, text: "Actual Marks" }, min: 20, max: 100, grid: { color: "#F1F5F9" } },
              y: { title: { display: true, text: "Predicted Marks" }, min: 20, max: 100, grid: { color: "#F1F5F9" } }
            }
          }
        });
      }
    } catch (e) {
      console.warn("Model eval load error:", e);
    }
  }

  // -------------------------------------------------------------
  // 10. History
  // -------------------------------------------------------------
  async function loadPredictionHistory() {
    try {
      const res = await fetch(`${API_BASE}/api/predictions?limit=30`);
      if (!res.ok) throw new Error("Failed to load history");
      const rows = await res.json();

      const tbody = document.getElementById("history-table-body");
      tbody.innerHTML = "";

      if (rows.length === 0) {
        tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; color: var(--text-sub); padding: 1.5rem;">No predictions yet. Predict from Dashboard to log records.</td></tr>`;
        return;
      }

      rows.forEach(r => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><code>${r.timestamp.split(" ")[1] || r.timestamp}</code></td>
          <td>${r.study_hours}h</td>
          <td>${r.attendance}%</td>
          <td>${r.assignment_score}%</td>
          <td>${r.previous_marks}</td>
          <td>${r.mock_test_score > 0 ? r.mock_test_score : '-'}</td>
          <td><strong style="color: #2563EB;">${r.predicted_marks.toFixed(1)}</strong></td>
          <td>${r.min_range.toFixed(0)}–${r.max_range.toFixed(0)}</td>
          <td><span style="font-weight: 600; color: #10B981;">${r.performance_level}</span></td>
        `;
        tbody.appendChild(tr);
      });
    } catch (e) {
      console.warn("History load error:", e);
    }
  }

  document.getElementById("btn-clear-history").addEventListener("click", async () => {
    if (confirm("Clear all prediction history?")) {
      try {
        await fetch(`${API_BASE}/api/predictions`, { method: "DELETE" });
        loadPredictionHistory();
      } catch (e) {
        console.error("Clear error:", e);
      }
    }
  });

  // Init
  fetchModelInfo();
  updateTopKpis();
  predictMarks();
});
