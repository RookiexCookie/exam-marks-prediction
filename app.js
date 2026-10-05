// Main Application Controller for Exam Marks Prediction Dashboard
document.addEventListener("DOMContentLoaded", () => {
  // Global State
  let dataset = DataManager.generateCohort(600, 42);
  let summaryStats = DataManager.calculateSummaryStats(dataset);
  let activeModelName = "Linear Regression";
  let trainedModels = {};
  let trainData = null;
  let testData = null;
  let charts = {};

  // Current Student Predict Inputs
  const currentInputs = {
    study_hours: 4.5,
    attendance_percentage: 82.0,
    assignment_score: 78.0,
    previous_exam_marks: 72.0,
    internal_marks: 38.0,
    study_days: 35
  };

  // Light SaaS Theme Colors
  const THEME = {
    blue: "#3B82F6",
    blueBg: "rgba(59, 130, 246, 0.12)",
    blueLight: "#DBEAFE",
    indigo: "#6366F1",
    green: "#10B981",
    greenBg: "rgba(16, 185, 129, 0.12)",
    amber: "#F59E0B",
    amberBg: "rgba(245, 158, 11, 0.12)",
    red: "#EF4444",
    redBg: "rgba(239, 68, 68, 0.12)",
    cyan: "#0EA5E9",
    textPrimary: "#0F172A",
    textMuted: "#64748B",
    textDim: "#94A3B8",
    border: "#E2E8F0",
    gridLine: "rgba(226, 232, 240, 0.6)"
  };

  // Configure Chart.js for light theme
  Chart.defaults.color = THEME.textMuted;
  Chart.defaults.font.family = "'Inter', sans-serif";
  Chart.defaults.font.size = 11;
  Chart.defaults.borderColor = THEME.gridLine;

  // -------------------------------------------------------------
  // 1. Train / Test Split & Model Training
  // -------------------------------------------------------------
  function trainAllModels() {
    const n = dataset.length;
    const testSize = Math.floor(n * 0.20);
    const shuffled = [...dataset].sort(() => 0.5 - Math.random());

    const testSet = shuffled.slice(0, testSize);
    const trainSet = shuffled.slice(testSize);

    trainData = {
      X: trainSet.map(d => DataManager.FEATURE_KEYS.map(k => d[k])),
      y: trainSet.map(d => d.exam_marks)
    };
    testData = {
      X: testSet.map(d => DataManager.FEATURE_KEYS.map(k => d[k])),
      y: testSet.map(d => d.exam_marks)
    };

    const linModel = new MLEngine.LinearRegressor(0.01);
    linModel.fit(trainData.X, trainData.y);
    const linPreds = linModel.predict(testData.X);
    const linEval = MLEngine.evaluate(testData.y, linPreds);

    const ridgeModel = new MLEngine.LinearRegressor(2.0);
    ridgeModel.fit(trainData.X, trainData.y);
    const ridgePreds = ridgeModel.predict(testData.X);
    const ridgeEval = MLEngine.evaluate(testData.y, ridgePreds);

    const rfModel = new MLEngine.RandomForestRegressor(12, 5);
    rfModel.fit(trainData.X, trainData.y);
    const rfPreds = rfModel.predict(testData.X);
    const rfEval = MLEngine.evaluate(testData.y, rfPreds);

    trainedModels = {
      "Linear Regression": { model: linModel, eval: linEval, preds: linPreds },
      "Ridge Regression": { model: ridgeModel, eval: ridgeEval, preds: ridgePreds },
      "Random Forest Regressor": { model: rfModel, eval: rfEval, preds: rfPreds }
    };

    updateLeaderboardTable();
    updateSidebarModelBadge();
  }

  function getActiveModel() {
    return trainedModels[activeModelName] || trainedModels["Linear Regression"];
  }

  function updateSidebarModelBadge() {
    const active = getActiveModel();
    const shortName = activeModelName === "Random Forest Regressor" ? "Random Forest" : (activeModelName === "Ridge Regression" ? "Ridge Reg." : "Linear Reg.");
    document.getElementById("sb-active-model").textContent = shortName;
    document.getElementById("sb-active-r2").textContent = `R² ${active.eval.r2.toFixed(3)}`;
  }

  function updateLeaderboardTable() {
    const tbody = document.getElementById("leaderboard-body");
    tbody.innerHTML = "";

    const rows = [
      { name: "Linear Regression", desc: "OLS with Standard Scaling", res: trainedModels["Linear Regression"].eval },
      { name: "Ridge Regression", desc: "L2 Regularization (α = 2.0)", res: trainedModels["Ridge Regression"].eval },
      { name: "Random Forest", desc: "Ensemble of 12 Decision Trees", res: trainedModels["Random Forest Regressor"].eval }
    ].sort((a, b) => b.res.r2 - a.res.r2);

    rows.forEach((r, idx) => {
      const tr = document.createElement("tr");
      const isBest = idx === 0;
      tr.innerHTML = `
        <td>
          <strong style="color: ${THEME.textPrimary};">${r.name}</strong>
          <div style="font-size: 0.68rem; color: ${THEME.textDim};">${r.desc}</div>
        </td>
        <td><strong style="color: ${THEME.green};">${r.res.r2.toFixed(4)}</strong></td>
        <td>${r.res.rmse.toFixed(2)}</td>
        <td>${r.res.mae.toFixed(2)}</td>
        <td>${r.res.mse.toFixed(2)}</td>
        <td>
          <span class="status-chip ${isBest ? 'best' : 'trained'}">
            ${isBest ? '⭐ Best' : 'Trained'}
          </span>
        </td>
      `;
      tbody.appendChild(tr);
    });
  }

  // -------------------------------------------------------------
  // 2. Overview Dashboard
  // -------------------------------------------------------------
  function updateOverviewKPIs() {
    summaryStats = DataManager.calculateSummaryStats(dataset);
    document.getElementById("kpi-cohort-size").textContent = summaryStats.count.toLocaleString();
    document.getElementById("kpi-mean-score").textContent = summaryStats.exam_marks.mean.toFixed(1);
    document.getElementById("kpi-pass-rate").textContent = `${summaryStats.pass_rate}%`;
    document.getElementById("kpi-study-hours").textContent = `${summaryStats.study_hours.mean.toFixed(1)}h`;
    document.getElementById("kpi-attendance").textContent = `${summaryStats.attendance_percentage.mean.toFixed(1)}%`;
  }

  function initOverviewCharts() {
    // 1. Marks Distribution
    const binLabels = ["0-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-90", "91-100"];
    const binCounts = new Array(binLabels.length).fill(0);

    dataset.forEach(d => {
      const m = d.exam_marks;
      if (m <= 20) binCounts[0]++;
      else if (m <= 30) binCounts[1]++;
      else if (m <= 40) binCounts[2]++;
      else if (m <= 50) binCounts[3]++;
      else if (m <= 60) binCounts[4]++;
      else if (m <= 70) binCounts[5]++;
      else if (m <= 80) binCounts[6]++;
      else if (m <= 90) binCounts[7]++;
      else binCounts[8]++;
    });

    const barColors = binLabels.map((_, i) =>
      i < 3 ? THEME.redBg : (i >= 6 ? THEME.greenBg : THEME.blueBg)
    );
    const barBorderColors = binLabels.map((_, i) =>
      i < 3 ? THEME.red : (i >= 6 ? THEME.green : THEME.blue)
    );

    const ctxMarks = document.getElementById("chart-marks-dist").getContext("2d");
    charts.marksDist = new Chart(ctxMarks, {
      type: "bar",
      data: {
        labels: binLabels,
        datasets: [{
          label: "Students",
          data: binCounts,
          backgroundColor: barColors,
          borderColor: barBorderColors,
          borderWidth: 1.5,
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { display: false }, ticks: { font: { size: 10 } } },
          y: { grid: { color: THEME.gridLine }, ticks: { font: { size: 10 } } }
        }
      }
    });

    // 2. Study Hours Scatter
    const scatterData = dataset.slice(0, 150).map(d => ({ x: d.study_hours, y: d.exam_marks }));
    const ctxStudy = document.getElementById("chart-study-scatter").getContext("2d");
    charts.studyScatter = new Chart(ctxStudy, {
      type: "scatter",
      data: {
        datasets: [{
          label: "Student",
          data: scatterData,
          backgroundColor: "rgba(59, 130, 246, 0.45)",
          borderColor: THEME.blue,
          borderWidth: 1,
          pointRadius: 3.5,
          pointHoverRadius: 5
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { title: { display: true, text: "Daily Study Hours", font: { size: 10, weight: '600' } }, grid: { color: THEME.gridLine } },
          y: { title: { display: true, text: "Exam Marks", font: { size: 10, weight: '600' } }, grid: { color: THEME.gridLine }, min: 20, max: 100 }
        }
      }
    });

    // 3. Attendance Bracket Means
    const brackets = [
      { label: "< 60%", filter: d => d.attendance_percentage < 60 },
      { label: "60-75%", filter: d => d.attendance_percentage >= 60 && d.attendance_percentage < 75 },
      { label: "75-85%", filter: d => d.attendance_percentage >= 75 && d.attendance_percentage < 85 },
      { label: "85-100%", filter: d => d.attendance_percentage >= 85 }
    ];
    const bracketMeans = brackets.map(b => {
      const match = dataset.filter(b.filter);
      return match.length > 0 ? Number((match.reduce((acc, d) => acc + d.exam_marks, 0) / match.length).toFixed(1)) : 0;
    });

    const ctxAtt = document.getElementById("chart-attendance-bracket").getContext("2d");
    charts.attBracket = new Chart(ctxAtt, {
      type: "bar",
      data: {
        labels: brackets.map(b => b.label),
        datasets: [{
          label: "Avg Marks",
          data: bracketMeans,
          backgroundColor: [THEME.redBg, THEME.amberBg, THEME.blueBg, THEME.greenBg],
          borderColor: [THEME.red, THEME.amber, THEME.blue, THEME.green],
          borderWidth: 1.5,
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { display: false }, ticks: { font: { size: 10 } } },
          y: { min: 30, max: 100, grid: { color: THEME.gridLine }, ticks: { font: { size: 10 } } }
        }
      }
    });

    // 4. Feature Correlations
    const corrKeys = ["study_hours", "attendance_percentage", "assignment_score", "previous_exam_marks", "internal_marks", "study_days"];
    const corrLabels = ["Study Hrs", "Attendance", "Assignments", "Prev Exam", "Internals", "Study Days"];

    function getPearson(k1, k2) {
      const mean1 = summaryStats[k1].mean, mean2 = summaryStats[k2].mean;
      let num = 0, den1 = 0, den2 = 0;
      dataset.forEach(d => {
        const diff1 = d[k1] - mean1, diff2 = d[k2] - mean2;
        num += diff1 * diff2;
        den1 += diff1 * diff1;
        den2 += diff2 * diff2;
      });
      return Number((num / (Math.sqrt(den1 * den2) || 1.0)).toFixed(2));
    }

    const correlations = corrKeys.map(k => getPearson(k, "exam_marks"));
    const ctxCorr = document.getElementById("chart-correlation-bar").getContext("2d");
    charts.corrBar = new Chart(ctxCorr, {
      type: "bar",
      data: {
        labels: corrLabels,
        datasets: [{
          label: "Correlation r",
          data: correlations,
          backgroundColor: THEME.blueBg,
          borderColor: THEME.blue,
          borderWidth: 1.5,
          borderRadius: 4
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { min: 0, max: 1, grid: { color: THEME.gridLine }, ticks: { font: { size: 10 } } },
          y: { grid: { display: false }, ticks: { font: { size: 10 } } }
        }
      }
    });
  }

  // -------------------------------------------------------------
  // 3. Prediction & Recommendations
  // -------------------------------------------------------------
  function runPrediction() {
    const active = getActiveModel();
    const inputVec = DataManager.FEATURE_KEYS.map(k => currentInputs[k]);
    const predScore = active.model.predictSample(inputVec);
    const rmse = active.eval.rmse || 3.0;

    const low95 = Math.max(0, Math.min(100, Number((predScore - 1.96 * rmse).toFixed(1))));
    const high95 = Math.max(0, Math.min(100, Number((predScore + 1.96 * rmse).toFixed(1))));

    const gradeInfo = MLEngine.getGradeInfo(predScore);

    document.getElementById("res-score").textContent = predScore.toFixed(1);

    const gradePill = document.getElementById("res-grade-pill");
    gradePill.textContent = `Grade ${gradeInfo.grade} • ${gradeInfo.class}`;
    gradePill.style.color = gradeInfo.color;
    gradePill.style.borderColor = gradeInfo.color;
    gradePill.style.background = `${gradeInfo.color}12`;

    document.getElementById("res-interval").innerHTML =
      `🎯 <strong>95% Interval:</strong> ${low95} &mdash; ${high95} Marks (&plusmn;${(1.96 * rmse).toFixed(1)})`;

    const cohortMean = summaryStats.exam_marks.mean;
    const diff = predScore - cohortMean;
    document.getElementById("res-benchmark").innerHTML =
      `📊 <strong>Benchmark:</strong> ${diff >= 0 ? '+' : ''}${diff.toFixed(1)} marks ${diff >= 0 ? 'above' : 'below'} cohort mean (${cohortMean.toFixed(1)}).`;

    renderSmartRecommendations(currentInputs, predScore);
  }

  function renderSmartRecommendations(inputs, score) {
    const container = document.getElementById("recs-container");
    container.innerHTML = "";
    const recs = [];

    if (inputs.study_hours < 3.0) {
      recs.push({
        type: "warn", icon: "⚠️",
        title: "Low Study Hours",
        body: `Currently ${inputs.study_hours}h/day. Increasing by 1.5–2.0 hours can yield ~+6–8 marks.`
      });
    } else if (inputs.study_hours >= 7.0) {
      recs.push({
        type: "success", icon: "🌟",
        title: "Strong Study Commitment",
        body: `${inputs.study_hours}h daily is excellent. Maintain with periodic review tests.`
      });
    }

    if (inputs.attendance_percentage < 75.0) {
      recs.push({
        type: "danger", icon: "🚨",
        title: "Attendance Warning (< 75%)",
        body: `At ${inputs.attendance_percentage}%, below mandatory threshold. Attend remaining lectures.`
      });
    } else {
      recs.push({
        type: "success", icon: "✅",
        title: "Good Attendance",
        body: `${inputs.attendance_percentage}% is above threshold. Keep it up.`
      });
    }

    if (inputs.internal_marks < 30.0) {
      recs.push({
        type: "warn", icon: "📝",
        title: "Boost Internals",
        body: `${inputs.internal_marks}/50 is below average. Focus on continuous evaluation.`
      });
    }

    recs.forEach(r => {
      const div = document.createElement("div");
      div.className = `rec-item ${r.type}`;
      div.innerHTML = `
        <span class="rec-icon">${r.icon}</span>
        <div class="rec-content">
          <h4>${r.title}</h4>
          <p>${r.body}</p>
        </div>
      `;
      container.appendChild(div);
    });
  }

  // -------------------------------------------------------------
  // 4. Goal Planner
  // -------------------------------------------------------------
  function updateGoalPlanner() {
    const target = Number(document.getElementById("input-target-goal").value);
    const currH = Number(document.getElementById("input-goal-curr-hrs").value);
    const currA = Number(document.getElementById("input-goal-curr-att").value);

    document.getElementById("val-target-goal").textContent = `${target} / 100`;
    document.getElementById("val-goal-curr-hrs").textContent = `${currH} hrs/day`;
    document.getElementById("val-goal-curr-att").textContent = `${currA} %`;
    document.getElementById("goal-target-display").textContent = target.toFixed(1);

    const active = getActiveModel();
    const baselineVec = [currH, currA, 75, 70, 35, 30];
    const currEst = active.model.predictSample(baselineVec);
    document.getElementById("goal-curr-pred").textContent = currEst.toFixed(1);

    const gap = target - currEst;
    const gapElem = document.getElementById("goal-gap-display");
    gapElem.textContent = `${gap >= 0 ? '+' : ''}${gap.toFixed(1)}`;
    gapElem.style.color = gap <= 0 ? THEME.green : THEME.amber;

    const solBox = document.getElementById("goal-solution-box");

    if (gap <= 0) {
      solBox.innerHTML = `
        <div style="color: ${THEME.green}; font-weight: 700; margin-bottom: 0.25rem;">🎉 Goal Already Met!</div>
        Your current projected score of <strong>${currEst.toFixed(1)}</strong> meets your target of <strong>${target}</strong>.
      `;
    } else {
      let reqH = null;
      for (let h = currH; h <= 14.0; h += 0.2) {
        const testVec = [h, currA, 75, 70, 35, 30];
        if (active.model.predictSample(testVec) >= target) { reqH = Number(h.toFixed(1)); break; }
      }

      let reqA = null;
      for (let a = currA; a <= 100.0; a += 0.5) {
        const testVec = [currH, a, 75, 70, 35, 30];
        if (active.model.predictSample(testVec) >= target) { reqA = Number(a.toFixed(1)); break; }
      }

      let solHtml = `<div style="font-weight: 700; color: ${THEME.textPrimary}; margin-bottom: 0.35rem;">Action Plan (+${gap.toFixed(1)} marks needed):</div>`;
      if (reqH) {
        solHtml += `<div>• <strong>Study:</strong> Increase from <code>${currH}h</code> to <code>${reqH}h/day</code> (+${(reqH - currH).toFixed(1)}h).</div>`;
      } else {
        solHtml += `<div>• Study hours alone plateau at diminishing returns.</div>`;
      }
      if (reqA) {
        solHtml += `<div>• <strong>Attendance:</strong> Raise from <code>${currA}%</code> to <code>${reqA}%</code> (+${(reqA - currA).toFixed(1)}%).</div>`;
      } else {
        solHtml += `<div>• Attendance alone insufficient — combine with more study hours.</div>`;
      }
      solBox.innerHTML = solHtml;
    }

    renderSensitivityCurve(baselineVec, target, currH);
  }

  function renderSensitivityCurve(baselineVec, target, currH) {
    const active = getActiveModel();
    const hRange = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    const curvePoints = hRange.map(h => {
      const v = [...baselineVec];
      v[0] = h;
      return active.model.predictSample(v);
    });

    if (charts.sensitivity) charts.sensitivity.destroy();

    const ctx = document.getElementById("chart-sensitivity-curve").getContext("2d");
    charts.sensitivity = new Chart(ctx, {
      type: "line",
      data: {
        labels: hRange.map(h => `${h}h`),
        datasets: [
          {
            label: "Projected Marks",
            data: curvePoints,
            borderColor: THEME.blue,
            backgroundColor: THEME.blueBg,
            fill: true,
            tension: 0.35,
            pointRadius: 3,
            borderWidth: 2
          },
          {
            label: "Target",
            data: new Array(hRange.length).fill(target),
            borderColor: THEME.green,
            borderDash: [5, 5],
            pointRadius: 0,
            borderWidth: 1.5
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { title: { display: true, text: "Study Hours", font: { size: 10, weight: '600' } }, grid: { color: THEME.gridLine } },
          y: { title: { display: true, text: "Marks", font: { size: 10, weight: '600' } }, min: 30, max: 100, grid: { color: THEME.gridLine } }
        }
      }
    });
  }

  // -------------------------------------------------------------
  // 5. Training Charts
  // -------------------------------------------------------------
  function initTrainingCharts() {
    const active = getActiveModel();
    const yTrue = testData.y;
    const yPred = active.preds;

    const scatterPoints = Array.from({ length: yTrue.length }, (_, i) => ({
      x: yTrue[i],
      y: yPred[i]
    }));

    if (charts.actualVsPred) charts.actualVsPred.destroy();
    const ctxAvP = document.getElementById("chart-actual-vs-pred").getContext("2d");
    charts.actualVsPred = new Chart(ctxAvP, {
      type: "scatter",
      data: {
        datasets: [
          {
            label: "Test Student",
            data: scatterPoints,
            backgroundColor: "rgba(14, 165, 233, 0.45)",
            borderColor: THEME.cyan,
            borderWidth: 1,
            pointRadius: 3.5
          },
          {
            label: "Ideal (y = x)",
            data: [{ x: 20, y: 20 }, { x: 100, y: 100 }],
            type: "line",
            borderColor: THEME.red,
            borderDash: [4, 4],
            borderWidth: 1.5,
            pointRadius: 0
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { title: { display: true, text: "Actual Marks", font: { size: 10, weight: '600' } }, min: 20, max: 100, grid: { color: THEME.gridLine } },
          y: { title: { display: true, text: "Predicted Marks", font: { size: 10, weight: '600' } }, min: 20, max: 100, grid: { color: THEME.gridLine } }
        }
      }
    });

    // Residuals Histogram
    const residuals = active.eval.residuals;
    const resLabels = ["<-6", "-6 to -4", "-4 to -2", "-2 to 0", "0 to 2", "2 to 4", "4 to 6", ">6"];
    const resCounts = new Array(resLabels.length).fill(0);

    residuals.forEach(r => {
      if (r < -6) resCounts[0]++;
      else if (r < -4) resCounts[1]++;
      else if (r < -2) resCounts[2]++;
      else if (r < 0) resCounts[3]++;
      else if (r < 2) resCounts[4]++;
      else if (r < 4) resCounts[5]++;
      else if (r < 6) resCounts[6]++;
      else resCounts[7]++;
    });

    if (charts.residuals) charts.residuals.destroy();
    const ctxRes = document.getElementById("chart-residuals").getContext("2d");
    charts.residuals = new Chart(ctxRes, {
      type: "bar",
      data: {
        labels: resLabels,
        datasets: [{
          label: "Count",
          data: resCounts,
          backgroundColor: THEME.greenBg,
          borderColor: THEME.green,
          borderWidth: 1.5,
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { display: false }, ticks: { font: { size: 10 } } },
          y: { grid: { color: THEME.gridLine }, ticks: { font: { size: 10 } } }
        }
      }
    });

    initFeatureImportanceChart();
  }

  function initFeatureImportanceChart() {
    const active = getActiveModel();
    const featureLabels = ["Study Hours", "Attendance", "Assignment", "Prev Exam", "Internal", "Study Days"];
    const importances = active.model.getFeatureImportances(featureLabels);

    if (charts.featureImp) charts.featureImp.destroy();
    const ctx = document.getElementById("chart-feature-importance").getContext("2d");
    charts.featureImp = new Chart(ctx, {
      type: "bar",
      data: {
        labels: importances.map(f => f.name),
        datasets: [{
          label: "Importance (%)",
          data: importances.map(f => f.pct),
          backgroundColor: THEME.blueBg,
          borderColor: THEME.blue,
          borderWidth: 1.5,
          borderRadius: 4
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { min: 0, max: 40, grid: { color: THEME.gridLine }, ticks: { font: { size: 10 } } },
          y: { grid: { display: false }, ticks: { font: { size: 10 } } }
        }
      }
    });
  }

  // -------------------------------------------------------------
  // 6. Dataset Explorer
  // -------------------------------------------------------------
  function renderDatasetTable(filterQuery = "") {
    const tbody = document.getElementById("dataset-body");
    tbody.innerHTML = "";

    const filtered = filterQuery
      ? dataset.filter(d => d.student_id.toLowerCase().includes(filterQuery.toLowerCase()))
      : dataset.slice(0, 50);

    filtered.forEach(d => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><code style="color: ${THEME.blue}; font-size: 0.74rem;">${d.student_id}</code></td>
        <td>${d.study_hours}h</td>
        <td>${d.attendance_percentage}%</td>
        <td>${d.assignment_score}</td>
        <td>${d.previous_exam_marks}</td>
        <td>${d.internal_marks}/50</td>
        <td>${d.study_days}d</td>
        <td><strong style="color: ${d.exam_marks >= 40 ? THEME.textPrimary : THEME.red};">${d.exam_marks}</strong></td>
      `;
      tbody.appendChild(tr);
    });
  }

  // -------------------------------------------------------------
  // 7. Event Listeners
  // -------------------------------------------------------------
  function bindUIEvents() {
    // Navigation
    document.querySelectorAll(".nav-item").forEach(item => {
      item.addEventListener("click", () => {
        document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));
        document.querySelectorAll(".view-container").forEach(v => v.classList.remove("active"));

        item.classList.add("active");
        const targetView = item.getAttribute("data-view");
        document.getElementById(targetView).classList.add("active");

        const titleMap = {
          "view-overview": "Overview Dashboard",
          "view-predict": "Predict Marks",
          "view-planner": "Goal Planner",
          "view-training": "Model Training",
          "view-insights": "Insights & Reports",
          "view-data": "Dataset Explorer"
        };
        document.getElementById("page-title").textContent = titleMap[targetView] || "Dashboard";

        if (targetView === "view-planner") updateGoalPlanner();
      });
    });

    // Model Selector
    document.getElementById("model-select").addEventListener("change", (e) => {
      activeModelName = e.target.value;
      updateSidebarModelBadge();
      runPrediction();
      initTrainingCharts();
      if (document.getElementById("view-planner").classList.contains("active")) {
        updateGoalPlanner();
      }
    });

    // Slider Bindings
    const sliderBindings = [
      { id: "input-study-hours", disp: "val-study-hours", key: "study_hours", suffix: " hrs" },
      { id: "input-attendance", disp: "val-attendance", key: "attendance_percentage", suffix: " %" },
      { id: "input-assignment", disp: "val-assignment", key: "assignment_score", suffix: "" },
      { id: "input-previous", disp: "val-previous", key: "previous_exam_marks", suffix: "" },
      { id: "input-internal", disp: "val-internal", key: "internal_marks", suffix: " /50" },
      { id: "input-days", disp: "val-days", key: "study_days", suffix: " days" }
    ];

    sliderBindings.forEach(s => {
      const slider = document.getElementById(s.id);
      slider.addEventListener("input", (e) => {
        const val = Number(e.target.value);
        currentInputs[s.key] = val;
        document.getElementById(s.disp).textContent = `${val}${s.suffix}`;
        runPrediction();
      });
    });

    // Presets
    document.getElementById("preset-topper").addEventListener("click", () => {
      applyPreset({ study_hours: 8.0, attendance_percentage: 94.0, assignment_score: 92.0, previous_exam_marks: 90.0, internal_marks: 46.0, study_days: 50 });
    });
    document.getElementById("preset-average").addEventListener("click", () => {
      applyPreset({ study_hours: 4.5, attendance_percentage: 80.0, assignment_score: 72.0, previous_exam_marks: 68.0, internal_marks: 35.0, study_days: 30 });
    });
    document.getElementById("preset-atrisk").addEventListener("click", () => {
      applyPreset({ study_hours: 1.5, attendance_percentage: 52.0, assignment_score: 42.0, previous_exam_marks: 38.0, internal_marks: 18.0, study_days: 12 });
    });

    function applyPreset(preset) {
      Object.assign(currentInputs, preset);
      sliderBindings.forEach(s => {
        document.getElementById(s.id).value = currentInputs[s.key];
        document.getElementById(s.disp).textContent = `${currentInputs[s.key]}${s.suffix}`;
      });
      runPrediction();
    }

    // Goal Planner
    document.getElementById("input-target-goal").addEventListener("input", updateGoalPlanner);
    document.getElementById("input-goal-curr-hrs").addEventListener("input", updateGoalPlanner);
    document.getElementById("input-goal-curr-att").addEventListener("input", updateGoalPlanner);

    // Dataset Search
    document.getElementById("data-search").addEventListener("input", (e) => {
      renderDatasetTable(e.target.value);
    });

    // CSV Export
    document.getElementById("btn-export-csv").addEventListener("click", () => {
      const headers = ["student_id", ...DataManager.FEATURE_KEYS, DataManager.TARGET_KEY];
      const rows = dataset.map(d => headers.map(h => d[h]).join(","));
      const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows].join("\n");
      const encodedUri = encodeURI(csvContent);
      const link = document.createElement("a");
      link.setAttribute("href", encodedUri);
      link.setAttribute("download", "student_exam_data.csv");
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    });

    // Reset Data
    document.getElementById("btn-regenerate-data").addEventListener("click", () => {
      dataset = DataManager.generateCohort(600, Math.floor(Math.random() * 1000));
      updateOverviewKPIs();
      trainAllModels();
      runPrediction();
      initTrainingCharts();
      renderDatasetTable();
    });

    // Download Report
    document.getElementById("btn-download-report").addEventListener("click", () => {
      const name = document.getElementById("report-name").value || "Student";
      const active = getActiveModel();
      const inputVec = DataManager.FEATURE_KEYS.map(k => currentInputs[k]);
      const predScore = active.model.predictSample(inputVec);
      const grade = MLEngine.getGradeInfo(predScore);

      const report = `============================================================
EXAM MARKS PREDICTION — ACADEMIC REPORT
============================================================
Student       : ${name}
Algorithm     : ${activeModelName} (R² = ${active.eval.r2})
Predicted Mark: ${predScore.toFixed(1)} / 100
95% Interval  : ${(predScore - 1.96 * active.eval.rmse).toFixed(1)} – ${(predScore + 1.96 * active.eval.rmse).toFixed(1)} Marks
Grade         : ${grade.grade} (${grade.status} — ${grade.class})

--- ACADEMIC PROFILE ---
• Study Hours     : ${currentInputs.study_hours} hrs/day
• Attendance      : ${currentInputs.attendance_percentage}%
• Assignment Score: ${currentInputs.assignment_score} / 100
• Previous Exam   : ${currentInputs.previous_exam_marks} / 100
• Internal Marks  : ${currentInputs.internal_marks} / 50
• Study Days Left : ${currentInputs.study_days} days

--- RECOMMENDATIONS ---
1. ${currentInputs.study_hours < 3.5 ? 'Increase daily study by 1.5h to gain ~6 marks.' : 'Maintain study consistency with practice exams.'}
2. ${currentInputs.attendance_percentage < 75 ? 'URGENT: Attend remaining lectures for criteria.' : 'Attendance satisfies requirements.'}
3. Continuous evaluation prevents last-minute panic.
============================================================`;

      const blob = new Blob([report], { type: "text/plain;charset=utf-8" });
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = `academic_report_${name.replace(/[^a-zA-Z0-9]/g, "_")}.txt`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    });
  }

  // -------------------------------------------------------------
  // Initialization
  // -------------------------------------------------------------
  updateOverviewKPIs();
  trainAllModels();
  initOverviewCharts();
  initTrainingCharts();
  runPrediction();
  renderDatasetTable();
  bindUIEvents();
});
