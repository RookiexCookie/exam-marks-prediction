// Student Dataset Generator and Default Synthetic Cohort
const DataManager = (() => {
  const FEATURE_METADATA = {
    study_hours: { label: "Daily Study Hours", min: 0.5, max: 14.0, step: 0.5, default: 4.5, unit: "hrs/day" },
    attendance_percentage: { label: "Attendance Rate", min: 40.0, max: 100.0, step: 1.0, default: 82.0, unit: "%" },
    assignment_score: { label: "Assignment Score", min: 20.0, max: 100.0, step: 1.0, default: 78.0, unit: "/100" },
    previous_exam_marks: { label: "Previous Exam Marks", min: 20.0, max: 100.0, step: 1.0, default: 72.0, unit: "/100" },
    internal_marks: { label: "Internal Assessment", min: 10.0, max: 50.0, step: 1.0, default: 38.0, unit: "/50" },
    study_days: { label: "Preparation Study Days", min: 5, max: 90, step: 1, default: 35, unit: "days" }
  };

  const FEATURE_KEYS = [
    "study_hours",
    "attendance_percentage",
    "assignment_score",
    "previous_exam_marks",
    "internal_marks",
    "study_days"
  ];
  const TARGET_KEY = "exam_marks";

  // Pseudo-random Gaussian generator (Box-Muller)
  function randomGaussian(mean = 0, stdev = 1, rng = Math.random) {
    const u = 1 - rng();
    const v = rng();
    const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    return z * stdev + mean;
  }

  // Seeded random number generator (xorshift32)
  function createSeededRNG(seed = 42) {
    let s = seed ? seed : 42;
    return function() {
      s ^= s << 13;
      s ^= s >>> 17;
      s ^= s << 5;
      return ((s >>> 0) / 4294967296);
    };
  }

  function generateCohort(nSamples = 600, seed = 42) {
    const rng = createSeededRNG(seed);
    const data = [];

    for (let i = 0; i < nSamples; i++) {
      const aptitude = randomGaussian(0, 1.0, rng);

      const study_hours = Math.max(0.5, Math.min(14.0, Number((4.5 + 0.8 * aptitude + randomGaussian(0, 1.8, rng)).toFixed(1))));
      const attendance = Math.max(45.0, Math.min(100.0, Number((80.0 + 5.0 * aptitude + randomGaussian(0, 9.0, rng)).toFixed(1))));
      const assignment = Math.max(25.0, Math.min(100.0, Number((74.0 + 8.0 * aptitude + 1.2 * study_hours + randomGaussian(0, 8.0, rng)).toFixed(1))));
      const prev_exam = Math.max(25.0, Math.min(100.0, Number((68.0 + 11.0 * aptitude + randomGaussian(0, 10.0, rng)).toFixed(1))));
      const internal = Math.max(10.0, Math.min(50.0, Number((35.0 + 4.5 * aptitude + 0.08 * attendance + 0.05 * assignment + randomGaussian(0, 4.5, rng)).toFixed(1))));
      const study_days = Math.max(5, Math.min(80, Math.round(35 + 5 * aptitude + randomGaussian(0, 13, rng))));

      // Realistic non-linear model with diminishing returns on study hours > 8
      const effective_study = study_hours > 8.0 ? 8.0 + (study_hours - 8.0) * 0.45 : study_hours;
      const internal_norm = (internal / 50.0) * 100.0;

      const raw_mark = (
        0.28 * prev_exam +
        0.24 * internal_norm +
        0.18 * assignment +
        1.75 * effective_study +
        0.12 * attendance +
        0.08 * study_days +
        randomGaussian(0, 3.0, rng)
      );

      const exam_marks = Math.max(15.0, Math.min(100.0, Number(raw_mark.toFixed(1))));

      data.push({
        student_id: `STU${1000 + i}`,
        study_hours,
        attendance_percentage: attendance,
        assignment_score: assignment,
        previous_exam_marks: prev_exam,
        internal_marks: internal,
        study_days,
        exam_marks
      });
    }

    return data;
  }

  function calculateSummaryStats(dataset) {
    if (!dataset || dataset.length === 0) return null;
    const n = dataset.length;

    const stats = {};
    const cols = [...FEATURE_KEYS, TARGET_KEY];

    cols.forEach(col => {
      const vals = dataset.map(d => d[col]).sort((a, b) => a - b);
      const sum = vals.reduce((acc, v) => acc + v, 0);
      const mean = sum / n;
      const variance = vals.reduce((acc, v) => acc + Math.pow(v - mean, 2), 0) / n;
      const std = Math.sqrt(variance);

      stats[col] = {
        mean: Number(mean.toFixed(2)),
        std: Number(std.toFixed(2)),
        min: Number(vals[0].toFixed(2)),
        p25: Number(vals[Math.floor(n * 0.25)].toFixed(2)),
        median: Number(vals[Math.floor(n * 0.50)].toFixed(2)),
        p75: Number(vals[Math.floor(n * 0.75)].toFixed(2)),
        max: Number(vals[n - 1].toFixed(2))
      };
    });

    const passCount = dataset.filter(d => d.exam_marks >= 40).length;
    stats.pass_rate = Number(((passCount / n) * 100).toFixed(1));
    stats.count = n;

    return stats;
  }

  return {
    FEATURE_METADATA,
    FEATURE_KEYS,
    TARGET_KEY,
    generateCohort,
    calculateSummaryStats
  };
})();
