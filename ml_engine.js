// Machine Learning Regression Engine for Exam Marks Prediction
const MLEngine = (() => {

  // Matrix math helpers
  function transpose(A) {
    const rows = A.length, cols = A[0].length;
    const AT = Array.from({ length: cols }, () => new Float64Array(rows));
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        AT[c][r] = A[r][c];
      }
    }
    return AT;
  }

  function matMul(A, B) {
    const rowsA = A.length, colsA = A[0].length, colsB = B[0].length;
    const C = Array.from({ length: rowsA }, () => new Float64Array(colsB));
    for (let i = 0; i < rowsA; i++) {
      for (let k = 0; k < colsA; k++) {
        const aik = A[i][k];
        for (let j = 0; j < colsB; j++) {
          C[i][j] += aik * B[k][j];
        }
      }
    }
    return C;
  }

  function matVecMul(A, v) {
    const rows = A.length, cols = A[0].length;
    const res = new Float64Array(rows);
    for (let r = 0; r < rows; r++) {
      let sum = 0;
      for (let c = 0; c < cols; c++) {
        sum += A[r][c] * v[c];
      }
      res[r] = sum;
    }
    return res;
  }

  // Gauss-Jordan matrix inversion with L2 ridge regularization
  function invertSymmetricRegularized(A, lambda = 1e-4) {
    const n = A.length;
    const M = Array.from({ length: n }, (row, i) => {
      const arr = new Float64Array(2 * n);
      for (let j = 0; j < n; j++) {
        arr[j] = A[i][j] + (i === j ? lambda : 0);
      }
      arr[n + i] = 1.0;
      return arr;
    });

    for (let i = 0; i < n; i++) {
      let pivot = i;
      let maxVal = Math.abs(M[i][i]);
      for (let r = i + 1; r < n; r++) {
        if (Math.abs(M[r][i]) > maxVal) {
          maxVal = Math.abs(M[r][i]);
          pivot = r;
        }
      }

      if (pivot !== i) {
        const temp = M[i];
        M[i] = M[pivot];
        M[pivot] = temp;
      }

      const pVal = M[i][i];
      if (Math.abs(pVal) < 1e-12) continue; // Singular fallback

      const invP = 1.0 / pVal;
      for (let c = 0; c < 2 * n; c++) M[i][c] *= invP;

      for (let r = 0; r < n; r++) {
        if (r !== i) {
          const factor = M[r][i];
          if (factor !== 0) {
            for (let c = 0; c < 2 * n; c++) {
              M[r][c] -= factor * M[i][c];
            }
          }
        }
      }
    }

    const inv = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        inv[i][j] = M[i][n + j];
      }
    }
    return inv;
  }

  // Feature Scaler
  class StandardScaler {
    constructor() {
      this.means = [];
      this.stds = [];
    }

    fit(X) {
      const n = X.length;
      const d = X[0].length;
      this.means = new Float64Array(d);
      this.stds = new Float64Array(d);

      for (let j = 0; j < d; j++) {
        let sum = 0;
        for (let i = 0; i < n; i++) sum += X[i][j];
        this.means[j] = sum / n;

        let varSum = 0;
        for (let i = 0; i < n; i++) varSum += Math.pow(X[i][j] - this.means[j], 2);
        this.stds[j] = Math.sqrt(varSum / n) || 1.0;
      }
    }

    transform(X) {
      const n = X.length, d = X[0].length;
      const scaled = Array.from({ length: n }, () => new Float64Array(d));
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < d; j++) {
          scaled[i][j] = (X[i][j] - this.means[j]) / this.stds[j];
        }
      }
      return scaled;
    }

    transformSample(x) {
      const res = new Float64Array(x.length);
      for (let j = 0; j < x.length; j++) {
        res[j] = (x[j] - this.means[j]) / this.stds[j];
      }
      return res;
    }
  }

  // Linear Regression with Ridge Regularization
  class LinearRegressor {
    constructor(alpha = 0.05) {
      this.alpha = alpha;
      this.weights = null; // includes intercept at index 0
      this.scaler = new StandardScaler();
    }

    fit(X, y) {
      this.scaler.fit(X);
      const X_scaled = this.scaler.transform(X);
      const n = X_scaled.length, d = X_scaled[0].length;

      // Add bias column (1s)
      const X_bias = Array.from({ length: n }, (row, i) => {
        const rowArr = new Float64Array(d + 1);
        rowArr[0] = 1.0;
        for (let j = 0; j < d; j++) rowArr[j + 1] = X_scaled[i][j];
        return rowArr;
      });

      const XT = transpose(X_bias);
      const XTX = matMul(XT, X_bias);
      const XTX_inv = invertSymmetricRegularized(XTX, this.alpha);
      const XTy = matVecMul(XT, y);

      this.weights = matVecMul(XTX_inv, XTy);
    }

    predict(X) {
      const X_scaled = this.scaler.transform(X);
      const preds = new Float64Array(X.length);
      for (let i = 0; i < X.length; i++) {
        let val = this.weights[0];
        for (let j = 0; j < X_scaled[0].length; j++) {
          val += this.weights[j + 1] * X_scaled[i][j];
        }
        preds[i] = Math.max(0, Math.min(100, val));
      }
      return preds;
    }

    predictSample(x) {
      const x_scaled = this.scaler.transformSample(x);
      let val = this.weights[0];
      for (let j = 0; j < x_scaled.length; j++) {
        val += this.weights[j + 1] * x_scaled[j];
      }
      return Math.max(0, Math.min(100, Number(val.toFixed(1))));
    }

    getFeatureImportances(featureNames) {
      // Use absolute standardized coefficients
      const absWeights = [];
      for (let j = 0; j < featureNames.length; j++) {
        absWeights.push(Math.abs(this.weights[j + 1]));
      }
      const sum = absWeights.reduce((a, b) => a + b, 0) || 1.0;
      return featureNames.map((name, i) => ({
        name,
        importance: absWeights[i] / sum,
        pct: Number(((absWeights[i] / sum) * 100).toFixed(1))
      })).sort((a, b) => b.importance - a.importance);
    }
  }

  // Random Forest Regressor
  class SimpleDecisionTree {
    constructor(maxDepth = 5, minSamplesSplit = 5) {
      this.maxDepth = maxDepth;
      this.minSamplesSplit = minSamplesSplit;
      this.root = null;
    }

    fit(X, y) {
      this.root = this._buildTree(X, y, 0);
    }

    _buildTree(X, y, depth) {
      const n = X.length;
      if (n === 0) return { isLeaf: true, val: 50 };

      let meanY = 0;
      for (let i = 0; i < n; i++) meanY += y[i];
      meanY /= n;

      if (depth >= this.maxDepth || n < this.minSamplesSplit) {
        return { isLeaf: true, val: meanY };
      }

      const numFeatures = X[0].length;
      let bestFeature = -1;
      let bestThreshold = 0;
      let bestVarianceReduction = -1;
      let bestLeft = null, bestRight = null;

      // Base variance
      let baseVar = 0;
      for (let i = 0; i < n; i++) baseVar += Math.pow(y[i] - meanY, 2);

      // Subsample features
      const sampleFeatures = [];
      while (sampleFeatures.length < Math.max(2, Math.floor(Math.sqrt(numFeatures)))) {
        const randF = Math.floor(Math.random() * numFeatures);
        if (!sampleFeatures.includes(randF)) sampleFeatures.push(randF);
      }

      for (const f of sampleFeatures) {
        // Sample 5 random cut candidates
        for (let step = 0; step < 5; step++) {
          const randIdx = Math.floor(Math.random() * n);
          const threshold = X[randIdx][f];

          const leftX = [], leftY = [];
          const rightX = [], rightY = [];

          for (let i = 0; i < n; i++) {
            if (X[i][f] <= threshold) {
              leftX.push(X[i]); leftY.push(y[i]);
            } else {
              rightX.push(X[i]); rightY.push(y[i]);
            }
          }

          if (leftX.length < 2 || rightX.length < 2) continue;

          let meanL = leftY.reduce((a, b) => a + b, 0) / leftY.length;
          let varL = leftY.reduce((a, b) => a + Math.pow(b - meanL, 2), 0);

          let meanR = rightY.reduce((a, b) => a + b, 0) / rightY.length;
          let varR = rightY.reduce((a, b) => a + Math.pow(b - meanR, 2), 0);

          const vr = baseVar - (varL + varR);
          if (vr > bestVarianceReduction) {
            bestVarianceReduction = vr;
            bestFeature = f;
            bestThreshold = threshold;
            bestLeft = { X: leftX, y: leftY };
            bestRight = { X: rightX, y: rightY };
          }
        }
      }

      if (bestVarianceReduction <= 0 || !bestLeft) {
        return { isLeaf: true, val: meanY };
      }

      return {
        isLeaf: false,
        feature: bestFeature,
        threshold: bestThreshold,
        left: this._buildTree(bestLeft.X, bestLeft.y, depth + 1),
        right: this._buildTree(bestRight.X, bestRight.y, depth + 1)
      };
    }

    predictSample(x) {
      let node = this.root;
      while (!node.isLeaf) {
        if (x[node.feature] <= node.threshold) {
          node = node.left;
        } else {
          node = node.right;
        }
      }
      return node.val;
    }
  }

  class RandomForestRegressor {
    constructor(nEstimators = 15, maxDepth = 6) {
      this.nEstimators = nEstimators;
      this.maxDepth = maxDepth;
      this.trees = [];
      this.linearAnchor = new LinearRegressor(0.1);
    }

    fit(X, y) {
      this.linearAnchor.fit(X, y);
      const n = X.length;
      this.trees = [];

      for (let t = 0; t < this.nEstimators; t++) {
        // Bootstrap sample
        const bX = [], bY = [];
        for (let i = 0; i < n; i++) {
          const idx = Math.floor(Math.random() * n);
          bX.push(X[idx]);
          bY.push(y[idx]);
        }
        const tree = new SimpleDecisionTree(this.maxDepth, 4);
        tree.fit(bX, bY);
        this.trees.push(tree);
      }
    }

    predict(X) {
      const preds = new Float64Array(X.length);
      for (let i = 0; i < X.length; i++) {
        preds[i] = this.predictSample(X[i]);
      }
      return preds;
    }

    predictSample(x) {
      let sum = 0;
      for (let t = 0; t < this.trees.length; t++) {
        sum += this.trees[t].predictSample(x);
      }
      const rfVal = sum / this.trees.length;
      const linVal = this.linearAnchor.predictSample(x);
      // Ensemble blend
      const finalVal = 0.55 * rfVal + 0.45 * linVal;
      return Math.max(0, Math.min(100, Number(finalVal.toFixed(1))));
    }

    getFeatureImportances(featureNames) {
      return this.linearAnchor.getFeatureImportances(featureNames);
    }
  }

  // Evaluation Metrics
  function evaluate(yTrue, yPred) {
    const n = yTrue.length;
    let sumAbsErr = 0;
    let sumSqErr = 0;
    let sumTrue = 0;

    for (let i = 0; i < n; i++) {
      const err = yTrue[i] - yPred[i];
      sumAbsErr += Math.abs(err);
      sumSqErr += Math.pow(err, 2);
      sumTrue += yTrue[i];
    }

    const meanTrue = sumTrue / n;
    let totalVar = 0;
    for (let i = 0; i < n; i++) {
      totalVar += Math.pow(yTrue[i] - meanTrue, 2);
    }

    const mae = sumAbsErr / n;
    const mse = sumSqErr / n;
    const rmse = Math.sqrt(mse);
    const r2 = 1 - (sumSqErr / (totalVar || 1.0));

    const residuals = Array.from({ length: n }, (_, i) => Number((yTrue[i] - yPred[i]).toFixed(2)));

    return {
      mae: Number(mae.toFixed(2)),
      mse: Number(mse.toFixed(2)),
      rmse: Number(rmse.toFixed(2)),
      r2: Number(r2.toFixed(4)),
      residuals
    };
  }

  function getGradeInfo(mark) {
    if (mark >= 90) return { grade: "O", status: "Outstanding", class: "Distinction", color: "#10B981" };
    if (mark >= 80) return { grade: "A+", status: "Excellent", class: "Distinction", color: "#10B981" };
    if (mark >= 70) return { grade: "A", status: "Very Good", class: "First Class", color: "#3B82F6" };
    if (mark >= 60) return { grade: "B+", status: "Good", class: "First Class", color: "#6366F1" };
    if (mark >= 50) return { grade: "B", status: "Above Average", class: "Second Class", color: "#F59E0B" };
    if (mark >= 40) return { grade: "C", status: "Pass", class: "Pass Division", color: "#EAB308" };
    return { grade: "F", status: "At-Risk", class: "Needs Remedial Action", color: "#EF4444" };
  }

  return {
    LinearRegressor,
    RandomForestRegressor,
    evaluate,
    getGradeInfo
  };
})();
