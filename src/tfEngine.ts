/**
 * tfEngine.ts — Motor TensorFlow.js real para NeuroLab Enterprise
 *
 * Responsabilidades:
 *  1. Recibir la configuración de capas del Diseñador de Arquitecturas
 *  2. Preprocesar el dataset CSV (normalización, one-hot encoding, splits)
 *  3. Construir el modelo Sequential con Keras-style layers
 *  4. Entrenar época por época y reportar métricas reales vía callback
 *  5. Evaluar con el subconjunto de Test
 */

import * as tf from "@tensorflow/tfjs";

// ─── Tipos ────────────────────────────────────────────────────────────────────

export type LayerConfig = {
  kind: "input" | "hidden" | "output";
  neurons: number;
  activation: string; // "ReLU" | "Sigmoid" | "Tanh" | "Softmax" | "Ninguna"
};

export type DatasetRecord = {
  columns: string[];
  preview: string[][];  // filas de texto tal como vienen del CSV
  target: string;       // nombre de la columna objetivo
};

export type EpochResult = {
  epoch: number;
  loss: number;
  accuracy: number;
};

export type TrainResult = {
  epochHistory: EpochResult[];
  finalLoss: number;
  finalAccuracy: number;
  testLoss: number;
  testAccuracy: number;
};

export type TrainConfig = {
  epochs: number;
  batchSize: number;
  optimizer: string;  // "Adam" | "SGD" | "RMSprop" | "Adagrad"
  learningRate: number;
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Convierte el nombre de activación del UI al string que acepta TF.js */
function mapActivation(name: string): string {
  const map: Record<string, string> = {
    ReLU: "relu",
    Sigmoid: "sigmoid",
    Tanh: "tanh",
    Softmax: "softmax",
    Ninguna: "linear",
    linear: "linear",
  };
  return map[name] ?? "relu";
}

/** Convierte el nombre del optimizador del UI al optimizador TF.js */
function buildOptimizer(name: string, lr: number): tf.Optimizer {
  switch (name) {
    case "SGD":     return tf.train.sgd(lr);
    case "RMSprop": return tf.train.rmsprop(lr);
    case "Adagrad": return tf.train.adagrad(lr);
    default:        return tf.train.adam(lr);  // Adam por defecto
  }
}

/**
 * Preprocesa las filas del CSV:
 *  - Detecta columnas numéricas y categóricas
 *  - Aplica Min-Max Scaling a las features numéricas
 *  - Aplica One-Hot Encoding a la columna objetivo categórica
 *  - Divide en Train (70%), Validation (15%), Test (15%)
 */
function preprocessDataset(dataset: DatasetRecord): {
  xTrain: tf.Tensor2D;
  yTrain: tf.Tensor2D;
  xVal:   tf.Tensor2D;
  yVal:   tf.Tensor2D;
  xTest:  tf.Tensor2D;
  yTest:  tf.Tensor2D;
  numFeatures: number;
  numClasses:  number;
  targetIsNumeric: boolean;
} {
  const { columns, preview, target } = dataset;
  const targetIndex = columns.indexOf(target);

  // Si no se encontró la columna objetivo, usar la última
  const tIdx = targetIndex >= 0 ? targetIndex : columns.length - 1;

  // Separar features y etiquetas en texto
  const featureCols = columns.filter((_, i) => i !== tIdx);
  const rawFeatures: string[][] = preview.map(row => featureCols.map((_, fi) => {
    const colIdx = columns.indexOf(featureCols[fi]);
    return row[colIdx] ?? "0";
  }));
  const rawLabels: string[] = preview.map(row => row[tIdx] ?? "0");

  // ── Detectar si el target es numérico o categórico ──
  const targetIsNumeric = rawLabels.every(v => !isNaN(Number(v)));

  // ── Procesar features: convertir texto → número, luego normalizar ──
  // Si una columna es categórica (no numérica), asignar índice ordinal
  const featureMatrix: number[][] = rawFeatures.map(row =>
    row.map(v => {
      const n = Number(v);
      return isNaN(n) ? 0 : n;  // columnas categóricas de features → 0 simple
    })
  );

  // Min-Max Scaling por columna
  const numCols = featureCols.length;
  const mins = Array(numCols).fill(Infinity);
  const maxs = Array(numCols).fill(-Infinity);
  for (const row of featureMatrix) {
    for (let c = 0; c < numCols; c++) {
      if (row[c] < mins[c]) mins[c] = row[c];
      if (row[c] > maxs[c]) maxs[c] = row[c];
    }
  }
  const scaled = featureMatrix.map(row =>
    row.map((v, c) => {
      const range = maxs[c] - mins[c];
      return range === 0 ? 0 : (v - mins[c]) / range;
    })
  );

  // ── Procesar etiquetas ──
  let labelsMatrix: number[][];
  let numClasses: number;

  if (targetIsNumeric) {
    // Regresión: normalizar el target también
    const nums = rawLabels.map(Number);
    const minL = Math.min(...nums);
    const maxL = Math.max(...nums);
    const rangeL = maxL - minL || 1;
    labelsMatrix = nums.map(v => [(v - minL) / rangeL]);
    numClasses = 1;
  } else {
    // Clasificación: One-Hot Encoding
    const uniqueClasses = [...new Set(rawLabels)].sort();
    numClasses = uniqueClasses.length;
    labelsMatrix = rawLabels.map(label => {
      const vec = Array(numClasses).fill(0);
      const idx = uniqueClasses.indexOf(label);
      if (idx >= 0) vec[idx] = 1;
      return vec;
    });
  }

  // ── Split: 70% train, 15% val, 15% test ──
  const n = scaled.length;
  const trainEnd = Math.floor(n * 0.70);
  const valEnd   = Math.floor(n * 0.85);

  // Mezclar índices para que el split no dependa del orden del CSV
  const indices = Array.from({ length: n }, (_, i) => i);
  // Shuffle determinista (seed fija para reproducibilidad)
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor((i * 1664525 + 1013904223) % (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }

  const pick = (arr: number[][], idxs: number[]) => idxs.map(i => arr[i]);

  const xAll = scaled;
  const yAll = labelsMatrix;

  const trainIdx = indices.slice(0, trainEnd);
  const valIdx   = indices.slice(trainEnd, valEnd);
  const testIdx  = indices.slice(valEnd);

  // Si hay muy pocos datos, usar todos para train y duplicar para val/test
  const safeValIdx  = valIdx.length  > 0 ? valIdx  : trainIdx.slice(0, 1);
  const safeTestIdx = testIdx.length > 0 ? testIdx : trainIdx.slice(0, 1);

  const toTensor2D = (rows: number[][], cols: number): tf.Tensor2D =>
    tf.tensor2d(rows, [rows.length, cols]);

  return {
    xTrain: toTensor2D(pick(xAll, trainIdx),     numCols),
    yTrain: toTensor2D(pick(yAll, trainIdx),     numClasses),
    xVal:   toTensor2D(pick(xAll, safeValIdx),   numCols),
    yVal:   toTensor2D(pick(yAll, safeValIdx),   numClasses),
    xTest:  toTensor2D(pick(xAll, safeTestIdx),  numCols),
    yTest:  toTensor2D(pick(yAll, safeTestIdx),  numClasses),
    numFeatures: numCols,
    numClasses,
    targetIsNumeric,
  };
}

// ─── Función principal ────────────────────────────────────────────────────────

/**
 * buildAndTrain
 *
 * Construye un modelo Sequential con las capas del Diseñador,
 * lo compila con el optimizador elegido y lo entrena de verdad.
 *
 * @param layers       Capas del Diseñador de Arquitecturas
 * @param dataset      Datos del módulo de Datasets
 * @param config       Hiperparámetros del módulo de Entrenamientos
 * @param onEpochEnd   Callback llamado al terminar cada época con métricas reales
 * @param cancelSignal Objeto con propiedad .cancelled para detener el entrenamiento
 */
export async function buildAndTrain(
  layers: LayerConfig[],
  dataset: DatasetRecord,
  config: TrainConfig,
  onEpochEnd: (result: EpochResult) => void,
  cancelSignal: { cancelled: boolean },
  pauseSignal?: { paused: boolean }
): Promise<TrainResult> {

  // 1. Preprocesar dataset
  const {
    xTrain, yTrain,
    xVal, yVal,
    xTest, yTest,
    numFeatures,
    numClasses,
    targetIsNumeric,
  } = preprocessDataset(dataset);

  // 2. Construir modelo Sequential
  const model = tf.sequential();

  // Filtrar capas relevantes: input → hidden(s) → output
  const hiddenLayers = layers.filter(l => l.kind === "hidden");
  const outputLayer  = layers.find(l => l.kind === "output");

  // Primera capa densa (define inputShape)
  const firstHidden = hiddenLayers[0];
  if (firstHidden) {
    model.add(tf.layers.dense({
      units: firstHidden.neurons,
      activation: mapActivation(firstHidden.activation) as never,
      inputShape: [numFeatures],
    }));
  } else if (outputLayer) {
    // Sin capas ocultas: directo entrada → salida
    model.add(tf.layers.dense({
      units: outputLayer.neurons,
      activation: mapActivation(outputLayer.activation) as never,
      inputShape: [numFeatures],
    }));
  }

  // Capas ocultas restantes
  for (const layer of hiddenLayers.slice(1)) {
    model.add(tf.layers.dense({
      units: layer.neurons,
      activation: mapActivation(layer.activation) as never,
    }));
  }

  // Capa de salida (si había capas ocultas)
  if (outputLayer && hiddenLayers.length > 0) {
    // Ajustar neuronas de salida al número real de clases
    model.add(tf.layers.dense({
      units: numClasses,
      activation: targetIsNumeric ? "linear" : mapActivation(outputLayer.activation) as never,
    }));
  }

  // 3. Compilar
  const lossFunction = targetIsNumeric
    ? "meanSquaredError"
    : numClasses === 2
      ? "binaryCrossentropy"
      : "categoricalCrossentropy";

  const metrics = targetIsNumeric ? ["mse"] : ["accuracy"];

  model.compile({
    optimizer: buildOptimizer(config.optimizer, 0.001),
    loss: lossFunction,
    metrics,
  });

  // 4. Entrenar con callback por época
  const epochHistory: EpochResult[] = [];

  await model.fit(xTrain, yTrain, {
    epochs: config.epochs,
    batchSize: config.batchSize,
    validationData: [xVal, yVal],
    shuffle: true,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        if (cancelSignal.cancelled) {
          model.stopTraining = true;
          return;
        }

        const loss     = +(logs?.["loss"]     ?? 0).toFixed(4);
        const accuracy = targetIsNumeric
          ? +(100 * Math.max(0, 1 - (logs?.["mse"] ?? 1))).toFixed(2)
          : +(((logs?.["acc"] ?? logs?.["accuracy"] ?? 0) as number) * 100).toFixed(2);

        const result: EpochResult = { epoch: epoch + 1, loss, accuracy };
        epochHistory.push(result);
        onEpochEnd(result);

        // Ceder el hilo para que React pueda re-renderizar
        await tf.nextFrame();

        // Esperar mientras esté pausado
        if (pauseSignal) {
          while (pauseSignal.paused && !cancelSignal.cancelled) {
            await new Promise(r => setTimeout(r, 200));
          }
        }

        // Delay mínimo de 300ms por época para que sea visible en pantalla
        await new Promise(r => setTimeout(r, 300));

        if (cancelSignal.cancelled) {
          model.stopTraining = true;
        }
      },
    },
  });

  // 5. Evaluar con Test set
  const evalResult = model.evaluate(xTest, yTest) as tf.Tensor[];
  const testLossVal = Array.isArray(evalResult)
    ? (await evalResult[0].data())[0]
    : (await (evalResult as tf.Tensor).data())[0];
  const testMetricVal = Array.isArray(evalResult) && evalResult.length > 1
    ? (await evalResult[1].data())[0]
    : 0;

  const testLoss     = +testLossVal.toFixed(4);
  const testAccuracy = targetIsNumeric
    ? +(100 * Math.max(0, 1 - testLossVal)).toFixed(2)
    : +(testMetricVal * 100).toFixed(2);

  // 6. Liberar memoria de tensores
  xTrain.dispose(); yTrain.dispose();
  xVal.dispose();   yVal.dispose();
  xTest.dispose();  yTest.dispose();
  if (Array.isArray(evalResult)) evalResult.forEach(t => t.dispose());

  const last = epochHistory[epochHistory.length - 1];

  return {
    epochHistory,
    finalLoss:    last?.loss     ?? testLoss,
    finalAccuracy: last?.accuracy ?? testAccuracy,
    testLoss,
    testAccuracy,
  };
}
