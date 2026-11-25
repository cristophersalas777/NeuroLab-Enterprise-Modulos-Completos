/**
 * NetworkCanvas3D.tsx — Visualización profesional de red neuronal
 *
 * TRES FASES con comportamiento distinto (realismo de empresa):
 *
 * 1. IDLE (reposo)
 *    - Pulsos GRISES muy lentos y tenues
 *    - Representa: red lista para inferencia, pesos cargados
 *    - Texto: "Red en reposo"
 *
 * 2. TRAINING (entrenando con TF.js)
 *    - Pulsos AZULES rápidos: forward pass (datos → predicción)
 *    - Pulsos NARANJAS: backpropagation (error → ajuste de pesos)
 *    - Neuronas brillan en AMARILLO al recibir señal
 *    - Métricas en vivo (época, loss, accuracy)
 *    - Texto: "Entrenando..."
 *
 * 3. COMPLETED (entrenamiento terminado)
 *    - Pulsos VERDES fluyendo de entrada a salida (inferencia lista)
 *    - Representa: modelo entrenado haciendo predicciones
 *    - Dura 4 segundos y luego vuelve a idle con fade
 *    - Texto: "Modelo listo · Accuracy X%"
 */

import { useEffect, useRef, useState } from "react";

type LayerKind = "input" | "hidden" | "output";

export type Layer3D = {
  id: number;
  name: string;
  kind: LayerKind;
  neurons: number;
  activation: string;
};

type Props = {
  layers: Layer3D[];
  isTraining: boolean;
  trainingCompleted?: boolean;
  epochProgress: number;
  currentEpoch?: number;
  currentLoss?: number;
  currentAccuracy?: number;
};

type AnimPhase = "idle" | "training" | "completed";

// ─── Colores por capa ─────────────────────────────────────────────────────────
const LAYER_COLOR: Record<LayerKind, { fill: string; stroke: string; glow: string }> = {
  input:  { fill: "#1e3a5f", stroke: "#3b82f6", glow: "#60a5fa" },
  hidden: { fill: "#2e1f4f", stroke: "#8b5cf6", glow: "#a78bfa" },
  output: { fill: "#0f2f24", stroke: "#10b981", glow: "#34d399" },
};

// Colores de pulsos por fase
const PULSE_COLORS: Record<AnimPhase, { forward: string; back: string }> = {
  idle:      { forward: "#2d3f52", back: "#2d3f52" },   // gris muy tenue
  training:  { forward: "#60a5fa", back: "#f97316" },   // azul + naranja
  completed: { forward: "#34d399", back: "#34d399" },   // verde
};

const MAX_NEURONS = 7;

type NeuronPos = { x: number; y: number; li: number; ni: number; kind: LayerKind };

type Pulse = {
  id: number;
  x1: number; y1: number;
  x2: number; y2: number;
  cx: number; cy: number;
  progress: number;
  speed: number;
  isForward: boolean;
  targetKey: string;
  color: string;
};

function buildPositions(layers: Layer3D[], W: number, H: number): NeuronPos[][] {
  const PAD_X   = 80;
  const PAD_TOP = 68;
  const PAD_BOT = 36;
  const usableH = H - PAD_TOP - PAD_BOT;
  const usableW = W - PAD_X * 2;

  return layers.map((layer, li) => {
    const shown   = Math.min(layer.neurons, MAX_NEURONS);
    const x       = layers.length === 1 ? W / 2 : PAD_X + (li / (layers.length - 1)) * usableW;
    const spacing = shown > 1 ? usableH / (shown - 1) : 0;
    const startY  = shown > 1 ? PAD_TOP : H / 2;
    return Array.from({ length: shown }, (_, ni) => ({
      x, y: startY + ni * spacing, li, ni, kind: layer.kind,
    }));
  });
}

export function NetworkCanvas3D({
  layers,
  isTraining,
  trainingCompleted = false,
  epochProgress,
  currentEpoch,
  currentLoss,
  currentAccuracy,
}: Props) {
  const W = 900;
  const H = 380;

  const positions = buildPositions(layers, W, H);

  // Fase actual de animación
  const [phase, setPhase]         = useState<AnimPhase>("idle");
  const [pulses, setPulses]       = useState<Pulse[]>([]);
  const [glows, setGlows]         = useState<Map<string, number>>(new Map());
  // Opacidad global para transición suave entre fases
  const [globalAlpha, setGlobalAlpha] = useState(1);

  const pulsesRef    = useRef<Pulse[]>([]);
  const glowsRef     = useRef<Map<string, number>>(new Map());
  const nextId       = useRef(0);
  const lastSpawn    = useRef(0);
  const rafRef       = useRef<number>(0);
  const phaseRef     = useRef<AnimPhase>("idle");

  pulsesRef.current = pulses;
  glowsRef.current  = glows;
  phaseRef.current  = phase;

  // ── Detectar cambio de fase ───────────────────────────────────────────────
  useEffect(() => {
    if (isTraining) {
      setPhase("training");
      setGlobalAlpha(1);
    } else if (trainingCompleted) {
      // Fase completado: 4 segundos con pulsos verdes, luego fade a idle
      setPhase("completed");
      setGlobalAlpha(1);
      const timer = setTimeout(() => {
        // Fade out suave
        let alpha = 1;
        const fadeInterval = setInterval(() => {
          alpha -= 0.05;
          setGlobalAlpha(Math.max(0, alpha));
          if (alpha <= 0) {
            clearInterval(fadeInterval);
            setPhase("idle");
            setGlobalAlpha(1);
            setPulses([]);
            setGlows(new Map());
          }
        }, 80);
      }, 4000);
      return () => clearTimeout(timer);
    } else {
      setPhase("idle");
      setGlobalAlpha(1);
    }
  }, [isTraining, trainingCompleted]);

  // ── Loop de animación ─────────────────────────────────────────────────────
  useEffect(() => {
    cancelAnimationFrame(rafRef.current);
    pulsesRef.current = [];
    setPulses([]);
    setGlows(new Map());
    lastSpawn.current = 0;

    const currentPhase = phase;
    let lastTime = performance.now();

    // Al entrar en "training", lanzar primera oleada forward completa
    if (currentPhase === "training" && layers.length >= 2) {
      for (let li = 0; li < layers.length - 1; li++) {
        setTimeout(() => {
          if (phaseRef.current !== "training") return;
          const from = positions[li];
          const to   = positions[li + 1];
          if (!from?.length || !to?.length) return;
          const count = Math.min(from.length, 3);
          const newPulses: Pulse[] = [];
          for (let i = 0; i < count; i++) {
            const f = from[Math.floor(Math.random() * from.length)];
            const t = to[Math.floor(Math.random() * to.length)];
            newPulses.push({
              id: nextId.current++,
              x1: f.x, y1: f.y, x2: t.x, y2: t.y,
              cx: f.x, cy: f.y, progress: 0,
              speed: 0.020 + Math.random() * 0.008,
              isForward: true,
              targetKey: `${li + 1}-${t.ni}`,
              color: PULSE_COLORS.training.forward,
            });
          }
          pulsesRef.current = [...pulsesRef.current, ...newPulses];
          setPulses([...pulsesRef.current]);
        }, li * 180);
      }
    }

    // Al entrar en "completed", lanzar oleada verde de entrada a salida
    if (currentPhase === "completed" && layers.length >= 2) {
      for (let li = 0; li < layers.length - 1; li++) {
        setTimeout(() => {
          if (phaseRef.current !== "completed") return;
          const from = positions[li];
          const to   = positions[li + 1];
          if (!from?.length || !to?.length) return;
          const newPulses: Pulse[] = from.map(f => {
            const t = to[Math.floor(Math.random() * to.length)];
            return {
              id: nextId.current++,
              x1: f.x, y1: f.y, x2: t.x, y2: t.y,
              cx: f.x, cy: f.y, progress: 0,
              speed: 0.014,
              isForward: true,
              targetKey: `${li + 1}-${t.ni}`,
              color: PULSE_COLORS.completed.forward,
            };
          });
          pulsesRef.current = [...pulsesRef.current, ...newPulses];
          setPulses([...pulsesRef.current]);
        }, li * 300);
      }
    }

    const spawnPulse = () => {
      if (layers.length < 2) return;
      const colors = PULSE_COLORS[currentPhase];

      // Idle: solo forward lento y tenue
      // Training: forward 65%, backprop 35%
      // Completed: solo forward verde
      const isForward = currentPhase === "idle"      ? true
                      : currentPhase === "completed"  ? true
                      : Math.random() < 0.65;

      const maxLi = layers.length - 1;
      const fromLi = isForward
        ? Math.floor(Math.random() * maxLi)
        : Math.floor(Math.random() * maxLi) + 1;
      const toLi = isForward ? fromLi + 1 : fromLi - 1;

      const fromNodes = positions[fromLi];
      const toNodes   = positions[toLi];
      if (!fromNodes?.length || !toNodes?.length) return;

      const f = fromNodes[Math.floor(Math.random() * fromNodes.length)];
      const t = toNodes[Math.floor(Math.random() * toNodes.length)];

      const spawnInterval = currentPhase === "idle"      ? 700
                          : currentPhase === "training"  ? 180
                          : 250;

      pulsesRef.current = [...pulsesRef.current, {
        id: nextId.current++,
        x1: f.x, y1: f.y, x2: t.x, y2: t.y,
        cx: f.x, cy: f.y, progress: 0,
        speed: currentPhase === "idle"
          ? 0.006 + Math.random() * 0.004          // muy lento en idle
          : currentPhase === "completed"
          ? 0.012 + Math.random() * 0.006           // medio en completed
          : 0.016 + Math.random() * 0.012,          // rápido en training
        isForward,
        targetKey: `${toLi}-${t.ni}`,
        color: isForward ? colors.forward : colors.back,
      }];
      setPulses([...pulsesRef.current]);

      return spawnInterval;
    };

    const tick = (now: number) => {
      const dt = Math.min(now - lastTime, 50);
      lastTime = now;

      const spawnInterval = currentPhase === "idle"      ? 700
                          : currentPhase === "training"  ? 180
                          : 260;

      if (now - lastSpawn.current > spawnInterval) {
        lastSpawn.current = now;
        spawnPulse();
      }

      // Avanzar pulsos
      const arrived: string[] = [];
      const next = pulsesRef.current
        .map(p => {
          const prog = p.progress + p.speed * (dt / 16);
          return {
            ...p,
            progress: prog,
            cx: p.x1 + (p.x2 - p.x1) * Math.min(prog, 1),
            cy: p.y1 + (p.y2 - p.y1) * Math.min(prog, 1),
          };
        })
        .filter(p => {
          if (p.progress >= 1) { arrived.push(p.targetKey); return false; }
          return true;
        });

      pulsesRef.current = next;
      setPulses([...next]);

      // Activar brillo en neuronas
      if (arrived.length > 0 && currentPhase !== "idle") {
        const updated = new Map(glowsRef.current);
        for (const key of arrived) updated.set(key, 1.0);
        glowsRef.current = updated;
      }

      // Desvanecer brillos
      if (glowsRef.current.size > 0) {
        const updatedGlows = new Map(glowsRef.current);
        let changed = false;
        for (const [k, v] of updatedGlows) {
          const nv = v - 0.025 * (dt / 16);
          if (nv <= 0.01) { updatedGlows.delete(k); changed = true; }
          else { updatedGlows.set(k, nv); changed = true; }
        }
        if (changed) {
          glowsRef.current = updatedGlows;
          setGlows(new Map(updatedGlows));
        }
      }

      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase]);

  // ─── Colores activos según fase ───────────────────────────────────────────
  const glowColor = phase === "completed" ? "#34d399"
                  : phase === "training"  ? "#facc15"
                  : "#334155";

  const R = 18;

  // ─── Texto de estado ──────────────────────────────────────────────────────
  const statusText = phase === "training"  ? "Entrenando con TensorFlow.js..."
                   : phase === "completed" ? `Modelo listo · Accuracy ${(currentAccuracy ?? 0).toFixed(1)}%`
                   : "Red en reposo — lista para inferencia";

  const statusColor = phase === "training"  ? "#60a5fa"
                    : phase === "completed" ? "#34d399"
                    : "#334155";

  return (
    <div style={{
      background: "linear-gradient(160deg, #0d1117 0%, #0f172a 60%, #0a0f1e 100%)",
      borderRadius: "12px",
      position: "relative",
      overflow: "hidden",
    }}>

      {/* Barra de progreso — solo durante entrenamiento */}
      <div style={{
        position: "absolute", top: 0, left: 0, right: 0,
        height: "3px", background: "rgba(255,255,255,0.05)", zIndex: 10,
      }}>
        {phase === "training" && (
          <div style={{
            height: "100%", width: `${epochProgress}%`,
            background: "linear-gradient(90deg, #3b82f6, #8b5cf6, #10b981)",
            transition: "width 0.4s ease",
            boxShadow: "0 0 12px rgba(139,92,246,0.6)",
          }} />
        )}
        {phase === "completed" && (
          <div style={{
            height: "100%", width: "100%",
            background: "linear-gradient(90deg, #10b981, #34d399)",
            boxShadow: "0 0 12px rgba(16,185,129,0.8)",
          }} />
        )}
      </div>

      {/* Métricas en vivo — solo durante entrenamiento */}
      {phase === "training" && currentEpoch !== undefined && (
        <div style={{
          position: "absolute", top: "10px", left: "50%",
          transform: "translateX(-50%)",
          display: "flex", gap: "20px", zIndex: 10,
          background: "rgba(15,23,42,0.9)",
          padding: "5px 16px", borderRadius: "20px",
          border: "1px solid rgba(139,92,246,0.35)",
        }}>
          <span style={{ color: "#94a3b8", fontSize: "11px", fontWeight: 600 }}>
            ÉPOCA <span style={{ color: "#a78bfa" }}>{currentEpoch}</span>
          </span>
          <span style={{ color: "#94a3b8", fontSize: "11px", fontWeight: 600 }}>
            LOSS <span style={{ color: "#f97316" }}>{(currentLoss ?? 0).toFixed(4)}</span>
          </span>
          <span style={{ color: "#94a3b8", fontSize: "11px", fontWeight: 600 }}>
            ACC <span style={{ color: "#34d399" }}>{(currentAccuracy ?? 0).toFixed(1)}%</span>
          </span>
        </div>
      )}

      {/* Badge de modelo listo — fase completed */}
      {phase === "completed" && (
        <div style={{
          position: "absolute", top: "10px", left: "50%",
          transform: "translateX(-50%)",
          display: "flex", alignItems: "center", gap: "8px", zIndex: 10,
          background: "rgba(16,185,129,0.15)",
          padding: "5px 16px", borderRadius: "20px",
          border: "1px solid rgba(52,211,153,0.5)",
        }}>
          <span style={{
            width: 8, height: 8, borderRadius: "50%",
            background: "#34d399", display: "inline-block",
            boxShadow: "0 0 8px #34d399",
            animation: "pulse-green 1s ease-in-out infinite",
          }} />
          <span style={{ color: "#34d399", fontSize: "11px", fontWeight: 700, letterSpacing: "0.06em" }}>
            MODELO LISTO · ACC {(currentAccuracy ?? 0).toFixed(1)}%
          </span>
        </div>
      )}

      {/* Leyenda — solo cuando entrena */}
      {phase === "training" && (
        <div style={{
          position: "absolute", bottom: "10px", right: "14px",
          display: "flex", gap: "14px", zIndex: 10,
        }}>
          {[
            { color: "#60a5fa", label: "Forward pass" },
            { color: "#f97316", label: "Backprop" },
            { color: "#facc15", label: "Activación" },
          ].map(item => (
            <span key={item.label} style={{
              display: "flex", alignItems: "center", gap: "5px",
              fontSize: "10px", fontWeight: 600, color: item.color, opacity: 0.9,
            }}>
              <span style={{
                width: 7, height: 7, borderRadius: "50%",
                background: item.color, display: "inline-block",
                boxShadow: `0 0 6px ${item.color}`,
              }} />
              {item.label}
            </span>
          ))}
        </div>
      )}

      {/* Texto de estado inferior */}
      <div style={{
        position: "absolute", bottom: "10px", left: "50%",
        transform: "translateX(-50%)",
        color: statusColor, fontSize: "11px", fontWeight: 600,
        zIndex: 10, whiteSpace: "nowrap",
        transition: "color 0.6s ease",
        letterSpacing: "0.04em",
      }}>
        {statusText}
      </div>

      {/* SVG principal */}
      <svg
        viewBox={`0 0 ${W} ${H}`}
        style={{ width: "100%", display: "block", opacity: globalAlpha, transition: "opacity 0.3s ease" }}
        aria-label="Red neuronal — visualización del flujo de datos"
      >
        <defs>
          <filter id="glow-strong" x="-60%" y="-60%" width="220%" height="220%">
            <feGaussianBlur stdDeviation="6" result="b1" />
            <feGaussianBlur stdDeviation="2" result="b2" />
            <feMerge><feMergeNode in="b1" /><feMergeNode in="b2" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <filter id="glow-soft" x="-40%" y="-40%" width="180%" height="180%">
            <feGaussianBlur stdDeviation="3" result="b" />
            <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <filter id="glow-pulse" x="-80%" y="-80%" width="260%" height="260%">
            <feGaussianBlur stdDeviation="4" result="b" />
            <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          {["input","hidden","output"].map(k => (
            <radialGradient key={k} id={`grad-${k}`} cx="35%" cy="35%" r="65%">
              <stop offset="0%" stopColor={LAYER_COLOR[k as LayerKind].glow} stopOpacity="0.6" />
              <stop offset="100%" stopColor={LAYER_COLOR[k as LayerKind].fill} stopOpacity="1" />
            </radialGradient>
          ))}
        </defs>

        {/* Conexiones */}
        {positions.slice(0, -1).map((fromLayer, li) =>
          fromLayer.flatMap(from =>
            positions[li + 1].map(to => (
              <line
                key={`c-${li}-${from.ni}-${to.ni}`}
                x1={from.x} y1={from.y} x2={to.x} y2={to.y}
                stroke={phase === "completed" ? "#1a3a2a" : "#1e293b"}
                strokeWidth="1"
                opacity={phase === "training" ? 0.4 : 0.6}
              />
            ))
          )
        )}

        {/* Pulsos */}
        {pulses.map(p => {
          const trailX = p.x1 + (p.x2 - p.x1) * Math.max(0, p.progress - 0.22);
          const trailY = p.y1 + (p.y2 - p.y1) * Math.max(0, p.progress - 0.22);
          const size   = 0.5 + Math.sin(p.progress * Math.PI) * 0.6;
          const isIdle = phase === "idle";

          return (
            <g key={p.id} opacity={isIdle ? 0.4 : 1}>
              <line
                x1={trailX} y1={trailY} x2={p.cx} y2={p.cy}
                stroke={p.color} strokeWidth={isIdle ? 1 : 1.8 * size}
                strokeLinecap="round" opacity={isIdle ? 0.3 : 0.5 * size}
              />
              <circle
                cx={p.cx} cy={p.cy} r={isIdle ? 2.5 : 4 * size}
                fill={p.color}
                filter={isIdle ? undefined : "url(#glow-pulse)"}
                opacity={isIdle ? 0.5 : 0.95}
              />
              {!isIdle && (
                <circle cx={p.cx} cy={p.cy} r={1.8 * size} fill="white" opacity={0.7} />
              )}
            </g>
          );
        })}

        {/* Neuronas */}
        {positions.map((layerNodes, li) => {
          const layer = layers[li];
          const col   = LAYER_COLOR[layer.kind];

          return (
            <g key={layer.id}>
              <text x={layerNodes[0]?.x ?? 0} y={22}
                textAnchor="middle" fill={col.glow}
                fontSize="12" fontWeight="700" fontFamily="system-ui, sans-serif" letterSpacing="1">
                {layer.name.toUpperCase()}
              </text>
              <text x={layerNodes[0]?.x ?? 0} y={38}
                textAnchor="middle" fill="#475569"
                fontSize="10" fontFamily="system-ui, sans-serif">
                {layer.activation !== "Ninguna" ? layer.activation : "lineal"}
              </text>
              {layer.neurons > MAX_NEURONS && (
                <text x={layerNodes[0]?.x ?? 0} y={H - 12}
                  textAnchor="middle" fill="#334155" fontSize="10" fontFamily="system-ui, sans-serif">
                  +{layer.neurons - MAX_NEURONS} más
                </text>
              )}

              {layerNodes.map(node => {
                const key    = `${li}-${node.ni}`;
                const alpha  = glows.get(key) ?? 0;
                const active = alpha > 0.05;
                // En completed, todas las neuronas de salida brillan verde
                const isOutput = layer.kind === "output" && phase === "completed";

                return (
                  <g key={key}>
                    {(active || isOutput) && (
                      <circle cx={node.x} cy={node.y} r={R + 10}
                        fill="none"
                        stroke={isOutput ? "#34d399" : glowColor}
                        strokeWidth="1.5"
                        opacity={isOutput ? 0.5 : alpha * 0.6}
                        filter="url(#glow-strong)"
                      />
                    )}
                    <circle cx={node.x} cy={node.y} r={R + 4}
                      fill={col.stroke}
                      opacity={active || isOutput ? 0.22 : phase === "training" ? 0.08 : 0.05}
                      filter="url(#glow-soft)"
                    />
                    <circle cx={node.x} cy={node.y} r={R}
                      fill={`url(#grad-${layer.kind})`}
                      stroke={isOutput ? "#34d399" : active ? glowColor : col.stroke}
                      strokeWidth={active || isOutput ? 2 : 1.5}
                      filter={(active || isOutput) ? "url(#glow-soft)" : undefined}
                    />
                    <circle cx={node.x - 5} cy={node.y - 5} r={4}
                      fill="white" opacity={active || isOutput ? 0.5 : 0.15}
                    />
                    <text x={node.x} y={node.y + 4}
                      textAnchor="middle"
                      fill={isOutput ? "#34d399" : active ? glowColor : col.glow}
                      fontSize="11" fontWeight="600" fontFamily="system-ui, sans-serif">
                      {node.ni + 1}
                    </text>
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>

      <style>{`
        @keyframes pulse-green {
          0%, 100% { box-shadow: 0 0 8px #34d399; }
          50%       { box-shadow: 0 0 18px #34d399; }
        }
      `}</style>
    </div>
  );
}
