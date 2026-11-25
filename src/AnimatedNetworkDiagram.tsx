/**
 * AnimatedNetworkDiagram.tsx
 *
 * Diagrama SVG animado de red neuronal que muestra:
 *  - Forward pass: pulsos AZULES fluyendo izquierda → derecha (propagación de datos)
 *  - Backpropagation: pulsos NARANJAS fluyendo derecha → izquierda (ajuste de pesos)
 *  - Neuronas que "pulsan" cuando reciben una señal
 *  - Estado en reposo cuando no hay entrenamiento activo
 *
 * La animación solo se activa cuando `isTraining === true`,
 * lo que la sincroniza con el entrenamiento real de TF.js.
 */

import { useEffect, useRef, useState } from "react";

type LayerKind = "input" | "hidden" | "output";

type Layer = {
  id: number;
  name: string;
  kind: LayerKind;
  neurons: number;
  activation: string;
};

type Props = {
  layers: Layer[];
  isTraining: boolean;   // true mientras TF.js entrena de verdad
  epochProgress: number; // 0–100, para colorear la barra de fondo
  compact?: boolean;
};

// ─── Colores por tipo de capa ────────────────────────────────────────────────
const LAYER_COLOR: Record<LayerKind, string> = {
  input:  "#3478db",
  hidden: "#7954c8",
  output: "#0d9b72",
};

const FORWARD_COLOR  = "#60a5fa"; // azul claro — forward pass
const BACK_COLOR     = "#f97316"; // naranja     — backpropagation
const ACTIVE_GLOW    = "#facc15"; // amarillo    — neurona activa

// ─── Helpers ─────────────────────────────────────────────────────────────────

function clamp(v: number, min: number, max: number) {
  return Math.max(min, Math.min(max, v));
}

/** Calcula posiciones (x, y) de cada neurona en el SVG */
function computePositions(layers: Layer[], w: number, h: number, compact: boolean) {
  const topPad    = compact ? 55 : 70;
  const usableH   = compact ? 150 : 260;
  const sidePad   = 70;

  return layers.map((layer, li) => {
    const x = layers.length === 1
      ? w / 2
      : sidePad + (li * (w - sidePad * 2)) / (layers.length - 1);

    const maxN   = layer.neurons;
    const spacing = Math.min(compact ? 32 : 46, usableH / Math.max(maxN - 1, 1));
    const startY  = h / 2 - ((maxN - 1) * spacing) / 2;

    return Array.from({ length: maxN }, (_, ni) => ({
      x,
      y: startY + ni * spacing,
      li,
      ni,
    }));
  });
}

// ─── Tipos internos de animación ─────────────────────────────────────────────

type Pulse = {
  id: number;
  x1: number; y1: number;
  x2: number; y2: number;
  progress: number;  // 0 → 1
  color: string;
  speed: number;
};

type GlowNeuron = {
  key: string;   // `${li}-${ni}`
  alpha: number; // 0 → 1 (opacidad del glow)
};

// ─── Componente ───────────────────────────────────────────────────────────────

export function AnimatedNetworkDiagram({ layers, isTraining, epochProgress, compact = false }: Props) {
  const W = 1000;
  const H = compact ? 280 : 420;

  const positions = computePositions(layers, W, H, compact);

  // Lista de pulsos activos en pantalla
  const [pulses, setPulses]       = useState<Pulse[]>([]);
  const [glows, setGlows]         = useState<GlowNeuron[]>([]);
  const pulsesRef                 = useRef<Pulse[]>([]);
  const glowsRef                  = useRef<GlowNeuron[]>([]);
  const nextIdRef                 = useRef(0);
  const rafRef                    = useRef<number>(0);
  const lastSpawnRef              = useRef(0);

  // Sincronizar refs con state (evita stale closure en rAF)
  pulsesRef.current = pulses;
  glowsRef.current  = glows;

  useEffect(() => {
    if (!isTraining) {
      // Limpiar todo al detener
      setPulses([]);
      setGlows([]);
      cancelAnimationFrame(rafRef.current);
      return;
    }

    let lastTime = performance.now();

    const spawnPulse = (direction: "forward" | "back") => {
      if (layers.length < 2) return;

      // Elegir conexión aleatoria entre capas adyacentes
      const maxLayerIdx = layers.length - 1;
      let li: number;
      let fromNodes: { x: number; y: number }[];
      let toNodes:   { x: number; y: number }[];

      if (direction === "forward") {
        li        = Math.floor(Math.random() * maxLayerIdx);
        fromNodes = positions[li];
        toNodes   = positions[li + 1];
      } else {
        li        = Math.floor(Math.random() * maxLayerIdx) + 1;
        fromNodes = positions[li];
        toNodes   = positions[li - 1];
      }

      const from = fromNodes[Math.floor(Math.random() * fromNodes.length)];
      const to   = toNodes[Math.floor(Math.random() * toNodes.length)];

      const newPulse: Pulse = {
        id:       nextIdRef.current++,
        x1: from.x, y1: from.y,
        x2: to.x,   y2: to.y,
        progress: 0,
        color:    direction === "forward" ? FORWARD_COLOR : BACK_COLOR,
        speed:    0.012 + Math.random() * 0.010, // velocidad variable
      };

      pulsesRef.current = [...pulsesRef.current, newPulse];
      setPulses([...pulsesRef.current]);
    };

    const tick = (now: number) => {
      const dt = clamp(now - lastTime, 0, 50); // cap a 50ms para evitar saltos
      lastTime = now;

      // ── Spawn nuevos pulsos periódicamente ──
      const spawnInterval = 180; // ms entre spawns
      if (now - lastSpawnRef.current > spawnInterval) {
        lastSpawnRef.current = now;
        // Mayoritariamente forward, ocasionalmente backprop
        spawnPulse(Math.random() < 0.65 ? "forward" : "back");
      }

      // ── Avanzar pulsos existentes ──
      const arrivals: { li: number; ni: number }[] = [];
      const nextPulses = pulsesRef.current
        .map(p => ({ ...p, progress: p.progress + p.speed * (dt / 16) }))
        .filter(p => {
          if (p.progress >= 1) {
            // El pulso llegó a destino — registrar neurona que se activa
            const toLayerIdx = positions.findIndex(layerNodes =>
              layerNodes.some(n => Math.abs(n.x - p.x2) < 2 && Math.abs(n.y - p.y2) < 2)
            );
            if (toLayerIdx >= 0) {
              const ni = positions[toLayerIdx].findIndex(
                n => Math.abs(n.x - p.x2) < 2 && Math.abs(n.y - p.y2) < 2
              );
              if (ni >= 0) arrivals.push({ li: toLayerIdx, ni });
            }
            return false; // eliminar pulso completado
          }
          return true;
        });

      pulsesRef.current = nextPulses;
      setPulses([...nextPulses]);

      // ── Activar brillo en neuronas que recibieron señal ──
      if (arrivals.length > 0) {
        const newGlows: GlowNeuron[] = arrivals.map(a => ({
          key: `${a.li}-${a.ni}`,
          alpha: 1.0,
        }));
        glowsRef.current = [
          ...glowsRef.current.filter(g => !newGlows.some(ng => ng.key === g.key)),
          ...newGlows,
        ];
      }

      // ── Desvanecer brillos existentes ──
      const nextGlows = glowsRef.current
        .map(g => ({ ...g, alpha: g.alpha - 0.025 * (dt / 16) }))
        .filter(g => g.alpha > 0.01);

      glowsRef.current = nextGlows;
      setGlows([...nextGlows]);

      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isTraining]);

  // ─── Renderizado SVG ───────────────────────────────────────────────────────

  const r = compact ? 11 : 15; // radio de neuronas

  return (
    <div className="animated-network-wrapper" style={{ position: "relative" }}>
      {/* Barra de progreso de fondo */}
      {isTraining && (
        <div className="animated-network-progress-bg">
          <div
            className="animated-network-progress-fill"
            style={{ width: `${epochProgress}%` }}
          />
        </div>
      )}

      <svg
        viewBox={`0 0 ${W} ${H}`}
        className={`network-svg ${compact ? "network-svg-compact" : ""}`}
        role="img"
        aria-label={isTraining
          ? "Red neuronal animada — entrenamiento en progreso"
          : "Red neuronal artificial con capas y neuronas conectadas"}
      >
        <defs>
          {/* Filtro de brillo para neuronas activas */}
          <filter id="neuron-glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>

          {/* Gradiente de fondo sutil */}
          <radialGradient id="bg-gradient" cx="50%" cy="50%" r="50%">
            <stop offset="0%"   stopColor="#f0f4ff" stopOpacity="0.6" />
            <stop offset="100%" stopColor="#e8edf8" stopOpacity="0.2" />
          </radialGradient>

          {/* Patrón de puntos */}
          <pattern id="netDots" width="18" height="18" patternUnits="userSpaceOnUse">
            <circle cx="1" cy="1" r="1" fill="#cbd7e7" />
          </pattern>

          {/* Marcador de punta de flecha para forward */}
          <marker id="arrow-forward" markerWidth="6" markerHeight="6"
            refX="3" refY="3" orient="auto">
            <path d="M0,0 L0,6 L6,3 z" fill={FORWARD_COLOR} opacity="0.7" />
          </marker>

          {/* Marcador de punta de flecha para backprop */}
          <marker id="arrow-back" markerWidth="6" markerHeight="6"
            refX="3" refY="3" orient="auto">
            <path d="M6,0 L6,6 L0,3 z" fill={BACK_COLOR} opacity="0.7" />
          </marker>
        </defs>

        {/* Fondo */}
        <rect x="0" y="0" width={W} height={H} fill="url(#netDots)" opacity="0.5" />
        {isTraining && (
          <rect x="0" y="0" width={W} height={H} fill="url(#bg-gradient)" />
        )}

        {/* ── Conexiones estáticas ── */}
        {layers.slice(0, -1).map((_, li) =>
          positions[li].flatMap((from, fi) =>
            positions[li + 1].map((to, ti) => (
              <line
                key={`conn-${li}-${fi}-${ti}`}
                x1={from.x + r} y1={from.y}
                x2={to.x - r}   y2={to.y}
                className="network-connection"
                opacity={isTraining ? 0.25 : 0.45}
              />
            ))
          )
        )}

        {/* ── Pulsos animados ── */}
        {pulses.map(pulse => {
          const cx = pulse.x1 + (pulse.x2 - pulse.x1) * pulse.progress;
          const cy = pulse.y1 + (pulse.y2 - pulse.y1) * pulse.progress;
          const isForward = pulse.color === FORWARD_COLOR;
          return (
            <g key={pulse.id}>
              {/* Estela del pulso */}
              <line
                x1={pulse.x1 + (pulse.x2 - pulse.x1) * Math.max(0, pulse.progress - 0.18)}
                y1={pulse.y1 + (pulse.y2 - pulse.y1) * Math.max(0, pulse.progress - 0.18)}
                x2={cx} y2={cy}
                stroke={pulse.color}
                strokeWidth="1.5"
                strokeLinecap="round"
                opacity={0.4}
              />
              {/* Punto del pulso */}
              <circle
                cx={cx} cy={cy}
                r={isForward ? 5 : 4}
                fill={pulse.color}
                opacity={0.9}
                filter="url(#neuron-glow)"
              />
            </g>
          );
        })}

        {/* ── Neuronas ── */}
        {layers.map((layer, li) => {
          const layerNodes = positions[li];
          return (
            <g key={layer.id}>
              {/* Etiqueta de capa */}
              <text
                x={layerNodes[0].x}
                y={compact ? 24 : 28}
                textAnchor="middle"
                className="network-layer-title"
              >
                {layer.name}
              </text>
              <text
                x={layerNodes[0].x}
                y={compact ? 40 : 46}
                textAnchor="middle"
                className="network-layer-subtitle"
              >
                {layer.neurons} neuronas · {layer.activation !== "Ninguna" ? layer.activation : "lineal"}
              </text>

              {layerNodes.map(node => {
                const glowEntry = glows.find(g => g.key === `${li}-${node.ni}`);
                const isActive  = !!glowEntry && glowEntry.alpha > 0.05;
                const baseColor = LAYER_COLOR[layer.kind];

                return (
                  <g key={`${li}-${node.ni}`} className="network-node">
                    {/* Halo de brillo cuando la neurona recibe señal */}
                    {isActive && (
                      <circle
                        cx={node.x} cy={node.y}
                        r={r + 10}
                        fill={ACTIVE_GLOW}
                        opacity={glowEntry!.alpha * 0.45}
                        filter="url(#neuron-glow)"
                      />
                    )}

                    {/* Círculo exterior cuando está activo */}
                    {isActive && (
                      <circle
                        cx={node.x} cy={node.y}
                        r={r + 5}
                        fill="none"
                        stroke={ACTIVE_GLOW}
                        strokeWidth="1.5"
                        opacity={glowEntry!.alpha * 0.8}
                      />
                    )}

                    {/* Neurona principal */}
                    <circle
                      cx={node.x} cy={node.y}
                      r={r}
                      fill={isActive
                        ? `color-mix(in srgb, ${baseColor} 60%, ${ACTIVE_GLOW} 40%)`
                        : baseColor}
                      stroke={isActive ? ACTIVE_GLOW : baseColor}
                      strokeWidth={isActive ? 2 : 1}
                      style={{
                        transition: "fill 0.1s ease",
                        filter: isActive ? "url(#neuron-glow)" : "none",
                      }}
                    />

                    {/* Número de neurona (solo vista normal) */}
                    {!compact && (
                      <text
                        x={node.x} y={node.y + 4}
                        textAnchor="middle"
                        className="neuron-number"
                      >
                        {node.ni + 1}
                      </text>
                    )}
                  </g>
                );
              })}
            </g>
          );
        })}

        {/* ── Leyenda de animación ── */}
        {isTraining && (
          <g>
            <circle cx={compact ? W - 240 : W - 260} cy={H - 20} r="5" fill={FORWARD_COLOR} />
            <text x={compact ? W - 230 : W - 250} y={H - 15}
              className="network-layer-subtitle" textAnchor="start">
              Forward pass
            </text>
            <circle cx={compact ? W - 130 : W - 150} cy={H - 20} r="5" fill={BACK_COLOR} />
            <text x={compact ? W - 120 : W - 140} y={H - 15}
              className="network-layer-subtitle" textAnchor="start">
              Backpropagation
            </text>
          </g>
        )}

        {/* ── Texto cuando está en reposo ── */}
        {!isTraining && compact && (
          <text x="500" y={H - 10} textAnchor="middle" className="network-bottom-caption">
            REPRESENTACIÓN DE ARQUITECTURA — INICIA UN ENTRENAMIENTO PARA VER LA ANIMACIÓN
          </text>
        )}
      </svg>
    </div>
  );
}
