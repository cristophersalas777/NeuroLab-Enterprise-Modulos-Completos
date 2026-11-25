import { useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent, FormEvent, ReactNode } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import jsPDF from "jspdf";
import { buildAndTrain } from "./tfEngine";
import { NetworkCanvas3D } from "./NetworkCanvas3D";
import {
  Activity,
  ArrowRight,
  Bell,
  BookOpen,
  Boxes,
  BrainCircuit,
  Check,
  CheckCircle2,
  ChevronDown,
  CircleHelp,
  Database,
  Download,
  FileText,
  FlaskConical,
  FolderKanban,
  Layers,
  LayoutDashboard,
  LockKeyhole,
  Network,
  Pause,
  Pencil,
  Play,
  Plus,
  RotateCcw,
  Save,
  Search,
  Settings,
  ShieldCheck,
  SlidersHorizontal,
  Trash2,
  Upload,
  Users,
  X,
  Zap,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

type ModuleName =
  | "Dashboard"
  | "Proyectos de IA"
  | "Fundamentos neuronales"
  | "Diseñador de arquitecturas"
  | "Datasets"
  | "Entrenamientos"
  | "Evaluación y métricas"
  | "Registro de modelos"
  | "Reportes"
  | "Administración y seguridad";

type LayerKind = "input" | "hidden" | "output";
type Layer = { id: number; name: string; kind: LayerKind; neurons: number; activation: string };
type Project = { id: number; name: string; problem: string; owner: string; status: "Activo" | "En revisión" | "Archivado"; createdAt: string };
type Dataset = { id: number; name: string; description: string; rows: number; columns: string[]; preview: string[][]; target: string; split: string; source: string; createdAt: string };
type Training = { id: number; project: string; dataset: string; architecture: string; epochs: number; batchSize: number; optimizer: string; status: "En curso" | "Completado" | "Cancelado"; loss: number; accuracy: number; createdAt: string };
type ModelRecord = { id: number; name: string; version: string; architecture: string; dataset: string; status: "Borrador" | "En revisión" | "Aprobado" | "Retirado"; accuracy: number; createdAt: string };
type UserRecord = { id: number; name: string; email: string; role: string; status: "Activo" | "Suspendido" };
type NeuronSelection = { layerId: number; index: number };

const initialLayers: Layer[] = [
  { id: 1, name: "Entrada", kind: "input", neurons: 4, activation: "Ninguna" },
  { id: 2, name: "Oculta 1", kind: "hidden", neurons: 5, activation: "ReLU" },
  { id: 3, name: "Oculta 2", kind: "hidden", neurons: 4, activation: "ReLU" },
  { id: 4, name: "Salida", kind: "output", neurons: 2, activation: "Softmax" },
];

const seedProjects: Project[] = [
  { id: 1, name: "Clasificación de clientes", problem: "Identificar grupos de clientes según sus características", owner: "Administrador", status: "Activo", createdAt: "23/11/2026" },
  { id: 2, name: "Predicción de demanda", problem: "Estimar la demanda a partir de datos históricos", owner: "Administrador", status: "En revisión", createdAt: "22/11/2026" },
];

const seedDatasets: Dataset[] = [
  {
    id: 1, name: "clientes_demo.csv",
    description: "Clasificación de clientes por segmento según edad, ingresos y compras",
    rows: 40,
    columns: ["edad", "ingresos", "compras", "segmento"],
    preview: [
      ["23","1800","2","Básico"],["35","3200","5","Premium"],["29","2500","3","Básico"],
      ["42","5100","8","Premium"],["31","2900","4","Estándar"],["55","6200","10","Premium"],
      ["27","1500","1","Básico"],["38","3800","6","Estándar"],["45","4900","7","Premium"],
      ["22","1200","1","Básico"],["33","3000","4","Estándar"],["50","5800","9","Premium"],
      ["28","2100","2","Básico"],["41","4200","6","Premium"],["36","3500","5","Estándar"],
      ["24","1700","2","Básico"],["48","5500","8","Premium"],["30","2700","3","Estándar"],
      ["26","1900","2","Básico"],["43","4700","7","Premium"],["37","3600","5","Estándar"],
      ["21","1100","1","Básico"],["52","6000","9","Premium"],["32","2800","4","Estándar"],
      ["25","1600","2","Básico"],["46","5200","8","Premium"],["34","3100","4","Estándar"],
      ["20","1000","1","Básico"],["49","5600","8","Premium"],["39","3900","6","Estándar"],
      ["44","4800","7","Premium"],["27","2000","3","Básico"],["53","6100","10","Premium"],
      ["31","2600","3","Estándar"],["23","1400","2","Básico"],["47","5300","8","Premium"],
      ["36","3400","5","Estándar"],["22","1300","1","Básico"],["54","6300","10","Premium"],
      ["40","4000","6","Estándar"],["28","2200","3","Básico"],
    ],
    target: "segmento", split: "70 / 15 / 15",
    source: "Dataset de demostración — 40 registros", createdAt: "23/11/2026",
  },
  {
    id: 2, name: "diabetes_demo.csv",
    description: "Predicción de diabetes según glucosa, presión arterial e IMC (clasificación binaria)",
    rows: 40,
    columns: ["glucosa", "presion", "imc", "edad", "diabetes"],
    preview: [
      ["148","72","33.6","50","1"],["85","66","26.6","31","0"],["183","64","23.3","32","1"],
      ["89","66","28.1","21","0"],["137","40","43.1","33","1"],["116","74","25.6","30","0"],
      ["78","50","31.0","26","1"],["115","0","35.3","29","0"],["197","70","30.5","53","1"],
      ["125","96","0.0","54","1"],["110","0","37.6","24","0"],["168","74","38.0","34","1"],
      ["139","80","27.1","57","0"],["189","60","30.1","59","1"],["166","72","25.8","51","1"],
      ["100","0","30.0","32","0"],["167","74","37.6","38","1"],["145","82","26.2","26","0"],
      ["102","86","28.4","24","0"],["120","54","35.1","45","0"],["141","86","21.2","30","0"],
      ["174","55","25.6","52","1"],["99","76","23.2","21","0"],["109","75","36.0","54","0"],
      ["119","0","32.4","24","0"],["166","60","34.2","49","1"],["107","80","26.1","24","0"],
      ["173","70","29.5","51","1"],["97","60","39.0","25","0"],["83","78","31.4","30","0"],
      ["193","50","25.9","60","1"],["123","62","35.1","25","0"],["155","76","35.1","52","1"],
      ["93","100","33.6","43","0"],["143","94","33.1","46","1"],["147","76","39.4","43","1"],
      ["97","66","27.2","23","0"],["145","80","44.2","31","1"],["117","92","34.1","23","0"],
      ["109","75","30.0","55","0"],
    ],
    target: "diabetes", split: "70 / 15 / 15",
    source: "Dataset de demostración — 40 registros", createdAt: "23/11/2026",
  },
];

const seedTrainings: Training[] = [
  { id: 1, project: "Clasificación de clientes", dataset: "clientes_demo.csv", architecture: "Feedforward · 4-5-4-2", epochs: 20, batchSize: 16, optimizer: "Adam", status: "Completado", loss: 0.084, accuracy: 94.2, createdAt: "23/11/2026 09:30" },
];

const seedModels: ModelRecord[] = [
  { id: 1, name: "Clasificador de clientes", version: "v1.0.0", architecture: "Feedforward · 4-5-4-2", dataset: "clientes_demo.csv", status: "Aprobado", accuracy: 94.2, createdAt: "23/11/2026" },
];

const seedUsers: UserRecord[] = [
  { id: 1, name: "Administrador", email: "admin@neurolab.local", role: "Administrador", status: "Activo" },
  { id: 2, name: "Analista IA", email: "analista@neurolab.local", role: "Ingeniero/analista de IA", status: "Activo" },
  { id: 3, name: "Revisor", email: "revisor@neurolab.local", role: "Revisor", status: "Activo" },
];

const navigation: { title: string; items: { name: ModuleName; icon: LucideIcon }[] }[] = [
  { title: "PLATAFORMA", items: [
    { name: "Dashboard", icon: LayoutDashboard },
    { name: "Proyectos de IA", icon: FolderKanban },
    { name: "Fundamentos neuronales", icon: BookOpen },
    { name: "Diseñador de arquitecturas", icon: Network },
    { name: "Datasets", icon: Database },
    { name: "Entrenamientos", icon: Play },
    { name: "Evaluación y métricas", icon: FlaskConical },
  ] },
  { title: "GESTIÓN", items: [
    { name: "Registro de modelos", icon: Boxes },
    { name: "Reportes", icon: FileText },
    { name: "Administración y seguridad", icon: ShieldCheck },
  ] },
];

const moduleDescriptions: Record<ModuleName, string> = {
  Dashboard: "Espacio de trabajo y acceso rápido a la arquitectura neuronal.",
  "Proyectos de IA": "Crea proyectos y describe el problema de inteligencia artificial que deseas resolver.",
  "Fundamentos neuronales": "Aprende cómo trabajan las neuronas, capas, conexiones, pesos, sesgos y activaciones.",
  "Diseñador de arquitecturas": "Configura las capas y observa la estructura de tu red neuronal.",
  Datasets: "Carga y revisa conjuntos de datos que servirán como entrada para los modelos.",
  Entrenamientos: "Configura los parámetros y ejecuta una demostración del proceso de aprendizaje.",
  "Evaluación y métricas": "Consulta los resultados de las ejecuciones y el comportamiento del modelo.",
  "Registro de modelos": "Guarda, versiona y administra los modelos desarrollados.",
  Reportes: "Genera un resumen técnico de los proyectos, arquitecturas y entrenamientos.",
  "Administración y seguridad": "Administra usuarios, roles y permisos de acceso al entorno.",
};

function readStorage<T,>(key: string, fallback: T): T {
  try {
    const value = localStorage.getItem(key);
    return value ? (JSON.parse(value) as T) : fallback;
  } catch {
    return fallback;
  }
}

function useStoredState<T,>(key: string, initial: T) {
  const [value, setValue] = useState<T>(() => readStorage(key, initial));
  useEffect(() => {
    localStorage.setItem(key, JSON.stringify(value));
  }, [key, value]);
  return [value, setValue] as const;
}

function today() {
  return new Date().toLocaleDateString("es-PE");
}

function parseCsvLine(line: string): string[] {
  const cells: string[] = [];
  let current = "";
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    if (character === '"') {
      if (quoted && line[index + 1] === '"') {
        current += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if ((character === "," || character === ";") && !quoted) {
      cells.push(current.trim());
      current = "";
    } else {
      current += character;
    }
  }
  cells.push(current.trim());
  return cells;
}

function NetworkDiagram({ layers, selected, onSelect, compact = false }: { layers: Layer[]; selected?: NeuronSelection; onSelect?: (selection: NeuronSelection) => void; compact?: boolean }) {
  const width = 1000;
  const height = compact ? 285 : 430;
  const usableTop = compact ? 75 : 85;
  const usableHeight = compact ? 165 : 270;
  const positions = layers.map((layer, layerIndex) => {
    const x = layers.length === 1 ? width / 2 : 75 + (layerIndex * 850) / (layers.length - 1);
    const spacing = Math.min(compact ? 34 : 49, usableHeight / Math.max(layer.neurons - 1, 1));
    const startY = height / 2 - ((layer.neurons - 1) * spacing) / 2;
    return Array.from({ length: layer.neurons }, (_, index) => ({ x, y: startY + index * spacing, index }));
  });
  const colorFor = (kind: LayerKind) => kind === "input" ? "#3478db" : kind === "output" ? "#0d9b72" : "#7954c8";

  return (
    <svg className={`network-svg ${compact ? "network-svg-compact" : ""}`} viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Red neuronal artificial con capas y neuronas conectadas">
      <defs>
        <pattern id={compact ? "networkDotsSmall" : "networkDots"} width="18" height="18" patternUnits="userSpaceOnUse">
          <circle cx="1" cy="1" r="1" fill="#cbd7e7" />
        </pattern>
      </defs>
      <rect x="0" y="0" width={width} height={height} fill={`url(#${compact ? "networkDotsSmall" : "networkDots"})`} opacity=".53" />
      {layers.slice(0, -1).map((layer, layerIndex) => positions[layerIndex].flatMap((from) => positions[layerIndex + 1].map((to) => (
        <line key={`${layer.id}-${from.index}-${to.index}`} x1={from.x + (compact ? 12 : 16)} y1={from.y} x2={to.x - (compact ? 12 : 16)} y2={to.y} className="network-connection" />
      ))))}
      {layers.map((layer, layerIndex) => (
        <g key={layer.id}>
          <text x={positions[layerIndex][0].x} y={compact ? 26 : 29} textAnchor="middle" className="network-layer-title">{layer.name}</text>
          <text x={positions[layerIndex][0].x} y={compact ? 43 : 48} textAnchor="middle" className="network-layer-subtitle">{layer.neurons} neuronas</text>
          {positions[layerIndex].map((node) => {
            const isSelected = selected?.layerId === layer.id && selected.index === node.index;
            const radius = compact ? 11 : 15;
            return (
              <g key={`${layer.id}-${node.index}`} className={onSelect ? "network-node clickable" : "network-node"} onClick={() => onSelect?.({ layerId: layer.id, index: node.index })} role={onSelect ? "button" : undefined} tabIndex={onSelect ? 0 : undefined} onKeyDown={(event) => { if (onSelect && (event.key === "Enter" || event.key === " ")) onSelect({ layerId: layer.id, index: node.index }); }}>
                {isSelected && <circle cx={node.x} cy={node.y} r={radius + 8} fill="none" stroke="#efa92d" strokeWidth="2.5" />}
                <circle cx={node.x} cy={node.y} r={radius} fill={colorFor(layer.kind)} stroke={isSelected ? "#ffffff" : colorFor(layer.kind)} strokeWidth={isSelected ? 2.5 : 1} />
                {!compact && <text x={node.x} y={node.y + 3.5} textAnchor="middle" className="neuron-number">{node.index + 1}</text>}
              </g>
            );
          })}
        </g>
      ))}
      {compact && <text x="500" y="270" textAnchor="middle" className="network-bottom-caption">REPRESENTACIÓN DE ARQUITECTURA FEEDFORWARD</text>}
    </svg>
  );
}

function SectionHeader({ eyebrow, title, description, actions }: { eyebrow: string; title: string; description: string; actions?: ReactNode }) {
  return (
    <div className="page-heading">
      <div>
        <div className="eyebrow"><BrainCircuit size={14} />{eyebrow}</div>
        <h2>{title}</h2>
        <p>{description}</p>
      </div>
      {actions && <div className="heading-actions">{actions}</div>}
    </div>
  );
}

function StatusPill({ children }: { children: string }) {
  const normalized = children.toLowerCase().replaceAll(" ", "-");
  return <span className={`status-pill status-${normalized}`}>{children}</span>;
}

function App() {
  const [activeModule, setActiveModule] = useState<ModuleName>("Diseñador de arquitecturas");
  const [projects, setProjects] = useStoredState<Project[]>("neurolab-projects", seedProjects);
  const [datasets, setDatasets] = useStoredState<Dataset[]>("neurolab-datasets", seedDatasets);
  const [trainingHistory, setTrainingHistory] = useStoredState<Training[]>("neurolab-trainings", seedTrainings);
  const [models, setModels] = useStoredState<ModelRecord[]>("neurolab-models", seedModels);
  const [users, setUsers] = useStoredState<UserRecord[]>("neurolab-users", seedUsers);
  const [layers, setLayers] = useStoredState<Layer[]>("neurolab-architecture", initialLayers);
  const [architectureName, setArchitectureName] = useStoredState("neurolab-architecture-name", "Red neuronal principal");
  const [networkFamily, setNetworkFamily] = useStoredState("neurolab-network-family", "Feedforward");
  const [activeLayerId, setActiveLayerId] = useState(2);
  const [selectedNeuron, setSelectedNeuron] = useState<NeuronSelection>({ layerId: 2, index: 2 });
  const [notice, setNotice] = useState("");

  // Proyectos
  const [showProjectForm, setShowProjectForm] = useState(false);
  const [projectName, setProjectName] = useState("");
  const [projectProblem, setProjectProblem] = useState("");
  const [projectSearch, setProjectSearch] = useState("");
  const [editingProjectId, setEditingProjectId] = useState<number | null>(null);

  // Fundamentos
  const [activationName, setActivationName] = useState("ReLU");
  const [activationInput, setActivationInput] = useState("-1.5");

  // Datasets
  const [datasetDescription, setDatasetDescription] = useState("");
  const [datasetTarget, setDatasetTarget] = useState("");

  // Entrenamiento
  const [trainingProject, setTrainingProject] = useState(seedProjects[0]?.name ?? "");
  const [trainingDataset, setTrainingDataset] = useState(seedDatasets[0]?.name ?? "");
  const [epochs, setEpochs] = useState(50);
  const [batchSize, setBatchSize] = useState(16);
  const [optimizer, setOptimizer] = useState("Adam");
  const [trainingRunning, setTrainingRunning] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [activeTraining, setActiveTraining] = useStoredState<Training | null>("neurolab-active-training", null);
  const [epochData, setEpochData] = useStoredState<{ epoch: number; loss: number; accuracy: number }[]>("neurolab-epoch-data", []);
  const [trainingError, setTrainingError] = useState("");
  const [canCancel, setCanCancel] = useState(false);
  const cancelSignalRef = useRef<{ cancelled: boolean }>({ cancelled: false });

  // Administración
  const [showUserForm, setShowUserForm] = useState(false);
  const [newUserName, setNewUserName] = useState("");
  const [newUserEmail, setNewUserEmail] = useState("");
  const [newUserRole, setNewUserRole] = useState("Ingeniero/analista de IA");

  const activeLayer = layers.find((layer) => layer.id === activeLayerId) ?? layers[0];
  const hiddenCount = layers.filter((layer) => layer.kind === "hidden").length;
  const totalNeurons = layers.reduce((sum, layer) => sum + layer.neurons, 0);
  const selectedDataset = datasets.find((dataset) => dataset.name === trainingDataset) ?? datasets[0];
  const latestTraining = trainingHistory[0];
  const completedTrainings = trainingHistory.filter((training) => training.status === "Completado");
  const filteredProjects = projects.filter((project) => `${project.name} ${project.problem}`.toLowerCase().includes(projectSearch.toLowerCase()));

  // Cleanup: si el componente se desmonta mientras entrena, cancelar
  useEffect(() => {
    return () => { cancelSignalRef.current.cancelled = true; };
  }, []);

  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => setNotice(""), 3500);
    return () => window.clearTimeout(timer);
  }, [notice]);

  const selectModule = (name: ModuleName) => {
    setActiveModule(name);
    setNotice("");
  };

  const saveArchitecture = () => {
    localStorage.setItem("neurolab-architecture", JSON.stringify(layers));
    localStorage.setItem("neurolab-architecture-name", JSON.stringify(architectureName));
    localStorage.setItem("neurolab-network-family", JSON.stringify(networkFamily));
    setNotice("Arquitectura guardada en este navegador.");
  };

  const updateActiveLayer = (changes: Partial<Layer>) => {
    setLayers((current) => current.map((layer) => layer.id === activeLayer.id ? { ...layer, ...changes } : layer));
  };

  const addLayer = () => {
    if (layers.length >= 7) { setNotice("La demostración permite un máximo de 7 capas."); return; }
    const newId = Math.max(...layers.map((layer) => layer.id)) + 1;
    const newLayer: Layer = { id: newId, name: `Oculta ${hiddenCount + 1}`, kind: "hidden", neurons: 3, activation: "ReLU" };
    const outputIndex = layers.findIndex((layer) => layer.kind === "output");
    const next = [...layers];
    next.splice(outputIndex, 0, newLayer);
    setLayers(next);
    setActiveLayerId(newId);
    setSelectedNeuron({ layerId: newId, index: 0 });
  };

  const removeActiveLayer = () => {
    if (activeLayer.kind !== "hidden") return;
    const next = layers.filter((layer) => layer.id !== activeLayer.id);
    setLayers(next);
    const nextLayer = next.find((layer) => layer.kind === "hidden") ?? next[0];
    setActiveLayerId(nextLayer.id);
    setSelectedNeuron({ layerId: nextLayer.id, index: 0 });
  };

  const resetArchitecture = () => {
    setLayers(initialLayers);
    setArchitectureName("Red neuronal principal");
    setNetworkFamily("Feedforward");
    setActiveLayerId(2);
    setSelectedNeuron({ layerId: 2, index: 2 });
    setNotice("Se restauró la arquitectura inicial.");
  };

  const changeNeuronCount = (amount: number) => {
    const count = Math.max(1, Math.min(12, activeLayer.neurons + amount));
    updateActiveLayer({ neurons: count });
    setSelectedNeuron((current) => ({ layerId: activeLayer.id, index: Math.min(current.index, count - 1) }));
  };

  const submitProject = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!projectName.trim() || !projectProblem.trim()) return;
    if (editingProjectId !== null) {
      setProjects((current) => current.map((project) => project.id === editingProjectId ? { ...project, name: projectName.trim(), problem: projectProblem.trim() } : project));
      setNotice("Proyecto actualizado.");
    } else {
      const project: Project = { id: Date.now(), name: projectName.trim(), problem: projectProblem.trim(), owner: "Administrador", status: "Activo", createdAt: today() };
      setProjects((current) => [project, ...current]);
      setTrainingProject(project.name);
      setNotice("Proyecto creado correctamente.");
    }
    setProjectName("");
    setProjectProblem("");
    setEditingProjectId(null);
    setShowProjectForm(false);
  };

  const editProject = (project: Project) => {
    setProjectName(project.name);
    setProjectProblem(project.problem);
    setEditingProjectId(project.id);
    setShowProjectForm(true);
  };

  const deleteProject = (id: number) => {
    setProjects((current) => current.filter((project) => project.id !== id));
    setNotice("Proyecto eliminado de la demostración.");
  };

  const handleDatasetUpload = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const content = String(reader.result ?? "");
      const lines = content.split(/\r?\n/).filter((line) => line.trim().length > 0);
      if (lines.length < 2) { setNotice("El archivo debe contener encabezados y al menos una fila de datos."); return; }
      const columns = parseCsvLine(lines[0]);
      const rows = lines.slice(1).map(parseCsvLine);
      const dataset: Dataset = {
        id: Date.now(), name: file.name, description: datasetDescription || "Dataset importado desde archivo CSV", rows: rows.length,
        columns, preview: rows.slice(0, 500), target: datasetTarget || columns[columns.length - 1] || "", split: "70 / 15 / 15", source: "Archivo cargado por el usuario", createdAt: today(),
      };
      setDatasets((current) => [dataset, ...current]);
      setTrainingDataset(dataset.name);
      setDatasetDescription("");
      setDatasetTarget("");
      setNotice(`Se importó ${file.name} con ${rows.length} filas y ${columns.length} columnas.`);
      event.target.value = "";
    };
    reader.onerror = () => setNotice("No se pudo leer el archivo seleccionado.");
    reader.readAsText(file);
  };

  const removeDataset = (id: number) => {
    setDatasets((current) => current.filter((dataset) => dataset.id !== id));
    setNotice("Dataset eliminado.");
  };

  const startTraining = async () => {
    if (!trainingProject || !trainingDataset) { setNotice("Selecciona un proyecto y un dataset antes de iniciar."); return; }
    if (trainingRunning) return;

    // Buscar el dataset completo
    const ds = datasets.find(d => d.name === trainingDataset);
    if (!ds || ds.preview.length < 2) {
      setNotice("El dataset seleccionado no tiene suficientes filas para entrenar (mínimo 2).");
      return;
    }

    const training: Training = {
      id: Date.now(), project: trainingProject, dataset: trainingDataset,
      architecture: `${networkFamily} · ${layers.map(l => l.neurons).join("-")}`,
      epochs, batchSize, optimizer, status: "En curso", loss: 0, accuracy: 0,
      createdAt: new Date().toLocaleString("es-PE"),
    };

    setActiveTraining(training);
    setTrainingProgress(0);
    setEpochData([]);
    setTrainingError("");
    setTrainingRunning(true);
    setCanCancel(true);
    setNotice("Entrenamiento real iniciado con TensorFlow.js.");

    // Resetear señal de cancelación
    cancelSignalRef.current = { cancelled: false };

    try {
      const result = await buildAndTrain(
        // Capas del diseñador convertidas al formato del motor
        layers.map(l => ({ kind: l.kind, neurons: l.neurons, activation: l.activation })),
        // Dataset: columnas, filas y variable objetivo
        { columns: ds.columns, preview: ds.preview, target: ds.target || ds.columns[ds.columns.length - 1] },
        // Hiperparámetros
        { epochs, batchSize, optimizer, learningRate: 0.001 },
        // Callback por época → actualiza barra y gráfico en tiempo real
        (epochResult) => {
          setEpochData(prev => [...prev, epochResult]);
          setTrainingProgress(Math.round((epochResult.epoch / epochs) * 100));
        },
        cancelSignalRef.current
      );

      if (cancelSignalRef.current.cancelled) return; // ya manejado en cancelTraining

      const completed: Training = {
        ...training,
        status: "Completado",
        loss: result.testLoss,
        accuracy: result.testAccuracy,
        createdAt: new Date().toLocaleString("es-PE"),
      };
      setTrainingHistory(current => [completed, ...current]);
      setActiveTraining(completed);
      setTrainingProgress(100);
      setNotice(`Entrenamiento completado. Accuracy en Test: ${result.testAccuracy.toFixed(1)}% · Loss: ${result.testLoss.toFixed(4)}`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setTrainingError(msg);
      const failed: Training = {
        ...training,
        status: "Cancelado",
        createdAt: new Date().toLocaleString("es-PE"),
      };
      setTrainingHistory(current => [failed, ...current]);
      setActiveTraining(failed);
      setNotice("Error durante el entrenamiento. Revisa la consola para detalles.");
    } finally {
      setTrainingRunning(false);
      setCanCancel(false);
    }
  };

  const cancelTraining = () => {
    if (!activeTraining) return;
    cancelSignalRef.current.cancelled = true;
    const cancelled = { ...activeTraining, status: "Cancelado" as const, createdAt: new Date().toLocaleString("es-PE") };
    setTrainingHistory(current => [cancelled, ...current]);
    setActiveTraining(cancelled);
    setTrainingRunning(false);
    setCanCancel(false);
    setTrainingProgress(0);
    setNotice("Entrenamiento cancelado por el usuario.");
  };

  const createModelVersion = () => {
    const versions = models.filter((model) => model.name === architectureName).length;
    const model: ModelRecord = {
      id: Date.now(), name: architectureName || "Modelo sin nombre", version: `v1.${versions}.0`,
      architecture: `${networkFamily} · ${layers.map((layer) => layer.neurons).join("-")}`,
      dataset: latestTraining?.dataset ?? datasets[0]?.name ?? "Sin dataset", status: "Borrador",
      accuracy: latestTraining?.status === "Completado" ? latestTraining.accuracy : 0, createdAt: today(),
    };
    setModels((current) => [model, ...current]);
    setNotice(`Se registró la versión ${model.version} del modelo.`);
  };

  const updateModelStatus = (id: number, status: ModelRecord["status"]) => {
    setModels((current) => current.map((model) => model.id === id ? { ...model, status } : model));
  };

  const createReport = () => {
    const doc = new jsPDF();
    const margin = 18;
    let y = 20;

    // Encabezado
    doc.setFillColor(30, 41, 59);
    doc.rect(0, 0, 210, 28, "F");
    doc.setTextColor(255, 255, 255);
    doc.setFontSize(16);
    doc.setFont("helvetica", "bold");
    doc.text("NEUROLAB ENTERPRISE", margin, 12);
    doc.setFontSize(9);
    doc.setFont("helvetica", "normal");
    doc.text("Reporte técnico de demostración", margin, 20);
    doc.text(`Fecha: ${today()}`, 210 - margin, 20, { align: "right" });

    y = 38;
    doc.setTextColor(30, 41, 59);

    // Sección 1 - Arquitectura
    doc.setFontSize(12);
    doc.setFont("helvetica", "bold");
    doc.setTextColor(52, 120, 219);
    doc.text("1. Resumen de arquitectura", margin, y);
    y += 7;
    doc.setDrawColor(52, 120, 219);
    doc.line(margin, y, 210 - margin, y);
    y += 6;
    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    doc.setTextColor(30, 41, 59);
    const archLines: [string, string][] = [
      ["Nombre:", architectureName],
      ["Familia:", networkFamily],
      ["Total de capas:", String(layers.length)],
      ["Total de neuronas:", String(totalNeurons)],
      ["Estructura:", layers.map((l) => `${l.name} (${l.neurons})`).join(" → ")],
    ];
    for (const [label, value] of archLines) {
      doc.setFont("helvetica", "bold");
      doc.text(label, margin, y);
      doc.setFont("helvetica", "normal");
      doc.text(value, margin + 42, y);
      y += 7;
    }

    // Sección 2 - Recursos
    y += 4;
    doc.setFontSize(12);
    doc.setFont("helvetica", "bold");
    doc.setTextColor(52, 120, 219);
    doc.text("2. Recursos registrados", margin, y);
    y += 7;
    doc.setDrawColor(52, 120, 219);
    doc.line(margin, y, 210 - margin, y);
    y += 6;
    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    doc.setTextColor(30, 41, 59);
    const resourceLines: [string, string][] = [
      ["Proyectos:", String(projects.length)],
      ["Datasets:", String(datasets.length)],
      ["Entrenamientos:", String(trainingHistory.length)],
      ["Modelos registrados:", String(models.length)],
    ];
    for (const [label, value] of resourceLines) {
      doc.setFont("helvetica", "bold");
      doc.text(label, margin, y);
      doc.setFont("helvetica", "normal");
      doc.text(value, margin + 42, y);
      y += 7;
    }

    // Sección 3 - Última ejecución
    y += 4;
    doc.setFontSize(12);
    doc.setFont("helvetica", "bold");
    doc.setTextColor(52, 120, 219);
    doc.text("3. Última ejecución", margin, y);
    y += 7;
    doc.setDrawColor(52, 120, 219);
    doc.line(margin, y, 210 - margin, y);
    y += 6;
    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    doc.setTextColor(30, 41, 59);
    if (latestTraining) {
      const execLines: [string, string][] = [
        ["Proyecto:", latestTraining.project],
        ["Dataset:", latestTraining.dataset],
        ["Arquitectura:", latestTraining.architecture],
        ["Estado:", latestTraining.status],
        ["Resultado:", latestTraining.status === "Completado"
          ? `Loss ${latestTraining.loss.toFixed(3)}  ·  Accuracy ${latestTraining.accuracy.toFixed(1)}%`
          : "Sin resultado final"],
      ];
      for (const [label, value] of execLines) {
        doc.setFont("helvetica", "bold");
        doc.text(label, margin, y);
        doc.setFont("helvetica", "normal");
        doc.text(value, margin + 42, y);
        y += 7;
      }
    } else {
      doc.text("No se registran ejecuciones.", margin, y);
      y += 7;
    }

    // Sección 4 - Métricas por época
    if (epochData.length > 0) {
      y += 4;
      doc.setFontSize(12);
      doc.setFont("helvetica", "bold");
      doc.setTextColor(52, 120, 219);
      doc.text("4. Métricas por época", margin, y);
      y += 7;
      doc.setDrawColor(52, 120, 219);
      doc.line(margin, y, 210 - margin, y);
      y += 6;
      doc.setFontSize(9);
      doc.setFont("helvetica", "bold");
      doc.setTextColor(255, 255, 255);
      doc.setFillColor(52, 120, 219);
      doc.rect(margin, y - 4, 170, 7, "F");
      doc.text("Época", margin + 2, y);
      doc.text("Loss", margin + 40, y);
      doc.text("Accuracy (%)", margin + 90, y);
      y += 5;
      doc.setTextColor(30, 41, 59);
      doc.setFont("helvetica", "normal");
      for (const point of epochData) {
        if (y > 270) { doc.addPage(); y = 20; }
        doc.setFillColor(248, 250, 252);
        doc.rect(margin, y - 4, 170, 6, "F");
        doc.text(String(point.epoch), margin + 2, y);
        doc.text(point.loss.toFixed(4), margin + 40, y);
        doc.text(`${point.accuracy.toFixed(2)}%`, margin + 90, y);
        y += 6;
      }
    }

    // Pie de página
    const pageCount = doc.getNumberOfPages();
    for (let i = 1; i <= pageCount; i++) {
      doc.setPage(i);
      doc.setFontSize(8);
      doc.setTextColor(148, 163, 184);
      doc.text(
        "Generado por NeuroLab Enterprise · Modo de demostración · Datos almacenados localmente",
        105, 290, { align: "center" }
      );
      doc.text(`Página ${i} de ${pageCount}`, 210 - margin, 290, { align: "right" });
    }

    doc.save("reporte-neurolab.pdf");
    setNotice("Se descargó el reporte en PDF.");
  };

  const addUser = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!newUserName.trim() || !newUserEmail.trim()) return;
    setUsers((current) => [...current, { id: Date.now(), name: newUserName.trim(), email: newUserEmail.trim(), role: newUserRole, status: "Activo" }]);
    setNewUserName("");
    setNewUserEmail("");
    setShowUserForm(false);
    setNotice("Usuario agregado al entorno de demostración.");
  };

  const activationResult = useMemo(() => {
    const x = Number(activationInput);
    if (!Number.isFinite(x)) return "Ingresa un número válido";
    // Clamp para evitar Infinity/NaN con valores extremos
    const xc = Math.max(-500, Math.min(500, x));
    if (activationName === "ReLU")    return Math.max(0, xc).toFixed(4);
    if (activationName === "Sigmoid") return (1 / (1 + Math.exp(-xc))).toFixed(4);
    if (activationName === "Tanh")    return Math.tanh(xc).toFixed(4);
    if (activationName === "Softmax") {
      // Softmax real sobre vector [x, 0, 0] — muestra el peso de x frente a otras 2 neuronas en 0
      const expX = Math.exp(xc);
      const sum  = expX + Math.exp(0) + Math.exp(0); // 3 neuronas: x, 0, 0
      return (expX / sum).toFixed(4) + " (en vector [x, 0, 0])";
    }
    return xc.toFixed(4);
  }, [activationInput, activationName]);

  const renderDashboard = () => (
    <>
      <SectionHeader eyebrow="CENTRO DE EXPERIMENTACIÓN" title="Bienvenido a NeuroLab Enterprise" description="Tu espacio para explorar, diseñar y comprender redes neuronales artificiales." actions={<button className="button-primary" onClick={() => selectModule("Diseñador de arquitecturas")}><Network size={16} /> Abrir diseñador</button>} />
      <section className="panel dashboard-network-panel">
        <div className="panel-heading">
          <div>
            <span className="toolbar-label">ARQUITECTURA EN EDICIÓN</span>
            <h3>{architectureName}</h3>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            {trainingRunning && (
              <span className="training-live-label">
                <span className="live-dot" />
                ENTRENANDO
              </span>
            )}
            <span className="family-tag"><BrainCircuit size={14} />{networkFamily}</span>
          </div>
        </div>
        <div className="dashboard-network-canvas">
          <NetworkCanvas3D
            layers={layers}
            isTraining={trainingRunning}
            trainingCompleted={activeTraining?.status === "Completado"}
            epochProgress={trainingProgress}
            currentEpoch={epochData.length > 0 ? epochData[epochData.length - 1].epoch : undefined}
            currentLoss={epochData.length > 0 ? epochData[epochData.length - 1].loss : undefined}
            currentAccuracy={epochData.length > 0 ? epochData[epochData.length - 1].accuracy : undefined}
          />
        </div>
        <div className="dashboard-network-footer">
          <span>{layers.length} capas · {totalNeurons} neuronas · {hiddenCount} capas ocultas</span>
          <button className="text-button" onClick={() => selectModule("Diseñador de arquitecturas")}>
            Modificar arquitectura <ArrowRight size={15} />
          </button>
        </div>
      </section>
      <div className="dashboard-lower-grid">
        <section className="panel info-panel"><div className="panel-heading"><div><span className="toolbar-label">CONTINUAR TRABAJANDO</span><h3>Módulos de la plataforma</h3></div><Layers size={18} /></div>
          <div className="quick-module-list">{navigation.flatMap((group) => group.items).filter((item) => item.name !== "Dashboard").slice(0, 6).map((item) => { const Icon = item.icon; return <button key={item.name} onClick={() => selectModule(item.name)}><Icon size={18} /><span>{item.name}</span><ArrowRight size={14} /></button>; })}</div>
        </section>
        <section className="panel info-panel"><div className="panel-heading"><div><span className="toolbar-label">RESUMEN DEL ENTORNO</span><h3>Actividad registrada</h3></div><Activity size={18} /></div>
          <div className="summary-list"><div><span>Proyectos</span><strong>{projects.length}</strong></div><div><span>Datasets</span><strong>{datasets.length}</strong></div><div><span>Entrenamientos</span><strong>{trainingHistory.length}</strong></div><div><span>Versiones de modelos</span><strong>{models.length}</strong></div></div>
          <p className="muted-note">Los datos y resultados de esta versión se guardan en el almacenamiento local del navegador.</p>
        </section>
      </div>
    </>
  );

  const renderProjects = () => (
    <>
      <SectionHeader eyebrow="GESTIÓN DE EXPERIMENTOS" title="Proyectos de IA" description={moduleDescriptions["Proyectos de IA"]} actions={<button className="button-primary" onClick={() => { setEditingProjectId(null); setProjectName(""); setProjectProblem(""); setShowProjectForm((value) => !value); }}><Plus size={16} /> Nuevo proyecto</button>} />
      {showProjectForm && <form className="panel form-panel" onSubmit={submitProject}><div className="panel-heading"><div><span className="toolbar-label">DATOS DEL PROYECTO</span><h3>{editingProjectId ? "Editar proyecto" : "Crear nuevo proyecto"}</h3></div><button type="button" className="icon-button" onClick={() => setShowProjectForm(false)}><X size={17} /></button></div><div className="form-grid"><label>Nombre del proyecto<input required value={projectName} onChange={(event) => setProjectName(event.target.value)} placeholder="Ej. Clasificación de imágenes" /></label><label className="field-wide">Problema que se desea resolver<textarea required rows={3} value={projectProblem} onChange={(event) => setProjectProblem(event.target.value)} placeholder="Describe qué se busca resolver con inteligencia artificial..." /></label></div><div className="form-actions"><button type="button" className="button-secondary" onClick={() => setShowProjectForm(false)}>Cancelar</button><button className="button-primary" type="submit"><Save size={15} /> Guardar proyecto</button></div></form>}
      <section className="panel table-panel"><div className="panel-heading"><div><span className="toolbar-label">WORKSPACE</span><h3>Proyectos registrados</h3></div><div className="search-field"><Search size={15} /><input value={projectSearch} onChange={(event) => setProjectSearch(event.target.value)} placeholder="Buscar proyecto..." /></div></div>
        {filteredProjects.length ? <div className="table-scroll"><table><thead><tr><th>Proyecto</th><th>Problema</th><th>Responsable</th><th>Estado</th><th>Creación</th><th>Acciones</th></tr></thead><tbody>{filteredProjects.map((project) => <tr key={project.id}><td><strong>{project.name}</strong><small>ID-{project.id}</small></td><td className="problem-cell">{project.problem}</td><td>{project.owner}</td><td><StatusPill>{project.status}</StatusPill></td><td>{project.createdAt}</td><td><div className="table-actions"><button className="icon-button" title="Editar" onClick={() => editProject(project)}><Pencil size={15} /></button><button className="icon-button danger-icon" title="Eliminar" onClick={() => deleteProject(project.id)}><Trash2 size={15} /></button></div></td></tr>)}</tbody></table></div> : <EmptyState title="No hay proyectos" description="Crea un proyecto para organizar una experimentación de IA." />}
      </section>
      <div className="detail-note"><CircleHelp size={17} /><span>Un proyecto funciona como espacio de trabajo para organizar datasets, arquitecturas y ejecuciones de entrenamiento.</span></div>
    </>
  );

  const renderFundamentals = () => {
    const concepts = [
      { title: "Neurona", description: "Unidad básica que recibe valores de entrada, los combina y produce una salida.", formula: "salida = activación(Σ entrada × peso + sesgo)" },
      { title: "Capas", description: "Agrupan neuronas. La entrada recibe datos, las ocultas transforman la información y la salida entrega la predicción.", formula: "Entrada → Ocultas → Salida" },
      { title: "Conexiones y pesos", description: "Las conexiones transmiten valores entre neuronas. Cada peso determina la influencia de una entrada.", formula: "Σ (xᵢ × wᵢ)" },
      { title: "Sesgo (bias)", description: "Valor adicional que permite ajustar el resultado de una neurona antes de aplicar la activación.", formula: "z = Σ (xᵢ × wᵢ) + b" },
    ];
    return <>
      <SectionHeader eyebrow="APRENDIZAJE Y CONCEPTOS" title="Fundamentos neuronales" description={moduleDescriptions["Fundamentos neuronales"]} />
      <section className="panel fundamentals-network"><div className="panel-heading"><div><span className="toolbar-label">ESTRUCTURA GENERAL</span><h3>¿Cómo se organiza una red neuronal?</h3></div><BrainCircuit size={20} /></div><div className="dashboard-network-canvas"><NetworkDiagram layers={layers} compact /></div><div className="fundamentals-caption"><span><i className="legend-dot input-color" /> Entrada</span><ArrowRight size={14} /><span><i className="legend-dot hidden-color" /> Procesamiento en capas ocultas</span><ArrowRight size={14} /><span><i className="legend-dot output-color" /> Salida</span></div></section>
      <div className="concept-grid">{concepts.map((concept, index) => <article className="panel concept-card" key={concept.title}><span className="concept-index">0{index + 1}</span><h3>{concept.title}</h3><p>{concept.description}</p><div className="formula-box">{concept.formula}</div></article>)}</div>
      <section className="panel activation-lab"><div className="panel-heading"><div><span className="toolbar-label">EXPERIMENTACIÓN GUIADA</span><h3>Explorador de funciones de activación</h3></div><Zap size={18} /></div><div className="activation-controls"><label>Función<select value={activationName} onChange={(event) => setActivationName(event.target.value)}><option>ReLU</option><option>Sigmoid</option><option>Tanh</option><option>Softmax</option></select></label><label>Valor de entrada (x)<input type="number" step="0.1" value={activationInput} onChange={(event) => setActivationInput(event.target.value)} /></label><div className="activation-result"><span>Resultado aproximado</span><strong>{activationResult}</strong></div></div><p className="muted-note">ReLU, Sigmoid y Tanh operan sobre un solo valor. Softmax opera sobre un vector — el resultado muestra el peso de x frente a otras 2 neuronas con valor 0 en un vector de 3 clases.</p></section>
    </>;
  };

  const renderDesigner = () => (
    <>
      <SectionHeader eyebrow="LABORATORIO DE INTELIGENCIA ARTIFICIAL" title="Diseñador de arquitecturas" description={moduleDescriptions["Diseñador de arquitecturas"]} actions={<><button className="button-secondary" onClick={resetArchitecture}><RotateCcw size={15} /> Restablecer</button><button className="button-primary" onClick={saveArchitecture}><Save size={15} /> Guardar red</button></>} />
      <div className="designer-name-row"><label>Nombre de la arquitectura<input value={architectureName} onChange={(event) => setArchitectureName(event.target.value)} /></label><label>Familia de red<select value={networkFamily} onChange={(event) => setNetworkFamily(event.target.value)}><option>Feedforward</option><option>Recurrente (RNN)</option><option>Convolucional (CNN)</option><option>GAN</option></select></label><span className="family-tag"><Network size={14} />{networkFamily}</span></div>
      <div className="builder-layout"><section className="panel network-workspace"><div className="panel-heading"><div><span className="toolbar-label">VISTA ESTRUCTURAL</span><h3>{architectureName || "Red neuronal sin nombre"}</h3></div><span className="live-tag"><span /> Vista editable</span></div><div className="network-canvas"><NetworkDiagram layers={layers} selected={selectedNeuron} onSelect={(selection) => { setActiveLayerId(selection.layerId); setSelectedNeuron(selection); }} /></div><div className="network-legend"><span><i className="legend-dot input-color" /> Entrada</span><span><i className="legend-dot hidden-color" /> Capas ocultas</span><span><i className="legend-dot output-color" /> Salida</span><span><i className="legend-ring" /> Seleccionada</span></div><div className="architecture-footer"><div><span className="footer-label">ESTRUCTURA</span><strong>{layers.map((layer) => layer.neurons).join(" — ")}</strong></div><div><span className="footer-label">CAPAS</span><strong>{layers.length}</strong></div><div><span className="footer-label">NEURONAS</span><strong>{totalNeurons}</strong></div><div><span className="footer-label">CONEXIONES</span><strong>{layers.slice(0, -1).reduce((sum, layer, index) => sum + layer.neurons * layers[index + 1].neurons, 0)}</strong></div></div></section>
        <aside className="panel properties-panel"><div className="panel-heading"><div><span className="toolbar-label">CONFIGURACIÓN</span><h3>Propiedades de capa</h3></div><SlidersHorizontal size={18} /></div><label className="stacked-field">Seleccionar capa<select value={activeLayer.id} onChange={(event) => { const id = Number(event.target.value); setActiveLayerId(id); setSelectedNeuron({ layerId: id, index: 0 }); }}>{layers.map((layer) => <option key={layer.id} value={layer.id}>{layer.name}</option>)}</select></label><div className="selected-neuron-card"><div className={`selected-neuron-icon ${activeLayer.kind}`}><Network size={20} /></div><div><strong>Neurona {selectedNeuron.index + 1}</strong><span>{activeLayer.name}</span></div></div><div className="property-block"><div className="property-label-row"><label>Número de neuronas</label><strong>{activeLayer.neurons}</strong></div><div className="number-control"><button disabled={activeLayer.neurons <= 1} onClick={() => changeNeuronCount(-1)}><span>−</span></button><strong>{activeLayer.neurons} neuronas</strong><button disabled={activeLayer.neurons >= 12} onClick={() => changeNeuronCount(1)}><Plus size={15} /></button></div><small>Rango permitido en esta demostración: 1 a 12.</small></div><label className="stacked-field">Función de activación<select value={activeLayer.activation} disabled={activeLayer.kind === "input"} onChange={(event) => updateActiveLayer({ activation: event.target.value })}><option>Ninguna</option><option>ReLU</option><option>Sigmoid</option><option>Tanh</option><option>Softmax</option></select></label><div className="layer-kind-label"><Layers size={16} /><span>Tipo: {activeLayer.kind === "input" ? "Entrada" : activeLayer.kind === "output" ? "Salida" : "Capa oculta"}</span></div><div className="panel-separator" /><button className="button-primary full-button" onClick={addLayer}><Plus size={16} /> Agregar capa oculta</button><button className="button-danger full-button" disabled={activeLayer.kind !== "hidden"} onClick={removeActiveLayer}><Trash2 size={15} /> Eliminar capa seleccionada</button><p className="muted-note">Selecciona una neurona en el lienzo para identificarla. Los cambios actualizan la estructura visible.</p></aside></div>
      <div className="detail-note"><CircleHelp size={17} /><span>
        {networkFamily === "Feedforward"
          ? "Red Feedforward (MLP): las señales fluyen en una sola dirección, entrada → capas ocultas → salida. Compatible con datos tabulares. TF.js entrenará este modelo con capas densas."
          : networkFamily === "Recurrente (RNN)"
          ? "Red Recurrente (RNN): diseñada para datos secuenciales o temporales (texto, series de tiempo). En esta plataforma el entrenamiento usa capas densas como aproximación; una implementación completa requeriría capas LSTM o GRU."
          : networkFamily === "Convolucional (CNN)"
          ? "Red Convolucional (CNN): diseñada para imágenes y datos con estructura espacial. En esta plataforma el entrenamiento usa capas densas como aproximación; una CNN completa requiere capas Conv2D y Pooling con datasets de imágenes."
          : "GAN (Red Generativa Adversarial): compuesta por un generador y un discriminador que compiten entre sí. Requiere una arquitectura y proceso de entrenamiento especializado. Esta vista muestra la estructura base de referencia."
        }
      </span></div>
    </>
  );

  const renderDatasets = () => (
    <>
      <SectionHeader eyebrow="GESTIÓN DE DATOS" title="Datasets" description={moduleDescriptions.Datasets} />
      <section className="panel upload-panel"><div className="upload-intro"><div className="upload-icon"><Upload size={23} /></div><div><h3>Incorporar un dataset CSV</h3><p>Selecciona un archivo CSV o TXT delimitado por comas o punto y coma. La aplicación mostrará una vista previa local.</p></div></div><div className="upload-fields"><label>Descripción del dataset<input value={datasetDescription} onChange={(event) => setDatasetDescription(event.target.value)} placeholder="Ej. Registros de ventas por mes" /></label><label>Variable objetivo (opcional)<input value={datasetTarget} onChange={(event) => setDatasetTarget(event.target.value)} placeholder="Ej. categoria" /></label><label className="file-picker"><Upload size={16} /> Seleccionar archivo CSV<input type="file" accept=".csv,.txt,text/csv,text/plain" onChange={handleDatasetUpload} /></label></div><p className="muted-note">Los archivos se leen en el navegador. Se guardan hasta 500 filas para entrenamiento con TensorFlow.js y una vista previa de los primeros registros.</p></section>
      <div className="dataset-grid">{datasets.map((dataset) => <article className="panel dataset-card" key={dataset.id}><div className="dataset-card-top"><div className="dataset-file-icon"><FileText size={20} /></div><button className="icon-button danger-icon" onClick={() => removeDataset(dataset.id)} title="Eliminar dataset"><Trash2 size={15} /></button></div><h3>{dataset.name}</h3><p>{dataset.description}</p><div className="dataset-meta"><span><Database size={14} /> {dataset.rows} filas</span><span><Layers size={14} /> {dataset.columns.length} columnas</span></div><div className="dataset-target"><span>Variable objetivo</span><strong>{dataset.target || "No definida"}</strong></div><div className="dataset-card-footer"><span>{dataset.source}</span><button className="text-button" onClick={() => { setTrainingDataset(dataset.name); setNotice(`${dataset.name} seleccionado para entrenamiento.`); }}>Usar dataset <ArrowRight size={14} /></button></div></article>)}</div>
      {datasets.length === 0 && <EmptyState title="No hay datasets registrados" description="Carga un archivo CSV para comenzar." />}
      {datasets.map((dataset) => <section className="panel preview-panel" key={`preview-${dataset.id}`}><div className="panel-heading"><div><span className="toolbar-label">VISTA PREVIA</span><h3>{dataset.name}</h3></div><span className="soft-tag">División propuesta: {dataset.split}</span></div><div className="table-scroll"><table><thead><tr>{dataset.columns.map((column) => <th key={column}>{column}</th>)}</tr></thead><tbody>{dataset.preview.slice(0, 10).map((row, index) => <tr key={`${dataset.id}-${index}`}>{dataset.columns.map((column, cellIndex) => <td key={`${column}-${cellIndex}`}>{row[cellIndex] ?? "—"}</td>)}</tr>)}</tbody></table></div><p className="muted-note">Mostrando 10 de {dataset.rows} filas. Total disponible para entrenamiento: {dataset.preview.length} filas.</p></section>)}
    </>
  );

  const renderTraining = () => (
    <>
      <SectionHeader eyebrow="EJECUCIÓN DE APRENDIZAJE" title="Entrenamientos" description={moduleDescriptions.Entrenamientos} />
      <div className="training-layout"><section className="panel training-config"><div className="panel-heading"><div><span className="toolbar-label">NUEVA EJECUCIÓN</span><h3>Configuración del entrenamiento</h3></div><Settings size={18} /></div><label className="stacked-field">Proyecto<select value={trainingProject} onChange={(event) => setTrainingProject(event.target.value)}>{projects.map((project) => <option key={project.id}>{project.name}</option>)}</select></label><label className="stacked-field">Dataset<select value={trainingDataset} onChange={(event) => setTrainingDataset(event.target.value)}>{datasets.map((dataset) => <option key={dataset.id}>{dataset.name}</option>)}</select></label><div className="training-readonly"><span>Arquitectura asociada</span><strong>{networkFamily} · {layers.map((layer) => layer.neurons).join("-")}</strong><button className="text-button" onClick={() => selectModule("Diseñador de arquitecturas")}>Editar arquitectura</button></div><div className="training-parameter-grid"><label>Épocas<input type="number" min="1" max="100" value={epochs} onChange={(event) => setEpochs(Math.max(1, Math.min(100, Number(event.target.value))))} /></label><label>Batch size<select value={batchSize} onChange={(event) => setBatchSize(Number(event.target.value))}><option value={8}>8</option><option value={16}>16</option><option value={32}>32</option><option value={64}>64</option></select></label><label>Optimizador<select value={optimizer} onChange={(event) => setOptimizer(event.target.value)}><option>Adam</option><option>SGD</option><option>RMSprop</option><option>Adagrad</option></select></label></div><div className="training-actions"><button className="button-primary" disabled={trainingRunning || projects.length === 0 || datasets.length === 0} onClick={() => { void startTraining(); }}><Play size={16} /> Iniciar entrenamiento</button><button className="button-secondary" disabled={!canCancel} onClick={cancelTraining} style={canCancel ? { opacity: 1, cursor: "pointer", borderColor: "#ef4444", color: "#ef4444" } : {}}><X size={15} /> Cancelar</button></div><div className="simulation-warning"><CircleHelp size={16} /><span>Entrenamiento real con TensorFlow.js: se entrena directamente en tu navegador con el dataset cargado.</span></div>{trainingError && <div className="simulation-warning" style={{color:"#e05c5c"}}><CircleHelp size={16} /><span>Error: {trainingError}</span></div>}</section>
        <section className={`panel execution-panel ${trainingRunning ? "is-training" : ""}`}>
          <div className="panel-heading">
            <div><span className="toolbar-label">SEGUIMIENTO</span><h3>Estado de ejecución</h3></div>
            <span className={`execution-dot ${trainingRunning ? "running" : ""}`} />
          </div>

          {/* Red neuronal — animación sincronizada con TF.js */}
          <NetworkCanvas3D
            layers={layers}
            isTraining={trainingRunning}
            trainingCompleted={activeTraining?.status === "Completado"}
            epochProgress={trainingProgress}
            currentEpoch={epochData.length > 0 ? epochData[epochData.length - 1].epoch : undefined}
            currentLoss={epochData.length > 0 ? epochData[epochData.length - 1].loss : undefined}
            currentAccuracy={epochData.length > 0 ? epochData[epochData.length - 1].accuracy : undefined}
          />

          <div className="execution-state">
            <div className="execution-state-icon">
              <span className={trainingRunning ? "brain-training" : ""}>
                <BrainCircuit size={28} />
              </span>
            </div>
            <strong>{trainingRunning ? "Entrenamiento en curso" : activeTraining?.status === "Completado" ? "Entrenamiento completado" : activeTraining?.status === "Cancelado" ? "Ejecución cancelada" : "Listo para ejecutar"}</strong>
            <span>{trainingRunning ? "TensorFlow.js está ajustando los pesos de la red en tu navegador." : "Configura los parámetros e inicia el entrenamiento real."}</span>
          </div>

          <div className="progress-heading">
            <span className={`epoch-counter ${trainingRunning ? "active" : ""}`}>Progreso de épocas</span>
            <strong>{trainingProgress}%</strong>
          </div>
          <div className="progress-track">
            <div className="progress-value" style={{ width: `${trainingProgress}%` }} />
          </div>

          <div className="execution-details">
            <div><span>Proyecto</span><strong>{(activeTraining?.project ?? trainingProject) || "Sin seleccionar"}</strong></div>
            <div><span>Dataset</span><strong>{(activeTraining?.dataset ?? trainingDataset) || "Sin seleccionar"}</strong></div>
            <div><span className={`epoch-counter ${trainingRunning ? "active" : ""}`}>Épocas configuradas</span><strong>{activeTraining?.epochs ?? epochs}</strong></div>
            <div><span>Optimizador</span><strong>{activeTraining?.optimizer ?? optimizer}</strong></div>
          </div>

          {/* Métricas en vivo de la última época */}
          {trainingRunning && epochData.length > 0 && (
            <div className="completion-message metric-live" style={{ background: "rgba(52,120,219,0.07)", border: "1px solid rgba(52,120,219,0.2)" }}>
              <Activity size={17} />
              Época {epochData[epochData.length - 1].epoch} / {epochs} —
              Loss: <strong>{epochData[epochData.length - 1].loss.toFixed(4)}</strong> ·
              Accuracy: <strong>{epochData[epochData.length - 1].accuracy.toFixed(1)}%</strong>
            </div>
          )}

          {activeTraining?.status === "Completado" && (
            <div className="completion-message">
              <CheckCircle2 size={17} />
              Resultado final (Test set): accuracy {activeTraining.accuracy.toFixed(1)}% · loss {activeTraining.loss.toFixed(4)}.
            </div>
          )}
        </section></div>
      <section className="panel table-panel"><div className="panel-heading"><div><span className="toolbar-label">HISTORIAL</span><h3>Ejecuciones registradas</h3></div><span className="soft-tag">{trainingHistory.length} registros</span></div><div className="table-scroll"><table><thead><tr><th>Proyecto</th><th>Dataset</th><th>Arquitectura</th><th>Estado</th><th>Loss</th><th>Accuracy</th><th>Fecha</th></tr></thead><tbody>{trainingHistory.map((training) => <tr key={training.id}><td><strong>{training.project}</strong></td><td>{training.dataset}</td><td>{training.architecture}</td><td><StatusPill>{training.status}</StatusPill></td><td>{training.status === "Completado" ? training.loss.toFixed(3) : "—"}</td><td>{training.status === "Completado" ? `${training.accuracy.toFixed(1)}%` : "—"}</td><td>{training.createdAt}</td></tr>)}</tbody></table></div></section>
    </>
  );

  const renderEvaluation = () => (
    <>
      <SectionHeader eyebrow="REVISIÓN DE RESULTADOS" title="Evaluación y métricas" description={moduleDescriptions["Evaluación y métricas"]} />
      {latestTraining && latestTraining.status === "Completado" ? <>
        <section className="panel evaluation-summary">
          <div className="panel-heading"><div><span className="toolbar-label">ÚLTIMA EJECUCIÓN COMPLETADA</span><h3>{latestTraining.project}</h3></div><StatusPill>Completado</StatusPill></div>
          <div className="evaluation-values">
            <div><span>Accuracy</span><strong>{latestTraining.accuracy.toFixed(1)}%</strong><small>Precisión demostrativa</small></div>
            <div><span>Loss</span><strong>{latestTraining.loss.toFixed(3)}</strong><small>Pérdida demostrativa</small></div>
            <div><span>Épocas</span><strong>{latestTraining.epochs}</strong><small>Configuradas</small></div>
            <div><span>Optimizador</span><strong>{latestTraining.optimizer}</strong><small>Parámetro usado</small></div>
          </div>
        </section>

        {epochData.length > 0 && (
          <div className="evaluation-grid">
            <section className="panel info-panel">
              <div className="panel-heading"><div><span className="toolbar-label">CURVA DE APRENDIZAJE</span><h3>Pérdida por época (Loss)</h3></div><Activity size={18} /></div>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={epochData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="epoch" label={{ value: "Época", position: "insideBottom", offset: -2 }} tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} domain={[0, 1]} />
                  <Tooltip formatter={(v: number) => v.toFixed(4)} />
                  <Legend />
                  <Line type="monotone" dataKey="loss" stroke="#e05c5c" strokeWidth={2} dot={false} name="Loss" />
                </LineChart>
              </ResponsiveContainer>
            </section>
            <section className="panel info-panel">
              <div className="panel-heading"><div><span className="toolbar-label">CURVA DE APRENDIZAJE</span><h3>Precisión por época (Accuracy)</h3></div><Activity size={18} /></div>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={epochData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="epoch" label={{ value: "Época", position: "insideBottom", offset: -2 }} tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} domain={[0, 100]} unit="%" />
                  <Tooltip formatter={(v: number) => `${v.toFixed(2)}%`} />
                  <Legend />
                  <Line type="monotone" dataKey="accuracy" stroke="#0d9b72" strokeWidth={2} dot={false} name="Accuracy %" />
                </LineChart>
              </ResponsiveContainer>
            </section>
          </div>
        )}

        <div className="evaluation-grid">
          <section className="panel info-panel">
            <div className="panel-heading"><div><span className="toolbar-label">INTERPRETACIÓN</span><h3>Lectura de resultados</h3></div><BookOpen size={18} /></div>
            <div className="interpretation-list">
              <div><CheckCircle2 size={17} /><p><strong>Accuracy:</strong> representa la proporción de predicciones correctas en la evaluación de demostración.</p></div>
              <div><Activity size={17} /><p><strong>Loss:</strong> representa el valor de error utilizado como referencia para ilustrar el aprendizaje.</p></div>
              <div><Database size={17} /><p><strong>Dataset:</strong> {latestTraining.dataset}.</p></div>
              <div><Network size={17} /><p><strong>Arquitectura:</strong> {latestTraining.architecture}.</p></div>
            </div>
          </section>
          <section className="panel info-panel">
            <div className="panel-heading"><div><span className="toolbar-label">EVALUACIÓN CONCEPTUAL</span><h3>Elementos que deben revisarse</h3></div><FlaskConical size={18} /></div>
            <ul className="check-list">
              <li><Check size={15} /> Separación de datos de entrenamiento, validación y prueba.</li>
              <li><Check size={15} /> Comparación entre pérdida y precisión.</li>
              <li><Check size={15} /> Revisión de la clase objetivo del dataset.</li>
              <li><Check size={15} /> Registro de limitaciones y observaciones.</li>
            </ul>
          </section>
        </div>
      </> : <EmptyState title="Todavía no hay una ejecución completada" description="Ve a Entrenamientos, inicia una simulación y vuelve aquí para consultar los resultados de demostración." action={<button className="button-primary" onClick={() => selectModule("Entrenamientos")}>Ir a entrenamientos <ArrowRight size={15} /></button>} />}
      <section className="panel table-panel">
        <div className="panel-heading"><div><span className="toolbar-label">HISTORIAL DE EVALUACIÓN</span><h3>Resultados de las ejecuciones</h3></div></div>
        <div className="table-scroll"><table><thead><tr><th>Proyecto</th><th>Arquitectura</th><th>Dataset</th><th>Estado</th><th>Loss</th><th>Accuracy</th></tr></thead><tbody>{completedTrainings.map((training) => <tr key={training.id}><td>{training.project}</td><td>{training.architecture}</td><td>{training.dataset}</td><td><StatusPill>{training.status}</StatusPill></td><td>{training.loss.toFixed(3)}</td><td>{training.accuracy.toFixed(1)}%</td></tr>)}</tbody></table></div>
      </section>
    </>
  );

  const renderModels = () => (
    <>
      <SectionHeader eyebrow="GESTIÓN DE VERSIONES" title="Registro de modelos" description={moduleDescriptions["Registro de modelos"]} actions={<button className="button-primary" onClick={createModelVersion}><Plus size={16} /> Registrar versión</button>} />
      <div className="detail-note"><Boxes size={17} /><span>El registro conserva una ficha de cada versión. En esta modalidad se guardan metadatos demostrativos; no se genera un archivo real de pesos .keras o .h5.</span></div>
      <section className="panel table-panel"><div className="panel-heading"><div><span className="toolbar-label">MODEL REGISTRY</span><h3>Versiones registradas</h3></div><span className="soft-tag">{models.length} versiones</span></div>{models.length ? <div className="table-scroll"><table><thead><tr><th>Modelo</th><th>Versión</th><th>Arquitectura</th><th>Dataset</th><th>Accuracy</th><th>Estado</th><th>Creación</th></tr></thead><tbody>{models.map((model) => <tr key={model.id}><td><strong>{model.name}</strong></td><td><span className="version-tag">{model.version}</span></td><td>{model.architecture}</td><td>{model.dataset}</td><td>{model.accuracy ? `${model.accuracy.toFixed(1)}%` : "Pendiente"}</td><td><select className="inline-select" value={model.status} onChange={(event) => updateModelStatus(model.id, event.target.value as ModelRecord["status"])}><option>Borrador</option><option>En revisión</option><option>Aprobado</option><option>Retirado</option></select></td><td>{model.createdAt}</td></tr>)}</tbody></table></div> : <EmptyState title="Sin modelos registrados" description="Guarda una versión de tu arquitectura para incorporarla al registro." />}</section>
      <section className="panel model-lifecycle"><div className="panel-heading"><div><span className="toolbar-label">CICLO DE VIDA</span><h3>Estados de una versión</h3></div></div><div className="lifecycle-steps"><div><span>01</span><strong>Borrador</strong><small>Versión creada</small></div><ArrowRight size={16} /><div><span>02</span><strong>En revisión</strong><small>Revisión de resultados</small></div><ArrowRight size={16} /><div><span>03</span><strong>Aprobado</strong><small>Versión aceptada</small></div><ArrowRight size={16} /><div><span>04</span><strong>Retirado</strong><small>Fuera de uso</small></div></div></section>
    </>
  );

  const renderReports = () => (
    <>
      <SectionHeader eyebrow="DOCUMENTACIÓN DEL PROYECTO" title="Reportes" description={moduleDescriptions.Reportes} actions={<button className="button-primary" onClick={createReport}><Download size={16} /> Exportar reporte PDF</button>} />
      <section className="panel report-preview"><div className="report-document-head"><div className="brand-symbol report-brand"><BrainCircuit size={23} /></div><div><span>NEUROLAB ENTERPRISE</span><strong>Reporte técnico de demostración</strong></div><span className="report-date">{today()}</span></div><div className="report-section"><h3>1. Resumen de arquitectura</h3><p><strong>Nombre:</strong> {architectureName}</p><p><strong>Familia:</strong> {networkFamily}</p><p><strong>Capas:</strong> {layers.map((layer) => `${layer.name} (${layer.neurons} neuronas, ${layer.activation})`).join(" → ")}</p><p><strong>Total de neuronas:</strong> {totalNeurons}</p></div><div className="report-section"><h3>2. Recursos registrados</h3><div className="report-counts"><span>Proyectos<strong>{projects.length}</strong></span><span>Datasets<strong>{datasets.length}</strong></span><span>Entrenamientos<strong>{trainingHistory.length}</strong></span><span>Modelos<strong>{models.length}</strong></span></div></div><div className="report-section"><h3>3. Última ejecución</h3>{latestTraining ? <><p><strong>Proyecto:</strong> {latestTraining.project}</p><p><strong>Estado:</strong> {latestTraining.status}</p><p><strong>Dataset:</strong> {latestTraining.dataset}</p><p><strong>Resultado:</strong> {latestTraining.status === "Completado" ? `Loss ${latestTraining.loss.toFixed(3)} · Accuracy ${latestTraining.accuracy.toFixed(1)}%` : "Sin resultado final"}</p></> : <p>No se registran ejecuciones.</p>}</div><div className="report-disclaimer">Documento generado en modo de demostración. Los entrenamientos, las métricas y los registros son simulados y se almacenan localmente en el navegador.</div></section>
      <div className="detail-note"><FileText size={17} /><span>El botón de exportación descarga un archivo PDF con el resumen actual, arquitectura, métricas y tabla de épocas.</span></div>
    </>
  );

  const renderAdministration = () => (
    <>
      <SectionHeader eyebrow="GOBIERNO DE LA PLATAFORMA" title="Administración y seguridad" description={moduleDescriptions["Administración y seguridad"]} actions={<button className="button-primary" onClick={() => setShowUserForm((value) => !value)}><Plus size={16} /> Agregar usuario</button>} />
      <div className="security-overview"><div className="panel security-card"><LockKeyhole size={20} /><div><strong>Control por roles</strong><span>Acceso organizado por perfil</span></div></div><div className="panel security-card"><Users size={20} /><div><strong>{users.length} usuarios</strong><span>Registrados en esta demostración</span></div></div><div className="panel security-card"><ShieldCheck size={20} /><div><strong>Entorno local</strong><span>Sin autenticación de servidor</span></div></div></div>
      {showUserForm && <form className="panel form-panel" onSubmit={addUser}><div className="panel-heading"><div><span className="toolbar-label">NUEVO REGISTRO</span><h3>Agregar usuario</h3></div><button type="button" className="icon-button" onClick={() => setShowUserForm(false)}><X size={16} /></button></div><div className="form-grid"><label>Nombre completo<input required value={newUserName} onChange={(event) => setNewUserName(event.target.value)} placeholder="Nombre del usuario" /></label><label>Correo electrónico<input required type="email" value={newUserEmail} onChange={(event) => setNewUserEmail(event.target.value)} placeholder="usuario@neurolab.local" /></label><label>Rol<select value={newUserRole} onChange={(event) => setNewUserRole(event.target.value)}><option>Administrador</option><option>Responsable de proyecto</option><option>Ingeniero/analista de IA</option><option>Revisor</option><option>Lector</option></select></label></div><div className="form-actions"><button type="button" className="button-secondary" onClick={() => setShowUserForm(false)}>Cancelar</button><button className="button-primary" type="submit"><Save size={15} /> Guardar usuario</button></div></form>}
      <section className="panel table-panel"><div className="panel-heading"><div><span className="toolbar-label">CONTROL DE ACCESO</span><h3>Usuarios y roles</h3></div></div><div className="table-scroll"><table><thead><tr><th>Usuario</th><th>Correo</th><th>Rol</th><th>Estado</th><th>Acción</th></tr></thead><tbody>{users.map((user) => <tr key={user.id}><td><div className="user-cell"><span className="mini-avatar">{user.name.slice(0, 1).toUpperCase()}</span><strong>{user.name}</strong></div></td><td>{user.email}</td><td><select className="inline-select" value={user.role} onChange={(event) => setUsers((current) => current.map((entry) => entry.id === user.id ? { ...entry, role: event.target.value } : entry))}><option>Administrador</option><option>Responsable de proyecto</option><option>Ingeniero/analista de IA</option><option>Revisor</option><option>Lector</option></select></td><td><StatusPill>{user.status}</StatusPill></td><td><button className="button-small" onClick={() => setUsers((current) => current.map((entry) => entry.id === user.id ? { ...entry, status: entry.status === "Activo" ? "Suspendido" : "Activo" } : entry))}>{user.status === "Activo" ? "Suspender" : "Activar"}</button></td></tr>)}</tbody></table></div></section>
      <section className="panel roles-panel"><div className="panel-heading"><div><span className="toolbar-label">REFERENCIA DE PERMISOS</span><h3>Roles disponibles</h3></div></div><div className="role-grid"><div><strong>Administrador</strong><p>Administra usuarios, roles y configuración general.</p></div><div><strong>Responsable de proyecto</strong><p>Crea proyectos, asigna integrantes y administra modelos.</p></div><div><strong>Ingeniero/analista de IA</strong><p>Configura arquitecturas, ejecuta y evalúa modelos.</p></div><div><strong>Revisor</strong><p>Consulta resultados y revisa versiones de modelos.</p></div><div><strong>Lector</strong><p>Consulta los proyectos y reportes autorizados.</p></div></div></section>
      <div className="simulation-warning"><CircleHelp size={16} /><span>Este módulo representa la administración de forma local. Los roles no aplican seguridad real ni sustituyen autenticación y autorización de servidor.</span></div>
    </>
  );

  const renderModule = () => {
    switch (activeModule) {
      case "Dashboard": return renderDashboard();
      case "Proyectos de IA": return renderProjects();
      case "Fundamentos neuronales": return renderFundamentals();
      case "Diseñador de arquitecturas": return renderDesigner();
      case "Datasets": return renderDatasets();
      case "Entrenamientos": return renderTraining();
      case "Evaluación y métricas": return renderEvaluation();
      case "Registro de modelos": return renderModels();
      case "Reportes": return renderReports();
      case "Administración y seguridad": return renderAdministration();
      default: return renderDashboard();
    }
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><div className="brand-symbol"><BrainCircuit size={24} /></div><div><h1>NEUROLAB</h1><span>ENTERPRISE</span></div></div>
        <div className="workspace-selector"><div className="workspace-icon"><BrainCircuit size={19} /></div><div className="workspace-text"><strong>Neural Workspace</strong><span>Entorno de demostración</span></div><ChevronDown size={15} /></div>
        <nav className="sidebar-navigation">{navigation.map((group) => <div className="menu-group" key={group.title}><span className="menu-heading">{group.title}</span>{group.items.map((item) => { const Icon = item.icon; return <button key={item.name} className={`menu-item ${activeModule === item.name ? "active" : ""}`} onClick={() => selectModule(item.name)}><Icon size={17} strokeWidth={1.9} /><span>{item.name}</span></button>; })}</div>)}</nav>
        <div className="sidebar-bottom"><div className="system-status"><span className="status-dot" /><div><strong>Sistema operativo</strong><span>TensorFlow.js activo</span></div></div><div className="sidebar-user"><div className="user-avatar">AD</div><div className="user-data"><strong>Administrador</strong><span>admin@neurolab.local</span></div><Settings size={16} /></div></div>
      </aside>
      <main className="main-area"><header className="topbar"><div className="breadcrumb"><span>NeuroLab</span><span className="breadcrumb-separator">/</span><strong>{activeModule}</strong></div><div className="topbar-actions"><span className="simulation-badge"><i /> TF.js Local</span><button className="top-icon" title="Buscar módulos" onClick={() => selectModule("Proyectos de IA")}><Search size={18} /></button><button className="top-icon" title="Notificaciones" onClick={() => setNotice("No tienes notificaciones nuevas.")}><Bell size={18} /></button><div className="top-user"><div className="user-avatar">AD</div><span>Administrador</span></div></div></header><div className="page-content">{renderModule()}</div></main>
      {notice && <div className="toast-message"><CheckCircle2 size={17} />{notice}<button onClick={() => setNotice("")} aria-label="Cerrar aviso"><X size={15} /></button></div>}
    </div>
  );
}

function EmptyState({ title, description, action }: { title: string; description: string; action?: ReactNode }) {
  return <div className="panel empty-state"><div className="empty-state-icon"><Database size={24} /></div><h3>{title}</h3><p>{description}</p>{action}</div>;
}

export default App;
