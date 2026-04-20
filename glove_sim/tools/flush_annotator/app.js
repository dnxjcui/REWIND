import * as THREE from "https://unpkg.com/three@0.160.0/build/three.module.js";
import { OrbitControls } from "https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "https://unpkg.com/three@0.160.0/examples/jsm/loaders/GLTFLoader.js";

const canvas = document.getElementById("view");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio || 1);
renderer.setSize(window.innerWidth - 320, window.innerHeight);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x20242a);
const camera = new THREE.PerspectiveCamera(55, (window.innerWidth - 320) / window.innerHeight, 0.01, 100);
camera.position.set(0.2, 0.15, 0.25);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0);
controls.update();

scene.add(new THREE.AxesHelper(0.05));
scene.add(new THREE.GridHelper(0.4, 20));
scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 1.2));

let handRoot = null;
let gloveRoot = null;
const pickable = [];
const ray = new THREE.Raycaster();
const pointer = new THREE.Vector2();

const annotations = {
  schema_version: 1,
  reference_frame_idx: 119,
  hand_points: [],
  glove_points: [],
};
let activeTarget = "hand";
const markerStack = [];

const statusEl = document.getElementById("status");
const countsEl = document.getElementById("counts");
const frameIdxEl = document.getElementById("frameIdx");

function setStatus(msg) {
  statusEl.textContent = msg;
}

function refreshCounts() {
  countsEl.textContent =
    `active_target: ${activeTarget}\n` +
    `hand_points: ${annotations.hand_points.length}\n` +
    `glove_points: ${annotations.glove_points.length}\n` +
    `paired_points: ${Math.min(annotations.hand_points.length, annotations.glove_points.length)}`;
}

function addMarker(point, color) {
  const geo = new THREE.SphereGeometry(0.0025, 16, 16);
  const mat = new THREE.MeshStandardMaterial({ color });
  const m = new THREE.Mesh(geo, mat);
  m.position.copy(point);
  scene.add(m);
  markerStack.push(m);
}

function clearMarkers() {
  while (markerStack.length) {
    const m = markerStack.pop();
    scene.remove(m);
  }
}

function rebuildMarkers() {
  clearMarkers();
  for (const p of annotations.hand_points) addMarker(new THREE.Vector3(...p), 0x4caf50);
  for (const p of annotations.glove_points) addMarker(new THREE.Vector3(...p), 0xff5252);
}

function loadGLB(file, targetName) {
  const url = URL.createObjectURL(file);
  const loader = new GLTFLoader();
  loader.load(url, (gltf) => {
    const root = gltf.scene;
    root.traverse((obj) => {
      if (obj.isMesh) {
        obj.userData.targetName = targetName;
        pickable.push(obj);
      }
    });
    if (targetName === "hand") {
      if (handRoot) scene.remove(handRoot);
      handRoot = root;
    } else {
      if (gloveRoot) scene.remove(gloveRoot);
      gloveRoot = root;
    }
    scene.add(root);
    setStatus(`${targetName} GLB loaded.`);
  });
}

window.addEventListener("resize", () => {
  const w = window.innerWidth - 320;
  const h = window.innerHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
});

canvas.addEventListener("click", (ev) => {
  if (!handRoot || !gloveRoot) {
    setStatus("Load both hand and glove GLBs first.");
    return;
  }
  const rect = canvas.getBoundingClientRect();
  pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
  ray.setFromCamera(pointer, camera);
  const hits = ray.intersectObjects(pickable, true);
  if (!hits.length) return;
  const hit = hits.find((h) => h.object.userData.targetName === activeTarget);
  if (!hit) {
    setStatus(`Click landed on wrong mesh. Active target: ${activeTarget}`);
    return;
  }
  const p = [hit.point.x, hit.point.y, hit.point.z];
  if (activeTarget === "hand") annotations.hand_points.push(p);
  else annotations.glove_points.push(p);
  addMarker(hit.point, activeTarget === "hand" ? 0x4caf50 : 0xff5252);
  refreshCounts();
});

document.getElementById("handFile").addEventListener("change", (e) => {
  const f = e.target.files?.[0];
  if (f) loadGLB(f, "hand");
});
document.getElementById("gloveFile").addEventListener("change", (e) => {
  const f = e.target.files?.[0];
  if (f) loadGLB(f, "glove");
});
document.getElementById("targetHand").addEventListener("click", () => {
  activeTarget = "hand";
  refreshCounts();
});
document.getElementById("targetGlove").addEventListener("click", () => {
  activeTarget = "glove";
  refreshCounts();
});
document.getElementById("undoBtn").addEventListener("click", () => {
  if (activeTarget === "hand" && annotations.hand_points.length) annotations.hand_points.pop();
  if (activeTarget === "glove" && annotations.glove_points.length) annotations.glove_points.pop();
  rebuildMarkers();
  refreshCounts();
});
document.getElementById("clearBtn").addEventListener("click", () => {
  annotations.hand_points = [];
  annotations.glove_points = [];
  rebuildMarkers();
  refreshCounts();
});
document.getElementById("saveBtn").addEventListener("click", () => {
  annotations.reference_frame_idx = Number(frameIdxEl.value || 119);
  const blob = new Blob([JSON.stringify(annotations, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "annotations.json";
  a.click();
  setStatus("Downloaded annotations.json. Place it in glove_sim/outputs/knot_alignment/.");
});
document.getElementById("loadAnn").addEventListener("change", async (e) => {
  const f = e.target.files?.[0];
  if (!f) return;
  const data = JSON.parse(await f.text());
  annotations.schema_version = data.schema_version ?? 1;
  annotations.reference_frame_idx = data.reference_frame_idx ?? 119;
  annotations.hand_points = data.hand_points ?? [];
  annotations.glove_points = data.glove_points ?? [];
  frameIdxEl.value = String(annotations.reference_frame_idx);
  rebuildMarkers();
  refreshCounts();
  setStatus("Loaded annotation JSON.");
});

refreshCounts();

function tick() {
  requestAnimationFrame(tick);
  renderer.render(scene, camera);
}
tick();
