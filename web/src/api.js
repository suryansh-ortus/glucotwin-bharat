const j = (r) => {
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
};
const post = (url, body) =>
  fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(j);

export const api = {
  patients: () => fetch("/api/patients").then(j),
  state: (pid) => fetch(`/api/state?pid=${pid}`).then(j),
  clock: (body) => post("/api/clock", body),
  whatif: (body) => post("/api/whatif", body),
  brief: (pid) => post("/api/brief", { pid }),
  chat: (pid, message, history) => post("/api/chat", { pid, message, history }),
  privacy: () => fetch("/api/privacy").then(j),
  learning: (pid) => fetch(`/api/learning?pid=${pid}`).then(j),
};
