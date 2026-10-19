const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file");
const statusEl = document.getElementById("upload-status");
const listEl = document.getElementById("file-list");
const logEl = document.getElementById("log");
const welcomeEl = document.getElementById("welcome");
const chatForm = document.getElementById("chat-form");
const questionEl = document.getElementById("question");

function setStatus(text, isError) {
  statusEl.textContent = text;
  statusEl.classList.toggle("error", Boolean(isError));
}

function renderFiles(files) {
  listEl.innerHTML = "";
  if (!files.length) {
    const li = document.createElement("li");
    li.className = "muted";
    li.textContent = "No PDFs yet.";
    listEl.appendChild(li);
    return;
  }
  for (const file of files) {
    const li = document.createElement("li");
    li.className = "file-card";
    const info = document.createElement("div");
    const name = document.createElement("div");
    name.className = "file-name";
    name.textContent = file.name;
    const meta = document.createElement("div");
    meta.className = "muted";
    const pages = file.pages ? `${file.pages} pages` : "PDF";
    const chunks = file.chunks ? ` · ${file.chunks} sections` : "";
    meta.textContent = `${file.status || "ready"} · ${pages}${chunks}`;
    info.append(name, meta);
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "ghost";
    remove.textContent = "Remove";
    remove.addEventListener("click", () => removeFile(file.name));
    li.append(info, remove);
    listEl.appendChild(li);
  }
}

async function uploadFile(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    setStatus("Upload a PDF. Other file types are not accepted.", true);
    return;
  }
  setStatus(`Uploading ${file.name}…`);
  const body = new FormData();
  body.append("file", file);
  setStatus(`Extracting text and indexing ${file.name}…`);
  const response = await fetch("/api/upload", { method: "POST", body });
  const data = await response.json();
  if (!response.ok) {
    setStatus(data.error || "Upload failed.", true);
    return;
  }
  const ready = data.duplicate
    ? `${data.name} was already indexed. Ready for questions.`
    : `${data.name} is ready for questions (${data.pages} pages, ${data.chunks} sections).`;
  setStatus(ready, false);
  renderFiles(data.files || []);
}

async function removeFile(name) {
  const response = await fetch(`/api/files/${encodeURIComponent(name)}`, { method: "DELETE" });
  const data = await response.json();
  if (!response.ok) {
    setStatus(data.error || "Could not remove that document.", true);
    return;
  }
  setStatus(`Removed ${name}.`, false);
  renderFiles(data.files || []);
}

function showChat() {
  welcomeEl.hidden = true;
  logEl.hidden = false;
}

function addMessage(role, text) {
  showChat();
  const p = document.createElement("p");
  p.className = `msg ${role}`;
  p.textContent = text;
  logEl.appendChild(p);
  logEl.scrollTop = logEl.scrollHeight;
  return p;
}

document.getElementById("pick").addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  uploadFile(fileInput.files[0]);
  fileInput.value = "";
});
["dragenter", "dragover"].forEach((name) => {
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.add("over");
  });
});
["dragleave", "drop"].forEach((name) => {
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.remove("over");
  });
});
dropzone.addEventListener("drop", (event) => {
  uploadFile(event.dataTransfer.files[0]);
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionEl.value.trim();
  if (!question) return;
  addMessage("user", question);
  questionEl.value = "";
  const bot = addMessage("bot", "");
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!response.ok || !response.body) {
    bot.textContent = "The chat request failed.";
    return;
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";
    for (const part of parts) {
      const line = part.split("\n").find((row) => row.startsWith("data: "));
      if (!line) continue;
      const payload = JSON.parse(line.slice(6));
      if (payload.error) bot.textContent += payload.error;
      if (payload.token) bot.textContent += payload.token;
      if (payload.sources && payload.sources.length) {
        const note = document.createElement("span");
        note.className = "sources";
        note.textContent = payload.sources
          .map((item) => `${item.name}${item.page ? `, page ${item.page}` : ""}`)
          .join(" · ");
        bot.appendChild(note);
      }
    }
  }
});

renderFiles(JSON.parse(document.getElementById("initial-files").textContent || "[]"));
