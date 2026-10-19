export async function fetchFiles() {
  const response = await fetch("/api/files");
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Could not load documents.");
  return data.files || [];
}

export async function uploadPdf(file) {
  const body = new FormData();
  body.append("file", file);
  const response = await fetch("/api/upload", { method: "POST", body });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Upload failed.");
  return data;
}

export async function removePdf(name) {
  const response = await fetch(`/api/files/${encodeURIComponent(name)}`, { method: "DELETE" });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Could not remove that document.");
  return data.files || [];
}

export async function streamAnswer(question, documents, onEvent) {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, documents }),
  });
  if (!response.ok || !response.body) {
    throw new Error("The chat request failed.");
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
      onEvent(JSON.parse(line.slice(6)));
    }
  }
}
