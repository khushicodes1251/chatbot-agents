import { useEffect, useMemo, useState } from "react";
import { fetchFiles, removePdf, streamAnswer, uploadPdf } from "./api.js";
import ChatPane from "./components/ChatPane.jsx";
import Sidebar from "./components/Sidebar.jsx";

export default function App() {
  const [files, setFiles] = useState([]);
  const [selected, setSelected] = useState([]);
  const [status, setStatus] = useState({ text: "", error: false });
  const [messages, setMessages] = useState([]);
  const [busy, setBusy] = useState(false);
  const [indexing, setIndexing] = useState(false);

  useEffect(() => {
    fetchFiles()
      .then(setFiles)
      .catch((error) => setStatus({ text: error.message, error: true }));
  }, []);

  const selectedReady = useMemo(
    () => files.filter((file) => selected.includes(file.name) && file.status === "ready"),
    [files, selected]
  );

  function toggle(name) {
    setSelected((current) =>
      current.includes(name) ? current.filter((item) => item !== name) : [...current, name]
    );
  }

  async function onUpload(file) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setStatus({ text: "Upload a PDF. Other file types are not accepted.", error: true });
      return;
    }
    setIndexing(true);
    setStatus({ text: `Indexing ${file.name}…`, error: false });
    try {
      const data = await uploadPdf(file);
      setFiles(data.files || []);
      setSelected((current) => (current.includes(data.name) ? current : [...current, data.name]));
      setStatus({
        text: data.duplicate
          ? `${data.name} is already indexed and selected.`
          : `${data.name} is ready. It is selected for questions.`,
        error: false,
      });
    } catch (error) {
      setStatus({ text: error.message, error: true });
    } finally {
      setIndexing(false);
    }
  }

  async function onRemove(name) {
    try {
      const next = await removePdf(name);
      setFiles(next);
      setSelected((current) => current.filter((item) => item !== name));
      setStatus({ text: `Removed ${name}.`, error: false });
    } catch (error) {
      setStatus({ text: error.message, error: true });
    }
  }

  async function onAsk(question) {
    const documents = selectedReady.map((file) => file.name);
    if (!documents.length) {
      setStatus({ text: "Select at least one indexed PDF before asking.", error: true });
      return;
    }
    setBusy(true);
    setMessages((current) => [
      ...current,
      { role: "user", text: question },
      { role: "assistant", text: "", sources: [] },
    ]);
    try {
      await streamAnswer(question, documents, (event) => {
        setMessages((current) => {
          const next = current.slice();
          const last = { ...next[next.length - 1] };
          if (event.token) last.text += event.token;
          if (event.error) last.text += event.error;
          if (event.sources) last.sources = event.sources;
          next[next.length - 1] = last;
          return next;
        });
      });
    } catch (error) {
      setMessages((current) => {
        const next = current.slice();
        next[next.length - 1] = { ...next[next.length - 1], text: error.message };
        return next;
      });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="shell">
      <Sidebar
        files={files}
        selected={selected}
        status={status}
        indexing={indexing}
        onToggle={toggle}
        onUpload={onUpload}
        onRemove={onRemove}
      />
      <ChatPane
        messages={messages}
        busy={busy}
        canAsk={selectedReady.length > 0}
        selectedNames={selectedReady.map((file) => file.name)}
        onAsk={onAsk}
      />
    </div>
  );
}
