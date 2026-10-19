import { useState } from "react";

export default function Sidebar({ files, selected, status, indexing, onToggle, onUpload, onRemove }) {
  const [over, setOver] = useState(false);

  function take(file) {
    if (file) onUpload(file);
  }

  return (
    <aside className="sidebar">
      <header className="brand">
        <p className="eyebrow">Local documents</p>
        <h1>Document chat</h1>
        <p className="lede">Answers come only from the PDFs you select.</p>
      </header>

      <label
        className={`dropzone ${over ? "over" : ""} ${indexing ? "busy" : ""}`}
        onDragEnter={(event) => {
          event.preventDefault();
          setOver(true);
        }}
        onDragOver={(event) => {
          event.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(event) => {
          event.preventDefault();
          setOver(false);
          take(event.dataTransfer.files[0]);
        }}
      >
        <span className="drop-title">{indexing ? "Indexing PDF…" : "Drop a PDF here"}</span>
        <span className="drop-copy">Text-based files only. Scanned pages need OCR, which is not enabled.</span>
        <span className="upload-button">Upload PDF</span>
        <input
          type="file"
          accept="application/pdf,.pdf"
          hidden
          disabled={indexing}
          onChange={(event) => {
            take(event.target.files[0]);
            event.target.value = "";
          }}
        />
      </label>

      <p className={`notice ${status.error ? "error" : ""}`} role="status">
        {status.text}
      </p>

      <div className="library-head">
        <h2>Library</h2>
        <span>{selected.length} selected</span>
      </div>
      {files.length === 0 ? (
        <p className="empty">No PDFs yet.</p>
      ) : (
        <ul className="library">
          {files.map((file) => (
            <li key={file.name} className={selected.includes(file.name) ? "selected" : ""}>
              <label>
                <input
                  type="checkbox"
                  checked={selected.includes(file.name)}
                  onChange={() => onToggle(file.name)}
                />
                <span>
                  <strong>{file.name}</strong>
                  <small>
                    {file.status || "ready"}
                    {file.pages ? ` · ${file.pages} page${file.pages === 1 ? "" : "s"}` : ""}
                  </small>
                </span>
              </label>
              <button type="button" className="text-button" onClick={() => onRemove(file.name)}>
                Remove
              </button>
            </li>
          ))}
        </ul>
      )}
    </aside>
  );
}
