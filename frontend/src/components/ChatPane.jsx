import { useState } from "react";

function Evidence({ source }) {
  const [open, setOpen] = useState(false);
  const text = source.text || "";
  const long = text.length > 180;

  return (
    <li>
      <p className="source-cite">
        {source.name}
        {source.page ? `, page ${source.page}` : ""}
      </p>
      {text ? (
        <blockquote className={open || !long ? "excerpt open" : "excerpt"}>{text}</blockquote>
      ) : null}
      {long ? (
        <button type="button" className="text-button" onClick={() => setOpen((value) => !value)}>
          {open ? "Show less" : "Show more"}
        </button>
      ) : null}
    </li>
  );
}

export default function ChatPane({ messages, busy, canAsk, selectedNames, onAsk }) {
  const [draft, setDraft] = useState("");

  function submit(event) {
    event.preventDefault();
    const question = draft.trim();
    if (!question || busy) return;
    setDraft("");
    onAsk(question);
  }

  return (
    <main className="chat">
      {messages.length === 0 ? (
        <section className="welcome">
          <p className="eyebrow">Ask the files in front of you</p>
          <h2>Chat with your own documents</h2>
          <p>
            Upload a PDF, select it in the library, then ask a question. Answers stay inside those
            files and cite the filename and page. Nothing is filled in from a built-in knowledge base.
          </p>
          <ol>
            <li>Upload one or more PDFs.</li>
            <li>Leave checked only the files you want to query.</li>
            <li>Ask when the status says the file is ready.</li>
          </ol>
        </section>
      ) : (
        <section className="transcript" aria-live="polite">
          {messages.map((message, index) => (
            <article key={index} className={`bubble ${message.role}`}>
              <p>{message.text || (busy && index === messages.length - 1 ? "Reading the selected pages…" : "")}</p>
              {message.sources?.length > 0 && (
                <ul className="sources">
                  {message.sources.map((source, sourceIndex) => (
                    <Evidence key={`${source.name}-${source.page}-${sourceIndex}`} source={source} />
                  ))}
                </ul>
              )}
            </article>
          ))}
        </section>
      )}

      <form className="composer" onSubmit={submit}>
        <p className="scope">
          {selectedNames.length
            ? `Asking ${selectedNames.join(", ")}`
            : "Select a ready PDF to enable questions."}
        </p>
        <div className="composer-row">
          <input
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            placeholder={canAsk ? "Ask about the selected PDFs…" : "Select a PDF first"}
            disabled={!canAsk || busy}
            aria-label="Question"
          />
          <button type="submit" disabled={!canAsk || busy || !draft.trim()}>
            {busy ? "Sending" : "Send"}
          </button>
        </div>
      </form>
    </main>
  );
}
