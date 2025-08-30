import React, { useState } from "react";
import axios from "axios";
import logoimg from "./assets/smartinternz_thumb.png";

function App() {
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);

  const handleFileChange = (e) => setFile(e.target.files[0]);

  const handleUpload = async () => {
    if (!file) return alert("Please select a PDF file.");
    const formData = new FormData();
    formData.append("file", file);

    try {
      setUploading(true);
      const res = await axios.post("http://localhost:8000/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      alert(`Upload successful! Processed ${res.data.pages} pages.`);
    } catch (err) {
      alert("Upload failed: " + err.message);
    } finally {
      setUploading(false);
    }
  };

  const handleAsk = async () => {
    if (!question) return alert("Please enter a question.");
    try {
      setAsking(true);
      const res = await axios.post("http://localhost:8000/ask", {
        question: question,
        k: 5,
      });
      setAnswer(res.data.answer);
    } catch (err) {
      alert("Failed to fetch answer: " + err.message);
    } finally {
      setAsking(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#FAFAFB] text-gray-800 flex flex-col">
      {/* Navbar */}
      <nav className="flex justify-between items-center bg-white shadow px-6 py-4">
        <div className="flex items-center space-x-3">
          <img src={logoimg} alt="Logo" className="h-20 w-32" />
        </div>
        <div>
          <input
            type="file"
            accept="application/pdf"
            onChange={handleFileChange}
            className="border border-gray-300 rounded p-2 text-sm"
          />
          <button
            onClick={handleUpload}
            className="ml-2 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition"
            disabled={uploading}
          >
            {uploading ? "Uploading..." : "Upload PDF"}
          </button>
        </div>
      </nav>

      {/* Answer Section */}
      <main className="flex-grow flex items-center justify-center p-6">
        {answer ? (
          <div className="bg-white border border-gray-300 rounded shadow p-6 w-full max-w-3xl">
            <strong className="text-gray-700">Answer:</strong>
            <p className="mt-2 whitespace-pre-line">{answer}</p>
          </div>
        ) : (
          <p className="text-gray-500 italic">
            Upload a PDF and ask a question to see the answer here.
          </p>
        )}
      </main>

      {/* Footer - Ask Question */}
      <footer className="border-t bg-white px-6 py-4">
        <div className="flex items-center gap-4 max-w-3xl mx-auto">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Type your question here..."
            className="flex-grow border border-gray-300 p-2 rounded text-sm"
          />
          <button
            onClick={handleAsk}
            className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700 transition"
            disabled={asking}
          >
            {asking ? "Asking..." : "Send"}
          </button>
        </div>
      </footer>
    </div>
  );
}

export default App;
