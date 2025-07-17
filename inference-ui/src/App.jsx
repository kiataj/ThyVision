import React, { useState } from "react";
import axios from "axios";

function App() {
  const [file, setFile] = useState(null);
  const [patch, setPatch] = useState({ d: 76, h: 141, w: 141 });
  const [stride, setStride] = useState({ d: 38, h: 70, w: 70 });
  const [loading, setLoading] = useState(false);

  const runInference = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    formData.append("patch_d", patch.d);
    formData.append("patch_h", patch.h);
    formData.append("patch_w", patch.w);
    formData.append("stride_d", stride.d);
    formData.append("stride_h", stride.h);
    formData.append("stride_w", stride.w);

    try {
      setLoading(true);
      await axios.post("/run-inference/", formData);
    } catch (err) {
      console.error("Inference failed:", err);
    } finally {
      setLoading(false);
    }
  };

  const savePredictions = async () => {
    const res = await fetch("/save-predictions/", { method: "POST" });
    const result = await res.json();

    if (result.status === "saved") {
      alert(`Predictions saved to ${result.path}`);
    } else {
      alert(`Failed to save: ${result.message}`);
    }
  };

  return (
    <div>
      {/* Logo at top-right */}
      <div style={{ position: "absolute", top: 10, right: 10 }}>
        <img src="/empa.png" alt="Empa Logo" style={{ height: 50 }} />
      </div>

      {/* Main content */}
      <div style={{ padding: 20 }}>
        <input type="file" onChange={(e) => setFile(e.target.files[0])} />

        <div>
          Patch: D {" "}
          <input
            type="number"
            value={patch.d}
            onChange={(e) =>
              setPatch({ ...patch, d: parseInt(e.target.value) || 0 })
            }
          />
          H {" "}
          <input
            type="number"
            value={patch.h}
            onChange={(e) =>
              setPatch({ ...patch, h: parseInt(e.target.value) || 0 })
            }
          />
          W {" "}
          <input
            type="number"
            value={patch.w}
            onChange={(e) =>
              setPatch({ ...patch, w: parseInt(e.target.value) || 0 })
            }
          />
        </div>

        <div>
          Stride: D {" "}
          <input
            type="number"
            value={stride.d}
            onChange={(e) =>
              setStride({ ...stride, d: parseInt(e.target.value) || 0 })
            }
          />
          H {" "}
          <input
            type="number"
            value={stride.h}
            onChange={(e) =>
              setStride({ ...stride, h: parseInt(e.target.value) || 0 })
            }
          />
          W {" "}
          <input
            type="number"
            value={stride.w}
            onChange={(e) =>
              setStride({ ...stride, w: parseInt(e.target.value) || 0 })
            }
          />
        </div>

        <div style={{ marginTop: 10 }}>
          <button onClick={runInference} disabled={loading || !file}>
            Run Inference
          </button>
          <button onClick={savePredictions} disabled={loading || !file} style={{ marginLeft: 10 }}>
            Save Predictions
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
