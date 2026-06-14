import { useState, useRef, useEffect } from 'react'
import { 
  ShieldCheck, UploadCloud, Loader2, Search, AlertTriangle, 
  Camera, Focus, Flame, BarChart3, ShieldAlert, CheckCircle2, Zap 
} from 'lucide-react'
import './App.css'

const ResultCard = ({ res }) => {
  const [stage, setStage] = useState(0);

  useEffect(() => {
    if (res.status === 'error') {
      setStage(4); // Show error immediately
      return;
    }

    // Staged animation: Original -> Cropped -> Heatmap -> Evaluation
    const t1 = setTimeout(() => setStage(1), 500);
    const t2 = setTimeout(() => setStage(2), 1500);
    const t3 = setTimeout(() => setStage(3), 2500);
    const t4 = setTimeout(() => setStage(4), 3500);

    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
      clearTimeout(t4);
    };
  }, [res]);

  const OPTIMAL_THRESHOLD = 0.78;

  return (
    <div className="result-card glass-panel">
      <div className="result-header">
        <Search size={24} className="icon-success" />
        <h3>Analysis Report: <code>{res.filename}</code></h3>
      </div>

      {res.status === 'error' ? (
        <div className={`error-box fade-in ${stage >= 4 ? 'visible' : ''}`}>
          <AlertTriangle size={20} />
          <span>{res.message}</span>
        </div>
      ) : (
        <>
          <div className="images-grid">
            <div className={`image-box fade-in ${stage >= 1 ? 'visible' : ''}`}>
              <div className="image-title">
                <Camera size={18} /> 1. Original
              </div>
              <div className="image-container">
                <img src={res.original_base64} alt="Original" />
              </div>
            </div>
            
            <div className={`image-box fade-in ${stage >= 2 ? 'visible' : ''}`}>
              <div className="image-title">
                <Focus size={18} /> 2. Cropped Face
              </div>
              <div className="image-container">
                <img src={res.cropped_base64} alt="Cropped Face" />
              </div>
            </div>

            <div className={`image-box fade-in ${stage >= 3 ? 'visible' : ''}`}>
              <div className="image-title">
                <Flame size={18} /> 3. Explainability (Grad-CAM)
              </div>
              <div className="image-container">
                <img src={res.heatmap_base64} alt="Heatmap" />
              </div>
              <div className="image-caption">
                <span className="color-red">Red</span>: Abnormal traces | <span className="color-blue">Blue</span>: Normal
              </div>
            </div>
          </div>

          <div className={`evaluation-box fade-in ${stage >= 4 ? 'visible' : ''}`}>
            <div className="eval-left">
              <h4 className="flex-title">
                <BarChart3 size={20} /> Evaluation Results
              </h4>
              <div className={`status-badge ${res.is_fake ? 'status-danger' : 'status-success'}`}>
                {res.is_fake ? (
                  <><ShieldAlert size={20} /> DEEPFAKE DETECTED</>
                ) : (
                  <><CheckCircle2 size={20} /> AUTHENTIC</>
                )}
              </div>
              <p>
                {res.is_fake 
                  ? `The AI found strong evidence of manipulation. (Threshold: ${OPTIMAL_THRESHOLD})`
                  : `The AI did not find significant manipulation traces. (Threshold: ${OPTIMAL_THRESHOLD})`
                }
              </p>
              
              <div className="progress-bg">
                <div 
                  className="progress-fill" 
                  style={{ 
                    width: `${res.fake_score * 100}%`,
                    background: res.is_fake ? 'var(--danger)' : 'var(--success)'
                  }}
                ></div>
              </div>
            </div>
            
            <div className="metric-box">
              <div className="metric-label">Forgery Confidence</div>
              <div className="metric-value" style={{ color: res.is_fake ? 'var(--danger)' : 'var(--success)'}}>
                {(res.fake_score * 100).toFixed(1)}%
              </div>
              <div className={`metric-delta ${res.is_fake ? 'delta-danger' : 'delta-success'}`}>
                {res.is_fake ? 'Suspicious' : 'Natural'}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

function App() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleProcess = async () => {
    if (selectedFiles.length === 0) return;
    
    setIsLoading(true);
    setResults([]); // Clear previous results when processing

    const formData = new FormData();
    for (let i = 0; i < selectedFiles.length; i++) {
      formData.append('files', selectedFiles[i]);
    }

    try {
      const response = await fetch('http://localhost:8000/api/analyze', {
        method: 'POST',
        body: formData,
      });
      
      const data = await response.json();
      if (data.error) {
        alert("Server Error: " + data.error);
      } else if (data.results) {
        setResults(data.results);
      }
    } catch (error) {
      console.error("Error analyzing images:", error);
      alert("Failed to connect to the backend server. Is FastAPI running?");
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileSelect = (files) => {
    if (!files || files.length === 0) return;
    setSelectedFiles(Array.from(files));
    setResults([]); // Automatically clear results when new files are selected
  };

  const onDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = () => setIsDragging(false);

  const onDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    handleFileSelect(e.dataTransfer.files);
  };

  const onFileSelectChange = (e) => {
    handleFileSelect(e.target.files);
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1 className="gradient-text flex-center">
          <ShieldCheck size={48} className="mr-2" color="var(--accent-primary)" />
          Deepfake Detection Engine
        </h1>
        <p>Powered by Self-Blended Images (SBI) & EfficientNet</p>
      </header>

      <main>
        <div 
          className={`upload-zone glass-panel ${isDragging ? 'drag-active' : ''}`}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
          onClick={() => fileInputRef.current.click()}
        >
          <UploadCloud size={64} className="upload-icon-svg" />
          <h3>Drag & Drop images here</h3>
          <p>or click to browse files (JPG, PNG)</p>
          <input 
            type="file" 
            ref={fileInputRef} 
            className="file-input" 
            multiple 
            accept="image/jpeg, image/png, image/jpg"
            onChange={onFileSelectChange}
          />
          
          {selectedFiles.length > 0 && (
            <div className="selected-files-list flex-center">
              <CheckCircle2 size={18} className="mr-2" />
              {selectedFiles.length} file(s) selected
            </div>
          )}
        </div>

        {selectedFiles.length > 0 && !isLoading && results.length === 0 && (
          <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '3rem' }}>
            <button className="process-btn flex-center" onClick={handleProcess}>
              <Zap size={20} className="mr-2" fill="white" />
              Start Processing
            </button>
          </div>
        )}

        {isLoading && (
          <div className="loader-container glass-panel">
            <Loader2 size={48} className="spinner-svg" color="var(--accent-primary)" />
            <h3>AI is analyzing images...</h3>
            <p>This might take a few seconds.</p>
          </div>
        )}

        <div className="results-list">
          {results.map((res, index) => (
            <ResultCard key={index} res={res} />
          ))}
        </div>
      </main>
    </div>
  )
}

export default App
