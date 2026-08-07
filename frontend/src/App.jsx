import React, { useEffect, useState, useRef } from 'react'
import { io } from 'socket.io-client'
import axios from 'axios'
import { Activity, Brain, Code2, Play, CheckCircle, AlertTriangle, MonitorPlay, MousePointer2, Satellite } from 'lucide-react'
import { useAgentStore } from './store/useAgentStore'

const SOCKET_URL = window.location.hostname === 'localhost' ? 'http://localhost:3000' : '/'
const API_URL = '/api'

// --- Components ---

const SourceBadge = ({ source }) => {
  if (!source) return null;
  const isOverlay = source === 'overlay';
  return (
    <div className={`badge ${isOverlay ? 'overlay' : 'browser'}`}>
      <Satellite size={10} style={{ marginRight: '4px' }} />
      Source: {isOverlay ? 'Hotkey Overlay' : 'Browser UI'}
    </div>
  );
}

const PipelineLogs = () => {
  const logs = useAgentStore((state) => state.progressLogs)
  const terminalRef = useRef(null)

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight
    }
  }, [logs])

  return (
    <div className="glass-panel" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
      <div className="panel-header">
        <Activity size={18} color="var(--accent-secondary)" /> Pipeline Status
      </div>
      <div className="panel-body" style={{ padding: '0.75rem' }}>
        <div className="terminal" ref={terminalRef}>
          {logs.map((log, i) => {
            const isError = log.includes('error') || log.includes('⚠');
            const isOverlay = log.includes('[Overlay]');
            return (
              <div key={i} className="terminal-line" style={{ color: isError ? 'var(--accent-danger)' : undefined }}>
                <span style={{ opacity: 0.5, marginRight: '8px' }}>
                  {isOverlay ? '🛰' : '🌍'} {new Date().toLocaleTimeString('en-US', {hour12:false})}
                </span>
                {log}
              </div>
            )
          })}
          {logs.length === 0 && <span style={{ opacity: 0.5 }}>Waiting for actions...</span>}
        </div>
      </div>
    </div>
  )
}

const AnalyzeResultPanel = ({ result }) => {
  if (!result) return null;
  return (
    <div className="glass-panel" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
      <div className="panel-header">
        <MousePointer2 size={18} color="var(--accent-secondary)" /> Screen Context
        <SourceBadge source={result.source} />
      </div>
      <div className="panel-body prose">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
          <div>
            <h3>Active Window</h3>
            <p className="highlight-text">{result.active_window}</p>
          </div>
          <div>
              <h3>Vision Logic</h3>
              <p>{result.vision_notes}</p>
          </div>
        </div>
        <div>
          <h3>OCR Ground Truth</h3>
          <div className="code-block">{result.ocr_text}</div>
        </div>
      </div>
    </div>
  )
}

const ConfidenceDisplay = ({ result }) => {
  if (!result || !result.confidence) return null;
  
  return (
    <div className="glass-panel" style={{ marginBottom: '1.5rem' }}>
      <div className="panel-header">
        <CheckCircle size={18} color="var(--accent-success)" /> Confidence Report
        <SourceBadge source={result.source} />
      </div>
      <div className="panel-body">
        <div className="confidence-wrapper">
          <div className="confidence-label">
            <span>OCR Ground Truth</span>
            <span>{result.confidence.text}%</span>
          </div>
          <div className="confidence-bar-bg">
            <div className="confidence-bar-fill" style={{ width: `${result.confidence.text}%` }} />
          </div>
        </div>
        <div className="confidence-wrapper">
          <div className="confidence-label">
            <span>Explanation Accuracy</span>
            <span>{result.confidence.explanation}%</span>
          </div>
          <div className="confidence-bar-bg">
            <div className="confidence-bar-fill" style={{ width: `${result.confidence.explanation}%`, background: 'linear-gradient(90deg, var(--accent-secondary), var(--accent-primary))' }} />
          </div>
        </div>
      </div>
    </div>
  )
}

const SuggestionsList = ({ result, selectedId, onSelect, onExecute }) => {
  const status = useAgentStore((state) => state.status)
  
  if (!result || !result.suggestions || result.suggestions.length === 0) {
    return (
      <div className="glass-panel" style={{ flex: 1 }}>
        <div className="panel-header">
          <Code2 size={18} color="var(--accent-primary)" /> Auto-Type Suggestions
        </div>
        <div className="panel-body" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
          Run an explanation to see suggestions
        </div>
      </div>
    )
  }

  return (
    <div className="glass-panel" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
      <div className="panel-header">
        <Code2 size={18} color="var(--accent-primary)" /> Auto-Type Suggestions
      </div>
      <div className="panel-body">
        {result.suggestions.map((sug) => (
          <div 
            key={sug.id} 
            className={`suggestion-card ${selectedId === sug.id ? 'selected' : ''}`}
            onClick={() => onSelect(sug.id)}
          >
            <div className="suggestion-title">{sug.title}</div>
            <div className="code-block">{sug.code}</div>
          </div>
        ))}
      </div>
      {selectedId && (
        <div style={{ padding: '1rem', borderTop: '1px solid var(--border-color)', background: 'rgba(0,0,0,0.2)' }}>
          <button 
            className="btn btn-primary" 
            style={{ width: '100%', justifyContent: 'center' }}
            disabled={status !== 'idle'}
            onClick={() => onExecute(selectedId)}
          >
            <Play size={16} /> Auto-Type Selected Suggestion
          </button>
        </div>
      )}
    </div>
  )
}

const ExplanationPanel = ({ result }) => {
  if (!result) {
    return (
      <div className="glass-panel" style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ textAlign: 'center', opacity: 0.5 }}>
          <Brain size={48} style={{ marginBottom: '1rem', opacity: 0.5 }} />
          <div>Awaiting screen analysis...</div>
        </div>
      </div>
    )
  }

  return (
    <div className="glass-panel" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
      <div className="panel-header">
        <MonitorPlay size={18} color="var(--accent-primary)" /> Screen Analysis
        <SourceBadge source={result.source} />
      </div>
      <div className="panel-body prose">
        {result.explanation && (
          <div>
            <h3>Explanation</h3>
            <p>{result.explanation}</p>
          </div>
        )}
        
        {result.issues && result.issues.length > 0 && (
          <div>
            <h3 style={{ color: 'var(--accent-warning)', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <AlertTriangle size={16} /> Issues Detected
            </h3>
            <ul>
              {result.issues.map((issue, i) => <li key={issue}>{issue}</li>)}
            </ul>
          </div>
        )}
        
        {result.rag_insights && result.rag_insights.length > 0 && (
          <div>
            <h3>RAG Insights</h3>
            <ul>
              {result.rag_insights.map((insight, i) => <li key={insight}>{insight}</li>)}
            </ul>
          </div>
        )}

        {result.noise && (
          <div>
            <h3>Ignored Noise</h3>
            <div className="code-block" style={{ color: 'var(--text-muted)' }}>{result.noise}</div>
          </div>
        )}
        
        {result.raw_text && (
          <div>
            <h3>Raw Extracted code</h3>
            <div className="code-block">{result.raw_text}</div>
          </div>
        )}
      </div>
    </div>
  )
}

export default function App() {
  const { 
    status, setStatus, addProgress, clearProgress, 
    pipelineResult, setPipelineResult, 
    analyzeResult, setAnalyzeResult,
    overlayConnected, setOverlayStatus,
    setError 
  } = useAgentStore()
  
  const [selectedSugId, setSelectedSugId] = useState(null)
  
  // Real-time Socket Setup
  useEffect(() => {
    const socket = io(SOCKET_URL)
    
    // Status from Node Bridge
    socket.on('overlay_status', (data) => {
      setOverlayStatus(data.connected);
    });

    // Hotkey events
    socket.on('overlay_hotkey', (data) => {
      addProgress(`> [Overlay] Hotkey: ${data.action.toUpperCase()}`);
    });

    // Pipeline updates (Logs)
    socket.on('pipeline_update', (data) => {
      if (data.progress) addProgress(data.progress);
    })
    
    // Explain complete (from Overlay or Browser)
    socket.on('pipeline_complete', (data) => {
      addProgress(`> [System] Explanation results synced (${data.source || 'overlay'}).`);
      setPipelineResult(data);
      setAnalyzeResult(null); // Clear analyze if we have explanation
      setSelectedSugId(null);
      setStatus('idle');
    })

    // Analyze complete (from Overlay or Browser)
    socket.on('analyze_complete', (data) => {
      addProgress(`> [System] Screen analysis synced (${data.source || 'overlay'}).`);
      setAnalyzeResult(data);
      setPipelineResult(null); // Clear explain if we have analyze
      setStatus('idle');
    });
    
    socket.on('pipeline_error', (err) => {
      addProgress(`> ⛔ [Error] ${err}`);
      setStatus('idle');
    })

    return () => socket.disconnect()
  }, [])

  const handleExplain = async () => {
    clearProgress()
    setStatus('explaining')
    addProgress('> [Browser] Initiated Explain Pipeline...')
    try {
       await axios.post(`${API_URL}/explain`)
    } catch (e) {
       addProgress(`> ⛔ HTTP Error: ${e.message}`)
       setStatus('idle')
    }
  }

  const handleAnalyze = async () => {
    clearProgress();
    setStatus('analyzing');
    addProgress('> [Browser] Initiated Analyze Pipeline...');
    try {
       await axios.post(`${API_URL}/analyze`);
    } catch (e) {
       addProgress(`> ⛔ HTTP Error: ${e.message}`);
       setStatus('idle');
    }
  }

  const handleAutoType = async (id) => {
    const sug = pipelineResult.suggestions.find(s => s.id === id)
    if (!sug) return
    
    setStatus('autotyping')
    addProgress(`> Initiating Auto-Type for suggestion: ${sug.title}`)
    
    try {
      const calcRes = await axios.post(`${API_URL}/autotype/calculate`, { selected_suggestion: sug.title + "\n" + sug.code })
      const calcData = calcRes.data
      
      let textToType = calcData.insert_continue_line;
      if (!calcData.is_incomplete_line) {
         textToType = calcData.insert_new_line;
      }
      
      if (!window.confirm(`Insert this snippet?\n\n${textToType}`)) {
         addProgress('> ⬛ Cancelled by user.')
         setStatus('idle')
         return
      }
      
      await axios.post(`${API_URL}/autotype/execute`, {
          action: "type",
          target: "auto-type",
          text: textToType,
          reasoning: "React Auto-Type flow"
      })
      
      addProgress('> ✅ Auto-Type complete.')
    } catch (e) {
      addProgress(`> ⛔ Auto-Type Failed: ${e.message}`)
    }
    
    setStatus('idle')
  }

  return (
    <div className="app-container">
      {/* LEFT SIDEBAR */}
      <div className="sidebar">
        
        <div className="glass-panel">
          <div className="panel-header">
            <Activity size={18} /> OS Agent Controls
            <div 
              className={`status-circle ${overlayConnected ? 'connected' : 'disconnected'}`} 
              title={overlayConnected ? 'Overlay Process Sync Active' : 'Overlay Process Not Detected'}
            />
          </div>
          <div className="panel-body" style={{ padding: '1rem' }}>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '1rem' }}>
              <button 
                className={`btn btn-primary ${status === 'explaining' ? 'pulse' : ''}`}
                onClick={handleExplain}
                disabled={status !== 'idle'}
                style={{ flex: 1, minWidth: '130px', justifyContent: 'center' }}
              >
                <Brain size={16} /> 
                {status === 'explaining' ? 'Wait...' : 'Explain'}
              </button>
              <button 
                className={`btn ${status === 'analyzing' ? 'pulse' : ''}`}
                onClick={handleAnalyze}
                disabled={status !== 'idle'}
                style={{ flex: 1, minWidth: '130px', justifyContent: 'center' }}
              >
                <MonitorPlay size={16} /> 
                {status === 'analyzing' ? 'Wait...' : 'Analyze'}
              </button>
            </div>
            {status !== 'idle' && (
              <div style={{ color: 'var(--accent-secondary)', fontSize: '0.85rem', textAlign: 'center' }}>
                {status.toUpperCase()} in progress...
              </div>
            )}
          </div>
        </div>

        <ConfidenceDisplay result={pipelineResult} />
        <PipelineLogs />
        
      </div>

      {/* RIGHT SIDE */}
      <div className="main-content">
        {analyzeResult ? (
          <AnalyzeResultPanel result={analyzeResult} />
        ) : (
          <ExplanationPanel result={pipelineResult} />
        )}
        
        <div style={{ height: '40%' }}>
          <SuggestionsList 
            result={pipelineResult} 
            selectedId={selectedSugId}
            onSelect={setSelectedSugId}
            onExecute={handleAutoType}
          />
        </div>
      </div>
    </div>
  )
}
