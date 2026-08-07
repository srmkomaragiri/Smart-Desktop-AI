import { create } from 'zustand'

export const useAgentStore = create((set) => ({
  status: 'idle', // idle | analyzing | explaining | autotyping
  progressLogs: [],
  pipelineResult: null, // Holds the parsed explanation object
  analyzeResult: null,  // Holds structured analyze results
  resultSource: null,   // 'overlay' | 'browser'
  overlayConnected: false,
  error: null,
  
  setStatus: (status) => set({ status }),
  
  setOverlayStatus: (connected) => set({ overlayConnected: connected }),

  addProgress: (log) => set((state) => ({ 
    progressLogs: [...state.progressLogs, typeof log === 'string' ? log : JSON.stringify(log)] 
  })),
  
  clearProgress: () => set({ progressLogs: [] }),
  
  setPipelineResult: (res) => set({ 
    pipelineResult: res, 
    resultSource: res.source || 'overlay',
    status: 'idle' 
  }),

  setAnalyzeResult: (res) => set({ 
    analyzeResult: res,
    resultSource: res.source || 'overlay',
    status: 'idle'
  }),

  setError: (error) => set({ error, status: 'idle' }),
  
  reset: () => set({ 
    status: 'idle', 
    progressLogs: [], 
    pipelineResult: null, 
    analyzeResult: null,
    resultSource: null,
    error: null 
  })
}))
