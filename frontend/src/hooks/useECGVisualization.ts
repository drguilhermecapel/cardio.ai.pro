/**
 * Custom hook for ECG 3D visualization management
 */

import { useState, useEffect, useCallback, useRef } from 'react';

interface ECGData {
  leads: { [leadName: string]: number[] };
  sampleRate: number;
  duration: number;
  timestamp: number;
}

interface ECGAnalysisResult {
  predictions: { [condition: string]: number };
  confidence: number;
  rhythm: string;
  interpretability?: {
    attention_maps: { [lead: string]: number[] };
    clinical_findings: Array<{
      condition: string;
      confidence: number;
      evidence: string[];
      lead_involvement: string[];
    }>;
  };
}

interface VisualizationSettings {
  showHeartModel: boolean;
  showECGWaveforms: boolean;
  showAttentionMaps: boolean;
  showClinicalFindings: boolean;
  animationSpeed: number;
  waveformScale: number;
  heartBeatRate: number;
  immersiveMode: boolean;
}

interface UseECGVisualizationProps {
  initialData?: ECGData;
  initialAnalysis?: ECGAnalysisResult;
  autoUpdate?: boolean;
  updateInterval?: number;
}

export const useECGVisualization = ({
  initialData,
  initialAnalysis,
  autoUpdate = false,
  updateInterval = 1000
}: UseECGVisualizationProps = {}): {
  ecgData: ECGData | null;
  analysisResult: ECGAnalysisResult | null;
  settings: VisualizationSettings;
  isLoading: boolean;
  error: string | null;
  isVRSupported: boolean;
  isARSupported: boolean;
  isVRActive: boolean;
  isARActive: boolean;
  setECGData: (data: ECGData | null) => void;
  setAnalysisResult: (result: ECGAnalysisResult | null) => void;
  updateSettings: (newSettings: Partial<VisualizationSettings>) => void;
  loadECGData: (dataSource: string | File) => Promise<void>;
  analyzeECG: (data?: ECGData) => Promise<void>;
  enterVR: () => Promise<boolean>;
  enterAR: () => Promise<boolean>;
  exportVisualization: (format: 'json' | 'csv' | 'png') => Promise<string | null>;
  clearData: () => void;
  clearError: () => void;
} => {
  const [ecgData, setECGData] = useState<ECGData | null>(initialData || null);
  const [analysisResult, setAnalysisResult] = useState<ECGAnalysisResult | null>(
    initialAnalysis || null
  );
  const [settings, setSettings] = useState<VisualizationSettings>({
    showHeartModel: true,
    showECGWaveforms: true,
    showAttentionMaps: true,
    showClinicalFindings: true,
    animationSpeed: 1.0,
    waveformScale: 1.0,
    heartBeatRate: 75,
    immersiveMode: false
  });

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isVRSupported, setIsVRSupported] = useState(false);
  const [isARSupported, setIsARSupported] = useState(false);
  const [isVRActive, setIsVRActive] = useState(false);
  const [isARActive, setIsARActive] = useState(false);

  const updateIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const checkXRSupport = async (): Promise<void> => {
      if ('xr' in navigator && navigator.xr) {
        try {
          const vrSupported = await navigator.xr.isSessionSupported('immersive-vr');
          const arSupported = await navigator.xr.isSessionSupported('immersive-ar');
          setIsVRSupported(vrSupported);
          setIsARSupported(arSupported);
        } catch (error) {
          console.warn('WebXR support check failed:', error);
        }
      }
    };

    checkXRSupport();
  }, []);

  useEffect(() => {
    if (!autoUpdate) return;

    const connectWebSocket = (): void => {
      try {
        const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws/ecg';
        wsRef.current = new WebSocket(wsUrl);

        wsRef.current.onopen = (): void => {
          console.log('ECG WebSocket connected');
          setError(null);
        };

        wsRef.current.onmessage = (event: MessageEvent): void => {
          try {
            const data = JSON.parse(event.data);
            
            if (data.type === 'ecg_data') {
              setECGData(data.payload);
            } else if (data.type === 'analysis_result') {
              setAnalysisResult(data.payload);
            } else if (data.type === 'error') {
              setError(data.message);
            }
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
          }
        };

        wsRef.current.onerror = (error: Event): void => {
          console.error('WebSocket error:', error);
          setError('Real-time connection failed');
        };

        wsRef.current.onclose = (): void => {
          console.log('ECG WebSocket disconnected');
          setTimeout(connectWebSocket, 5000);
        };
      } catch (error) {
        console.error('Failed to connect WebSocket:', error);
        setError('Failed to establish real-time connection');
      }
    };

    connectWebSocket();

    return (): void => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [autoUpdate]);

  useEffect(() => {
    if (!autoUpdate || wsRef.current?.readyState === WebSocket.OPEN) return;

    const pollForUpdates = async (): Promise<void> => {
      try {
        setIsLoading(true);
        
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
        const response = await fetch(`${apiUrl}/api/ecg/latest`);
        
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (data.ecg_data) {
          setECGData(data.ecg_data);
        }
        
        if (data.analysis_result) {
          setAnalysisResult(data.analysis_result);
        }
        
        setError(null);
      } catch (error) {
        console.error('Failed to fetch ECG updates:', error);
        setError(error instanceof Error ? error.message : 'Failed to fetch updates');
      } finally {
        setIsLoading(false);
      }
    };

    updateIntervalRef.current = setInterval(pollForUpdates, updateInterval);

    return (): void => {
      if (updateIntervalRef.current) {
        clearInterval(updateIntervalRef.current);
      }
    };
  }, [autoUpdate, updateInterval]);

  const updateSettings = useCallback((newSettings: Partial<VisualizationSettings>): void => {
    setSettings(prev => ({ ...prev, ...newSettings }));
  }, []);

  const loadECGData = useCallback(async (dataSource: string | File): Promise<void> => {
    setIsLoading(true);
    setError(null);

    try {
      let data: ECGData;

      if (typeof dataSource === 'string') {
        const response = await fetch(dataSource);
        if (!response.ok) {
          throw new Error(`Failed to load ECG data: ${response.statusText}`);
        }
        data = await response.json();
      } else {
        const text = await dataSource.text();
        data = JSON.parse(text);
      }

      setECGData(data);
      
      if (data) {
        await analyzeECG(data);
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to load ECG data';
      setError(errorMessage);
      console.error('ECG data loading error:', error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const analyzeECG = useCallback(async (data?: ECGData): Promise<void> => {
    const dataToAnalyze = data || ecgData;
    if (!dataToAnalyze) {
      setError('No ECG data available for analysis');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/ecg/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ ecg_data: dataToAnalyze }),
      });

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`);
      }

      const result = await response.json();
      setAnalysisResult(result);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'ECG analysis failed';
      setError(errorMessage);
      console.error('ECG analysis error:', error);
    } finally {
      setIsLoading(false);
    }
  }, [ecgData]);

  const enterVR = useCallback(async () => {
    if (!isVRSupported || !navigator.xr) {
      setError('VR not supported');
      return false;
    }

    try {
      const session = await navigator.xr.requestSession('immersive-vr');
      setIsVRActive(true);
      
      session.addEventListener('end', () => {
        setIsVRActive(false);
      });
      
      return true;
    } catch (error) {
      setError('Failed to enter VR mode');
      console.error('VR session error:', error);
      return false;
    }
  }, [isVRSupported]);

  const enterAR = useCallback(async () => {
    if (!isARSupported || !navigator.xr) {
      setError('AR not supported');
      return false;
    }

    try {
      const session = await navigator.xr.requestSession('immersive-ar');
      setIsARActive(true);
      
      session.addEventListener('end', () => {
        setIsARActive(false);
      });
      
      return true;
    } catch (error) {
      setError('Failed to enter AR mode');
      console.error('AR session error:', error);
      return false;
    }
  }, [isARSupported]);

  const exportVisualization = useCallback(async (format: 'json' | 'csv' | 'png') => {
    if (!ecgData || !analysisResult) {
      setError('No data available for export');
      return null;
    }

    try {
      const exportData = {
        ecg_data: ecgData,
        analysis_result: analysisResult,
        settings,
        timestamp: new Date().toISOString()
      };

      switch (format) {
        case 'json':
          return JSON.stringify(exportData, null, 2);
        
        case 'csv': {
          const csvRows = ['Lead,Time,Amplitude,Attention'];
          Object.entries(ecgData.leads).forEach(([leadName, leadData]) => {
            const attentionMap = analysisResult.interpretability?.attention_maps[leadName] || [];
            leadData.forEach((amplitude, index) => {
              const time = index / ecgData.sampleRate;
              const attention = attentionMap[index] || 0;
              csvRows.push(`${leadName},${time},${amplitude},${attention}`);
            });
          });
          return csvRows.join('\n');
        }
        
        case 'png':
          setError('PNG export not yet implemented');
          return null;
        
        default:
          setError('Unsupported export format');
          return null;
      }
    } catch (error) {
      setError('Export failed');
      console.error('Export error:', error);
      return null;
    }
  }, [ecgData, analysisResult, settings]);

  const clearData = useCallback((): void => {
    setECGData(null);
    setAnalysisResult(null);
    setError(null);
  }, []);

  return {
    ecgData,
    analysisResult,
    settings,
    
    isLoading,
    error,
    isVRSupported,
    isARSupported,
    isVRActive,
    isARActive,
    
    setECGData,
    setAnalysisResult,
    updateSettings,
    loadECGData,
    analyzeECG,
    enterVR,
    enterAR,
    exportVisualization,
    clearData,
    
    clearError: (): void => setError(null)
  };
};

export default useECGVisualization;
