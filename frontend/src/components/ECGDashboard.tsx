/**
 * Main ECG Dashboard Component
 * Integrates all ECG analysis features including 3D visualization and voice assistant
 */

import React, { useState, useEffect, useCallback } from 'react';
import { ECGVisualization3D } from './ECGVisualization3D';
import { VoiceAssistant } from './VoiceAssistant';
import { ARVRInterface } from './ARVRInterface';
import { useECGVisualization } from '../hooks/useECGVisualization';

interface ECGDashboardProps {
  patientId?: string;
  initialMode?: 'normal' | '3d' | 'vr' | 'ar';
  enableVoiceAssistant?: boolean;
  enableRealTimeUpdates?: boolean;
}

interface DashboardState {
  currentMode: 'normal' | '3d' | 'vr' | 'ar';
  isAnalyzing: boolean;
  showSettings: boolean;
  notifications: Array<{
    id: string;
    type: 'info' | 'warning' | 'error' | 'success';
    message: string;
    timestamp: number;
  }>;
}

export const ECGDashboard: React.FC<ECGDashboardProps> = ({
  initialMode = 'normal',
  enableVoiceAssistant = true,
  enableRealTimeUpdates = false
}) => {
  const [dashboardState, setDashboardState] = useState<DashboardState>({
    currentMode: initialMode,
    isAnalyzing: false,
    showSettings: false,
    notifications: []
  });

  const {
    ecgData,
    analysisResult,
    settings,
    isLoading,
    error,
    isVRSupported,
    isARSupported,
    updateSettings,
    analyzeECG,
    enterVR,
    enterAR,
    clearError
  } = useECGVisualization({
    autoUpdate: enableRealTimeUpdates,
    updateInterval: 1000
  });

  const handleModeChange = useCallback((newMode: 'normal' | '3d' | 'vr' | 'ar') => {
    setDashboardState(prev => ({ ...prev, currentMode: newMode }));
    
    if (newMode === 'vr' && isVRSupported) {
      enterVR();
    } else if (newMode === 'ar' && isARSupported) {
      enterAR();
    }
  }, [isVRSupported, isARSupported, enterVR, enterAR]);

  const handleVoiceCommand = useCallback((command: { action: string; parameters?: Record<string, unknown> }) => {
    const { action, parameters } = command;
    
    switch (action) {
      case 'analyze_ecg':
        if (ecgData) {
          setDashboardState(prev => ({ ...prev, isAnalyzing: true }));
          analyzeECG().finally(() => {
            setDashboardState(prev => ({ ...prev, isAnalyzing: false }));
          });
        }
        break;
        
      case 'enter_vr':
        handleModeChange('vr');
        break;
        
      case 'enter_ar':
        handleModeChange('ar');
        break;
        
      case 'show_3d':
        handleModeChange('3d');
        break;
        
      case 'show_normal':
        handleModeChange('normal');
        break;
        
      default:
        console.log('Unhandled voice command:', action, parameters);
    }
  }, [ecgData, analyzeECG, handleModeChange]);

  const handleVisualizationControl = useCallback((action: string, params?: Record<string, unknown>) => {
    switch (action) {
      case 'toggle_heart_model':
        updateSettings({ showHeartModel: (params?.show as boolean) ?? !settings.showHeartModel });
        break;
        
      case 'toggle_waveforms':
        updateSettings({ showECGWaveforms: !settings.showECGWaveforms });
        break;
        
      case 'zoom':
        console.log('Zoom action:', params?.direction);
        break;
        
      case 'enter_vr':
        handleModeChange('vr');
        break;
        
      case 'enter_ar':
        handleModeChange('ar');
        break;
        
      default:
        console.log('Unhandled visualization control:', action, params);
    }
  }, [settings, updateSettings, handleModeChange]);

  const addNotification = useCallback((type: 'info' | 'warning' | 'error' | 'success', message: string) => {
    const notification = {
      id: Date.now().toString(),
      type,
      message,
      timestamp: Date.now()
    };
    
    setDashboardState(prev => ({
      ...prev,
      notifications: [...prev.notifications, notification]
    }));
    
    setTimeout(() => {
      setDashboardState(prev => ({
        ...prev,
        notifications: prev.notifications.filter(n => n.id !== notification.id)
      }));
    }, 5000);
  }, []);

  useEffect(() => {
    if (error) {
      addNotification('error', error);
      clearError();
    }
  }, [error, addNotification, clearError]);

  useEffect(() => {
    if (analysisResult && dashboardState.isAnalyzing) {
      addNotification('success', `ECG analysis completed: ${analysisResult.rhythm}`);
      setDashboardState(prev => ({ ...prev, isAnalyzing: false }));
    }
  }, [analysisResult, dashboardState.isAnalyzing, addNotification]);

  const renderVisualization = (): JSX.Element => {
    if (!ecgData) {
      return (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100vh',
          background: 'linear-gradient(to bottom, #1a1a2e, #16213e)',
          color: 'white',
          fontSize: '18px'
        }}>
          {isLoading ? 'Loading ECG data...' : 'No ECG data available'}
        </div>
      );
    }

    switch (dashboardState.currentMode) {
      case '3d':
        return (
          <ECGVisualization3D
            ecgData={ecgData}
            analysisResult={analysisResult || {
              predictions: {},
              confidence: 0,
              rhythm: 'Unknown'
            }}
            onVisualizationUpdate={(data) => {
              console.log('Visualization updated:', data);
            }}
          />
        );
        
      case 'vr':
      case 'ar':
        return (
          <ARVRInterface
            ecgData={ecgData}
            analysisResult={analysisResult || undefined}
            onModeChange={handleModeChange}
            isEnabled={true}
          />
        );
        
      default:
        return (
          <div style={{
            padding: '20px',
            background: 'linear-gradient(to bottom, #1a1a2e, #16213e)',
            color: 'white',
            minHeight: '100vh'
          }}>
            <h1>ECG Analysis Dashboard</h1>
            
            {/* Mode Selection */}
            <div style={{ marginBottom: '20px' }}>
              <button
                onClick={() => handleModeChange('3d')}
                style={{
                  background: '#4CAF50',
                  color: 'white',
                  border: 'none',
                  padding: '10px 20px',
                  margin: '0 10px',
                  borderRadius: '5px',
                  cursor: 'pointer'
                }}
              >
                3D Visualization
              </button>
              
              {isVRSupported && (
                <button
                  onClick={() => handleModeChange('vr')}
                  style={{
                    background: '#2196F3',
                    color: 'white',
                    border: 'none',
                    padding: '10px 20px',
                    margin: '0 10px',
                    borderRadius: '5px',
                    cursor: 'pointer'
                  }}
                >
                  VR Mode
                </button>
              )}
              
              {isARSupported && (
                <button
                  onClick={() => handleModeChange('ar')}
                  style={{
                    background: '#FF9800',
                    color: 'white',
                    border: 'none',
                    padding: '10px 20px',
                    margin: '0 10px',
                    borderRadius: '5px',
                    cursor: 'pointer'
                  }}
                >
                  AR Mode
                </button>
              )}
            </div>
            
            {/* ECG Data Summary */}
            {ecgData && (
              <div style={{
                background: 'rgba(255, 255, 255, 0.1)',
                padding: '20px',
                borderRadius: '10px',
                marginBottom: '20px'
              }}>
                <h2>ECG Data Summary</h2>
                <p>Leads: {Object.keys(ecgData.leads).length}</p>
                <p>Sample Rate: {ecgData.sampleRate} Hz</p>
                <p>Duration: {ecgData.duration.toFixed(1)} seconds</p>
              </div>
            )}
            
            {/* Analysis Results */}
            {analysisResult && (
              <div style={{
                background: 'rgba(255, 255, 255, 0.1)',
                padding: '20px',
                borderRadius: '10px',
                marginBottom: '20px'
              }}>
                <h2>Analysis Results</h2>
                <p>Rhythm: {analysisResult.rhythm}</p>
                <p>Confidence: {(analysisResult.confidence * 100).toFixed(1)}%</p>
                
                {analysisResult.predictions && (
                  <div>
                    <h3>Predictions:</h3>
                    {Object.entries(analysisResult.predictions).map(([condition, probability]) => (
                      <p key={condition}>
                        {condition}: {((probability as number) * 100).toFixed(1)}%
                      </p>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        );
    }
  };

  return (
    <div className="ecg-dashboard" style={{ position: 'relative', width: '100%', height: '100vh' }}>
      {/* Main Visualization */}
      {renderVisualization()}
      
      {/* Voice Assistant */}
      {enableVoiceAssistant && (
        <VoiceAssistant
          ecgData={analysisResult || undefined}
          onCommand={handleVoiceCommand}
          onVisualizationControl={handleVisualizationControl}
          isEnabled={true}
        />
      )}
      
      {/* Notifications */}
      <div style={{
        position: 'fixed',
        top: '20px',
        right: '20px',
        zIndex: 2000,
        maxWidth: '300px'
      }}>
        {dashboardState.notifications.map(notification => (
          <div
            key={notification.id}
            style={{
              background: notification.type === 'error' ? '#f44336' :
                         notification.type === 'warning' ? '#ff9800' :
                         notification.type === 'success' ? '#4caf50' : '#2196f3',
              color: 'white',
              padding: '10px',
              borderRadius: '5px',
              marginBottom: '10px',
              fontSize: '14px'
            }}
          >
            {notification.message}
          </div>
        ))}
      </div>
      
      {/* Loading Overlay */}
      {(isLoading || dashboardState.isAnalyzing) && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0, 0, 0, 0.7)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 3000,
          color: 'white',
          fontSize: '18px'
        }}>
          {dashboardState.isAnalyzing ? 'Analyzing ECG...' : 'Loading...'}
        </div>
      )}
    </div>
  );
};

export default ECGDashboard;

export type { ECGDashboardProps, DashboardState };
