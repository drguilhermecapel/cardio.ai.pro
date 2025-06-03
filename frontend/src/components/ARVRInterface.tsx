/**
 * AR/VR Interface Component for ECG Analysis
 * Provides immersive ECG analysis experience
 */

import React, { useState, useEffect, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { VRButton, ARButton, XR, Controllers, Hands } from '@react-three/xr';
import { OrbitControls, Text, Environment } from '@react-three/drei';

interface ARVRInterfaceProps {
  ecgData?: ECGData;
  analysisResult?: ECGAnalysisResult;
  onModeChange?: (mode: 'normal' | 'vr' | 'ar') => void;
  isEnabled?: boolean;
}

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

interface XRSession {
  mode: 'immersive-vr' | 'immersive-ar';
  isActive: boolean;
  startTime: number;
}

const XRScene: React.FC<{
  ecgData: ECGData;
  analysisResult: ECGAnalysisResult;
  mode: 'vr' | 'ar';
}> = ({ ecgData, analysisResult, mode }): JSX.Element => {
  return (
    <>
      {/* Environment lighting */}
      <Environment preset="studio" />
      
      {/* Ambient lighting for AR/VR */}
      <ambientLight intensity={mode === 'ar' ? 0.8 : 0.4} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      
      {/* ECG Data Visualization in 3D Space */}
      {ecgData && (
        <group position={[0, 0, -2]}>
          <Text
            position={[0, 2, 0]}
            fontSize={0.5}
            color="white"
            anchorX="center"
            anchorY="middle"
          >
            ECG Analysis - {mode.toUpperCase()} Mode
          </Text>
          
          {/* Placeholder for ECG waveforms */}
          <mesh position={[0, 0, 0]}>
            <boxGeometry args={[2, 1, 0.1]} />
            <meshStandardMaterial color="blue" transparent opacity={0.7} />
          </mesh>
        </group>
      )}
      
      {/* Analysis Results Display */}
      {analysisResult && (
        <group position={[3, 0, -2]}>
          <Text
            position={[0, 1, 0]}
            fontSize={0.3}
            color="green"
            anchorX="center"
            anchorY="middle"
          >
            Rhythm: {analysisResult.rhythm || 'Unknown'}
          </Text>
          
          <Text
            position={[0, 0.5, 0]}
            fontSize={0.25}
            color="yellow"
            anchorX="center"
            anchorY="middle"
          >
            Confidence: {((analysisResult.confidence || 0) * 100).toFixed(1)}%
          </Text>
        </group>
      )}
      
      {/* Interactive Controls */}
      <Controllers />
      <Hands />
    </>
  );
};

export const ARVRInterface: React.FC<ARVRInterfaceProps> = ({
  ecgData,
  analysisResult,
  onModeChange,
  isEnabled = true
}) => {
  const [currentMode] = useState<'normal' | 'vr' | 'ar'>('normal');
  const [isVRSupported, setIsVRSupported] = useState(false);
  const [isARSupported, setIsARSupported] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect((): void => {
    const checkXRSupport = async (): Promise<void> => {
      if (!('xr' in navigator)) {
        setError('WebXR not supported in this browser');
        return;
      }

      try {
        const xr = (navigator as any).xr;
        if (xr) {
          const vrSupported = await xr.isSessionSupported('immersive-vr');
          const arSupported = await xr.isSessionSupported('immersive-ar');
          
          setIsVRSupported(vrSupported);
          setIsARSupported(arSupported);
          
          if (!vrSupported && !arSupported) {
            setError('No XR devices detected');
          }
        }
      } catch (err) {
        console.warn('XR support check failed:', err);
        setError('XR support check failed');
      }
    };

    if (isEnabled) {
      checkXRSupport();
    }
  }, [isEnabled]);

  useEffect((): void => {
    if (onModeChange) {
      onModeChange(currentMode);
    }
  }, [currentMode, onModeChange]);



  if (!isEnabled) {
    return null;
  }

  return (
    <div className="arvr-interface" style={{
      position: 'relative',
      width: '100%',
      height: '100vh',
      background: currentMode === 'ar' ? 'transparent' : 'linear-gradient(to bottom, #0a0a0a, #1a1a2e)'
    }}>
      {/* XR Controls */}
      <div style={{
        position: 'absolute',
        top: '20px',
        right: '20px',
        zIndex: 1000,
        display: 'flex',
        gap: '10px'
      }}>
        {isVRSupported && (
          <VRButton />
        )}
        
        {isARSupported && (
          <ARButton />
        )}
      </div>

      {/* Status Display */}
      <div style={{
        position: 'absolute',
        top: '20px',
        left: '20px',
        zIndex: 1000,
        background: 'rgba(0, 0, 0, 0.8)',
        padding: '15px',
        borderRadius: '10px',
        color: 'white'
      }}>
        <div>Mode: {currentMode.toUpperCase()}</div>
        <div>VR Support: {isVRSupported ? '✓' : '✗'}</div>
        <div>AR Support: {isARSupported ? '✓' : '✗'}</div>
        <div>Session: Active</div>
      </div>

      {/* Error Display */}
      {error && (
        <div style={{
          position: 'absolute',
          bottom: '20px',
          left: '20px',
          right: '20px',
          zIndex: 1000,
          background: 'rgba(255, 0, 0, 0.8)',
          padding: '15px',
          borderRadius: '10px',
          color: 'white',
          textAlign: 'center'
        }}>
          {error}
        </div>
      )}

      {/* 3D Canvas */}
      <Canvas
        ref={canvasRef}
        camera={{ position: [0, 0, 5], fov: 75 }}
        style={{ width: '100%', height: '100%' }}
      >
        <XR>
          <XRScene
            ecgData={ecgData || { leads: {}, sampleRate: 500, duration: 0, timestamp: 0 }}
            analysisResult={analysisResult || { predictions: {}, confidence: 0, rhythm: 'unknown' }}
            mode={currentMode === 'normal' ? 'vr' : currentMode}
          />
          
          {/* Camera controls for non-XR mode */}
          {currentMode === 'normal' && (
            <OrbitControls
              enablePan={true}
              enableZoom={true}
              enableRotate={true}
              maxDistance={10}
              minDistance={1}
            />
          )}
        </XR>
      </Canvas>

      {/* Instructions */}
      {currentMode === 'normal' && (
        <div style={{
          position: 'absolute',
          bottom: '20px',
          right: '20px',
          zIndex: 1000,
          background: 'rgba(0, 0, 0, 0.8)',
          padding: '15px',
          borderRadius: '10px',
          color: 'white',
          maxWidth: '300px'
        }}>
          <h4 style={{ margin: '0 0 10px 0' }}>XR Instructions</h4>
          <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '14px' }}>
            <li>Click VR button to enter virtual reality mode</li>
            <li>Click AR button to enter augmented reality mode</li>
            <li>Use hand tracking or controllers to interact</li>
            <li>Voice commands are available in XR mode</li>
          </ul>
        </div>
      )}
    </div>
  );
};

export default ARVRInterface;

export type { ARVRInterfaceProps, XRSession };
