/**
 * ECG 3D Visualization Component with AR/VR Support
 * Advanced 3D visualization of ECG data with immersive capabilities
 */

import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import * as THREE from 'three';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Line, Sphere } from '@react-three/drei';
import { VRButton, ARButton, XR, Controllers, Hands } from '@react-three/xr';

interface ECGData {
  leads: {
    [leadName: string]: number[];
  };
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

interface HeartModelProps {
  ecgData: ECGData;
  analysisResult: ECGAnalysisResult;
  isBeating: boolean;
}

interface ECGWaveformProps {
  leadData: number[];
  leadName: string;
  position: [number, number, number];
  color: string;
  attentionMap?: number[];
  isHighlighted: boolean;
}

interface ECGVisualization3DProps {
  ecgData: ECGData;
  analysisResult: ECGAnalysisResult;
  onVisualizationUpdate?: (data: Record<string, unknown>) => void;
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

const HeartModel: React.FC<HeartModelProps> = ({ analysisResult, isBeating }) => {
  const heartRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (!heartRef.current || !isBeating) return;

    const time = state.clock.getElapsedTime();
    const heartRate = analysisResult.rhythm === 'Atrial Fibrillation' ? 
      Math.random() * 0.3 + 0.7 : // Irregular for AF
      Math.sin(time * 2) * 0.1 + 1; // Regular rhythm

    const scale = 1 + heartRate * 0.1;
    heartRef.current.scale.setScalar(scale);

    heartRef.current.rotation.y = Math.sin(time * 0.5) * 0.1;

  });

  const heartGeometry = useMemo(() => {
    const geometry = new THREE.Group();
    
    const leftVentricle = new THREE.SphereGeometry(1.2, 32, 32);
    const leftVentricleMesh = new THREE.Mesh(
      leftVentricle,
      new THREE.MeshPhongMaterial({ 
        color: analysisResult.predictions.stemi > 0.5 ? 0xff4444 : 0xff6b6b,
        transparent: true,
        opacity: 0.8
      })
    );
    leftVentricleMesh.position.set(-0.3, -0.2, 0);
    geometry.add(leftVentricleMesh);

    const rightVentricle = new THREE.SphereGeometry(0.9, 32, 32);
    const rightVentricleMesh = new THREE.Mesh(
      rightVentricle,
      new THREE.MeshPhongMaterial({ 
        color: 0xff8888,
        transparent: true,
        opacity: 0.7
      })
    );
    rightVentricleMesh.position.set(0.4, -0.1, 0.2);
    geometry.add(rightVentricleMesh);

    const leftAtrium = new THREE.SphereGeometry(0.6, 32, 32);
    const leftAtriumMesh = new THREE.Mesh(
      leftAtrium,
      new THREE.MeshPhongMaterial({ 
        color: analysisResult.predictions.atrial_fibrillation > 0.5 ? 0xffaa44 : 0xffaaaa,
        transparent: true,
        opacity: 0.6
      })
    );
    leftAtriumMesh.position.set(-0.2, 0.8, 0);
    geometry.add(leftAtriumMesh);

    const rightAtrium = new THREE.SphereGeometry(0.6, 32, 32);
    const rightAtriumMesh = new THREE.Mesh(
      rightAtrium,
      new THREE.MeshPhongMaterial({ 
        color: 0xffcccc,
        transparent: true,
        opacity: 0.6
      })
    );
    rightAtriumMesh.position.set(0.3, 0.7, 0.2);
    geometry.add(rightAtriumMesh);

    return geometry;
  }, [analysisResult]);

  return (
    <group ref={heartRef} position={[0, 0, 0]}>
      <primitive object={heartGeometry} />
      
      {/* Electrical conduction system visualization */}
      {analysisResult.predictions.left_bundle_branch_block > 0.3 && (
        <Line
          points={[[-0.5, 0, 0], [0, -0.5, 0], [0.5, -0.3, 0]]}
          color="yellow"
          lineWidth={3}
        />
      )}
      
      {/* Coronary arteries */}
      <Line
        points={[[-0.8, 0.2, 0], [-0.3, -0.2, 0], [0.2, -0.4, 0]]}
        color={analysisResult.predictions.stemi > 0.3 ? "red" : "pink"}
        lineWidth={2}
      />
    </group>
  );
};

const ECGWaveform3D: React.FC<ECGWaveformProps> = ({ 
  leadData, 
  leadName, 
  position, 
  color, 
  attentionMap,
  isHighlighted 
}) => {
  const waveformRef = useRef<THREE.Group>(null);

  const waveformPoints = useMemo(() => {
    const points: THREE.Vector3[] = [];
    const timeScale = 0.01; // Scale factor for time axis
    const amplitudeScale = 2; // Scale factor for amplitude

    for (let i = 0; i < leadData.length; i++) {
      const x = i * timeScale;
      const y = leadData[i] * amplitudeScale;
      const z = attentionMap ? attentionMap[i] * 0.5 : 0; // Use attention for Z-axis
      points.push(new THREE.Vector3(x, y, z));
    }

    return points;
  }, [leadData, attentionMap]);

  useFrame((state) => {
    if (!waveformRef.current) return;

    const time = state.clock.getElapsedTime();

    if (isHighlighted) {
      const pulse = Math.sin(time * 4) * 0.1 + 1;
      waveformRef.current.scale.setScalar(pulse);
    }
  });

  return (
    <group ref={waveformRef} position={position}>
      {/* Lead label */}
      <Text
        position={[0, 2, 0]}
        fontSize={0.3}
        color={color}
        anchorX="center"
        anchorY="middle"
      >
        {leadName}
      </Text>

      {/* Waveform line */}
      <Line
        points={waveformPoints}
        color={color}
        lineWidth={isHighlighted ? 4 : 2}
        transparent
        opacity={isHighlighted ? 1 : 0.8}
      />

      {/* Attention visualization spheres */}
      {attentionMap && attentionMap.map((attention, index) => (
        attention > 0.7 && (
          <Sphere
            key={index}
            position={[index * 0.01, leadData[index] * 2, attention * 0.5]}
            args={[0.02]}
          >
            <meshBasicMaterial color="yellow" transparent opacity={attention} />
          </Sphere>
        )
      ))}
    </group>
  );
};

interface ClinicalFinding {
  condition: string;
  confidence: number;
  evidence: string[];
  lead_involvement: string[];
}

const ClinicalFindings: React.FC<{ 
  findings: ClinicalFinding[] | undefined;
  position: [number, number, number];
}> = ({ findings, position }) => {
  return (
    <group position={position}>
      {findings?.map((finding: ClinicalFinding, index: number) => (
        <group key={index} position={[0, -index * 0.8, 0]}>
          <Text
            fontSize={0.2}
            color={finding.confidence > 0.7 ? "red" : "orange"}
            anchorX="left"
            anchorY="middle"
          >
            {finding.condition}: {(finding.confidence * 100).toFixed(1)}%
          </Text>
          
          {/* Evidence indicators */}
          {finding.evidence.slice(0, 3).map((evidence: string, evidenceIndex: number) => (
            <Text
              key={evidenceIndex}
              position={[0, -0.3 - evidenceIndex * 0.2, 0]}
              fontSize={0.15}
              color="white"
              anchorX="left"
              anchorY="middle"
            >
              • {evidence}
            </Text>
          ))}
        </group>
      ))}
    </group>
  );
};

const ECGScene: React.FC<{
  ecgData: ECGData;
  analysisResult: ECGAnalysisResult;
  settings: VisualizationSettings;
}> = ({ ecgData, analysisResult, settings }) => {
  const { camera } = useThree();

  const leadPositions: { [key: string]: [number, number, number] } = {
    'I': [-4, 2, 0],
    'II': [-4, 0, 0],
    'III': [-4, -2, 0],
    'aVR': [-2, 2, 0],
    'aVL': [-2, 0, 0],
    'aVF': [-2, -2, 0],
    'V1': [2, 2, 0],
    'V2': [2, 0, 0],
    'V3': [2, -2, 0],
    'V4': [4, 2, 0],
    'V5': [4, 0, 0],
    'V6': [4, -2, 0],
  };

  const leadColors: { [key: string]: string } = {
    'I': '#ff6b6b', 'II': '#ff6b6b', 'III': '#ff6b6b', // Limb leads
    'aVR': '#4ecdc4', 'aVL': '#4ecdc4', 'aVF': '#4ecdc4', // Augmented leads
    'V1': '#45b7d1', 'V2': '#45b7d1', 'V3': '#45b7d1', // Septal leads
    'V4': '#96ceb4', 'V5': '#96ceb4', 'V6': '#96ceb4', // Lateral leads
  };

  useEffect(() => {
    if (settings.immersiveMode) {
      camera.position.set(0, 0, 8);
    } else {
      camera.position.set(0, 2, 10);
    }
  }, [settings.immersiveMode, camera]);

  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.6} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={0.5} />

      {/* Heart model */}
      {settings.showHeartModel && (
        <HeartModel
          ecgData={ecgData}
          analysisResult={analysisResult}
          isBeating={true}
        />
      )}

      {/* ECG Waveforms */}
      {settings.showECGWaveforms && Object.entries(ecgData.leads).map(([leadName, leadData]) => {
        const position = leadPositions[leadName] || [0, 0, 0];
        const color = leadColors[leadName] || '#ffffff';
        const attentionMap = settings.showAttentionMaps ? 
          analysisResult.interpretability?.attention_maps[leadName] : undefined;
        const isHighlighted = analysisResult.interpretability?.clinical_findings
          ?.some(finding => finding.lead_involvement.includes(leadName)) || false;

        return (
          <ECGWaveform3D
            key={leadName}
            leadData={leadData}
            leadName={leadName}
            position={position}
            color={color}
            attentionMap={attentionMap}
            isHighlighted={isHighlighted}
          />
        );
      })}

      {/* Clinical findings */}
      {settings.showClinicalFindings && analysisResult.interpretability?.clinical_findings && (
        <ClinicalFindings
          findings={analysisResult.interpretability.clinical_findings}
          position={[6, 2, 0]}
        />
      )}

      {/* Analysis summary */}
      <group position={[0, 4, 0]}>
        <Text
          fontSize={0.4}
          color="white"
          anchorX="center"
          anchorY="middle"
        >
          ECG Analysis: {analysisResult.rhythm}
        </Text>
        <Text
          position={[0, -0.5, 0]}
          fontSize={0.3}
          color={analysisResult.confidence > 0.8 ? "green" : "yellow"}
          anchorX="center"
          anchorY="middle"
        >
          Confidence: {(analysisResult.confidence * 100).toFixed(1)}%
        </Text>
      </group>

      {/* Grid for reference */}
      <gridHelper args={[20, 20]} position={[0, -4, 0]} />
    </>
  );
};

const ControlPanel: React.FC<{
  settings: VisualizationSettings;
  onSettingsChange: (settings: Partial<VisualizationSettings>) => void;
}> = ({ settings, onSettingsChange }) => {
  return (
    <div className="control-panel" style={{
      position: 'absolute',
      top: '20px',
      left: '20px',
      background: 'rgba(0, 0, 0, 0.8)',
      padding: '20px',
      borderRadius: '10px',
      color: 'white',
      zIndex: 1000,
      minWidth: '250px'
    }}>
      <h3>Visualization Controls</h3>
      
      <div className="control-group">
        <label>
          <input
            type="checkbox"
            checked={settings.showHeartModel}
            onChange={(e) => onSettingsChange({ showHeartModel: e.target.checked })}
          />
          Show Heart Model
        </label>
      </div>

      <div className="control-group">
        <label>
          <input
            type="checkbox"
            checked={settings.showECGWaveforms}
            onChange={(e) => onSettingsChange({ showECGWaveforms: e.target.checked })}
          />
          Show ECG Waveforms
        </label>
      </div>

      <div className="control-group">
        <label>
          <input
            type="checkbox"
            checked={settings.showAttentionMaps}
            onChange={(e) => onSettingsChange({ showAttentionMaps: e.target.checked })}
          />
          Show Attention Maps
        </label>
      </div>

      <div className="control-group">
        <label>
          <input
            type="checkbox"
            checked={settings.showClinicalFindings}
            onChange={(e) => onSettingsChange({ showClinicalFindings: e.target.checked })}
          />
          Show Clinical Findings
        </label>
      </div>

      <div className="control-group">
        <label>
          Animation Speed:
          <input
            type="range"
            min="0.1"
            max="2"
            step="0.1"
            value={settings.animationSpeed}
            onChange={(e) => onSettingsChange({ animationSpeed: parseFloat(e.target.value) })}
          />
          {settings.animationSpeed.toFixed(1)}x
        </label>
      </div>

      <div className="control-group">
        <label>
          Waveform Scale:
          <input
            type="range"
            min="0.5"
            max="3"
            step="0.1"
            value={settings.waveformScale}
            onChange={(e) => onSettingsChange({ waveformScale: parseFloat(e.target.value) })}
          />
          {settings.waveformScale.toFixed(1)}x
        </label>
      </div>

      <div className="control-group">
        <label>
          <input
            type="checkbox"
            checked={settings.immersiveMode}
            onChange={(e) => onSettingsChange({ immersiveMode: e.target.checked })}
          />
          Immersive Mode
        </label>
      </div>
    </div>
  );
};

export const ECGVisualization3D: React.FC<{
  ecgData: ECGData;
  analysisResult: ECGAnalysisResult;
  onVisualizationUpdate?: (data: Record<string, unknown>) => void;
}> = ({ ecgData, analysisResult, onVisualizationUpdate }) => {
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

  const [isVRSupported, setIsVRSupported] = useState(false);
  const [isARSupported, setIsARSupported] = useState(false);

  useEffect(() => {
    if ('xr' in navigator) {
      navigator.xr?.isSessionSupported('immersive-vr').then(setIsVRSupported);
      navigator.xr?.isSessionSupported('immersive-ar').then(setIsARSupported);
    }
  }, []);

  const handleSettingsChange = useCallback((newSettings: Partial<VisualizationSettings>) => {
    setSettings(prev => ({ ...prev, ...newSettings }));
    
    if (onVisualizationUpdate) {
      onVisualizationUpdate({
        settings: { ...settings, ...newSettings },
        timestamp: Date.now()
      });
    }
  }, [settings, onVisualizationUpdate]);

  return (
    <div className="ecg-visualization-3d" style={{ width: '100%', height: '100vh', position: 'relative' }}>
      {/* Control Panel */}
      <ControlPanel settings={settings} onSettingsChange={handleSettingsChange} />

      {/* VR/AR Buttons */}
      <div style={{ position: 'absolute', top: '20px', right: '20px', zIndex: 1000 }}>
        {isVRSupported && <VRButton />}
        {isARSupported && <ARButton />}
      </div>

      {/* 3D Canvas */}
      <Canvas
        camera={{ position: [0, 2, 10], fov: 75 }}
        style={{ background: 'linear-gradient(to bottom, #1a1a2e, #16213e)' }}
      >
        <XR>
          {/* VR/AR Controllers */}
          <Controllers />
          <Hands />

          {/* Main Scene */}
          <ECGScene
            ecgData={ecgData}
            analysisResult={analysisResult}
            settings={settings}
          />

          {/* Camera Controls */}
          <OrbitControls
            enablePan={true}
            enableZoom={true}
            enableRotate={true}
            maxDistance={20}
            minDistance={2}
          />
        </XR>
      </Canvas>

      {/* Performance Monitor */}
      <div style={{
        position: 'absolute',
        bottom: '20px',
        right: '20px',
        background: 'rgba(0, 0, 0, 0.7)',
        padding: '10px',
        borderRadius: '5px',
        color: 'white',
        fontSize: '12px'
      }}>
        <div>Leads: {Object.keys(ecgData.leads).length}</div>
        <div>Sample Rate: {ecgData.sampleRate} Hz</div>
        <div>Duration: {ecgData.duration.toFixed(1)}s</div>
        <div>Confidence: {(analysisResult.confidence * 100).toFixed(1)}%</div>
      </div>

      {/* Loading indicator for heavy computations */}
      {!ecgData.leads && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          color: 'white',
          fontSize: '18px'
        }}>
          Loading ECG Visualization...
        </div>
      )}
    </div>
  );
};

export default ECGVisualization3D;

export type { ECGVisualization3DProps, HeartModelProps };

export type { ECGData, ECGAnalysisResult, VisualizationSettings };
