import React, { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Sphere, Line, Points } from '@react-three/drei'
import * as THREE from 'three'
import { futuristicTheme } from '../theme/futuristicTheme'

interface ECGData {
  timestamp: number
  leads: {
    I: number[]
    II: number[]
    III: number[]
    aVR: number[]
    aVL: number[]
    aVF: number[]
    V1: number[]
    V2: number[]
    V3: number[]
    V4: number[]
    V5: number[]
    V6: number[]
  }
  heartRate: number
  rhythm: string
  confidence: number
}

interface Heart3DModelProps {
  ecgData: ECGData | null
}

export const Heart3DModel: React.FC<Heart3DModelProps> = ({ ecgData }) => {
  const heartRef = useRef<THREE.Group>(null)
  const neuralConnectionsRef = useRef<THREE.Group>(null)
  const electricalPathwaysRef = useRef<THREE.Group>(null)

  const heartGeometry = useMemo(() => {
    const points: THREE.Vector3[] = []
    const segments = 64
    
    for (let i = 0; i < segments; i++) {
      const t = (i / segments) * Math.PI * 2
      
      const x = 16 * Math.pow(Math.sin(t), 3)
      const y = 13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t)
      const z = Math.sin(t) * 2
      
      points.push(new THREE.Vector3(x * 0.05, y * 0.05, z * 0.05))
    }
    
    return points
  }, [])

  const neuralConnections = useMemo(() => {
    const connections: THREE.Vector3[] = []
    const nodeCount = 20
    
    for (let i = 0; i < nodeCount; i++) {
      const angle = (i / nodeCount) * Math.PI * 2
      const radius = 1.5 + Math.sin(angle * 3) * 0.3
      
      const x = Math.cos(angle) * radius
      const y = Math.sin(angle) * radius
      const z = Math.sin(angle * 2) * 0.5
      
      connections.push(new THREE.Vector3(x, y, z))
      
      if (i < nodeCount - 1) {
        const nextAngle = ((i + 1) / nodeCount) * Math.PI * 2
        const nextRadius = 1.5 + Math.sin(nextAngle * 3) * 0.3
        const nextX = Math.cos(nextAngle) * nextRadius
        const nextY = Math.sin(nextAngle) * nextRadius
        const nextZ = Math.sin(nextAngle * 2) * 0.5
        
        connections.push(new THREE.Vector3(nextX, nextY, nextZ))
      }
    }
    
    return connections
  }, [])

  const electricalPathways = useMemo(() => {
    if (!ecgData) return []
    
    const pathways: THREE.Vector3[][] = []
    
    pathways.push([
      new THREE.Vector3(0.3, 0.8, 0.1),
      new THREE.Vector3(0.1, 0.2, 0.05),
      new THREE.Vector3(-0.1, -0.2, 0)
    ])
    
    pathways.push([
      new THREE.Vector3(-0.1, -0.2, 0),
      new THREE.Vector3(-0.2, -0.6, -0.1),
      new THREE.Vector3(-0.3, -0.8, -0.2)
    ])
    
    pathways.push([
      new THREE.Vector3(-0.3, -0.8, -0.2),
      new THREE.Vector3(-0.6, -0.9, -0.1),
      new THREE.Vector3(-0.8, -0.7, 0.1)
    ])
    
    pathways.push([
      new THREE.Vector3(-0.3, -0.8, -0.2),
      new THREE.Vector3(0.2, -0.9, -0.1),
      new THREE.Vector3(0.6, -0.7, 0.1)
    ])
    
    return pathways
  }, [ecgData])

  useFrame((state) => {
    if (heartRef.current) {
      heartRef.current.rotation.y += 0.005
      
      const heartRate = ecgData?.heartRate || 72
      const beatFrequency = heartRate / 60
      const scale = 1 + Math.sin(state.clock.elapsedTime * beatFrequency * Math.PI * 2) * 0.1
      heartRef.current.scale.setScalar(scale)
    }
    
    if (neuralConnectionsRef.current) {
      neuralConnectionsRef.current.rotation.z += 0.002
    }
    
    if (electricalPathwaysRef.current) {
      electricalPathwaysRef.current.children.forEach((child, index) => {
        if (child instanceof THREE.Mesh) {
          const material = child.material as THREE.MeshBasicMaterial
          const opacity = 0.5 + Math.sin(state.clock.elapsedTime * 3 + index) * 0.3
          material.opacity = Math.max(0.2, opacity)
        }
      })
    }
  })

  return (
    <group ref={heartRef}>
      {/* Main Heart Structure */}
      <Line
        points={heartGeometry}
        color={futuristicTheme.colors.data.secondary}
        lineWidth={3}
        transparent
        opacity={0.8}
      />
      
      {/* Heart Surface */}
      <mesh>
        <sphereGeometry args={[1.2, 32, 32]} />
        <meshPhongMaterial
          color={futuristicTheme.colors.data.primary}
          transparent
          opacity={0.3}
          wireframe
        />
      </mesh>
      
      {/* Neural Network Connections */}
      <group ref={neuralConnectionsRef}>
        {neuralConnections.map((point, index) => (
          <Sphere
            key={`neural-${index}`}
            position={[point.x, point.y, point.z]}
            args={[0.02]}
          >
            <meshBasicMaterial
              color={futuristicTheme.colors.neural.nodes}
              transparent
              opacity={0.8}
            />
          </Sphere>
        ))}
        
        {/* Connection lines between neural nodes */}
        <Line
          points={neuralConnections}
          color={futuristicTheme.colors.neural.connections}
          lineWidth={1}
          transparent
          opacity={0.6}
        />
      </group>
      
      {/* Electrical Pathways */}
      <group ref={electricalPathwaysRef}>
        {electricalPathways.map((pathway, index) => (
          <Line
            key={`pathway-${index}`}
            points={pathway}
            color={futuristicTheme.colors.neural.pathways}
            lineWidth={2}
            transparent
            opacity={0.7}
          />
        ))}
      </group>
      
      {/* Ambient lighting effects */}
      <pointLight
        position={[2, 2, 2]}
        color={futuristicTheme.colors.data.primary}
        intensity={0.5}
      />
      <pointLight
        position={[-2, -2, 2]}
        color={futuristicTheme.colors.data.secondary}
        intensity={0.3}
      />
      
      {/* Particle effects for data flow */}
      <Points limit={1000}>
        <pointsMaterial
          size={0.01}
          color={futuristicTheme.colors.data.primary}
          transparent
          opacity={0.6}
        />
      </Points>
    </group>
  )
}

export default Heart3DModel
