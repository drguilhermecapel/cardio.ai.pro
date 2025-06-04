import React, { useState, useEffect, useRef } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Environment, PerspectiveCamera } from '@react-three/drei'
import { motion } from 'framer-motion'
import { Box, Typography, LinearProgress } from '@mui/material'
import { styled } from '@mui/material/styles'
import { futuristicTheme } from '../theme/futuristicTheme'

const DashboardContainer = styled(Box)(() => ({
  width: '100vw',
  height: '100vh',
  background: `linear-gradient(135deg, ${futuristicTheme.colors.background.primary} 0%, ${futuristicTheme.colors.background.secondary} 50%, ${futuristicTheme.colors.background.tertiary} 100%)`,
  position: 'relative',
  overflow: 'hidden',
  fontFamily: futuristicTheme.typography.fontFamily.primary,
  '&::before': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background: 'radial-gradient(circle at 50% 50%, rgba(0, 191, 255, 0.1) 0%, transparent 70%)',
    pointerEvents: 'none',
    zIndex: futuristicTheme.zIndex.background
  }
}))



const CentralCanvas = styled(Box)(() => ({
  position: 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  width: '600px',
  height: '600px',
  zIndex: futuristicTheme.zIndex.base,
  borderRadius: '50%',
  background: 'radial-gradient(circle, rgba(0, 191, 255, 0.1) 0%, transparent 70%)',
  '&::before': {
    content: '""',
    position: 'absolute',
    top: '-10px',
    left: '-10px',
    right: '-10px',
    bottom: '-10px',
    borderRadius: '50%',
    background: `conic-gradient(from 0deg, ${futuristicTheme.colors.neural.connections}, ${futuristicTheme.colors.neural.nodes}, ${futuristicTheme.colors.neural.pathways}, ${futuristicTheme.colors.neural.connections})`,
    opacity: 0.3,
    animation: 'rotate 20s linear infinite',
    zIndex: -1
  }
}))

const LeftPanel = styled(Box)(() => ({
  position: 'absolute',
  left: futuristicTheme.spacing.lg,
  top: '50%',
  transform: 'translateY(-50%)',
  width: '300px',
  height: '80vh',
  zIndex: futuristicTheme.zIndex.overlay
}))

const RightPanel = styled(Box)(() => ({
  position: 'absolute',
  right: futuristicTheme.spacing.lg,
  top: '50%',
  transform: 'translateY(-50%)',
  width: '300px',
  height: '80vh',
  zIndex: futuristicTheme.zIndex.overlay
}))

const BottomTimeline = styled(Box)(() => ({
  position: 'absolute',
  bottom: futuristicTheme.spacing.lg,
  left: '50%',
  transform: 'translateX(-50%)',
  width: '80vw',
  height: '120px',
  zIndex: futuristicTheme.zIndex.overlay
}))

const FloatingPanels = styled(Box)(() => ({
  position: 'absolute',
  top: futuristicTheme.spacing.xl,
  left: '50%',
  transform: 'translateX(-50%)',
  width: '90vw',
  height: '200px',
  zIndex: futuristicTheme.zIndex.floating,
  display: 'flex',
  justifyContent: 'space-around',
  alignItems: 'flex-start'
}))





export const FuturisticDashboard: React.FC = () => {

  const [isLoading, setIsLoading] = useState(true)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    setTimeout(() => setIsLoading(false), 2000)
  }, [])

  if (isLoading) {
    return (
      <DashboardContainer>
        <Box
          sx={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            textAlign: 'center',
            color: futuristicTheme.colors.data.text
          }}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 1 }}
          >
            <Typography variant="h3" sx={{ mb: 2, fontFamily: futuristicTheme.typography.fontFamily.primary }}>
              Initializing Hybrid ECG AI System
            </Typography>
            <LinearProgress
              sx={{
                width: '400px',
                height: '8px',
                borderRadius: '4px',
                backgroundColor: futuristicTheme.colors.ui.glass,
                '& .MuiLinearProgress-bar': {
                  backgroundColor: futuristicTheme.colors.data.primary,
                  boxShadow: futuristicTheme.effects.glow.primary
                }
              }}
            />
            <Typography variant="body1" sx={{ mt: 2, opacity: 0.8 }}>
              Loading neural networks and quantum encryption modules...
            </Typography>
          </motion.div>
        </Box>
      </DashboardContainer>
    )
  }

  return (
    <DashboardContainer>
      {/* Central 3D Heart Model */}
      <CentralCanvas>
        <Canvas ref={canvasRef}>
          <PerspectiveCamera makeDefault position={[0, 0, 5]} />
          <OrbitControls enableZoom={false} enablePan={false} />
          <Environment preset="night" />
          
          {/* 3D Heart Model Placeholder */}
          <mesh>
            <sphereGeometry args={[1.5, 32, 32]} />
            <meshPhongMaterial
              color={futuristicTheme.colors.data.primary}
              transparent
              opacity={0.6}
              wireframe
            />
          </mesh>
        </Canvas>
      </CentralCanvas>

      {/* Floating Holographic Panels */}
      <FloatingPanels>
        <Typography
          sx={{
            color: futuristicTheme.colors.data.text,
            fontSize: futuristicTheme.typography.sizes.lg,
            fontFamily: futuristicTheme.typography.fontFamily.primary,
            textAlign: 'center'
          }}
        >
          Futuristic Medical AI Dashboard
        </Typography>
      </FloatingPanels>

      {/* Left Panel - Edge AI Metrics */}
      <LeftPanel>
        <Typography
          sx={{
            color: futuristicTheme.colors.data.secondary,
            fontSize: futuristicTheme.typography.sizes.base,
            fontFamily: futuristicTheme.typography.fontFamily.primary,
            textAlign: 'center'
          }}
        >
          Edge AI Metrics
        </Typography>
      </LeftPanel>

      {/* Right Panel - Explainable AI */}
      <RightPanel>
        <Typography
          sx={{
            color: futuristicTheme.colors.neural.pathways,
            fontSize: futuristicTheme.typography.sizes.base,
            fontFamily: futuristicTheme.typography.fontFamily.primary,
            textAlign: 'center'
          }}
        >
          Explainable AI
        </Typography>
      </RightPanel>

      {/* Bottom Timeline */}
      <BottomTimeline>
        <Typography
          sx={{
            color: futuristicTheme.colors.data.text,
            fontSize: futuristicTheme.typography.sizes.base,
            fontFamily: futuristicTheme.typography.fontFamily.primary,
            textAlign: 'center'
          }}
        >
          Continuous Learning Timeline
        </Typography>
      </BottomTimeline>

      {/* Global Styles */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600&family=Fira+Code:wght@400;500&display=swap');
        
        @keyframes rotate {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.7; }
        }
        
        @keyframes glow {
          0%, 100% { box-shadow: 0 0 20px rgba(0, 191, 255, 0.5); }
          50% { box-shadow: 0 0 30px rgba(0, 191, 255, 0.8); }
        }
        
        body {
          margin: 0;
          padding: 0;
          overflow: hidden;
        }
        
        .holographic-text {
          background: linear-gradient(45deg, #00bfff, #00ff7f, #ffa500);
          background-size: 200% 200%;
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          animation: gradient 3s ease infinite;
        }
        
        @keyframes gradient {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
        
        .neural-connection {
          stroke: #00ff7f;
          stroke-width: 2;
          filter: drop-shadow(0 0 5px #00ff7f);
          animation: pulse 2s ease-in-out infinite;
        }
        
        .data-stream {
          background: linear-gradient(90deg, transparent, #00bfff, transparent);
          animation: stream 2s linear infinite;
        }
        
        @keyframes stream {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
      `}</style>
    </DashboardContainer>
  )
}

export default FuturisticDashboard
