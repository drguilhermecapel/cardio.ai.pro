import React from 'react'
import { Box, Typography } from '@mui/material'
import { styled } from '@mui/material/styles'
import { motion } from 'framer-motion'
import { futuristicTheme } from '../theme/futuristicTheme'

const PanelContainer = styled(motion.div)<{ width?: string }>(({ width }) => ({
  width: width || '250px',
  background: `linear-gradient(135deg, rgba(0, 191, 255, 0.1) 0%, rgba(0, 255, 127, 0.05) 100%)`,
  backdropFilter: futuristicTheme.effects.blur.glass,
  border: `1px solid ${futuristicTheme.colors.ui.border}`,
  borderRadius: futuristicTheme.borderRadius.lg,
  padding: futuristicTheme.spacing.md,
  boxShadow: futuristicTheme.effects.glow.primary,
  position: 'relative',
  overflow: 'hidden',
  '&::before': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background: `linear-gradient(45deg, transparent 30%, ${futuristicTheme.colors.ui.glow} 50%, transparent 70%)`,
    opacity: 0.1,
    animation: 'holographic-sweep 3s ease-in-out infinite',
    pointerEvents: 'none'
  },
  '&::after': {
    content: '""',
    position: 'absolute',
    top: '-2px',
    left: '-2px',
    right: '-2px',
    bottom: '-2px',
    background: `conic-gradient(from 0deg, ${futuristicTheme.colors.data.primary}, ${futuristicTheme.colors.data.secondary}, ${futuristicTheme.colors.neural.pathways}, ${futuristicTheme.colors.data.primary})`,
    borderRadius: futuristicTheme.borderRadius.lg,
    opacity: 0.3,
    animation: 'border-glow 4s linear infinite',
    zIndex: -1
  }
}))

const PanelHeader = styled(Box)(() => ({
  borderBottom: `1px solid ${futuristicTheme.colors.ui.border}`,
  paddingBottom: futuristicTheme.spacing.sm,
  marginBottom: futuristicTheme.spacing.md,
  position: 'relative'
}))

const PanelTitle = styled(Typography)(() => ({
  color: futuristicTheme.colors.data.text,
  fontFamily: futuristicTheme.typography.fontFamily.primary,
  fontSize: futuristicTheme.typography.sizes.sm,
  fontWeight: 'bold',
  textTransform: 'uppercase',
  letterSpacing: '0.1em',
  background: `linear-gradient(45deg, ${futuristicTheme.colors.data.primary}, ${futuristicTheme.colors.data.secondary})`,
  backgroundClip: 'text',
  WebkitBackgroundClip: 'text',
  WebkitTextFillColor: 'transparent',
  position: 'relative',
  '&::after': {
    content: '""',
    position: 'absolute',
    bottom: '-4px',
    left: 0,
    width: '100%',
    height: '1px',
    background: `linear-gradient(90deg, ${futuristicTheme.colors.data.primary}, transparent)`,
    animation: 'title-glow 2s ease-in-out infinite alternate'
  }
}))

const PanelContent = styled(Box)(() => ({
  color: futuristicTheme.colors.data.text,
  fontSize: futuristicTheme.typography.sizes.sm,
  lineHeight: 1.4,
  position: 'relative',
  zIndex: 1
}))

const StatusIndicator = styled(Box)<{ status?: 'active' | 'warning' | 'critical' | 'normal' }>(({ status }) => {
  const getStatusColor = (): string => {
    switch (status) {
      case 'active':
        return futuristicTheme.colors.data.secondary
      case 'warning':
        return futuristicTheme.colors.data.warning
      case 'critical':
        return futuristicTheme.colors.data.critical
      default:
        return futuristicTheme.colors.data.primary
    }
  }

  return {
    position: 'absolute',
    top: futuristicTheme.spacing.sm,
    right: futuristicTheme.spacing.sm,
    width: '8px',
    height: '8px',
    borderRadius: '50%',
    backgroundColor: getStatusColor(),
    boxShadow: `0 0 10px ${getStatusColor()}`,
    animation: 'pulse 2s ease-in-out infinite'
  }
})

interface HolographicPanelProps {
  title: string
  children: React.ReactNode
  width?: string
  status?: 'active' | 'warning' | 'critical' | 'normal'
  className?: string
}

export const HolographicPanel: React.FC<HolographicPanelProps> = ({
  title,
  children,
  width,
  status = 'normal',
  className
}) => {
  return (
    <PanelContainer
      width={width}
      className={className}
      initial={{ opacity: 0, y: 20, scale: 0.9 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
      whileHover={{
        scale: 1.02,
        boxShadow: futuristicTheme.effects.glow.secondary,
        transition: { duration: 0.2 }
      }}
    >
      <StatusIndicator status={status} />
      
      <PanelHeader>
        <PanelTitle variant="h6">
          {title}
        </PanelTitle>
      </PanelHeader>
      
      <PanelContent>
        {children}
      </PanelContent>
      
      {/* Holographic scan lines effect */}
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: `repeating-linear-gradient(
            0deg,
            transparent,
            transparent 2px,
            rgba(0, 191, 255, 0.03) 2px,
            rgba(0, 191, 255, 0.03) 4px
          )`,
          pointerEvents: 'none',
          animation: 'scan-lines 3s linear infinite'
        }}
      />
      
      {/* Global styles for animations */}
      <style>{`
        @keyframes holographic-sweep {
          0% { transform: translateX(-100%) skewX(-15deg); }
          100% { transform: translateX(200%) skewX(-15deg); }
        }
        
        @keyframes border-glow {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        
        @keyframes title-glow {
          0% { opacity: 0.5; transform: scaleX(0.8); }
          100% { opacity: 1; transform: scaleX(1); }
        }
        
        @keyframes scan-lines {
          0% { transform: translateY(-100%); }
          100% { transform: translateY(100%); }
        }
        
        @keyframes pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.7; transform: scale(1.2); }
        }
      `}</style>
    </PanelContainer>
  )
}

export default HolographicPanel
