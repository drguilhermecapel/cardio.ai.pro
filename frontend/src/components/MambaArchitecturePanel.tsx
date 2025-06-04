import React, { useState, useEffect } from 'react'
import { Box, Typography } from '@mui/material'
import { styled } from '@mui/material/styles'
import { HolographicPanel } from './HolographicPanel'
import { futuristicTheme } from '../theme/futuristicTheme'

const ArchitectureContainer = styled(Box)(({ theme: _theme }) => ({
  position: 'relative',
  height: '200px',
  overflow: 'hidden',
  background: `radial-gradient(circle at center, rgba(0, 191, 255, 0.1) 0%, transparent 70%)`,
  borderRadius: futuristicTheme.borderRadius.md,
}))

const DataFlowPath = styled(Box)<{ direction: 'forward' | 'backward' }>(
  ({ theme: _theme, direction }) => ({
    position: 'absolute',
    width: '100%',
    height: '2px',
    background: `linear-gradient(90deg, transparent, ${futuristicTheme.colors.neural.connections}, transparent)`,
    animation: `dataFlow${direction === 'forward' ? 'Forward' : 'Backward'} 2s ease-in-out infinite`,
    boxShadow: `0 0 5px ${futuristicTheme.colors.neural.connections}`,
  })
)

const ProcessingNode = styled(Box)<{ active?: boolean; size?: 'small' | 'medium' | 'large' }>(({
  theme: _theme,
  active,
  size = 'medium',
}) => {
  const getSize = (): string => {
    switch (size) {
      case 'small':
        return '8px'
      case 'large':
        return '16px'
      default:
        return '12px'
    }
  }

  return {
    position: 'absolute',
    width: getSize(),
    height: getSize(),
    borderRadius: '50%',
    backgroundColor: active
      ? futuristicTheme.colors.neural.nodes
      : futuristicTheme.colors.ui.border,
    boxShadow: active ? `0 0 10px ${futuristicTheme.colors.neural.nodes}` : 'none',
    animation: active ? 'pulse 1.5s ease-in-out infinite' : 'none',
    transition: 'all 0.3s ease',
  }
})

const MetricDisplay = styled(Box)(({ theme: _theme }) => ({
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  padding: futuristicTheme.spacing.sm,
  background: `rgba(0, 0, 0, 0.4)`,
  borderRadius: futuristicTheme.borderRadius.sm,
  border: `1px solid ${futuristicTheme.colors.ui.border}`,
  marginBottom: futuristicTheme.spacing.sm,
}))

const LayerVisualization = styled(Box)(({ theme: _theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  gap: futuristicTheme.spacing.xs,
  marginTop: futuristicTheme.spacing.md,
}))

const LayerBar = styled(Box)<{ activity: number }>(({ theme: _theme, activity }) => ({
  height: '4px',
  background: `linear-gradient(90deg, 
    ${futuristicTheme.colors.neural.pathways} 0%, 
    ${futuristicTheme.colors.neural.pathways} ${activity}%, 
    rgba(255, 165, 0, 0.2) ${activity}%, 
    rgba(255, 165, 0, 0.2) 100%)`,
  borderRadius: '2px',
  position: 'relative',
  overflow: 'hidden',
  '&::after': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background: `linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent)`,
    animation: 'shimmer 2s ease-in-out infinite',
  },
}))

interface MambaMetrics {
  sequenceLength: number
  hiddenStates: number
  attentionHeads: number
  processingLatency: number
  memoryEfficiency: number
  bidirectionalFlow: boolean
}

interface MambaArchitecturePanelProps {
  className?: string
}

export const MambaArchitecturePanel: React.FC<MambaArchitecturePanelProps> = ({ className }) => {
  const [metrics, setMetrics] = useState<MambaMetrics>({
    sequenceLength: 1024,
    hiddenStates: 768,
    attentionHeads: 12,
    processingLatency: 2.3,
    memoryEfficiency: 94.7,
    bidirectionalFlow: true,
  })

  const [layerActivities, setLayerActivities] = useState<number[]>([])
  const [activeNodes, setActiveNodes] = useState<number[]>([])

  useEffect(() => {
    const interval = setInterval(() => {
      setMetrics(prev => ({
        ...prev,
        processingLatency: 2.1 + Math.random() * 0.4,
        memoryEfficiency: 93.5 + Math.random() * 2.0,
        sequenceLength: 1024 + Math.floor(Math.random() * 512),
        bidirectionalFlow: Math.random() > 0.1, // Mostly true, occasionally false
      }))

      const newActivities = Array.from({ length: 12 }, () => 60 + Math.random() * 40)
      setLayerActivities(newActivities)

      const nodeCount = 8
      const newActiveNodes = Array.from({ length: nodeCount }, (_, i) =>
        Math.random() > 0.3 ? i : -1
      ).filter(i => i !== -1)
      setActiveNodes(newActiveNodes)
    }, 1500)

    return () => clearInterval(interval)
  }, [])

  const nodePositions = [
    { top: '20%', left: '15%' },
    { top: '35%', left: '25%' },
    { top: '50%', left: '35%' },
    { top: '65%', left: '45%' },
    { top: '65%', left: '55%' },
    { top: '50%', left: '65%' },
    { top: '35%', left: '75%' },
    { top: '20%', left: '85%' },
  ]

  return (
    <HolographicPanel
      title="Mamba Neural Architecture"
      status="active"
      className={className}
      width="350px"
    >
      <ArchitectureContainer>
        {/* Bidirectional data flow paths */}
        <DataFlowPath direction="forward" sx={{ top: '30%' }} />
        <DataFlowPath direction="backward" sx={{ top: '70%' }} />

        {/* Processing nodes */}
        {nodePositions.map((pos, index) => (
          <ProcessingNode
            key={index}
            active={activeNodes.includes(index)}
            size={index === 3 || index === 4 ? 'large' : 'medium'}
            sx={{
              top: pos.top,
              left: pos.left,
              transform: 'translate(-50%, -50%)',
            }}
          />
        ))}

        {/* Central processing indicator */}
        <Box
          sx={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            width: '40px',
            height: '40px',
            border: `2px solid ${futuristicTheme.colors.neural.connections}`,
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: `radial-gradient(circle, rgba(0, 191, 255, 0.2), transparent)`,
            animation: 'pulse 2s ease-in-out infinite',
          }}
        >
          <Typography
            sx={{
              color: futuristicTheme.colors.neural.connections,
              fontSize: futuristicTheme.typography.sizes.xs,
              fontFamily: futuristicTheme.typography.fontFamily.mono,
              fontWeight: 'bold',
            }}
          >
            MAMBA
          </Typography>
        </Box>
      </ArchitectureContainer>

      {/* Architecture metrics */}
      <Box sx={{ mt: 2 }}>
        <MetricDisplay>
          <Typography
            variant="body2"
            sx={{
              color: futuristicTheme.colors.data.text,
              fontSize: futuristicTheme.typography.sizes.xs,
              fontFamily: futuristicTheme.typography.fontFamily.mono,
            }}
          >
            Sequence Length
          </Typography>
          <Typography
            variant="body2"
            sx={{
              color: futuristicTheme.colors.data.primary,
              fontSize: futuristicTheme.typography.sizes.sm,
              fontFamily: futuristicTheme.typography.fontFamily.primary,
              fontWeight: 'bold',
            }}
          >
            {metrics.sequenceLength}
          </Typography>
        </MetricDisplay>

        <MetricDisplay>
          <Typography
            variant="body2"
            sx={{
              color: futuristicTheme.colors.data.text,
              fontSize: futuristicTheme.typography.sizes.xs,
              fontFamily: futuristicTheme.typography.fontFamily.mono,
            }}
          >
            Processing Latency
          </Typography>
          <Typography
            variant="body2"
            sx={{
              color: futuristicTheme.colors.data.secondary,
              fontSize: futuristicTheme.typography.sizes.sm,
              fontFamily: futuristicTheme.typography.fontFamily.primary,
              fontWeight: 'bold',
            }}
          >
            {metrics.processingLatency.toFixed(1)}ms
          </Typography>
        </MetricDisplay>

        <MetricDisplay>
          <Typography
            variant="body2"
            sx={{
              color: futuristicTheme.colors.data.text,
              fontSize: futuristicTheme.typography.sizes.xs,
              fontFamily: futuristicTheme.typography.fontFamily.mono,
            }}
          >
            Memory Efficiency
          </Typography>
          <Typography
            variant="body2"
            sx={{
              color: futuristicTheme.colors.data.secondary,
              fontSize: futuristicTheme.typography.sizes.sm,
              fontFamily: futuristicTheme.typography.fontFamily.primary,
              fontWeight: 'bold',
            }}
          >
            {metrics.memoryEfficiency.toFixed(1)}%
          </Typography>
        </MetricDisplay>
      </Box>

      {/* Layer activity visualization */}
      <Box sx={{ mt: 2 }}>
        <Typography
          variant="body2"
          sx={{
            color: futuristicTheme.colors.data.text,
            fontSize: futuristicTheme.typography.sizes.xs,
            fontFamily: futuristicTheme.typography.fontFamily.mono,
            mb: 1,
          }}
        >
          Layer Activities:
        </Typography>

        <LayerVisualization>
          {layerActivities.map((activity, index) => (
            <Box key={index} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Typography
                sx={{
                  color: futuristicTheme.colors.data.text,
                  fontSize: futuristicTheme.typography.sizes.xs,
                  fontFamily: futuristicTheme.typography.fontFamily.mono,
                  width: '20px',
                }}
              >
                L{index + 1}
              </Typography>
              <LayerBar activity={activity} sx={{ flex: 1 }} />
              <Typography
                sx={{
                  color: futuristicTheme.colors.neural.pathways,
                  fontSize: futuristicTheme.typography.sizes.xs,
                  fontFamily: futuristicTheme.typography.fontFamily.mono,
                  width: '35px',
                  textAlign: 'right',
                }}
              >
                {activity.toFixed(0)}%
              </Typography>
            </Box>
          ))}
        </LayerVisualization>
      </Box>

      {/* Global styles for animations */}
      <style>{`
        @keyframes dataFlowForward {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
        
        @keyframes dataFlowBackward {
          0% { transform: translateX(100%); }
          100% { transform: translateX(-100%); }
        }
        
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
      `}</style>
    </HolographicPanel>
  )
}

export default MambaArchitecturePanel
