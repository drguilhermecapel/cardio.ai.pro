import React, { useState, useEffect } from 'react'
import { Box, Typography, LinearProgress } from '@mui/material'
import { styled } from '@mui/material/styles'
import { HolographicPanel } from './HolographicPanel'
import { futuristicTheme } from '../theme/futuristicTheme'

const ConditionCard = styled(Box)(({ theme: _theme }) => ({
  background: `rgba(0, 0, 0, 0.4)`,
  border: `1px solid ${futuristicTheme.colors.ui.border}`,
  borderRadius: futuristicTheme.borderRadius.md,
  padding: futuristicTheme.spacing.sm,
  marginBottom: futuristicTheme.spacing.sm,
  position: 'relative',
  overflow: 'hidden',
  '&::before': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    width: '3px',
    height: '100%',
    background: `linear-gradient(180deg, ${futuristicTheme.colors.neural.pathways}, transparent)`,
    animation: 'conditionGlow 3s ease-in-out infinite'
  }
}))

const RarityIndicator = styled(Box)<{ rarity: 'ultra-rare' | 'rare' | 'uncommon' }>(({ theme: _theme, rarity }) => {
  const getRarityColor = (): string => {
    switch (rarity) {
      case 'ultra-rare':
        return futuristicTheme.colors.data.critical
      case 'rare':
        return futuristicTheme.colors.data.warning
      default:
        return futuristicTheme.colors.data.secondary
    }
  }

  return {
    display: 'inline-flex',
    alignItems: 'center',
    gap: futuristicTheme.spacing.xs,
    padding: `${futuristicTheme.spacing.xs} ${futuristicTheme.spacing.sm}`,
    background: `rgba(${getRarityColor().replace('#', '').match(/.{2}/g)?.map(hex => parseInt(hex, 16)).join(', ')}, 0.2)`,
    border: `1px solid ${getRarityColor()}`,
    borderRadius: futuristicTheme.borderRadius.sm,
    fontSize: futuristicTheme.typography.sizes.xs,
    color: getRarityColor(),
    fontFamily: futuristicTheme.typography.fontFamily.mono,
    fontWeight: 'bold',
    textTransform: 'uppercase',
    boxShadow: `0 0 5px ${getRarityColor()}`
  }
})

const LearningProgress = styled(Box)(({ theme: _theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: futuristicTheme.spacing.sm,
  marginTop: futuristicTheme.spacing.sm,
  padding: futuristicTheme.spacing.sm,
  background: `linear-gradient(45deg, rgba(0, 191, 255, 0.1), rgba(0, 255, 127, 0.1))`,
  borderRadius: futuristicTheme.borderRadius.sm,
  border: `1px solid ${futuristicTheme.colors.ui.border}`
}))

const NeuralNetworkVisualization = styled(Box)(({ theme: _theme }) => ({
  height: '80px',
  background: `radial-gradient(circle at center, rgba(0, 191, 255, 0.1) 0%, transparent 70%)`,
  borderRadius: futuristicTheme.borderRadius.md,
  position: 'relative',
  overflow: 'hidden',
  marginTop: futuristicTheme.spacing.md,
  border: `1px solid ${futuristicTheme.colors.ui.border}`
}))

const NeuralNode = styled(Box)<{ active?: boolean; size?: number }>(({ theme: _theme, active, size = 4 }) => ({
  position: 'absolute',
  width: `${size}px`,
  height: `${size}px`,
  borderRadius: '50%',
  backgroundColor: active ? futuristicTheme.colors.neural.nodes : futuristicTheme.colors.ui.border,
  boxShadow: active ? `0 0 8px ${futuristicTheme.colors.neural.nodes}` : 'none',
  animation: active ? 'pulse 2s ease-in-out infinite' : 'none',
  transition: 'all 0.3s ease'
}))

interface RareCondition {
  name: string
  confidence: number
  rarity: 'ultra-rare' | 'rare' | 'uncommon'
  prevalence: string
  description: string
  learningSource: string
}

interface ZeroShotLearningPanelProps {
  className?: string
}

export const ZeroShotLearningPanel: React.FC<ZeroShotLearningPanelProps> = ({ className }) => {
  const [detectedConditions, setDetectedConditions] = useState<RareCondition[]>([])
  const [learningProgress, setLearningProgress] = useState(0)
  const [isLearning, setIsLearning] = useState(false)
  const [activeNodes, setActiveNodes] = useState<number[]>([])

  useEffect(() => {
    const rareConditions: RareCondition[] = [
      {
        name: 'Brugada Syndrome Type 3',
        confidence: 0.89,
        rarity: 'ultra-rare',
        prevalence: '1:10,000',
        description: 'Rare genetic arrhythmia disorder',
        learningSource: 'Literature synthesis'
      },
      {
        name: 'Catecholaminergic Polymorphic VT',
        confidence: 0.76,
        rarity: 'ultra-rare',
        prevalence: '1:10,000',
        description: 'Exercise-induced ventricular arrhythmia',
        learningSource: 'Case study analysis'
      },
      {
        name: 'Long QT Syndrome Type 8',
        confidence: 0.82,
        rarity: 'rare',
        prevalence: '1:5,000',
        description: 'Rare variant of LQTS',
        learningSource: 'Genetic database'
      },
      {
        name: 'Arrhythmogenic Cardiomyopathy',
        confidence: 0.71,
        rarity: 'rare',
        prevalence: '1:2,500',
        description: 'Progressive heart muscle disease',
        learningSource: 'Multi-modal learning'
      },
      {
        name: 'Epsilon Wave Pattern',
        confidence: 0.94,
        rarity: 'uncommon',
        prevalence: '1:1,000',
        description: 'Characteristic ARVC finding',
        learningSource: 'Pattern recognition'
      }
    ]

    const interval = setInterval(() => {
      setIsLearning(Math.random() > 0.7)
      setLearningProgress(prev => {
        const newProgress = Math.min(100, prev + Math.random() * 5)
        return newProgress
      })

      if (Math.random() > 0.6) {
        const randomCondition = rareConditions[Math.floor(Math.random() * rareConditions.length)]
        setDetectedConditions(prev => {
          const exists = prev.find(c => c.name === randomCondition.name)
          if (!exists) {
            return [randomCondition, ...prev.slice(0, 2)] // Keep last 3 detections
          }
          return prev
        })
      }

      const nodeCount = 12
      const newActiveNodes = Array.from({ length: nodeCount }, (_, i) => 
        Math.random() > 0.4 ? i : -1
      ).filter(i => i !== -1)
      setActiveNodes(newActiveNodes)
    }, 2500)

    return () => clearInterval(interval)
  }, [])

  const nodePositions = [
    { top: '15%', left: '10%' },
    { top: '15%', left: '30%' },
    { top: '15%', left: '50%' },
    { top: '15%', left: '70%' },
    { top: '15%', left: '90%' },
    { top: '50%', left: '20%' },
    { top: '50%', left: '40%' },
    { top: '50%', left: '60%' },
    { top: '50%', left: '80%' },
    { top: '85%', left: '25%' },
    { top: '85%', left: '50%' },
    { top: '85%', left: '75%' }
  ]

  return (
    <HolographicPanel
      title="Zero-Shot Learning Module"
      status={isLearning ? 'active' : 'normal'}
      className={className}
      width="380px"
    >
      <LearningProgress>
        <Box sx={{ flex: 1 }}>
          <Typography
            variant="body2"
            sx={{
              color: futuristicTheme.colors.data.text,
              fontSize: futuristicTheme.typography.sizes.xs,
              fontFamily: futuristicTheme.typography.fontFamily.mono,
              mb: 0.5
            }}
          >
            Knowledge Acquisition Progress
          </Typography>
          <LinearProgress
            variant="determinate"
            value={learningProgress}
            sx={{
              height: '6px',
              borderRadius: '3px',
              backgroundColor: `rgba(0, 191, 255, 0.2)`,
              '& .MuiLinearProgress-bar': {
                backgroundColor: futuristicTheme.colors.neural.connections,
                boxShadow: `0 0 8px ${futuristicTheme.colors.neural.connections}`
              }
            }}
          />
        </Box>
        <Typography
          sx={{
            color: futuristicTheme.colors.neural.connections,
            fontSize: futuristicTheme.typography.sizes.sm,
            fontFamily: futuristicTheme.typography.fontFamily.primary,
            fontWeight: 'bold'
          }}
        >
          {learningProgress.toFixed(1)}%
        </Typography>
      </LearningProgress>

      <NeuralNetworkVisualization>
        {nodePositions.map((pos, index) => (
          <NeuralNode
            key={index}
            active={activeNodes.includes(index)}
            size={index < 5 ? 6 : index < 9 ? 5 : 4}
            sx={{
              top: pos.top,
              left: pos.left,
              transform: 'translate(-50%, -50%)'
            }}
          />
        ))}
        
        {/* Learning indicator */}
        {isLearning && (
          <Box
            sx={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              color: futuristicTheme.colors.neural.connections,
              fontSize: futuristicTheme.typography.sizes.xs,
              fontFamily: futuristicTheme.typography.fontFamily.mono,
              fontWeight: 'bold',
              animation: 'pulse 1s ease-in-out infinite'
            }}
          >
            LEARNING...
          </Box>
        )}
      </NeuralNetworkVisualization>

      <Box sx={{ mt: 2 }}>
        <Typography
          variant="body2"
          sx={{
            color: futuristicTheme.colors.data.text,
            fontSize: futuristicTheme.typography.sizes.xs,
            fontFamily: futuristicTheme.typography.fontFamily.mono,
            mb: 1
          }}
        >
          Rare Condition Detections:
        </Typography>

        {detectedConditions.length === 0 ? (
          <Box
            sx={{
              textAlign: 'center',
              py: 2,
              color: futuristicTheme.colors.data.text,
              fontSize: futuristicTheme.typography.sizes.xs,
              opacity: 0.7
            }}
          >
            No rare conditions detected
          </Box>
        ) : (
          detectedConditions.map((condition, index) => (
            <ConditionCard key={`${condition.name}-${index}`}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                <Typography
                  variant="body2"
                  sx={{
                    color: futuristicTheme.colors.data.text,
                    fontSize: futuristicTheme.typography.sizes.sm,
                    fontFamily: futuristicTheme.typography.fontFamily.primary,
                    fontWeight: 'bold',
                    flex: 1
                  }}
                >
                  {condition.name}
                </Typography>
                <Typography
                  variant="body2"
                  sx={{
                    color: futuristicTheme.colors.data.secondary,
                    fontSize: futuristicTheme.typography.sizes.sm,
                    fontFamily: futuristicTheme.typography.fontFamily.primary,
                    fontWeight: 'bold'
                  }}
                >
                  {(condition.confidence * 100).toFixed(1)}%
                </Typography>
              </Box>

              <Typography
                variant="body2"
                sx={{
                  color: futuristicTheme.colors.data.text,
                  fontSize: futuristicTheme.typography.sizes.xs,
                  opacity: 0.8,
                  mb: 1
                }}
              >
                {condition.description}
              </Typography>

              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <RarityIndicator rarity={condition.rarity}>
                  {condition.rarity} • {condition.prevalence}
                </RarityIndicator>
                
                <Typography
                  variant="body2"
                  sx={{
                    color: futuristicTheme.colors.neural.pathways,
                    fontSize: futuristicTheme.typography.sizes.xs,
                    fontFamily: futuristicTheme.typography.fontFamily.mono,
                    fontStyle: 'italic'
                  }}
                >
                  {condition.learningSource}
                </Typography>
              </Box>
            </ConditionCard>
          ))
        )}
      </Box>

      {/* Learning status indicator */}
      <Box
        sx={{
          mt: 2,
          pt: 2,
          borderTop: `1px solid ${futuristicTheme.colors.ui.border}`,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box
            sx={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: isLearning ? futuristicTheme.colors.neural.connections : futuristicTheme.colors.ui.border,
              animation: isLearning ? 'pulse 1s ease-in-out infinite' : 'none',
              boxShadow: isLearning ? `0 0 10px ${futuristicTheme.colors.neural.connections}` : 'none'
            }}
          />
          <Typography
            variant="body2"
            sx={{
              color: isLearning ? futuristicTheme.colors.neural.connections : futuristicTheme.colors.data.text,
              fontSize: futuristicTheme.typography.sizes.xs,
              fontFamily: futuristicTheme.typography.fontFamily.mono
            }}
          >
            {isLearning ? 'ACTIVE LEARNING' : 'STANDBY'}
          </Typography>
        </Box>
        
        <Typography
          variant="body2"
          sx={{
            color: futuristicTheme.colors.data.text,
            fontSize: futuristicTheme.typography.sizes.xs,
            fontFamily: futuristicTheme.typography.fontFamily.mono,
            opacity: 0.7
          }}
        >
          {detectedConditions.length} conditions identified
        </Typography>
      </Box>

      {/* Global styles for animations */}
      <style>{`
        @keyframes conditionGlow {
          0%, 100% { opacity: 0.5; }
          50% { opacity: 1; }
        }
      `}</style>
    </HolographicPanel>
  )
}

export default ZeroShotLearningPanel
