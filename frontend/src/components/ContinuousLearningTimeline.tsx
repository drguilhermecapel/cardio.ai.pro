import React, { useState, useEffect } from 'react'
import { Box, Typography, LinearProgress } from '@mui/material'
import { styled } from '@mui/material/styles'
import { HolographicPanel } from './HolographicPanel'
import { futuristicTheme } from '../theme/futuristicTheme'
import { createFormatters } from '../utils/formatters'

const TimelineContainer = styled(Box)(() => ({
  height: '150px',
  background: `rgba(0, 0, 0, 0.3)`,
  border: `1px solid ${futuristicTheme.colors.ui.border}`,
  borderRadius: futuristicTheme.borderRadius.md,
  position: 'relative',
  overflow: 'hidden',
  padding: futuristicTheme.spacing.sm
}))

const TimelineAxis = styled(Box)(() => ({
  position: 'absolute',
  bottom: '30px',
  left: futuristicTheme.spacing.sm,
  right: futuristicTheme.spacing.sm,
  height: '2px',
  background: `linear-gradient(90deg, ${futuristicTheme.colors.ui.border}, ${futuristicTheme.colors.data.primary}, ${futuristicTheme.colors.ui.border})`,
  boxShadow: `0 0 5px ${futuristicTheme.colors.data.primary}`
}))

const TimelinePoint = styled(Box)<{ improvement: number; active?: boolean }>(({ improvement, active }) => {
  const getColor = (): string => {
    if (improvement > 0.8) return futuristicTheme.colors.data.secondary
    if (improvement > 0.5) return futuristicTheme.colors.data.primary
    return futuristicTheme.colors.data.warning
  }

  return {
    position: 'absolute',
    bottom: '20px',
    width: '12px',
    height: '12px',
    borderRadius: '50%',
    backgroundColor: getColor(),
    border: `2px solid ${active ? futuristicTheme.colors.data.holographic : getColor()}`,
    boxShadow: active ? 
      `0 0 15px ${futuristicTheme.colors.data.holographic}` : 
      `0 0 8px ${getColor()}`,
    animation: active ? 'pulse 2s ease-in-out infinite' : 'none',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
    '&::before': {
      content: '""',
      position: 'absolute',
      bottom: '20px',
      left: '50%',
      transform: 'translateX(-50%)',
      width: '2px',
      height: `${improvement * 60}px`,
      background: `linear-gradient(180deg, transparent, ${getColor()})`,
      borderRadius: '1px'
    }
  }
})

const ImprovementMetric = styled(Box)(() => ({
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  padding: futuristicTheme.spacing.sm,
  background: `rgba(0, 0, 0, 0.4)`,
  border: `1px solid ${futuristicTheme.colors.ui.border}`,
  borderRadius: futuristicTheme.borderRadius.sm,
  marginBottom: futuristicTheme.spacing.sm
}))

const LearningIndicator = styled(Box)<{ active?: boolean }>(({ active }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: futuristicTheme.spacing.sm,
  padding: futuristicTheme.spacing.sm,
  background: active ? 
    `linear-gradient(45deg, rgba(0, 255, 127, 0.2), rgba(0, 191, 255, 0.2))` :
    `rgba(0, 0, 0, 0.3)`,
  border: `1px solid ${active ? futuristicTheme.colors.data.secondary : futuristicTheme.colors.ui.border}`,
  borderRadius: futuristicTheme.borderRadius.md,
  marginTop: futuristicTheme.spacing.md
}))

interface LearningEvent {
  timestamp: number
  improvement: number
  modelVersion: string
  accuracy: number
  description: string
  dataPoints: number
}

interface ContinuousLearningTimelineProps {
  className?: string
}

export const ContinuousLearningTimeline: React.FC<ContinuousLearningTimelineProps> = ({ className }) => {
  const [learningEvents, setLearningEvents] = useState<LearningEvent[]>([])
  const [currentAccuracy, setCurrentAccuracy] = useState(94.2)
  const [isLearning, setIsLearning] = useState(false)
  const [selectedEvent, setSelectedEvent] = useState<LearningEvent | null>(null)
  const [learningProgress, setLearningProgress] = useState(0)
  
  const formatters = createFormatters(navigator.language || 'en')

  useEffect(() => {
    const initialEvents: LearningEvent[] = [
      {
        timestamp: Date.now() - 86400000 * 7, // 7 days ago
        improvement: 0.3,
        modelVersion: 'v1.0.0',
        accuracy: 89.2,
        description: 'Initial model deployment',
        dataPoints: 10000
      },
      {
        timestamp: Date.now() - 86400000 * 5, // 5 days ago
        improvement: 0.6,
        modelVersion: 'v1.1.0',
        accuracy: 91.5,
        description: 'Arrhythmia detection improvements',
        dataPoints: 15000
      },
      {
        timestamp: Date.now() - 86400000 * 3, // 3 days ago
        improvement: 0.8,
        modelVersion: 'v1.2.0',
        accuracy: 93.1,
        description: 'Rare condition recognition enhanced',
        dataPoints: 22000
      },
      {
        timestamp: Date.now() - 86400000 * 1, // 1 day ago
        improvement: 0.9,
        modelVersion: 'v1.3.0',
        accuracy: 94.2,
        description: 'Zero-shot learning integration',
        dataPoints: 28000
      }
    ]

    setLearningEvents(initialEvents)
    setSelectedEvent(initialEvents[initialEvents.length - 1])

    const interval = setInterval(() => {
      setIsLearning(Math.random() > 0.7)
      setLearningProgress(prev => {
        if (prev >= 100) {
          const newEvent: LearningEvent = {
            timestamp: Date.now(),
            improvement: 0.7 + Math.random() * 0.3,
            modelVersion: `v1.${Math.floor(Math.random() * 10)}.${Math.floor(Math.random() * 10)}`,
            accuracy: currentAccuracy + Math.random() * 1.5,
            description: [
              'Pattern recognition optimization',
              'Multi-lead correlation enhancement',
              'Noise reduction improvements',
              'Real-time processing acceleration',
              'Cross-patient generalization'
            ][Math.floor(Math.random() * 5)],
            dataPoints: 25000 + Math.floor(Math.random() * 10000)
          }

          setLearningEvents(prev => [...prev.slice(-3), newEvent])
          setCurrentAccuracy(newEvent.accuracy)
          setSelectedEvent(newEvent)
          return 0
        }
        return prev + Math.random() * 5
      })
    }, 2000)

    return () => clearInterval(interval)
  }, [currentAccuracy])

  const formatTimestamp = (timestamp: number): string => {
    const date = new Date(timestamp)
    const now = new Date()
    const diffDays = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24))
    
    if (diffDays === 0) return 'Today'
    if (diffDays === 1) return '1 day ago'
    return `${diffDays} days ago`
  }

  return (
    <HolographicPanel
      title="Continuous Learning Evolution"
      status={isLearning ? 'active' : 'normal'}
      className={className}
      width="100%"
    >
      {/* Current metrics */}
      <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 2, mb: 2 }}>
        <ImprovementMetric>
          <Typography
            variant="body2"
            sx={{
              color: futuristicTheme.colors.data.text,
              fontSize: futuristicTheme.typography.sizes.xs,
              fontFamily: futuristicTheme.typography.fontFamily.mono
            }}
          >
            Current Accuracy
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
            {currentAccuracy.toFixed(1)}%
          </Typography>
        </ImprovementMetric>

        <ImprovementMetric>
          <Typography
            variant="body2"
            sx={{
              color: futuristicTheme.colors.data.text,
              fontSize: futuristicTheme.typography.sizes.xs,
              fontFamily: futuristicTheme.typography.fontFamily.mono
            }}
          >
            Model Version
          </Typography>
          <Typography
            variant="body2"
            sx={{
              color: futuristicTheme.colors.data.primary,
              fontSize: futuristicTheme.typography.sizes.sm,
              fontFamily: futuristicTheme.typography.fontFamily.primary,
              fontWeight: 'bold'
            }}
          >
            {selectedEvent?.modelVersion || 'v1.3.0'}
          </Typography>
        </ImprovementMetric>

        <ImprovementMetric>
          <Typography
            variant="body2"
            sx={{
              color: futuristicTheme.colors.data.text,
              fontSize: futuristicTheme.typography.sizes.xs,
              fontFamily: futuristicTheme.typography.fontFamily.mono
            }}
          >
            Training Data
          </Typography>
          <Typography
            variant="body2"
            sx={{
              color: futuristicTheme.colors.neural.pathways,
              fontSize: futuristicTheme.typography.sizes.sm,
              fontFamily: futuristicTheme.typography.fontFamily.primary,
              fontWeight: 'bold'
            }}
          >
            {selectedEvent?.dataPoints ? formatters.formatNumber(selectedEvent.dataPoints) : '28,000'}
          </Typography>
        </ImprovementMetric>
      </Box>

      {/* Timeline visualization */}
      <TimelineContainer>
        <Typography
          sx={{
            color: futuristicTheme.colors.data.text,
            fontSize: futuristicTheme.typography.sizes.xs,
            fontFamily: futuristicTheme.typography.fontFamily.mono,
            mb: 1
          }}
        >
          Model Evolution Timeline:
        </Typography>

        <TimelineAxis />

        {learningEvents.map((event, index) => (
          <TimelinePoint
            key={event.timestamp}
            improvement={event.improvement}
            active={selectedEvent?.timestamp === event.timestamp}
            sx={{
              left: `${10 + (index / (learningEvents.length - 1)) * 80}%`,
              transform: 'translateX(-50%)'
            }}
            onClick={() => setSelectedEvent(event)}
          />
        ))}

        {/* Timeline labels */}
        {learningEvents.map((event, index) => (
          <Typography
            key={`label-${event.timestamp}`}
            sx={{
              position: 'absolute',
              bottom: '5px',
              left: `${10 + (index / (learningEvents.length - 1)) * 80}%`,
              transform: 'translateX(-50%)',
              color: futuristicTheme.colors.data.text,
              fontSize: futuristicTheme.typography.sizes.xs,
              fontFamily: futuristicTheme.typography.fontFamily.mono,
              opacity: 0.7,
              whiteSpace: 'nowrap'
            }}
          >
            {formatTimestamp(event.timestamp)}
          </Typography>
        ))}
      </TimelineContainer>

      {/* Selected event details */}
      {selectedEvent && (
        <Box
          sx={{
            mt: 2,
            p: 2,
            background: `rgba(0, 0, 0, 0.4)`,
            border: `1px solid ${futuristicTheme.colors.ui.border}`,
            borderRadius: futuristicTheme.borderRadius.md
          }}
        >
          <Typography
            sx={{
              color: futuristicTheme.colors.data.text,
              fontSize: futuristicTheme.typography.sizes.sm,
              fontFamily: futuristicTheme.typography.fontFamily.primary,
              fontWeight: 'bold',
              mb: 1
            }}
          >
            {selectedEvent.description}
          </Typography>
          
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography
              sx={{
                color: futuristicTheme.colors.data.text,
                fontSize: futuristicTheme.typography.sizes.xs,
                fontFamily: futuristicTheme.typography.fontFamily.mono,
                opacity: 0.8
              }}
            >
              Accuracy: {selectedEvent.accuracy.toFixed(1)}% • Data Points: {formatters.formatNumber(selectedEvent.dataPoints)}
            </Typography>
            
            <Typography
              sx={{
                color: futuristicTheme.colors.neural.connections,
                fontSize: futuristicTheme.typography.sizes.xs,
                fontFamily: futuristicTheme.typography.fontFamily.mono,
                fontWeight: 'bold'
              }}
            >
              {selectedEvent.modelVersion}
            </Typography>
          </Box>
        </Box>
      )}

      {/* Learning progress indicator */}
      <LearningIndicator active={isLearning}>
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
            {isLearning ? 'Active Learning in Progress...' : 'Learning System Ready'}
          </Typography>
          
          {isLearning && (
            <LinearProgress
              variant="determinate"
              value={learningProgress}
              sx={{
                height: '4px',
                borderRadius: '2px',
                backgroundColor: `rgba(0, 191, 255, 0.2)`,
                '& .MuiLinearProgress-bar': {
                  backgroundColor: futuristicTheme.colors.neural.connections,
                  boxShadow: `0 0 5px ${futuristicTheme.colors.neural.connections}`
                }
              }}
            />
          )}
        </Box>
        
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
      </LearningIndicator>
    </HolographicPanel>
  )
}

export default ContinuousLearningTimeline
