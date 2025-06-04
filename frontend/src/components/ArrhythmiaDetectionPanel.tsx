import React, { useState, useEffect } from 'react'
import { Box, Typography, Chip, LinearProgress } from '@mui/material'
import { styled } from '@mui/material/styles'
import { HolographicPanel } from './HolographicPanel'
import { futuristicTheme } from '../theme/futuristicTheme'

const MetricRow = styled(Box)(({ theme: _theme }) => ({
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  marginBottom: futuristicTheme.spacing.sm,
  padding: futuristicTheme.spacing.xs,
  background: `rgba(0, 0, 0, 0.3)`,
  borderRadius: futuristicTheme.borderRadius.sm,
  border: `1px solid ${futuristicTheme.colors.ui.border}`
}))

const MetricLabel = styled(Typography)(({ theme: _theme }) => ({
  color: futuristicTheme.colors.data.text,
  fontSize: futuristicTheme.typography.sizes.xs,
  fontFamily: futuristicTheme.typography.fontFamily.mono
}))

const MetricValue = styled(Typography)<{ severity?: 'normal' | 'warning' | 'critical' }>(({ theme: _theme, severity }) => {
  const getColor = (): string => {
    switch (severity) {
      case 'warning':
        return futuristicTheme.colors.data.warning
      case 'critical':
        return futuristicTheme.colors.data.critical
      default:
        return futuristicTheme.colors.data.secondary
    }
  }

  return {
    color: getColor(),
    fontSize: futuristicTheme.typography.sizes.sm,
    fontFamily: futuristicTheme.typography.fontFamily.primary,
    fontWeight: 'bold',
    textShadow: `0 0 5px ${getColor()}`
  }
})

const AccuracyIndicator = styled(Box)(({ theme: _theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: futuristicTheme.spacing.sm,
  marginTop: futuristicTheme.spacing.md,
  padding: futuristicTheme.spacing.sm,
  background: `linear-gradient(45deg, rgba(0, 255, 127, 0.1), rgba(0, 191, 255, 0.1))`,
  borderRadius: futuristicTheme.borderRadius.md,
  border: `1px solid ${futuristicTheme.colors.data.secondary}`
}))

const PulsingDot = styled(Box)<{ active?: boolean }>(({ theme: _theme, active }) => ({
  width: '8px',
  height: '8px',
  borderRadius: '50%',
  backgroundColor: active ? futuristicTheme.colors.data.secondary : futuristicTheme.colors.ui.border,
  animation: active ? 'pulse 1s ease-in-out infinite' : 'none',
  boxShadow: active ? `0 0 10px ${futuristicTheme.colors.data.secondary}` : 'none'
}))

interface ArrhythmiaData {
  type: string
  confidence: number
  severity: 'normal' | 'warning' | 'critical'
  timestamp: number
}

interface ArrhythmiaDetectionPanelProps {
  className?: string
}

export const ArrhythmiaDetectionPanel: React.FC<ArrhythmiaDetectionPanelProps> = ({ className }) => {
  const [detections, setDetections] = useState<ArrhythmiaData[]>([])
  const [accuracy, setAccuracy] = useState(99.5)
  const [isProcessing, setIsProcessing] = useState(true)

  useEffect(() => {
    const interval = setInterval(() => {
      const arrhythmiaTypes = [
        { type: 'Normal Sinus Rhythm', severity: 'normal' as const },
        { type: 'Atrial Fibrillation', severity: 'warning' as const },
        { type: 'Ventricular Tachycardia', severity: 'critical' as const },
        { type: 'Premature Ventricular Contractions', severity: 'warning' as const },
        { type: 'Bradycardia', severity: 'warning' as const },
        { type: 'Supraventricular Tachycardia', severity: 'critical' as const }
      ]

      const randomType = arrhythmiaTypes[Math.floor(Math.random() * arrhythmiaTypes.length)]
      const newDetection: ArrhythmiaData = {
        type: randomType.type,
        confidence: 0.85 + Math.random() * 0.14, // 85-99% confidence
        severity: randomType.severity,
        timestamp: Date.now()
      }

      setDetections(prev => [newDetection, ...prev.slice(0, 4)]) // Keep last 5 detections
      setAccuracy(99.2 + Math.random() * 0.6) // 99.2-99.8% accuracy
      setIsProcessing(Math.random() > 0.7) // Randomly show processing state
    }, 3000)

    return () => clearInterval(interval)
  }, [])

  const getSeverityChipColor = (severity: string): string => {
    switch (severity) {
      case 'critical':
        return futuristicTheme.colors.data.critical
      case 'warning':
        return futuristicTheme.colors.data.warning
      default:
        return futuristicTheme.colors.data.secondary
    }
  }

  return (
    <HolographicPanel
      title="Real-Time Arrhythmia Detection"
      status={detections.length > 0 && detections[0].severity === 'critical' ? 'critical' : 'active'}
      className={className}
      width="300px"
    >
      <AccuracyIndicator>
        <PulsingDot active={isProcessing} />
        <Box>
          <Typography
            variant="body2"
            sx={{
              color: futuristicTheme.colors.data.text,
              fontSize: futuristicTheme.typography.sizes.xs,
              fontFamily: futuristicTheme.typography.fontFamily.mono
            }}
          >
            AI Accuracy
          </Typography>
          <Typography
            variant="h6"
            sx={{
              color: futuristicTheme.colors.data.secondary,
              fontSize: futuristicTheme.typography.sizes.lg,
              fontFamily: futuristicTheme.typography.fontFamily.primary,
              fontWeight: 'bold',
              textShadow: `0 0 5px ${futuristicTheme.colors.data.secondary}`
            }}
          >
            {accuracy.toFixed(1)}%
          </Typography>
        </Box>
      </AccuracyIndicator>

      {isProcessing && (
        <Box sx={{ mt: 2, mb: 2 }}>
          <Typography
            variant="body2"
            sx={{
              color: futuristicTheme.colors.data.text,
              fontSize: futuristicTheme.typography.sizes.xs,
              mb: 1
            }}
          >
            Processing ECG Signal...
          </Typography>
          <LinearProgress
            sx={{
              backgroundColor: `rgba(0, 191, 255, 0.2)`,
              '& .MuiLinearProgress-bar': {
                backgroundColor: futuristicTheme.colors.data.primary,
                boxShadow: `0 0 10px ${futuristicTheme.colors.data.primary}`
              }
            }}
          />
        </Box>
      )}

      <Box sx={{ mt: 2 }}>
        <Typography
          variant="body2"
          sx={{
            color: futuristicTheme.colors.data.text,
            fontSize: futuristicTheme.typography.sizes.xs,
            mb: 1,
            fontFamily: futuristicTheme.typography.fontFamily.mono
          }}
        >
          Recent Detections:
        </Typography>

        {detections.map((detection, _index) => (
          <MetricRow key={detection.timestamp}>
            <Box sx={{ flex: 1 }}>
              <MetricLabel>{detection.type}</MetricLabel>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
                <Chip
                  label={detection.severity.toUpperCase()}
                  size="small"
                  sx={{
                    backgroundColor: getSeverityChipColor(detection.severity),
                    color: futuristicTheme.colors.background.primary,
                    fontSize: futuristicTheme.typography.sizes.xs,
                    fontWeight: 'bold',
                    boxShadow: `0 0 5px ${getSeverityChipColor(detection.severity)}`
                  }}
                />
                <MetricValue severity={detection.severity}>
                  {(detection.confidence * 100).toFixed(1)}%
                </MetricValue>
              </Box>
            </Box>
          </MetricRow>
        ))}
      </Box>

      {/* Real-time monitoring indicators */}
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
          <PulsingDot active />
          <Typography
            variant="body2"
            sx={{
              color: futuristicTheme.colors.data.secondary,
              fontSize: futuristicTheme.typography.sizes.xs,
              fontFamily: futuristicTheme.typography.fontFamily.mono
            }}
          >
            LIVE
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
          {detections.length > 0 ? `${Math.floor((Date.now() - detections[0].timestamp) / 1000)}s ago` : 'Initializing...'}
        </Typography>
      </Box>
    </HolographicPanel>
  )
}

export default ArrhythmiaDetectionPanel
