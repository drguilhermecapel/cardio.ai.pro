import React, { useState, useEffect } from 'react'
import { Box, Typography, Chip } from '@mui/material'
import { styled } from '@mui/material/styles'
import { HolographicPanel } from './HolographicPanel'
import { futuristicTheme } from '../theme/futuristicTheme'

const ComparisonContainer = styled(Box)(() => ({
  display: 'grid',
  gridTemplateColumns: '1fr 1fr',
  gap: futuristicTheme.spacing.md,
  marginBottom: futuristicTheme.spacing.md
}))

const ECGComparison = styled(Box)<{ type: 'original' | 'counterfactual' }>(({ type }) => ({
  background: `rgba(0, 0, 0, 0.4)`,
  border: `1px solid ${type === 'original' ? futuristicTheme.colors.data.primary : futuristicTheme.colors.neural.pathways}`,
  borderRadius: futuristicTheme.borderRadius.md,
  padding: futuristicTheme.spacing.sm,
  height: '120px',
  position: 'relative',
  overflow: 'hidden'
}))

const SHAPValue = styled(Box)<{ importance: number }>(({ importance }) => {
  const getColor = (): string => {
    if (importance > 0.7) return futuristicTheme.colors.data.critical
    if (importance > 0.4) return futuristicTheme.colors.data.warning
    return futuristicTheme.colors.data.secondary
  }

  return {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: futuristicTheme.spacing.xs,
    background: `rgba(${getColor().replace('#', '').match(/.{2}/g)?.map(hex => parseInt(hex, 16)).join(', ')}, 0.2)`,
    border: `1px solid ${getColor()}`,
    borderRadius: futuristicTheme.borderRadius.sm,
    marginBottom: futuristicTheme.spacing.xs,
    position: 'relative',
    overflow: 'hidden',
    '&::before': {
      content: '""',
      position: 'absolute',
      left: 0,
      top: 0,
      bottom: 0,
      width: `${importance * 100}%`,
      background: `linear-gradient(90deg, ${getColor()}, transparent)`,
      opacity: 0.3
    }
  }
})

const FeatureImportance = styled(Box)(() => ({
  background: `rgba(0, 0, 0, 0.3)`,
  border: `1px solid ${futuristicTheme.colors.ui.border}`,
  borderRadius: futuristicTheme.borderRadius.md,
  padding: futuristicTheme.spacing.sm,
  marginTop: futuristicTheme.spacing.md
}))

const ExplanationMetric = styled(Box)(() => ({
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  padding: futuristicTheme.spacing.sm,
  background: `rgba(0, 0, 0, 0.4)`,
  border: `1px solid ${futuristicTheme.colors.ui.border}`,
  borderRadius: futuristicTheme.borderRadius.sm,
  marginBottom: futuristicTheme.spacing.sm
}))

interface SHAPFeature {
  name: string
  importance: number
  contribution: 'positive' | 'negative'
  value: number
}

interface CounterfactualData {
  originalPrediction: string
  counterfactualPrediction: string
  confidence: number
  changedFeatures: string[]
  explanation: string
}

interface ExplainableAIPanelProps {
  className?: string
}

export const ExplainableAIPanel: React.FC<ExplainableAIPanelProps> = ({ className }) => {
  const [shapValues, setSHAPValues] = useState<SHAPFeature[]>([])
  const [counterfactual, setCounterfactual] = useState<CounterfactualData | null>(null)
  const [explainabilityScore, setExplainabilityScore] = useState(0.92)
  const [isAnalyzing, setIsAnalyzing] = useState(false)

  useEffect(() => {
    const features: SHAPFeature[] = [
      { name: 'QRS Duration', importance: 0.85, contribution: 'positive', value: 120 },
      { name: 'ST Elevation', importance: 0.72, contribution: 'negative', value: -0.8 },
      { name: 'T Wave Amplitude', importance: 0.68, contribution: 'positive', value: 0.45 },
      { name: 'PR Interval', importance: 0.54, contribution: 'negative', value: 180 },
      { name: 'Heart Rate Variability', importance: 0.41, contribution: 'positive', value: 0.32 },
      { name: 'QT Interval', importance: 0.38, contribution: 'negative', value: 420 },
      { name: 'P Wave Morphology', importance: 0.29, contribution: 'positive', value: 0.78 },
      { name: 'Baseline Drift', importance: 0.15, contribution: 'negative', value: 0.12 }
    ]

    const counterfactuals: CounterfactualData[] = [
      {
        originalPrediction: 'Atrial Fibrillation',
        counterfactualPrediction: 'Normal Sinus Rhythm',
        confidence: 0.89,
        changedFeatures: ['Heart Rate Variability', 'P Wave Morphology'],
        explanation: 'If HRV increased by 15% and P waves were more regular, prediction would change to NSR'
      },
      {
        originalPrediction: 'Ventricular Tachycardia',
        counterfactualPrediction: 'Sinus Tachycardia',
        confidence: 0.76,
        changedFeatures: ['QRS Duration', 'T Wave Amplitude'],
        explanation: 'Narrower QRS (<100ms) and normal T waves would indicate sinus origin'
      },
      {
        originalPrediction: 'ST Elevation MI',
        counterfactualPrediction: 'Normal ECG',
        confidence: 0.94,
        changedFeatures: ['ST Elevation', 'T Wave Amplitude'],
        explanation: 'Absence of ST elevation and normal T waves would rule out acute MI'
      }
    ]

    const interval = setInterval(() => {
      setSHAPValues(features.map(feature => ({
        ...feature,
        importance: Math.max(0.1, feature.importance + (Math.random() - 0.5) * 0.1),
        value: feature.value + (Math.random() - 0.5) * feature.value * 0.1
      })))

      setCounterfactual(counterfactuals[Math.floor(Math.random() * counterfactuals.length)])
      
      setExplainabilityScore(0.88 + Math.random() * 0.08)
      setIsAnalyzing(Math.random() > 0.7)
    }, 3000)

    return () => clearInterval(interval)
  }, [])

  const generateMockECGWaveform = (type: 'original' | 'counterfactual'): Array<{x: number, y: number}> => {
    const points: Array<{x: number, y: number}> = []
    const segments = 100
    
    for (let i = 0; i < segments; i++) {
      const t = (i / segments) * Math.PI * 4
      let y = Math.sin(t) * 0.3
      
      if (i % 25 === 0) {
        y += type === 'original' ? 0.8 : 0.6
      }
      
      y += (Math.random() - 0.5) * 0.1
      
      points.push({ x: (i / segments) * 100, y: 50 + y * 30 })
    }
    
    return points
  }

  return (
    <HolographicPanel
      title="Explainable AI & SHAP Analysis"
      status={isAnalyzing ? 'active' : 'normal'}
      className={className}
      width="450px"
    >
      {/* Explainability metrics */}
      <ExplanationMetric>
        <Typography
          variant="body2"
          sx={{
            color: futuristicTheme.colors.data.text,
            fontSize: futuristicTheme.typography.sizes.xs,
            fontFamily: futuristicTheme.typography.fontFamily.mono
          }}
        >
          Explainability Score
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
          {(explainabilityScore * 100).toFixed(1)}%
        </Typography>
      </ExplanationMetric>

      {/* ECG Counterfactual Comparison */}
      {counterfactual && (
        <Box sx={{ mb: 2 }}>
          <Typography
            variant="body2"
            sx={{
              color: futuristicTheme.colors.data.text,
              fontSize: futuristicTheme.typography.sizes.xs,
              fontFamily: futuristicTheme.typography.fontFamily.mono,
              mb: 1
            }}
          >
            Counterfactual ECG Comparison:
          </Typography>
          
          <ComparisonContainer>
            <ECGComparison type="original">
              <Typography
                sx={{
                  color: futuristicTheme.colors.data.primary,
                  fontSize: futuristicTheme.typography.sizes.xs,
                  fontFamily: futuristicTheme.typography.fontFamily.mono,
                  mb: 1
                }}
              >
                Original: {counterfactual.originalPrediction}
              </Typography>
              
              <svg width="100%" height="80" style={{ background: 'transparent' }}>
                <polyline
                  points={generateMockECGWaveform('original').map(p => `${p.x},${p.y}`).join(' ')}
                  fill="none"
                  stroke={futuristicTheme.colors.data.primary}
                  strokeWidth="2"
                  style={{ filter: `drop-shadow(0 0 3px ${futuristicTheme.colors.data.primary})` }}
                />
              </svg>
            </ECGComparison>

            <ECGComparison type="counterfactual">
              <Typography
                sx={{
                  color: futuristicTheme.colors.neural.pathways,
                  fontSize: futuristicTheme.typography.sizes.xs,
                  fontFamily: futuristicTheme.typography.fontFamily.mono,
                  mb: 1
                }}
              >
                Counterfactual: {counterfactual.counterfactualPrediction}
              </Typography>
              
              <svg width="100%" height="80" style={{ background: 'transparent' }}>
                <polyline
                  points={generateMockECGWaveform('counterfactual').map(p => `${p.x},${p.y}`).join(' ')}
                  fill="none"
                  stroke={futuristicTheme.colors.neural.pathways}
                  strokeWidth="2"
                  style={{ filter: `drop-shadow(0 0 3px ${futuristicTheme.colors.neural.pathways})` }}
                />
              </svg>
            </ECGComparison>
          </ComparisonContainer>

          <Box
            sx={{
              background: `rgba(0, 0, 0, 0.4)`,
              border: `1px solid ${futuristicTheme.colors.ui.border}`,
              borderRadius: futuristicTheme.borderRadius.sm,
              padding: futuristicTheme.spacing.sm,
              mb: 2
            }}
          >
            <Typography
              sx={{
                color: futuristicTheme.colors.data.text,
                fontSize: futuristicTheme.typography.sizes.xs,
                fontFamily: futuristicTheme.typography.fontFamily.mono,
                mb: 1
              }}
            >
              Explanation:
            </Typography>
            <Typography
              sx={{
                color: futuristicTheme.colors.data.text,
                fontSize: futuristicTheme.typography.sizes.sm,
                lineHeight: 1.4,
                mb: 1
              }}
            >
              {counterfactual.explanation}
            </Typography>
            
            <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
              {counterfactual.changedFeatures.map(feature => (
                <Chip
                  key={feature}
                  label={feature}
                  size="small"
                  sx={{
                    backgroundColor: futuristicTheme.colors.neural.pathways,
                    color: futuristicTheme.colors.background.primary,
                    fontSize: futuristicTheme.typography.sizes.xs,
                    fontWeight: 'bold'
                  }}
                />
              ))}
            </Box>
          </Box>
        </Box>
      )}

      {/* SHAP Feature Importance */}
      <FeatureImportance>
        <Typography
          variant="body2"
          sx={{
            color: futuristicTheme.colors.data.text,
            fontSize: futuristicTheme.typography.sizes.xs,
            fontFamily: futuristicTheme.typography.fontFamily.mono,
            mb: 1
          }}
        >
          SHAP Feature Importance:
        </Typography>

        {shapValues.slice(0, 6).map((feature) => (
          <SHAPValue key={feature.name} importance={feature.importance}>
            <Box sx={{ flex: 1, zIndex: 1 }}>
              <Typography
                sx={{
                  color: futuristicTheme.colors.data.text,
                  fontSize: futuristicTheme.typography.sizes.xs,
                  fontFamily: futuristicTheme.typography.fontFamily.mono
                }}
              >
                {feature.name}
              </Typography>
            </Box>
            
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, zIndex: 1 }}>
              <Typography
                sx={{
                  color: feature.contribution === 'positive' ? 
                    futuristicTheme.colors.data.secondary : 
                    futuristicTheme.colors.data.warning,
                  fontSize: futuristicTheme.typography.sizes.xs,
                  fontFamily: futuristicTheme.typography.fontFamily.mono
                }}
              >
                {feature.contribution === 'positive' ? '+' : '-'}{feature.importance.toFixed(2)}
              </Typography>
            </Box>
          </SHAPValue>
        ))}
      </FeatureImportance>

      {/* Analysis status */}
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
              backgroundColor: isAnalyzing ? futuristicTheme.colors.neural.connections : futuristicTheme.colors.ui.border,
              animation: isAnalyzing ? 'pulse 1s ease-in-out infinite' : 'none',
              boxShadow: isAnalyzing ? `0 0 10px ${futuristicTheme.colors.neural.connections}` : 'none'
            }}
          />
          <Typography
            variant="body2"
            sx={{
              color: isAnalyzing ? futuristicTheme.colors.neural.connections : futuristicTheme.colors.data.text,
              fontSize: futuristicTheme.typography.sizes.xs,
              fontFamily: futuristicTheme.typography.fontFamily.mono
            }}
          >
            {isAnalyzing ? 'ANALYZING' : 'READY'}
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
          {shapValues.length} features analyzed
        </Typography>
      </Box>
    </HolographicPanel>
  )
}

export default ExplainableAIPanel
