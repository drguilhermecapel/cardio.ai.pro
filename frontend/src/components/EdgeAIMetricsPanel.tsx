import React, { useState, useEffect } from 'react'
import { Box, Typography, LinearProgress } from '@mui/material'
import { styled } from '@mui/material/styles'
import { HolographicPanel } from './HolographicPanel'
import { futuristicTheme } from '../theme/futuristicTheme'

const MetricsGrid = styled(Box)(({ theme: _theme }) => ({
  display: 'grid',
  gridTemplateColumns: '1fr 1fr',
  gap: futuristicTheme.spacing.sm,
  marginBottom: futuristicTheme.spacing.md
}))

const MetricCard = styled(Box)(({ theme: _theme }) => ({
  background: `rgba(0, 0, 0, 0.4)`,
  border: `1px solid ${futuristicTheme.colors.ui.border}`,
  borderRadius: futuristicTheme.borderRadius.sm,
  padding: futuristicTheme.spacing.sm,
  position: 'relative',
  overflow: 'hidden'
}))

const MetricValue = styled(Typography)<{ status?: 'optimal' | 'warning' | 'critical' }>(({ theme: _theme, status }) => {
  const getColor = (): string => {
    switch (status) {
      case 'critical':
        return futuristicTheme.colors.data.critical
      case 'warning':
        return futuristicTheme.colors.data.warning
      default:
        return futuristicTheme.colors.data.secondary
    }
  }

  return {
    color: getColor(),
    fontSize: futuristicTheme.typography.sizes.lg,
    fontFamily: futuristicTheme.typography.fontFamily.primary,
    fontWeight: 'bold',
    textShadow: `0 0 5px ${getColor()}`
  }
})

const PerformanceGraph = styled(Box)(({ theme: _theme }) => ({
  height: '80px',
  background: `rgba(0, 0, 0, 0.3)`,
  border: `1px solid ${futuristicTheme.colors.ui.border}`,
  borderRadius: futuristicTheme.borderRadius.md,
  position: 'relative',
  overflow: 'hidden',
  marginTop: futuristicTheme.spacing.md
}))

const GraphBar = styled(Box)<{ height: number; delay: number }>(({ theme: _theme, height, delay }) => ({
  position: 'absolute',
  bottom: 0,
  width: '8px',
  height: `${height}%`,
  background: `linear-gradient(180deg, ${futuristicTheme.colors.data.primary}, ${futuristicTheme.colors.neural.connections})`,
  borderRadius: '2px 2px 0 0',
  animation: `barGrow 0.5s ease-out ${delay}s both`,
  boxShadow: `0 0 5px ${futuristicTheme.colors.data.primary}`
}))

const ProcessorIndicator = styled(Box)<{ active?: boolean }>(({ theme: _theme, active }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: futuristicTheme.spacing.xs,
  padding: futuristicTheme.spacing.xs,
  background: active ? 
    `linear-gradient(45deg, rgba(0, 255, 127, 0.2), rgba(0, 191, 255, 0.2))` :
    `rgba(0, 0, 0, 0.3)`,
  border: `1px solid ${active ? futuristicTheme.colors.data.secondary : futuristicTheme.colors.ui.border}`,
  borderRadius: futuristicTheme.borderRadius.sm,
  marginBottom: futuristicTheme.spacing.xs
}))

interface EdgeMetrics {
  latency: number
  throughput: number
  cpuUsage: number
  gpuUsage: number
  memoryUsage: number
  powerConsumption: number
  temperature: number
  inferenceRate: number
  jetsonCores: Array<{
    id: number
    usage: number
    active: boolean
  }>
  performanceHistory: number[]
}

interface EdgeAIMetricsPanelProps {
  className?: string
}

export const EdgeAIMetricsPanel: React.FC<EdgeAIMetricsPanelProps> = ({ className }) => {
  const [metrics, setMetrics] = useState<EdgeMetrics>({
    latency: 2.3,
    throughput: 1250,
    cpuUsage: 45,
    gpuUsage: 78,
    memoryUsage: 62,
    powerConsumption: 15.2,
    temperature: 42,
    inferenceRate: 340,
    jetsonCores: [
      { id: 0, usage: 85, active: true },
      { id: 1, usage: 72, active: true },
      { id: 2, usage: 91, active: true },
      { id: 3, usage: 68, active: true },
      { id: 4, usage: 45, active: false },
      { id: 5, usage: 38, active: false }
    ],
    performanceHistory: [65, 72, 68, 85, 91, 78, 82, 89, 76, 94, 88, 92]
  })

  useEffect(() => {
    const interval = setInterval(() => {
      setMetrics(prev => ({
        ...prev,
        latency: 1.8 + Math.random() * 1.0,
        throughput: 1200 + Math.random() * 200,
        cpuUsage: 40 + Math.random() * 30,
        gpuUsage: 70 + Math.random() * 25,
        memoryUsage: 55 + Math.random() * 20,
        powerConsumption: 14 + Math.random() * 3,
        temperature: 38 + Math.random() * 8,
        inferenceRate: 320 + Math.random() * 50,
        jetsonCores: prev.jetsonCores.map(core => ({
          ...core,
          usage: Math.max(20, core.usage + (Math.random() - 0.5) * 10),
          active: Math.random() > 0.1
        })),
        performanceHistory: [
          ...prev.performanceHistory.slice(1),
          70 + Math.random() * 30
        ]
      }))
    }, 1500)

    return () => clearInterval(interval)
  }, [])

  const getLatencyStatus = (latency: number): 'optimal' | 'warning' | 'critical' => {
    if (latency > 4) return 'critical'
    if (latency > 3) return 'warning'
    return 'optimal'
  }

  const getUsageStatus = (usage: number): 'optimal' | 'warning' | 'critical' => {
    if (usage > 90) return 'critical'
    if (usage > 75) return 'warning'
    return 'optimal'
  }

  return (
    <HolographicPanel
      title="Edge AI Processing Metrics"
      status={metrics.latency > 4 ? 'critical' : metrics.latency > 3 ? 'warning' : 'active'}
      className={className}
      width="400px"
    >
      {/* Primary metrics */}
      <MetricsGrid>
        <MetricCard>
          <Typography
            variant="body2"
            sx={{
              color: futuristicTheme.colors.data.text,
              fontSize: futuristicTheme.typography.sizes.xs,
              fontFamily: futuristicTheme.typography.fontFamily.mono,
              mb: 0.5
            }}
          >
            Latency
          </Typography>
          <MetricValue status={getLatencyStatus(metrics.latency)}>
            {metrics.latency.toFixed(1)}ms
          </MetricValue>
        </MetricCard>

        <MetricCard>
          <Typography
            variant="body2"
            sx={{
              color: futuristicTheme.colors.data.text,
              fontSize: futuristicTheme.typography.sizes.xs,
              fontFamily: futuristicTheme.typography.fontFamily.mono,
              mb: 0.5
            }}
          >
            Throughput
          </Typography>
          <MetricValue status="optimal">
            {metrics.throughput.toFixed(0)}/s
          </MetricValue>
        </MetricCard>

        <MetricCard>
          <Typography
            variant="body2"
            sx={{
              color: futuristicTheme.colors.data.text,
              fontSize: futuristicTheme.typography.sizes.xs,
              fontFamily: futuristicTheme.typography.fontFamily.mono,
              mb: 0.5
            }}
          >
            GPU Usage
          </Typography>
          <MetricValue status={getUsageStatus(metrics.gpuUsage)}>
            {metrics.gpuUsage.toFixed(0)}%
          </MetricValue>
        </MetricCard>

        <MetricCard>
          <Typography
            variant="body2"
            sx={{
              color: futuristicTheme.colors.data.text,
              fontSize: futuristicTheme.typography.sizes.xs,
              fontFamily: futuristicTheme.typography.fontFamily.mono,
              mb: 0.5
            }}
          >
            Temperature
          </Typography>
          <MetricValue status={metrics.temperature > 50 ? 'warning' : 'optimal'}>
            {metrics.temperature.toFixed(0)}°C
          </MetricValue>
        </MetricCard>
      </MetricsGrid>

      {/* NVIDIA Jetson cores */}
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
          NVIDIA Jetson Cores:
        </Typography>

        {metrics.jetsonCores.map(core => (
          <ProcessorIndicator key={core.id} active={core.active}>
            <Typography
              sx={{
                color: futuristicTheme.colors.data.text,
                fontSize: futuristicTheme.typography.sizes.xs,
                fontFamily: futuristicTheme.typography.fontFamily.mono,
                width: '40px'
              }}
            >
              Core {core.id}
            </Typography>
            
            <Box sx={{ flex: 1, mx: 1 }}>
              <LinearProgress
                variant="determinate"
                value={core.usage}
                sx={{
                  height: '4px',
                  borderRadius: '2px',
                  backgroundColor: `rgba(0, 191, 255, 0.2)`,
                  '& .MuiLinearProgress-bar': {
                    backgroundColor: core.active ? 
                      futuristicTheme.colors.data.secondary : 
                      futuristicTheme.colors.ui.border,
                    boxShadow: core.active ? 
                      `0 0 5px ${futuristicTheme.colors.data.secondary}` : 
                      'none'
                  }
                }}
              />
            </Box>
            
            <Typography
              sx={{
                color: core.active ? futuristicTheme.colors.data.secondary : futuristicTheme.colors.data.text,
                fontSize: futuristicTheme.typography.sizes.xs,
                fontFamily: futuristicTheme.typography.fontFamily.mono,
                width: '35px',
                textAlign: 'right'
              }}
            >
              {core.usage.toFixed(0)}%
            </Typography>
          </ProcessorIndicator>
        ))}
      </Box>

      {/* Performance graph */}
      <PerformanceGraph>
        <Typography
          sx={{
            position: 'absolute',
            top: futuristicTheme.spacing.xs,
            left: futuristicTheme.spacing.xs,
            color: futuristicTheme.colors.data.text,
            fontSize: futuristicTheme.typography.sizes.xs,
            fontFamily: futuristicTheme.typography.fontFamily.mono,
            zIndex: 2
          }}
        >
          Performance History
        </Typography>

        {metrics.performanceHistory.map((value, index) => (
          <GraphBar
            key={index}
            height={value}
            delay={index * 0.05}
            sx={{
              left: `${8 + index * 7}%`
            }}
          />
        ))}

        {/* Grid lines */}
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
              transparent 19px,
              rgba(0, 191, 255, 0.1) 20px
            )`,
            pointerEvents: 'none'
          }}
        />
      </PerformanceGraph>

      {/* System status */}
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
              backgroundColor: metrics.latency < 3 ? futuristicTheme.colors.data.secondary : futuristicTheme.colors.data.warning,
              animation: 'pulse 2s ease-in-out infinite',
              boxShadow: `0 0 10px ${metrics.latency < 3 ? futuristicTheme.colors.data.secondary : futuristicTheme.colors.data.warning}`
            }}
          />
          <Typography
            variant="body2"
            sx={{
              color: futuristicTheme.colors.data.text,
              fontSize: futuristicTheme.typography.sizes.xs,
              fontFamily: futuristicTheme.typography.fontFamily.mono
            }}
          >
            {metrics.latency < 3 ? 'OPTIMAL' : metrics.latency < 4 ? 'DEGRADED' : 'CRITICAL'}
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
          {metrics.powerConsumption.toFixed(1)}W • {metrics.inferenceRate.toFixed(0)} inf/s
        </Typography>
      </Box>

      {/* Global styles for animations */}
      <style>{`
        @keyframes barGrow {
          0% { height: 0%; }
          100% { height: var(--target-height); }
        }
      `}</style>
    </HolographicPanel>
  )
}

export default EdgeAIMetricsPanel
