import React, { useRef, useEffect } from 'react'
import { Box, Typography } from '@mui/material'
import { styled } from '@mui/material/styles'
import { futuristicTheme } from '../theme/futuristicTheme'
import * as d3 from 'd3'

const WaveformContainer = styled(Box)(({ theme: _theme }) => ({
  position: 'absolute',
  top: '10%',
  left: '10%',
  right: '10%',
  height: '25%',
  background: `rgba(0, 0, 0, 0.3)`,
  backdropFilter: futuristicTheme.effects.blur.glass,
  border: `1px solid ${futuristicTheme.colors.ui.border}`,
  borderRadius: futuristicTheme.borderRadius.lg,
  padding: futuristicTheme.spacing.md,
  zIndex: futuristicTheme.zIndex.overlay
}))

const LeadContainer = styled(Box)(({ theme: _theme }) => ({
  display: 'grid',
  gridTemplateColumns: 'repeat(4, 1fr)',
  gridTemplateRows: 'repeat(3, 1fr)',
  gap: futuristicTheme.spacing.sm,
  height: '100%'
}))

const LeadPanel = styled(Box)(({ theme: _theme }) => ({
  background: `rgba(0, 0, 0, 0.5)`,
  border: `1px solid ${futuristicTheme.colors.ui.border}`,
  borderRadius: futuristicTheme.borderRadius.sm,
  padding: futuristicTheme.spacing.xs,
  position: 'relative',
  overflow: 'hidden'
}))

const LeadLabel = styled(Typography)(({ theme: _theme }) => ({
  position: 'absolute',
  top: futuristicTheme.spacing.xs,
  left: futuristicTheme.spacing.xs,
  color: futuristicTheme.colors.data.text,
  fontSize: futuristicTheme.typography.sizes.xs,
  fontFamily: futuristicTheme.typography.fontFamily.mono,
  fontWeight: 'bold',
  zIndex: 2
}))

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

interface ECGWaveformDisplayProps {
  ecgData: ECGData | null
}

export const ECGWaveformDisplay: React.FC<ECGWaveformDisplayProps> = ({ ecgData }) => {
  const svgRefs = useRef<{ [key: string]: SVGSVGElement | null }>({})
  
  const leads = React.useMemo(() => ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'], [])

  useEffect(() => {
    if (!ecgData) return

    leads.forEach(lead => {
      const svg = svgRefs.current[lead]
      if (!svg) return

      const data = ecgData.leads[lead as keyof typeof ecgData.leads]
      if (!data || data.length === 0) return

      d3.select(svg).selectAll('*').remove()

      const margin = { top: 10, right: 10, bottom: 10, left: 10 }
      const width = svg.clientWidth - margin.left - margin.right
      const height = svg.clientHeight - margin.top - margin.bottom

      const svgSelection = d3.select(svg)
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom)

      const g = svgSelection.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`)

      const xScale = d3.scaleLinear()
        .domain([0, data.length - 1])
        .range([0, width])

      const yScale = d3.scaleLinear()
        .domain(d3.extent(data) as [number, number])
        .range([height, 0])

      const line = d3.line<number>()
        .x((_d, i) => xScale(i))
        .y(d => yScale(d))
        .curve(d3.curveCardinal)

      const xAxis = d3.axisBottom(xScale)
        .tickSize(-height)
        .tickFormat(() => '')
        .ticks(10)

      const yAxis = d3.axisLeft(yScale)
        .tickSize(-width)
        .tickFormat(() => '')
        .ticks(5)

      g.append('g')
        .attr('class', 'grid')
        .attr('transform', `translate(0,${height})`)
        .call(xAxis)
        .selectAll('line')
        .style('stroke', futuristicTheme.colors.ui.border)
        .style('stroke-width', 0.5)
        .style('opacity', 0.3)

      g.append('g')
        .attr('class', 'grid')
        .call(yAxis)
        .selectAll('line')
        .style('stroke', futuristicTheme.colors.ui.border)
        .style('stroke-width', 0.5)
        .style('opacity', 0.3)

      const path = g.append('path')
        .datum(data)
        .attr('fill', 'none')
        .attr('stroke', futuristicTheme.colors.data.primary)
        .attr('stroke-width', 2)
        .attr('d', line)
        .style('filter', `drop-shadow(0 0 3px ${futuristicTheme.colors.data.primary})`)

      const peaks = detectPeaks(data)
      peaks.forEach(peakIndex => {
        g.append('circle')
          .attr('cx', xScale(peakIndex))
          .attr('cy', yScale(data[peakIndex]))
          .attr('r', 3)
          .attr('fill', futuristicTheme.colors.neural.pathways)
          .style('filter', `drop-shadow(0 0 5px ${futuristicTheme.colors.neural.pathways})`)
          .style('opacity', 0.8)
      })

      const totalLength = path.node()?.getTotalLength() || 0
      path
        .attr('stroke-dasharray', `${totalLength} ${totalLength}`)
        .attr('stroke-dashoffset', totalLength)
        .transition()
        .duration(2000)
        .ease(d3.easeLinear)
        .attr('stroke-dashoffset', 0)

      const anomalies = detectAnomalies(data)
      anomalies.forEach(anomalyIndex => {
        g.append('rect')
          .attr('x', xScale(anomalyIndex) - 5)
          .attr('y', 0)
          .attr('width', 10)
          .attr('height', height)
          .attr('fill', futuristicTheme.colors.data.warning)
          .style('opacity', 0.2)
          .style('filter', `drop-shadow(0 0 5px ${futuristicTheme.colors.data.warning})`)
      })
    })
  }, [ecgData, leads])

  const detectPeaks = (data: number[]): number[] => {
    const peaks: number[] = []
    const threshold = Math.max(...data) * 0.6
    
    for (let i = 1; i < data.length - 1; i++) {
      if (data[i] > data[i - 1] && data[i] > data[i + 1] && data[i] > threshold) {
        peaks.push(i)
      }
    }
    
    return peaks
  }

  const detectAnomalies = (data: number[]): number[] => {
    const anomalies: number[] = []
    const mean = data.reduce((sum, val) => sum + val, 0) / data.length
    const stdDev = Math.sqrt(data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / data.length)
    const threshold = 2.5 * stdDev
    
    for (let i = 0; i < data.length; i++) {
      if (Math.abs(data[i] - mean) > threshold) {
        anomalies.push(i)
      }
    }
    
    return anomalies
  }

  return (
    <WaveformContainer>
      <Typography
        variant="h6"
        sx={{
          color: futuristicTheme.colors.data.text,
          fontFamily: futuristicTheme.typography.fontFamily.primary,
          mb: 1,
          textAlign: 'center'
        }}
      >
        Real-Time ECG Analysis - {ecgData?.rhythm || 'Analyzing...'}
      </Typography>
      
      <LeadContainer>
        {leads.map(lead => (
          <LeadPanel key={lead}>
            <LeadLabel>{lead}</LeadLabel>
            <svg
              ref={el => { svgRefs.current[lead] = el }}
              style={{
                width: '100%',
                height: '100%',
                background: 'transparent'
              }}
            />
          </LeadPanel>
        ))}
      </LeadContainer>
      
      {/* Real-time metrics overlay */}
      <Box
        sx={{
          position: 'absolute',
          bottom: futuristicTheme.spacing.sm,
          right: futuristicTheme.spacing.sm,
          display: 'flex',
          gap: futuristicTheme.spacing.md,
          color: futuristicTheme.colors.data.text,
          fontSize: futuristicTheme.typography.sizes.sm,
          fontFamily: futuristicTheme.typography.fontFamily.mono
        }}
      >
        <Box>
          HR: <span style={{ color: futuristicTheme.colors.data.secondary }}>
            {ecgData?.heartRate?.toFixed(0) || '--'} BPM
          </span>
        </Box>
        <Box>
          Confidence: <span style={{ color: futuristicTheme.colors.data.primary }}>
            {ecgData?.confidence ? (ecgData.confidence * 100).toFixed(1) : '--'}%
          </span>
        </Box>
      </Box>
    </WaveformContainer>
  )
}

export default ECGWaveformDisplay
