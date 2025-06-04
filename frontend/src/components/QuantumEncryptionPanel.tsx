import React, { useState, useEffect } from 'react'
import { Box, Typography, Chip } from '@mui/material'
import { styled } from '@mui/material/styles'
import { HolographicPanel } from './HolographicPanel'
import { futuristicTheme } from '../theme/futuristicTheme'

const EncryptionMatrix = styled(Box)(({ theme: _theme }) => ({
  height: '120px',
  background: `radial-gradient(circle at center, rgba(0, 255, 255, 0.1) 0%, transparent 70%)`,
  borderRadius: futuristicTheme.borderRadius.md,
  position: 'relative',
  overflow: 'hidden',
  border: `1px solid ${futuristicTheme.colors.ui.border}`,
  fontFamily: futuristicTheme.typography.fontFamily.mono,
}))

const QuantumBit = styled(Box)<{ state: 'superposition' | 'entangled' | 'measured' }>(({
  theme: _theme,
  state,
}) => {
  const getStateColor = (): string => {
    switch (state) {
      case 'superposition':
        return futuristicTheme.colors.data.holographic
      case 'entangled':
        return futuristicTheme.colors.neural.connections
      case 'measured':
        return futuristicTheme.colors.data.primary
      default:
        return futuristicTheme.colors.ui.border
    }
  }

  return {
    position: 'absolute',
    width: '6px',
    height: '6px',
    borderRadius: '50%',
    backgroundColor: getStateColor(),
    boxShadow: `0 0 8px ${getStateColor()}`,
    animation:
      state === 'superposition'
        ? 'quantumFlicker 0.5s ease-in-out infinite'
        : state === 'entangled'
          ? 'quantumPulse 1s ease-in-out infinite'
          : 'none',
    transition: 'all 0.3s ease',
  }
})

const SecurityMetric = styled(Box)(({ theme: _theme }) => ({
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  padding: futuristicTheme.spacing.sm,
  background: `rgba(0, 0, 0, 0.4)`,
  borderRadius: futuristicTheme.borderRadius.sm,
  border: `1px solid ${futuristicTheme.colors.ui.border}`,
  marginBottom: futuristicTheme.spacing.sm,
}))

const BlockchainBlock = styled(Box)<{ verified?: boolean }>(({ theme: _theme, verified }) => ({
  width: '20px',
  height: '20px',
  background: verified
    ? `linear-gradient(45deg, ${futuristicTheme.colors.data.secondary}, ${futuristicTheme.colors.neural.connections})`
    : `rgba(${futuristicTheme.colors.ui.border
        .replace('#', '')
        .match(/.{2}/g)
        ?.map(hex => parseInt(hex, 16))
        .join(', ')}, 0.3)`,
  border: `1px solid ${verified ? futuristicTheme.colors.data.secondary : futuristicTheme.colors.ui.border}`,
  borderRadius: futuristicTheme.borderRadius.sm,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  fontSize: futuristicTheme.typography.sizes.xs,
  color: verified ? futuristicTheme.colors.background.primary : futuristicTheme.colors.data.text,
  fontFamily: futuristicTheme.typography.fontFamily.mono,
  fontWeight: 'bold',
  boxShadow: verified ? `0 0 5px ${futuristicTheme.colors.data.secondary}` : 'none',
  animation: verified ? 'pulse 2s ease-in-out infinite' : 'none',
}))

const DataStreamIndicator = styled(Box)(({ theme: _theme }) => ({
  height: '2px',
  background: `linear-gradient(90deg, transparent, ${futuristicTheme.colors.data.holographic}, transparent)`,
  animation: 'dataStream 2s ease-in-out infinite',
  marginBottom: futuristicTheme.spacing.sm,
  borderRadius: '1px',
  boxShadow: `0 0 5px ${futuristicTheme.colors.data.holographic}`,
}))

interface QuantumState {
  keyStrength: number
  entanglementLevel: number
  decoherenceTime: number
  encryptionRate: number
  blockchainBlocks: boolean[]
  quantumBits: Array<{
    id: number
    state: 'superposition' | 'entangled' | 'measured'
    position: { x: number; y: number }
  }>
}

interface QuantumEncryptionPanelProps {
  className?: string
}

export const QuantumEncryptionPanel: React.FC<QuantumEncryptionPanelProps> = ({ className }) => {
  const [quantumState, setQuantumState] = useState<QuantumState>({
    keyStrength: 2048,
    entanglementLevel: 98.7,
    decoherenceTime: 15.3,
    encryptionRate: 1.2,
    blockchainBlocks: [true, true, true, true, false, false],
    quantumBits: [],
  })

  const [isEncrypting, setIsEncrypting] = useState(false)
  const [threatLevel, setThreatLevel] = useState<'low' | 'medium' | 'high'>('low')

  useEffect(() => {
    const bits = Array.from({ length: 24 }, (_, i) => ({
      id: i,
      state: 'superposition' as const,
      position: {
        x: 10 + (i % 6) * 15,
        y: 15 + Math.floor(i / 6) * 25,
      },
    }))
    setQuantumState(prev => ({ ...prev, quantumBits: bits }))
  }, [])

  useEffect(() => {
    const interval = setInterval(() => {
      setQuantumState(prev => ({
        ...prev,
        keyStrength: 2048 + Math.floor(Math.random() * 512),
        entanglementLevel: 97.5 + Math.random() * 2,
        decoherenceTime: 14 + Math.random() * 3,
        encryptionRate: 1.0 + Math.random() * 0.5,
        blockchainBlocks: prev.blockchainBlocks.map(() => Math.random() > 0.2),
        quantumBits: prev.quantumBits.map(bit => ({
          ...bit,
          state: (['superposition', 'entangled', 'measured'] as const)[
            Math.floor(Math.random() * 3)
          ],
        })),
      }))

      setIsEncrypting(Math.random() > 0.6)
      setThreatLevel((['low', 'medium', 'high'] as const)[Math.floor(Math.random() * 3)])
    }, 2000)

    return () => clearInterval(interval)
  }, [])

  const getThreatColor = (level: string): string => {
    switch (level) {
      case 'high':
        return futuristicTheme.colors.data.critical
      case 'medium':
        return futuristicTheme.colors.data.warning
      default:
        return futuristicTheme.colors.data.secondary
    }
  }

  return (
    <HolographicPanel
      title="Quantum Encryption & Blockchain"
      status={threatLevel === 'high' ? 'critical' : threatLevel === 'medium' ? 'warning' : 'active'}
      className={className}
      width="350px"
    >
      {isEncrypting && <DataStreamIndicator />}

      <EncryptionMatrix>
        {quantumState.quantumBits.map(bit => (
          <QuantumBit
            key={bit.id}
            state={bit.state}
            sx={{
              left: `${bit.position.x}%`,
              top: `${bit.position.y}%`,
              transform: 'translate(-50%, -50%)',
            }}
          />
        ))}

        {/* Quantum entanglement lines */}
        <svg
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            pointerEvents: 'none',
          }}
        >
          {quantumState.quantumBits
            .filter(bit => bit.state === 'entangled')
            .slice(0, 4)
            .map((bit, index, entangledBits) => {
              if (index === entangledBits.length - 1) return null
              const nextBit = entangledBits[index + 1]
              return (
                <line
                  key={`entanglement-${bit.id}-${nextBit.id}`}
                  x1={`${bit.position.x}%`}
                  y1={`${bit.position.y}%`}
                  x2={`${nextBit.position.x}%`}
                  y2={`${nextBit.position.y}%`}
                  stroke={futuristicTheme.colors.neural.connections}
                  strokeWidth="1"
                  opacity="0.6"
                  style={{
                    filter: `drop-shadow(0 0 3px ${futuristicTheme.colors.neural.connections})`,
                  }}
                />
              )
            })}
        </svg>

        {/* Central quantum processor */}
        <Box
          sx={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            width: '30px',
            height: '30px',
            border: `2px solid ${futuristicTheme.colors.data.holographic}`,
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: `radial-gradient(circle, rgba(0, 255, 255, 0.2), transparent)`,
            animation: 'quantumPulse 1.5s ease-in-out infinite',
          }}
        >
          <Typography
            sx={{
              color: futuristicTheme.colors.data.holographic,
              fontSize: futuristicTheme.typography.sizes.xs,
              fontFamily: futuristicTheme.typography.fontFamily.mono,
              fontWeight: 'bold',
            }}
          >
            Q
          </Typography>
        </Box>
      </EncryptionMatrix>

      {/* Security metrics */}
      <Box sx={{ mt: 2 }}>
        <SecurityMetric>
          <Typography
            variant="body2"
            sx={{
              color: futuristicTheme.colors.data.text,
              fontSize: futuristicTheme.typography.sizes.xs,
              fontFamily: futuristicTheme.typography.fontFamily.mono,
            }}
          >
            Key Strength
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
            {quantumState.keyStrength}-bit
          </Typography>
        </SecurityMetric>

        <SecurityMetric>
          <Typography
            variant="body2"
            sx={{
              color: futuristicTheme.colors.data.text,
              fontSize: futuristicTheme.typography.sizes.xs,
              fontFamily: futuristicTheme.typography.fontFamily.mono,
            }}
          >
            Entanglement Level
          </Typography>
          <Typography
            variant="body2"
            sx={{
              color: futuristicTheme.colors.neural.connections,
              fontSize: futuristicTheme.typography.sizes.sm,
              fontFamily: futuristicTheme.typography.fontFamily.primary,
              fontWeight: 'bold',
            }}
          >
            {quantumState.entanglementLevel.toFixed(1)}%
          </Typography>
        </SecurityMetric>

        <SecurityMetric>
          <Typography
            variant="body2"
            sx={{
              color: futuristicTheme.colors.data.text,
              fontSize: futuristicTheme.typography.sizes.xs,
              fontFamily: futuristicTheme.typography.fontFamily.mono,
            }}
          >
            Threat Level
          </Typography>
          <Chip
            label={threatLevel.toUpperCase()}
            size="small"
            sx={{
              backgroundColor: getThreatColor(threatLevel),
              color: futuristicTheme.colors.background.primary,
              fontSize: futuristicTheme.typography.sizes.xs,
              fontWeight: 'bold',
              boxShadow: `0 0 5px ${getThreatColor(threatLevel)}`,
            }}
          />
        </SecurityMetric>
      </Box>

      {/* Blockchain verification */}
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
          Blockchain Verification:
        </Typography>

        <Box sx={{ display: 'flex', gap: 0.5, mb: 2 }}>
          {quantumState.blockchainBlocks.map((verified, index) => (
            <BlockchainBlock key={index} verified={verified}>
              {verified ? '✓' : '○'}
            </BlockchainBlock>
          ))}
        </Box>
      </Box>

      {/* Encryption status */}
      <Box
        sx={{
          mt: 2,
          pt: 2,
          borderTop: `1px solid ${futuristicTheme.colors.ui.border}`,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box
            sx={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: isEncrypting
                ? futuristicTheme.colors.data.holographic
                : futuristicTheme.colors.ui.border,
              animation: isEncrypting ? 'pulse 1s ease-in-out infinite' : 'none',
              boxShadow: isEncrypting
                ? `0 0 10px ${futuristicTheme.colors.data.holographic}`
                : 'none',
            }}
          />
          <Typography
            variant="body2"
            sx={{
              color: isEncrypting
                ? futuristicTheme.colors.data.holographic
                : futuristicTheme.colors.data.text,
              fontSize: futuristicTheme.typography.sizes.xs,
              fontFamily: futuristicTheme.typography.fontFamily.mono,
            }}
          >
            {isEncrypting ? 'ENCRYPTING' : 'SECURE'}
          </Typography>
        </Box>

        <Typography
          variant="body2"
          sx={{
            color: futuristicTheme.colors.data.text,
            fontSize: futuristicTheme.typography.sizes.xs,
            fontFamily: futuristicTheme.typography.fontFamily.mono,
            opacity: 0.7,
          }}
        >
          {quantumState.encryptionRate.toFixed(1)} GB/s
        </Typography>
      </Box>

      {/* Global styles for animations */}
      <style>{`
        @keyframes quantumFlicker {
          0%, 100% { opacity: 0.3; transform: scale(0.8); }
          50% { opacity: 1; transform: scale(1.2); }
        }
        
        @keyframes quantumPulse {
          0%, 100% { opacity: 0.6; transform: scale(1); }
          50% { opacity: 1; transform: scale(1.1); }
        }
        
        @keyframes dataStream {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
      `}</style>
    </HolographicPanel>
  )
}

export default QuantumEncryptionPanel
