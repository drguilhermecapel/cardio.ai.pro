import React, { useState, useEffect } from 'react'
import { Box, Typography } from '@mui/material'
import { styled } from '@mui/material/styles'
import { futuristicTheme } from '../theme/futuristicTheme'

const MatrixContainer = styled(Box)(({ theme: _theme }) => ({
  position: 'fixed',
  top: 0,
  left: 0,
  right: 0,
  bottom: 0,
  pointerEvents: 'none',
  zIndex: futuristicTheme.zIndex.background,
  overflow: 'hidden',
}))

const BadgeElement = styled(Box)<{ verified?: boolean; floating?: boolean }>(
  ({ theme: _theme, verified, floating }) => ({
    position: 'absolute',
    padding: `${futuristicTheme.spacing.xs} ${futuristicTheme.spacing.sm}`,
    background: verified ? `rgba(0, 255, 127, 0.1)` : `rgba(255, 165, 0, 0.1)`,
    border: `1px solid ${verified ? futuristicTheme.colors.data.secondary : futuristicTheme.colors.data.warning}`,
    borderRadius: futuristicTheme.borderRadius.sm,
    backdropFilter: futuristicTheme.effects.blur.background,
    opacity: floating ? 0.6 : 0.3,
    animation: floating ? 'float 6s ease-in-out infinite' : 'none',
    transition: 'all 0.3s ease',
    fontSize: futuristicTheme.typography.sizes.xs,
    fontFamily: futuristicTheme.typography.fontFamily.mono,
    color: verified ? futuristicTheme.colors.data.secondary : futuristicTheme.colors.data.warning,
    fontWeight: 'bold',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
    boxShadow: verified ? `0 0 10px rgba(0, 255, 127, 0.3)` : `0 0 10px rgba(255, 165, 0, 0.3)`,
    '&::before': {
      content: '""',
      position: 'absolute',
      top: '-2px',
      left: '-2px',
      right: '-2px',
      bottom: '-2px',
      background: `linear-gradient(45deg, ${verified ? futuristicTheme.colors.data.secondary : futuristicTheme.colors.data.warning}, transparent, ${verified ? futuristicTheme.colors.data.secondary : futuristicTheme.colors.data.warning})`,
      borderRadius: futuristicTheme.borderRadius.sm,
      opacity: 0.3,
      zIndex: -1,
      animation: floating ? 'borderGlow 3s ease-in-out infinite' : 'none',
    },
  })
)

const ValidationMetric = styled(Box)<{ status: 'verified' | 'pending' | 'expired' }>(({
  theme: _theme,
  status,
}) => {
  const getStatusColor = (): string => {
    switch (status) {
      case 'verified':
        return futuristicTheme.colors.data.secondary
      case 'pending':
        return futuristicTheme.colors.data.warning
      case 'expired':
        return futuristicTheme.colors.data.critical
      default:
        return futuristicTheme.colors.ui.border
    }
  }

  return {
    display: 'flex',
    alignItems: 'center',
    gap: futuristicTheme.spacing.xs,
    fontSize: futuristicTheme.typography.sizes.xs,
    color: getStatusColor(),
    '&::before': {
      content: '""',
      width: '6px',
      height: '6px',
      borderRadius: '50%',
      backgroundColor: getStatusColor(),
      boxShadow: `0 0 5px ${getStatusColor()}`,
    },
  }
})

interface ComplianceBadge {
  id: string
  name: string
  authority: string
  status: 'verified' | 'pending' | 'expired'
  validUntil: string
  certificationNumber: string
  position: { x: number; y: number }
  floating: boolean
}

interface ComplianceBadgesMatrixProps {
  className?: string
}

export const ComplianceBadgesMatrix: React.FC<ComplianceBadgesMatrixProps> = ({ className }) => {
  const [badges, setBadges] = useState<ComplianceBadge[]>([])
  const [floatingBadges, setFloatingBadges] = useState<ComplianceBadge[]>([])

  useEffect(() => {
    const complianceBadges: ComplianceBadge[] = [
      {
        id: 'fda-510k',
        name: 'FDA 510(k)',
        authority: 'US FDA',
        status: 'verified',
        validUntil: '2026-12-31',
        certificationNumber: 'K243891',
        position: { x: 5, y: 10 },
        floating: false,
      },
      {
        id: 'ce-mdd',
        name: 'CE-MDD',
        authority: 'European Union',
        status: 'verified',
        validUntil: '2027-06-15',
        certificationNumber: 'CE-0123',
        position: { x: 85, y: 15 },
        floating: false,
      },
      {
        id: 'anvisa',
        name: 'ANVISA',
        authority: 'Brazil ANVISA',
        status: 'verified',
        validUntil: '2026-09-30',
        certificationNumber: 'BR-80146900001',
        position: { x: 10, y: 85 },
        floating: false,
      },
      {
        id: 'nmpa',
        name: 'NMPA',
        authority: 'China NMPA',
        status: 'pending',
        validUntil: '2025-03-15',
        certificationNumber: 'NMPA-2024-001',
        position: { x: 80, y: 80 },
        floating: false,
      },
      {
        id: 'iso13485',
        name: 'ISO 13485',
        authority: 'ISO',
        status: 'verified',
        validUntil: '2026-11-20',
        certificationNumber: 'ISO-13485-2024',
        position: { x: 45, y: 5 },
        floating: false,
      },
      {
        id: 'iso14155',
        name: 'ISO 14155',
        authority: 'ISO',
        status: 'verified',
        validUntil: '2025-08-10',
        certificationNumber: 'ISO-14155-2024',
        position: { x: 15, y: 50 },
        floating: false,
      },
      {
        id: 'iec62304',
        name: 'IEC 62304',
        authority: 'IEC',
        status: 'verified',
        validUntil: '2027-01-25',
        certificationNumber: 'IEC-62304-2024',
        position: { x: 75, y: 45 },
        floating: false,
      },
      {
        id: 'hipaa',
        name: 'HIPAA',
        authority: 'US HHS',
        status: 'verified',
        validUntil: '2025-12-31',
        certificationNumber: 'HIPAA-2024-COMP',
        position: { x: 50, y: 90 },
        floating: false,
      },
    ]

    const floatingValidationBadges: ComplianceBadge[] = [
      {
        id: 'gdpr',
        name: 'GDPR Compliant',
        authority: 'EU',
        status: 'verified',
        validUntil: '2025-05-25',
        certificationNumber: 'GDPR-2024-001',
        position: { x: 25, y: 25 },
        floating: true,
      },
      {
        id: 'soc2',
        name: 'SOC 2 Type II',
        authority: 'AICPA',
        status: 'verified',
        validUntil: '2025-07-15',
        certificationNumber: 'SOC2-2024-T2',
        position: { x: 65, y: 25 },
        floating: true,
      },
      {
        id: 'fips140',
        name: 'FIPS 140-2',
        authority: 'NIST',
        status: 'verified',
        validUntil: '2026-03-10',
        certificationNumber: 'FIPS-140-2-L3',
        position: { x: 35, y: 65 },
        floating: true,
      },
      {
        id: 'common-criteria',
        name: 'Common Criteria',
        authority: 'ISO/IEC',
        status: 'pending',
        validUntil: '2025-09-30',
        certificationNumber: 'CC-EAL4-2024',
        position: { x: 55, y: 35 },
        floating: true,
      },
    ]

    setBadges(complianceBadges)
    setFloatingBadges(floatingValidationBadges)

    const interval = setInterval(() => {
      setFloatingBadges(prev =>
        prev.map(badge => ({
          ...badge,
          position: {
            x: badge.position.x + (Math.random() - 0.5) * 2,
            y: badge.position.y + (Math.random() - 0.5) * 2,
          },
        }))
      )
    }, 5000)

    return () => clearInterval(interval)
  }, [])

  return (
    <MatrixContainer className={className}>
      {/* Static compliance badges */}
      {badges.map(badge => (
        <BadgeElement
          key={badge.id}
          verified={badge.status === 'verified'}
          floating={false}
          sx={{
            top: `${badge.position.y}%`,
            left: `${badge.position.x}%`,
            transform: 'translate(-50%, -50%)',
          }}
        >
          <ValidationMetric status={badge.status}>{badge.name}</ValidationMetric>

          <Typography
            sx={{
              fontSize: futuristicTheme.typography.sizes.xs,
              color: 'inherit',
              opacity: 0.7,
              mt: 0.5,
            }}
          >
            {badge.certificationNumber}
          </Typography>
        </BadgeElement>
      ))}

      {/* Floating validation metrics */}
      {floatingBadges.map(badge => (
        <BadgeElement
          key={badge.id}
          verified={badge.status === 'verified'}
          floating={true}
          sx={{
            top: `${badge.position.y}%`,
            left: `${badge.position.x}%`,
            transform: 'translate(-50%, -50%)',
            animationDelay: `${Math.random() * 2}s`,
          }}
        >
          <ValidationMetric status={badge.status}>{badge.name}</ValidationMetric>

          <Typography
            sx={{
              fontSize: futuristicTheme.typography.sizes.xs,
              color: 'inherit',
              opacity: 0.7,
              mt: 0.5,
            }}
          >
            Valid: {badge.validUntil}
          </Typography>
        </BadgeElement>
      ))}

      {/* Subtle data flow particles */}
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: `
            radial-gradient(circle at 20% 30%, rgba(0, 191, 255, 0.05) 0%, transparent 50%),
            radial-gradient(circle at 80% 70%, rgba(0, 255, 127, 0.05) 0%, transparent 50%),
            radial-gradient(circle at 50% 50%, rgba(255, 165, 0, 0.03) 0%, transparent 50%)
          `,
          animation: 'dataFlow 20s ease-in-out infinite',
        }}
      />

      {/* Global styles for animations */}
      <style>{`
        @keyframes float {
          0%, 100% { transform: translate(-50%, -50%) translateY(0px); }
          50% { transform: translate(-50%, -50%) translateY(-10px); }
        }
        
        @keyframes borderGlow {
          0%, 100% { opacity: 0.3; }
          50% { opacity: 0.6; }
        }
        
        @keyframes dataFlow {
          0%, 100% { opacity: 0.8; }
          50% { opacity: 1; }
        }
      `}</style>
    </MatrixContainer>
  )
}

export default ComplianceBadgesMatrix
