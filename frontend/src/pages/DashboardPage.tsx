import React, { useEffect } from 'react'
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  LinearProgress,
} from '@mui/material'
import {
  MonitorHeart,
  People,
  Assignment,
  Warning,
} from '@mui/icons-material'
import { useAppDispatch, useAppSelector } from '../hooks/redux'
import { fetchAnalyses } from '../store/slices/ecgSlice'
import { fetchUnreadCount } from '../store/slices/notificationSlice'

interface StatCardProps {
  title: string
  value: string | number
  icon: React.ReactNode
  color: string
  subtitle?: string
}

const StatCard: React.FC<StatCardProps> = ({ title, value, icon, color, subtitle }) => (
  <Card>
    <CardContent>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <Box sx={{ color, mr: 2 }}>{icon}</Box>
        <Typography variant="h6" component="div">
          {title}
        </Typography>
      </Box>
      <Typography variant="h4" component="div" gutterBottom>
        {value}
      </Typography>
      {subtitle && (
        <Typography variant="body2" color="text.secondary">
          {subtitle}
        </Typography>
      )}
    </CardContent>
  </Card>
)

const DashboardPage: React.FC = () => {
  const dispatch = useAppDispatch()
  const { analyses, isLoading } = useAppSelector(state => state.ecg)
  const { unreadCount } = useAppSelector(state => state.notification)

  useEffect(() => {
    dispatch(fetchAnalyses({ limit: 10 }))
    dispatch(fetchUnreadCount())
  }, [dispatch])

  const totalAnalyses = analyses.length
  const pendingAnalyses = analyses.filter(a => a.status === 'pending').length
  const criticalAnalyses = analyses.filter(a => a.clinicalUrgency === 'critical').length
  const completedAnalyses = analyses.filter(a => a.status === 'completed').length

  if (isLoading) {
    return (
      <Box sx={{ width: '100%' }}>
        <LinearProgress />
        <Typography sx={{ mt: 2 }}>Loading dashboard...</Typography>
      </Box>
    )
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>
      
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Total Analyses"
            value={totalAnalyses}
            icon={<MonitorHeart />}
            color="#1976d2"
            subtitle="All time"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Pending"
            value={pendingAnalyses}
            icon={<Assignment />}
            color="#ed6c02"
            subtitle="Awaiting processing"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Critical"
            value={criticalAnalyses}
            icon={<Warning />}
            color="#d32f2f"
            subtitle="Require immediate attention"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Notifications"
            value={unreadCount}
            icon={<People />}
            color="#2e7d32"
            subtitle="Unread messages"
          />
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent ECG Analyses
              </Typography>
              {analyses.slice(0, 5).map(analysis => (
                <Box
                  key={analysis.id}
                  sx={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    py: 1,
                    borderBottom: '1px solid #eee',
                  }}
                >
                  <Box>
                    <Typography variant="body1">
                      Analysis #{analysis.analysisId}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Patient ID: {analysis.patientId}
                    </Typography>
                  </Box>
                  <Box sx={{ textAlign: 'right' }}>
                    <Chip
                      label={analysis.status}
                      color={
                        analysis.status === 'completed'
                          ? 'success'
                          : analysis.status === 'pending'
                          ? 'warning'
                          : 'default'
                      }
                      size="small"
                    />
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                      {new Date(analysis.createdAt).toLocaleDateString()}
                    </Typography>
                  </Box>
                </Box>
              ))}
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Status
              </Typography>
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" gutterBottom>
                  Processing Queue
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={(completedAnalyses / (totalAnalyses || 1)) * 100}
                  sx={{ mb: 1 }}
                />
                <Typography variant="body2" color="text.secondary">
                  {completedAnalyses} of {totalAnalyses} completed
                </Typography>
              </Box>
              
              <Box>
                <Typography variant="body2" gutterBottom>
                  AI Model Status
                </Typography>
                <Chip label="Online" color="success" size="small" />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  )
}

export default DashboardPage
