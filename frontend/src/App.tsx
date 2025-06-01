import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { Box } from '@mui/material'

import { useAppSelector } from './hooks/redux'
import Layout from './components/Layout'
import LoginPage from './pages/LoginPage'
import DashboardPage from './pages/DashboardPage'
import ECGAnalysisPage from './pages/ECGAnalysisPage'
import PatientsPage from './pages/PatientsPage'
import ValidationsPage from './pages/ValidationsPage'
import NotificationsPage from './pages/NotificationsPage'
import ProfilePage from './pages/ProfilePage'

const App: React.FC = () => {
  const { isAuthenticated } = useAppSelector(state => state.auth)

  if (!isAuthenticated) {
    return <LoginPage />
  }

  return (
    <Layout>
      <Box sx={{ flexGrow: 1, p: 3 }}>
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/ecg-analysis" element={<ECGAnalysisPage />} />
          <Route path="/patients" element={<PatientsPage />} />
          <Route path="/validations" element={<ValidationsPage />} />
          <Route path="/notifications" element={<NotificationsPage />} />
          <Route path="/profile" element={<ProfilePage />} />
          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </Box>
    </Layout>
  )
}

export default App
