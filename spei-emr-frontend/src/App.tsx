import { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { Toaster } from '@/components/ui/toaster'
import { AuthProvider, useAuth } from '@/contexts/AuthContext'
import LoginPage from '@/pages/LoginPage'
import Dashboard from '@/pages/Dashboard'
import PatientManagement from '@/pages/PatientManagement'
import MedicalRecords from '@/pages/MedicalRecords'
import AIDiagnostics from '@/pages/AIDiagnostics'
import Telemedicine from '@/pages/Telemedicine'
import Layout from '@/components/Layout'
import './App.css'

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, loading } = useAuth()
  
  if (loading) {
    return <div className="flex items-center justify-center min-h-screen">
      <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600"></div>
    </div>
  }
  
  return isAuthenticated ? <>{children}</> : <Navigate to="/login" />
}

function AppRoutes() {
  const { isAuthenticated } = useAuth()
  
  return (
    <Routes>
      <Route 
        path="/login" 
        element={isAuthenticated ? <Navigate to="/dashboard" /> : <LoginPage />} 
      />
      <Route path="/" element={<Navigate to="/dashboard" />} />
      <Route
        path="/dashboard"
        element={
          <ProtectedRoute>
            <Layout>
              <Dashboard />
            </Layout>
          </ProtectedRoute>
        }
      />
      <Route
        path="/patients"
        element={
          <ProtectedRoute>
            <Layout>
              <PatientManagement />
            </Layout>
          </ProtectedRoute>
        }
      />
      <Route
        path="/medical-records"
        element={
          <ProtectedRoute>
            <Layout>
              <MedicalRecords />
            </Layout>
          </ProtectedRoute>
        }
      />
      <Route
        path="/ai-diagnostics"
        element={
          <ProtectedRoute>
            <Layout>
              <AIDiagnostics />
            </Layout>
          </ProtectedRoute>
        }
      />
      <Route
        path="/telemedicine"
        element={
          <ProtectedRoute>
            <Layout>
              <Telemedicine />
            </Layout>
          </ProtectedRoute>
        }
      />
    </Routes>
  )
}

function App() {
  return (
    <AuthProvider>
      <Router>
        <div className="min-h-screen bg-gray-50">
          <AppRoutes />
          <Toaster />
        </div>
      </Router>
    </AuthProvider>
  )
}

export default App
