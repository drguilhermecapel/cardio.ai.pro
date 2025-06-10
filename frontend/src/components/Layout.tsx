import React from 'react'
import { Outlet, Link, useNavigate, useLocation } from 'react-router-dom'
import { useAuth } from '../hooks/useAuth'

const Layout: React.FC = (): JSX.Element | null => {
  const { user, logout, isAuthenticated } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()

  React.useEffect(() => {
    if (!isAuthenticated) {
      navigate('/login')
    }
  }, [isAuthenticated, navigate])

  if (!isAuthenticated) {
    return null
  }

  const handleLogout = (): void => {
    logout()
    navigate('/login')
  }

  const isActive = (path: string): boolean => location.pathname === path

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-purple-50">
      <nav className="bg-gradient-to-r from-blue-600 via-purple-600 to-cyan-600 text-white shadow-2xl backdrop-blur-xl border-b border-white/20">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex justify-between h-20">
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 bg-gradient-to-br from-white/20 to-white/10 rounded-xl flex items-center justify-center backdrop-blur-sm">
                <span className="text-2xl">🫀</span>
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-white to-blue-100 bg-clip-text text-transparent">
                  CardioAI Pro
                </h1>
                <p className="text-sm text-blue-100">Sistema de Análise Cardíaca</p>
              </div>
            </div>
            <div className="flex items-center space-x-6">
              <div className="flex items-center space-x-2 bg-white/10 backdrop-blur-sm rounded-xl px-4 py-2">
                <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
                <span className="text-green-100">Online</span>
              </div>
              <span className="text-blue-100">Bem-vindo, {user?.username}</span>
              <button
                onClick={handleLogout}
                className="bg-white/10 hover:bg-white/20 backdrop-blur-sm px-4 py-2 rounded-xl transition-all duration-300 border border-white/20 hover:border-white/40"
              >
                Sair
              </button>
            </div>
          </div>
        </div>
      </nav>

      <div className="flex">
        <aside className="w-72 bg-white/80 backdrop-blur-xl shadow-2xl min-h-screen border-r border-gray-200/50">
          <nav className="mt-8">
            <div className="px-6 space-y-3">
              <Link
                to="/dashboard"
                className={`group flex items-center space-x-3 px-4 py-3 rounded-xl transition-all duration-300 ${
                  isActive('/dashboard')
                    ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg'
                    : 'text-gray-700 hover:bg-gradient-to-r hover:from-blue-50 hover:to-purple-50 hover:text-blue-700'
                }`}
              >
                <span className="text-xl">📊</span>
                <span className="font-medium">Dashboard</span>
              </Link>
              <Link
                to="/patients"
                className={`group flex items-center space-x-3 px-4 py-3 rounded-xl transition-all duration-300 ${
                  isActive('/patients')
                    ? 'bg-gradient-to-r from-green-500 to-emerald-600 text-white shadow-lg'
                    : 'text-gray-700 hover:bg-gradient-to-r hover:from-green-50 hover:to-emerald-50 hover:text-green-700'
                }`}
              >
                <span className="text-xl">👥</span>
                <span className="font-medium">Pacientes</span>
              </Link>
              <Link
                to="/medical-records"
                className={`group flex items-center space-x-3 px-4 py-3 rounded-xl transition-all duration-300 ${
                  isActive('/medical-records')
                    ? 'bg-gradient-to-r from-purple-500 to-pink-600 text-white shadow-lg'
                    : 'text-gray-700 hover:bg-gradient-to-r hover:from-purple-50 hover:to-pink-50 hover:text-purple-700'
                }`}
              >
                <span className="text-xl">📋</span>
                <span className="font-medium">Prontuários ECG</span>
              </Link>
              <Link
                to="/ai-diagnostics"
                className={`group flex items-center space-x-3 px-4 py-3 rounded-xl transition-all duration-300 ${
                  isActive('/ai-diagnostics')
                    ? 'bg-gradient-to-r from-red-500 to-pink-600 text-white shadow-lg'
                    : 'text-gray-700 hover:bg-gradient-to-r hover:from-red-50 hover:to-pink-50 hover:text-red-700'
                }`}
              >
                <span className="text-xl">🧠</span>
                <span className="font-medium">IA Cardíaca</span>
              </Link>
              <Link
                to="/telemedicine"
                className={`group flex items-center space-x-3 px-4 py-3 rounded-xl transition-all duration-300 ${
                  isActive('/telemedicine')
                    ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-lg'
                    : 'text-gray-700 hover:bg-gradient-to-r hover:from-cyan-50 hover:to-blue-50 hover:text-cyan-700'
                }`}
              >
                <span className="text-xl">📱</span>
                <span className="font-medium">Telemedicina</span>
              </Link>
            </div>
          </nav>
        </aside>

        <main className="flex-1">
          <Outlet />
        </main>
      </div>
    </div>
  )
}

export default Layout
