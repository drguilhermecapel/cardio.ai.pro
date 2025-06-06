import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Users, FileText, Brain, Activity, TrendingUp, Clock } from 'lucide-react'

interface DashboardStats {
  total_patients: number
  total_consultations: number
  ai_diagnoses_accuracy: number
  recent_activity: Array<{
    type: string
    count: number
    date: string
  }>
}

export default function Dashboard() {
  const [stats, setStats] = useState<DashboardStats | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchDashboardData()
  }, [])

  const fetchDashboardData = async () => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/analytics/dashboard`)
      const data = await response.json()
      setStats(data)
    } catch (error) {
      console.error('Error fetching dashboard data:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-600">Visão geral do sistema SPEI</p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total de Pacientes</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats?.total_patients || 0}</div>
            <p className="text-xs text-muted-foreground">
              Pacientes cadastrados no sistema
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Consultas</CardTitle>
            <FileText className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats?.total_consultations || 0}</div>
            <p className="text-xs text-muted-foreground">
              Consultas realizadas
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Precisão da IA</CardTitle>
            <Brain className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {stats?.ai_diagnoses_accuracy ? `${(stats.ai_diagnoses_accuracy * 100).toFixed(1)}%` : '0%'}
            </div>
            <p className="text-xs text-muted-foreground">
              Precisão dos diagnósticos de IA
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Status do Sistema</CardTitle>
            <Activity className="h-4 w-4 text-green-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">Online</div>
            <p className="text-xs text-muted-foreground">
              Todos os serviços operacionais
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Recent Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Atividade Recente</CardTitle>
            <CardDescription>
              Últimas atividades do sistema
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {stats?.recent_activity?.map((activity, index) => (
                <div key={index} className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
                    <div>
                      <p className="text-sm font-medium">
                        {activity.type === 'consultation' ? 'Consultas' : 'Diagnósticos'}
                      </p>
                      <p className="text-xs text-gray-500">{activity.date}</p>
                    </div>
                  </div>
                  <Badge variant="secondary">{activity.count}</Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Recursos do Sistema</CardTitle>
            <CardDescription>
              Funcionalidades disponíveis
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center space-x-3">
                <Brain className="h-5 w-5 text-blue-600" />
                <div>
                  <p className="text-sm font-medium">IA Diagnóstica</p>
                  <p className="text-xs text-gray-500">Diagnósticos assistidos por IA</p>
                </div>
                <Badge className="ml-auto">Ativo</Badge>
              </div>
              
              <div className="flex items-center space-x-3">
                <FileText className="h-5 w-5 text-green-600" />
                <div>
                  <p className="text-sm font-medium">Prontuário Eletrônico</p>
                  <p className="text-xs text-gray-500">Conformidade FHIR</p>
                </div>
                <Badge className="ml-auto">Ativo</Badge>
              </div>
              
              <div className="flex items-center space-x-3">
                <Users className="h-5 w-5 text-purple-600" />
                <div>
                  <p className="text-sm font-medium">Telemedicina</p>
                  <p className="text-xs text-gray-500">Consultas remotas</p>
                </div>
                <Badge className="ml-auto">Ativo</Badge>
              </div>
              
              <div className="flex items-center space-x-3">
                <TrendingUp className="h-5 w-5 text-orange-600" />
                <div>
                  <p className="text-sm font-medium">Analytics</p>
                  <p className="text-xs text-gray-500">Relatórios e métricas</p>
                </div>
                <Badge className="ml-auto">Ativo</Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Compliance Information */}
      <Card>
        <CardHeader>
          <CardTitle>Conformidade Regulatória</CardTitle>
          <CardDescription>
            Status de conformidade com regulamentações de saúde
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="flex items-center space-x-3">
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <div>
                <p className="text-sm font-medium">ANVISA</p>
                <p className="text-xs text-gray-500">Conforme regulamentações brasileiras</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <div>
                <p className="text-sm font-medium">FDA</p>
                <p className="text-xs text-gray-500">Conforme regulamentações americanas</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <div>
                <p className="text-sm font-medium">EU GDPR</p>
                <p className="text-xs text-gray-500">Conforme regulamentações europeias</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
