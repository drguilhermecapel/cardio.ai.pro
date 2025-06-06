import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { Badge } from '@/components/ui/badge'
import { Video, Phone, MessageSquare, Calendar, Clock, User, Plus } from 'lucide-react'

interface Consultation {
  id: string
  patient_id: string
  physician_id: string
  consultation_type: string
  status: string
  scheduled_time: string
  actual_start_time?: string
  actual_end_time?: string
  room_id?: string
  chief_complaint: string
  consultation_notes?: string
  diagnosis?: any
  ai_insights?: any
  created_at: string
}

export default function Telemedicine() {
  const [consultations, setConsultations] = useState<Consultation[]>([])
  const [loading, setLoading] = useState(true)
  const [showScheduleDialog, setShowScheduleDialog] = useState(false)
  const [newConsultation, setNewConsultation] = useState({
    patient_id: '',
    consultation_type: 'video',
    scheduled_time: '',
    chief_complaint: ''
  })

  useEffect(() => {
    setConsultations([
      {
        id: '1',
        patient_id: 'patient-1',
        physician_id: 'physician-1',
        consultation_type: 'video',
        status: 'scheduled',
        scheduled_time: new Date(Date.now() + 3600000).toISOString(), // 1 hour from now
        chief_complaint: 'Consulta de rotina',
        created_at: new Date().toISOString()
      },
      {
        id: '2',
        patient_id: 'patient-2',
        physician_id: 'physician-1',
        consultation_type: 'audio',
        status: 'completed',
        scheduled_time: new Date(Date.now() - 3600000).toISOString(), // 1 hour ago
        actual_start_time: new Date(Date.now() - 3600000).toISOString(),
        actual_end_time: new Date(Date.now() - 2700000).toISOString(),
        chief_complaint: 'Dor de cabeça',
        consultation_notes: 'Paciente apresenta cefaleia tensional. Prescrito analgésico.',
        created_at: new Date().toISOString()
      }
    ])
    setLoading(false)
  }, [])

  const handleScheduleConsultation = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      const token = localStorage.getItem('token')
      const response = await fetch(`${import.meta.env.VITE_API_URL}/telemedicine/consultations`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(newConsultation)
      })
      
      if (response.ok) {
        setShowScheduleDialog(false)
        setNewConsultation({
          patient_id: '',
          consultation_type: 'video',
          scheduled_time: '',
          chief_complaint: ''
        })
      }
    } catch (error) {
      console.error('Error scheduling consultation:', error)
    }
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'scheduled':
        return <Badge variant="outline">Agendada</Badge>
      case 'in_progress':
        return <Badge className="bg-green-600">Em Andamento</Badge>
      case 'completed':
        return <Badge variant="secondary">Concluída</Badge>
      case 'cancelled':
        return <Badge variant="destructive">Cancelada</Badge>
      default:
        return <Badge variant="outline">{status}</Badge>
    }
  }

  const getConsultationTypeIcon = (type: string) => {
    switch (type) {
      case 'video':
        return <Video className="h-4 w-4" />
      case 'audio':
        return <Phone className="h-4 w-4" />
      case 'chat':
        return <MessageSquare className="h-4 w-4" />
      default:
        return <Video className="h-4 w-4" />
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
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Telemedicina</h1>
          <p className="text-gray-600">Consultas médicas remotas</p>
        </div>
        
        <Dialog open={showScheduleDialog} onOpenChange={setShowScheduleDialog}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="h-4 w-4 mr-2" />
              Agendar Consulta
            </Button>
          </DialogTrigger>
          <DialogContent className="sm:max-w-[425px]">
            <DialogHeader>
              <DialogTitle>Agendar Nova Consulta</DialogTitle>
              <DialogDescription>
                Agende uma consulta de telemedicina
              </DialogDescription>
            </DialogHeader>
            <form onSubmit={handleScheduleConsultation} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="patient_id">ID do Paciente</Label>
                <Input
                  id="patient_id"
                  value={newConsultation.patient_id}
                  onChange={(e) => setNewConsultation({...newConsultation, patient_id: e.target.value})}
                  placeholder="ID do paciente"
                  required
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="consultation_type">Tipo de Consulta</Label>
                <select
                  id="consultation_type"
                  value={newConsultation.consultation_type}
                  onChange={(e) => setNewConsultation({...newConsultation, consultation_type: e.target.value})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                  required
                >
                  <option value="video">Videochamada</option>
                  <option value="audio">Chamada de Áudio</option>
                  <option value="chat">Chat</option>
                </select>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="scheduled_time">Data e Hora</Label>
                <Input
                  id="scheduled_time"
                  type="datetime-local"
                  value={newConsultation.scheduled_time}
                  onChange={(e) => setNewConsultation({...newConsultation, scheduled_time: e.target.value})}
                  required
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="chief_complaint">Motivo da Consulta</Label>
                <Textarea
                  id="chief_complaint"
                  value={newConsultation.chief_complaint}
                  onChange={(e) => setNewConsultation({...newConsultation, chief_complaint: e.target.value})}
                  placeholder="Descreva o motivo da consulta"
                  required
                />
              </div>
              
              <div className="flex justify-end space-x-2">
                <Button type="button" variant="outline" onClick={() => setShowScheduleDialog(false)}>
                  Cancelar
                </Button>
                <Button type="submit">
                  Agendar
                </Button>
              </div>
            </form>
          </DialogContent>
        </Dialog>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Calendar className="h-4 w-4 text-blue-600" />
              <div>
                <p className="text-sm font-medium">Agendadas</p>
                <p className="text-2xl font-bold">
                  {consultations.filter(c => c.status === 'scheduled').length}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Video className="h-4 w-4 text-green-600" />
              <div>
                <p className="text-sm font-medium">Em Andamento</p>
                <p className="text-2xl font-bold">
                  {consultations.filter(c => c.status === 'in_progress').length}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Clock className="h-4 w-4 text-purple-600" />
              <div>
                <p className="text-sm font-medium">Concluídas</p>
                <p className="text-2xl font-bold">
                  {consultations.filter(c => c.status === 'completed').length}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <User className="h-4 w-4 text-orange-600" />
              <div>
                <p className="text-sm font-medium">Total</p>
                <p className="text-2xl font-bold">{consultations.length}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Consultations List */}
      <Card>
        <CardHeader>
          <CardTitle>Consultas de Telemedicina</CardTitle>
          <CardDescription>
            Lista de consultas agendadas e realizadas
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {consultations.map((consultation) => (
              <div key={consultation.id} className="p-4 border rounded-lg">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-2">
                      {getConsultationTypeIcon(consultation.consultation_type)}
                      <h3 className="font-medium">{consultation.chief_complaint}</h3>
                      {getStatusBadge(consultation.status)}
                    </div>
                    
                    <div className="text-sm text-gray-600 space-y-1">
                      <p><strong>Paciente:</strong> {consultation.patient_id}</p>
                      <p><strong>Tipo:</strong> {
                        consultation.consultation_type === 'video' ? 'Videochamada' :
                        consultation.consultation_type === 'audio' ? 'Áudio' : 'Chat'
                      }</p>
                      <p><strong>Agendado para:</strong> {new Date(consultation.scheduled_time).toLocaleString('pt-BR')}</p>
                      
                      {consultation.actual_start_time && (
                        <p><strong>Iniciado:</strong> {new Date(consultation.actual_start_time).toLocaleString('pt-BR')}</p>
                      )}
                      
                      {consultation.actual_end_time && (
                        <p><strong>Finalizado:</strong> {new Date(consultation.actual_end_time).toLocaleString('pt-BR')}</p>
                      )}
                      
                      {consultation.consultation_notes && (
                        <p><strong>Notas:</strong> {consultation.consultation_notes}</p>
                      )}
                    </div>
                  </div>
                  
                  <div className="flex space-x-2">
                    {consultation.status === 'scheduled' && (
                      <Button size="sm" className="bg-green-600 hover:bg-green-700">
                        Iniciar
                      </Button>
                    )}
                    {consultation.status === 'in_progress' && (
                      <Button size="sm" variant="destructive">
                        Finalizar
                      </Button>
                    )}
                    <Button variant="outline" size="sm">
                      Detalhes
                    </Button>
                  </div>
                </div>
              </div>
            ))}
            
            {consultations.length === 0 && (
              <div className="text-center py-8 text-gray-500">
                Nenhuma consulta encontrada
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Compliance Notice */}
      <Card>
        <CardContent className="py-4">
          <div className="flex items-center space-x-2 text-sm text-gray-600">
            <Video className="h-4 w-4 text-blue-600" />
            <span>
              <strong>Telemedicina:</strong> Todas as consultas seguem as diretrizes do CFM e 
              regulamentações de telemedicina da ANVISA. Gravações e dados são protegidos 
              conforme LGPD e GDPR.
            </span>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
