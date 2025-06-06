import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { Badge } from '@/components/ui/badge'
import { Plus, Search, FileText, Calendar } from 'lucide-react'

interface MedicalRecord {
  id: string
  patient_id: string
  encounter_id: string
  document_type: string
  status: string
  chief_complaint: string
  history_present_illness?: string
  assessment?: string
  plan?: string
  created_at: string
  created_by: string
}

export default function MedicalRecords() {
  const [records, setRecords] = useState<MedicalRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState('')
  const [showAddDialog, setShowAddDialog] = useState(false)
  const [newRecord, setNewRecord] = useState({
    patient_id: '',
    encounter_id: '',
    document_type: 'anamnese',
    chief_complaint: '',
    history_present_illness: '',
    assessment: '',
    plan: ''
  })

  useEffect(() => {
    setRecords([
      {
        id: '1',
        patient_id: 'patient-1',
        encounter_id: 'enc-001',
        document_type: 'anamnese',
        status: 'draft',
        chief_complaint: 'Dor de cabeça há 3 dias',
        history_present_illness: 'Paciente relata cefaleia frontal, pulsátil, de intensidade moderada',
        assessment: 'Cefaleia tensional',
        plan: 'Analgésico e acompanhamento',
        created_at: new Date().toISOString(),
        created_by: 'Dr. Silva'
      }
    ])
    setLoading(false)
  }, [])

  const handleAddRecord = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      const token = localStorage.getItem('token')
      const response = await fetch(`${import.meta.env.VITE_API_URL}/medical-records`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          ...newRecord,
          encounter_id: `enc-${Date.now()}`
        })
      })
      
      if (response.ok) {
        setShowAddDialog(false)
        setNewRecord({
          patient_id: '',
          encounter_id: '',
          document_type: 'anamnese',
          chief_complaint: '',
          history_present_illness: '',
          assessment: '',
          plan: ''
        })
      }
    } catch (error) {
      console.error('Error adding medical record:', error)
    }
  }

  const filteredRecords = records.filter(record =>
    record.chief_complaint.toLowerCase().includes(searchTerm.toLowerCase()) ||
    record.document_type.toLowerCase().includes(searchTerm.toLowerCase())
  )

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
          <h1 className="text-2xl font-bold text-gray-900">Prontuários Médicos</h1>
          <p className="text-gray-600">Gestão de prontuários eletrônicos</p>
        </div>
        
        <Dialog open={showAddDialog} onOpenChange={setShowAddDialog}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="h-4 w-4 mr-2" />
              Novo Prontuário
            </Button>
          </DialogTrigger>
          <DialogContent className="sm:max-w-[600px]">
            <DialogHeader>
              <DialogTitle>Criar Novo Prontuário</DialogTitle>
              <DialogDescription>
                Preencha as informações do prontuário médico
              </DialogDescription>
            </DialogHeader>
            <form onSubmit={handleAddRecord} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="patient_id">ID do Paciente</Label>
                <Input
                  id="patient_id"
                  value={newRecord.patient_id}
                  onChange={(e) => setNewRecord({...newRecord, patient_id: e.target.value})}
                  placeholder="ID do paciente"
                  required
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="document_type">Tipo de Documento</Label>
                <select
                  id="document_type"
                  value={newRecord.document_type}
                  onChange={(e) => setNewRecord({...newRecord, document_type: e.target.value})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                  required
                >
                  <option value="anamnese">Anamnese</option>
                  <option value="evolucao">Evolução</option>
                  <option value="admissao">Admissão</option>
                  <option value="alta">Alta</option>
                  <option value="cirurgico">Cirúrgico</option>
                  <option value="consulta">Consulta</option>
                </select>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="chief_complaint">Queixa Principal</Label>
                <Textarea
                  id="chief_complaint"
                  value={newRecord.chief_complaint}
                  onChange={(e) => setNewRecord({...newRecord, chief_complaint: e.target.value})}
                  placeholder="Descreva a queixa principal do paciente"
                  required
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="history_present_illness">História da Doença Atual</Label>
                <Textarea
                  id="history_present_illness"
                  value={newRecord.history_present_illness}
                  onChange={(e) => setNewRecord({...newRecord, history_present_illness: e.target.value})}
                  placeholder="Descreva a história da doença atual"
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="assessment">Avaliação</Label>
                <Textarea
                  id="assessment"
                  value={newRecord.assessment}
                  onChange={(e) => setNewRecord({...newRecord, assessment: e.target.value})}
                  placeholder="Avaliação médica"
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="plan">Plano</Label>
                <Textarea
                  id="plan"
                  value={newRecord.plan}
                  onChange={(e) => setNewRecord({...newRecord, plan: e.target.value})}
                  placeholder="Plano de tratamento"
                />
              </div>
              
              <div className="flex justify-end space-x-2">
                <Button type="button" variant="outline" onClick={() => setShowAddDialog(false)}>
                  Cancelar
                </Button>
                <Button type="submit">
                  Criar Prontuário
                </Button>
              </div>
            </form>
          </DialogContent>
        </Dialog>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Lista de Prontuários</CardTitle>
          <CardDescription>
            {records.length} prontuários no sistema
          </CardDescription>
          <div className="flex items-center space-x-2">
            <Search className="h-4 w-4 text-gray-400" />
            <Input
              placeholder="Buscar por queixa ou tipo..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="max-w-sm"
            />
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {filteredRecords.map((record) => (
              <div key={record.id} className="p-4 border rounded-lg">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-2">
                      <FileText className="h-4 w-4 text-blue-600" />
                      <h3 className="font-medium">{record.chief_complaint}</h3>
                      <Badge variant="secondary">{record.document_type}</Badge>
                      <Badge variant={record.status === 'draft' ? 'outline' : 'default'}>
                        {record.status === 'draft' ? 'Rascunho' : 'Finalizado'}
                      </Badge>
                    </div>
                    
                    <div className="text-sm text-gray-600 space-y-1">
                      <p><strong>Paciente:</strong> {record.patient_id}</p>
                      <p><strong>Encontro:</strong> {record.encounter_id}</p>
                      {record.history_present_illness && (
                        <p><strong>História:</strong> {record.history_present_illness}</p>
                      )}
                      {record.assessment && (
                        <p><strong>Avaliação:</strong> {record.assessment}</p>
                      )}
                      {record.plan && (
                        <p><strong>Plano:</strong> {record.plan}</p>
                      )}
                    </div>
                    
                    <div className="flex items-center space-x-4 mt-3 text-xs text-gray-500">
                      <div className="flex items-center space-x-1">
                        <Calendar className="h-3 w-3" />
                        <span>{new Date(record.created_at).toLocaleDateString('pt-BR')}</span>
                      </div>
                      <span>Por: {record.created_by}</span>
                    </div>
                  </div>
                  
                  <div className="flex space-x-2">
                    <Button variant="outline" size="sm">
                      Visualizar
                    </Button>
                    <Button variant="outline" size="sm">
                      Editar
                    </Button>
                  </div>
                </div>
              </div>
            ))}
            
            {filteredRecords.length === 0 && (
              <div className="text-center py-8 text-gray-500">
                Nenhum prontuário encontrado
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
