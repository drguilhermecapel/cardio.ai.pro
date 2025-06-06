import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { Badge } from '@/components/ui/badge'
import { Plus, Search, Edit, Eye } from 'lucide-react'

interface Patient {
  id: string
  cpf: string
  name: string
  birth_date: string
  gender: string
  phone?: string
  email?: string
  created_at: string
}

export default function PatientManagement() {
  const [patients, setPatients] = useState<Patient[]>([])
  const [loading, setLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState('')
  const [showAddDialog, setShowAddDialog] = useState(false)
  const [newPatient, setNewPatient] = useState({
    cpf: '',
    name: '',
    birth_date: '',
    gender: '',
    phone: '',
    email: ''
  })

  useEffect(() => {
    fetchPatients()
  }, [])

  const fetchPatients = async () => {
    try {
      const token = localStorage.getItem('token')
      const response = await fetch(`${import.meta.env.VITE_API_URL}/patients`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
      const data = await response.json()
      setPatients(data)
    } catch (error) {
      console.error('Error fetching patients:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleAddPatient = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      const token = localStorage.getItem('token')
      const response = await fetch(`${import.meta.env.VITE_API_URL}/patients`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(newPatient)
      })
      
      if (response.ok) {
        setShowAddDialog(false)
        setNewPatient({
          cpf: '',
          name: '',
          birth_date: '',
          gender: '',
          phone: '',
          email: ''
        })
        fetchPatients()
      }
    } catch (error) {
      console.error('Error adding patient:', error)
    }
  }

  const filteredPatients = patients.filter(patient =>
    patient.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    patient.cpf.includes(searchTerm)
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
          <h1 className="text-2xl font-bold text-gray-900">Gerenciamento de Pacientes</h1>
          <p className="text-gray-600">Cadastro e consulta de pacientes</p>
        </div>
        
        <Dialog open={showAddDialog} onOpenChange={setShowAddDialog}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="h-4 w-4 mr-2" />
              Novo Paciente
            </Button>
          </DialogTrigger>
          <DialogContent className="sm:max-w-[425px]">
            <DialogHeader>
              <DialogTitle>Cadastrar Novo Paciente</DialogTitle>
              <DialogDescription>
                Preencha os dados do paciente para cadastro no sistema
              </DialogDescription>
            </DialogHeader>
            <form onSubmit={handleAddPatient} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="cpf">CPF</Label>
                <Input
                  id="cpf"
                  value={newPatient.cpf}
                  onChange={(e) => setNewPatient({...newPatient, cpf: e.target.value})}
                  placeholder="000.000.000-00"
                  required
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="name">Nome Completo</Label>
                <Input
                  id="name"
                  value={newPatient.name}
                  onChange={(e) => setNewPatient({...newPatient, name: e.target.value})}
                  placeholder="Nome completo do paciente"
                  required
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="birth_date">Data de Nascimento</Label>
                <Input
                  id="birth_date"
                  type="date"
                  value={newPatient.birth_date}
                  onChange={(e) => setNewPatient({...newPatient, birth_date: e.target.value})}
                  required
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="gender">Gênero</Label>
                <select
                  id="gender"
                  value={newPatient.gender}
                  onChange={(e) => setNewPatient({...newPatient, gender: e.target.value})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                  required
                >
                  <option value="">Selecione</option>
                  <option value="M">Masculino</option>
                  <option value="F">Feminino</option>
                  <option value="O">Outro</option>
                </select>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="phone">Telefone</Label>
                <Input
                  id="phone"
                  value={newPatient.phone}
                  onChange={(e) => setNewPatient({...newPatient, phone: e.target.value})}
                  placeholder="(11) 99999-9999"
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  type="email"
                  value={newPatient.email}
                  onChange={(e) => setNewPatient({...newPatient, email: e.target.value})}
                  placeholder="email@exemplo.com"
                />
              </div>
              
              <div className="flex justify-end space-x-2">
                <Button type="button" variant="outline" onClick={() => setShowAddDialog(false)}>
                  Cancelar
                </Button>
                <Button type="submit">
                  Cadastrar
                </Button>
              </div>
            </form>
          </DialogContent>
        </Dialog>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Lista de Pacientes</CardTitle>
          <CardDescription>
            {patients.length} pacientes cadastrados
          </CardDescription>
          <div className="flex items-center space-x-2">
            <Search className="h-4 w-4 text-gray-400" />
            <Input
              placeholder="Buscar por nome ou CPF..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="max-w-sm"
            />
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {filteredPatients.map((patient) => (
              <div key={patient.id} className="flex items-center justify-between p-4 border rounded-lg">
                <div className="flex-1">
                  <div className="flex items-center space-x-4">
                    <div>
                      <h3 className="font-medium">{patient.name}</h3>
                      <p className="text-sm text-gray-500">CPF: {patient.cpf}</p>
                    </div>
                    <Badge variant="secondary">
                      {patient.gender === 'M' ? 'Masculino' : patient.gender === 'F' ? 'Feminino' : 'Outro'}
                    </Badge>
                  </div>
                  <div className="mt-2 text-sm text-gray-600">
                    <p>Nascimento: {new Date(patient.birth_date).toLocaleDateString('pt-BR')}</p>
                    {patient.phone && <p>Telefone: {patient.phone}</p>}
                    {patient.email && <p>Email: {patient.email}</p>}
                  </div>
                </div>
                
                <div className="flex space-x-2">
                  <Button variant="outline" size="sm">
                    <Eye className="h-4 w-4" />
                  </Button>
                  <Button variant="outline" size="sm">
                    <Edit className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            ))}
            
            {filteredPatients.length === 0 && (
              <div className="text-center py-8 text-gray-500">
                Nenhum paciente encontrado
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
