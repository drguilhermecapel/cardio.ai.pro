import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Brain, Stethoscope, AlertTriangle, CheckCircle, Clock } from 'lucide-react'

interface DiagnosisHypothesis {
  icd_code: string
  icd_version: string
  description: string
  confidence: number
  evidence: string[]
  differential_features: string[]
  required_tests: string[]
  treatment_implications: string[]
  prognosis: string
}

interface DiagnosisResponse {
  session_id: string
  primary_diagnoses: DiagnosisHypothesis[]
  differential_diagnoses: DiagnosisHypothesis[]
  ruled_out_diagnoses: string[]
  ai_confidence_score: number
  processing_time: number
  recommendations: string[]
  red_flags: string[]
}

export default function AIDiagnostics() {
  const [loading, setLoading] = useState(false)
  const [diagnosisResult, setDiagnosisResult] = useState<DiagnosisResponse | null>(null)
  const [formData, setFormData] = useState({
    patient_id: '',
    chief_complaint: '',
    history_present_illness: '',
    symptoms: '',
    vital_signs: '',
    physical_examination: '',
    laboratory_results: '',
    imaging_results: ''
  })

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)

    try {
      const token = localStorage.getItem('token')
      const response = await fetch(`${import.meta.env.VITE_API_URL}/ai/diagnose`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          patient_id: formData.patient_id,
          chief_complaint: formData.chief_complaint,
          history_present_illness: formData.history_present_illness,
          symptoms: formData.symptoms.split(',').map(s => ({ symptom: s.trim() })),
          vital_signs: formData.vital_signs ? JSON.parse(formData.vital_signs) : {},
          physical_examination: formData.physical_examination ? JSON.parse(formData.physical_examination) : {},
          laboratory_results: formData.laboratory_results ? JSON.parse(formData.laboratory_results) : {},
          imaging_results: formData.imaging_results ? JSON.parse(formData.imaging_results) : {}
        })
      })

      const result = await response.json()
      setDiagnosisResult(result)
    } catch (error) {
      console.error('Error getting AI diagnosis:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">IA Diagnóstica</h1>
        <p className="text-gray-600">Sistema de diagnóstico assistido por inteligência artificial</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Form */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Brain className="h-5 w-5 text-blue-600" />
              <span>Dados Clínicos</span>
            </CardTitle>
            <CardDescription>
              Insira os dados clínicos para análise diagnóstica
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="patient_id">ID do Paciente</Label>
                <Input
                  id="patient_id"
                  value={formData.patient_id}
                  onChange={(e) => setFormData({...formData, patient_id: e.target.value})}
                  placeholder="ID do paciente"
                  required
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="chief_complaint">Queixa Principal</Label>
                <Textarea
                  id="chief_complaint"
                  value={formData.chief_complaint}
                  onChange={(e) => setFormData({...formData, chief_complaint: e.target.value})}
                  placeholder="Ex: Dor de cabeça há 3 dias"
                  required
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="history_present_illness">História da Doença Atual</Label>
                <Textarea
                  id="history_present_illness"
                  value={formData.history_present_illness}
                  onChange={(e) => setFormData({...formData, history_present_illness: e.target.value})}
                  placeholder="Descreva a evolução dos sintomas"
                  required
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="symptoms">Sintomas (separados por vírgula)</Label>
                <Input
                  id="symptoms"
                  value={formData.symptoms}
                  onChange={(e) => setFormData({...formData, symptoms: e.target.value})}
                  placeholder="febre, cefaleia, náusea"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="vital_signs">Sinais Vitais (JSON)</Label>
                <Textarea
                  id="vital_signs"
                  value={formData.vital_signs}
                  onChange={(e) => setFormData({...formData, vital_signs: e.target.value})}
                  placeholder='{"temperatura": 38.5, "pressao": "120/80", "frequencia_cardiaca": 90}'
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="physical_examination">Exame Físico (JSON)</Label>
                <Textarea
                  id="physical_examination"
                  value={formData.physical_examination}
                  onChange={(e) => setFormData({...formData, physical_examination: e.target.value})}
                  placeholder='{"cabeca_pescoco": "normal", "torax": "murmúrio vesicular presente"}'
                />
              </div>

              <Button type="submit" className="w-full" disabled={loading}>
                {loading ? (
                  <>
                    <Clock className="mr-2 h-4 w-4 animate-spin" />
                    Analisando...
                  </>
                ) : (
                  <>
                    <Brain className="mr-2 h-4 w-4" />
                    Analisar com IA
                  </>
                )}
              </Button>
            </form>
          </CardContent>
        </Card>

        {/* Results */}
        <div className="space-y-6">
          {diagnosisResult && (
            <>
              {/* AI Confidence Score */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Stethoscope className="h-5 w-5 text-green-600" />
                    <span>Análise Concluída</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between text-sm mb-2">
                        <span>Confiança da IA</span>
                        <span>{(diagnosisResult.ai_confidence_score * 100).toFixed(1)}%</span>
                      </div>
                      <Progress value={diagnosisResult.ai_confidence_score * 100} />
                    </div>
                    
                    <div className="text-sm text-gray-600">
                      <p>Tempo de processamento: {diagnosisResult.processing_time}s</p>
                      <p>Sessão: {diagnosisResult.session_id}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Primary Diagnoses */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <CheckCircle className="h-5 w-5 text-green-600" />
                    <span>Diagnósticos Primários</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {diagnosisResult.primary_diagnoses.map((diagnosis, index) => (
                      <div key={index} className="border rounded-lg p-4">
                        <div className="flex items-center justify-between mb-2">
                          <h4 className="font-medium">{diagnosis.description}</h4>
                          <Badge variant="default">
                            {(diagnosis.confidence * 100).toFixed(1)}%
                          </Badge>
                        </div>
                        
                        <div className="text-sm text-gray-600 space-y-2">
                          <p><strong>CID:</strong> {diagnosis.icd_code} (v{diagnosis.icd_version})</p>
                          <p><strong>Prognóstico:</strong> {diagnosis.prognosis}</p>
                          
                          {diagnosis.evidence.length > 0 && (
                            <div>
                              <strong>Evidências:</strong>
                              <ul className="list-disc list-inside ml-2">
                                {diagnosis.evidence.map((evidence, i) => (
                                  <li key={i}>{evidence}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                          
                          {diagnosis.required_tests.length > 0 && (
                            <div>
                              <strong>Exames recomendados:</strong>
                              <ul className="list-disc list-inside ml-2">
                                {diagnosis.required_tests.map((test, i) => (
                                  <li key={i}>{test}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* Differential Diagnoses */}
              {diagnosisResult.differential_diagnoses.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <AlertTriangle className="h-5 w-5 text-yellow-600" />
                      <span>Diagnósticos Diferenciais</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {diagnosisResult.differential_diagnoses.map((diagnosis, index) => (
                        <div key={index} className="border rounded-lg p-3">
                          <div className="flex items-center justify-between">
                            <h5 className="font-medium text-sm">{diagnosis.description}</h5>
                            <Badge variant="secondary">
                              {(diagnosis.confidence * 100).toFixed(1)}%
                            </Badge>
                          </div>
                          <p className="text-xs text-gray-600 mt-1">
                            CID: {diagnosis.icd_code}
                          </p>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Recommendations */}
              {diagnosisResult.recommendations.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle>Recomendações</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ul className="space-y-2">
                      {diagnosisResult.recommendations.map((recommendation, index) => (
                        <li key={index} className="flex items-start space-x-2">
                          <CheckCircle className="h-4 w-4 text-green-600 mt-0.5 flex-shrink-0" />
                          <span className="text-sm">{recommendation}</span>
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>
              )}
            </>
          )}

          {!diagnosisResult && (
            <Card>
              <CardContent className="text-center py-8">
                <Brain className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-500">
                  Preencha os dados clínicos e clique em "Analisar com IA" para obter o diagnóstico
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>

      {/* Compliance Notice */}
      <Card>
        <CardContent className="py-4">
          <div className="flex items-center space-x-2 text-sm text-gray-600">
            <AlertTriangle className="h-4 w-4 text-yellow-600" />
            <span>
              <strong>Aviso:</strong> Este sistema de IA é uma ferramenta de apoio diagnóstico. 
              O diagnóstico final deve sempre ser validado por um profissional médico qualificado.
              Conforme diretrizes da ANVISA, FDA e regulamentações da União Europeia.
            </span>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
