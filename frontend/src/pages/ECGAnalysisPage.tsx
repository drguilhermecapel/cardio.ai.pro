import React, { useState, useEffect } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Grid,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  LinearProgress,
  Alert,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from '@mui/material'
import { CloudUpload, Visibility } from '@mui/icons-material'
import { useTranslation } from 'react-i18next'
import { useAppDispatch, useAppSelector } from '../hooks/redux'
import { uploadECG, fetchAnalyses, clearError } from '../store/slices/ecgSlice'
import { fetchPatients } from '../store/slices/patientSlice'
import { useFormatters } from '../utils/formatters'

const ECGAnalysisPage: React.FC = () => {
  const { t } = useTranslation()
  const formatters = useFormatters()
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [selectedPatientId, setSelectedPatientId] = useState<number | ''>('')

  const dispatch = useAppDispatch()
  const { analyses, isLoading, error, uploadProgress } = useAppSelector(state => state.ecg)
  const { patients } = useAppSelector(state => state.patient)

  useEffect(() => {
    dispatch(fetchAnalyses({}))
    dispatch(fetchPatients({}))
  }, [dispatch])

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>): void => {
    const file = event.target.files?.[0]
    if (file) {
      setSelectedFile(file)
    }
  }

  const handleUpload = async (): Promise<void> => {
    if (selectedFile && selectedPatientId) {
      dispatch(clearError())
      await dispatch(
        uploadECG({
          patientId: selectedPatientId as number,
          file: selectedFile,
        })
      )
      setUploadDialogOpen(false)
      setSelectedFile(null)
      setSelectedPatientId('')
      dispatch(fetchAnalyses({}))
    }
  }

  const getStatusColor = (
    status: string
  ): 'default' | 'primary' | 'secondary' | 'error' | 'info' | 'success' | 'warning' => {
    switch (status) {
      case 'completed':
        return 'success'
      case 'processing':
        return 'info'
      case 'pending':
        return 'warning'
      case 'failed':
        return 'error'
      default:
        return 'default'
    }
  }

  const getUrgencyColor = (
    urgency: string
  ): 'default' | 'primary' | 'secondary' | 'error' | 'info' | 'success' | 'warning' => {
    switch (urgency) {
      case 'critical':
        return 'error'
      case 'high':
        return 'warning'
      case 'medium':
        return 'info'
      case 'low':
        return 'success'
      default:
        return 'default'
    }
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">{t('ecgAnalysis.title')}</Typography>
        <Button
          variant="contained"
          startIcon={<CloudUpload />}
          onClick={() => setUploadDialogOpen(true)}
        >
          {t('ecgAnalysis.uploadECG')}
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {isLoading && <LinearProgress sx={{ mb: 2 }} />}

      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            {t('ecgAnalysis.title')}
          </Typography>
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>{t('ecgAnalysis.analysisId')}</TableCell>
                  <TableCell>{t('patients.patientId')}</TableCell>
                  <TableCell>{t('validations.status')}</TableCell>
                  <TableCell>{t('ecgAnalysis.diagnosis')}</TableCell>
                  <TableCell>{t('ecgAnalysis.urgency')}</TableCell>
                  <TableCell>{t('ecgAnalysis.confidence')}</TableCell>
                  <TableCell>{t('ecgAnalysis.date')}</TableCell>
                  <TableCell>{t('patients.actions')}</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {analyses.map(analysis => (
                  <TableRow key={analysis.id}>
                    <TableCell>{analysis.analysisId}</TableCell>
                    <TableCell>{analysis.patientId}</TableCell>
                    <TableCell>
                      <Chip
                        label={analysis.status}
                        color={getStatusColor(analysis.status)}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>{analysis.diagnosis || t('validations.pending')}</TableCell>
                    <TableCell>
                      <Chip
                        label={analysis.clinicalUrgency}
                        color={getUrgencyColor(analysis.clinicalUrgency)}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      {analysis.confidence ? `${(analysis.confidence * 100).toFixed(1)}%` : t('common.notAvailable')}
                    </TableCell>
                    <TableCell>{formatters.formatDate(new Date(analysis.createdAt))}</TableCell>
                    <TableCell>
                      <Button size="small" startIcon={<Visibility />} onClick={() => {}}>
                        {t('validations.view')}
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      <Dialog
        open={uploadDialogOpen}
        onClose={() => setUploadDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>{t('ecgAnalysis.uploadECGFile')}</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>{t('ecgAnalysis.selectPatient')}</InputLabel>
                <Select
                  value={selectedPatientId}
                  onChange={e => setSelectedPatientId(e.target.value as number)}
                  label={t('ecgAnalysis.selectPatient')}
                >
                  {patients.map(patient => (
                    <MenuItem key={patient.id} value={patient.id}>
                      {patient.firstName} {patient.lastName} ({patient.patientId})
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <Button variant="outlined" component="label" fullWidth sx={{ height: 56 }}>
                {selectedFile ? selectedFile.name : t('ecgAnalysis.chooseECGFile')}
                <input
                  type="file"
                  hidden
                  accept=".csv,.txt,.xml,.dat"
                  onChange={handleFileSelect}
                />
              </Button>
            </Grid>
            {uploadProgress > 0 && uploadProgress < 100 && (
              <Grid item xs={12}>
                <LinearProgress variant="determinate" value={uploadProgress} />
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  {t('ecgAnalysis.uploading', { progress: uploadProgress })}
                </Typography>
              </Grid>
            )}
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setUploadDialogOpen(false)}>{t('common.cancel')}</Button>
          <Button
            onClick={handleUpload}
            variant="contained"
            disabled={!selectedFile || !selectedPatientId || isLoading}
          >
            {t('ecgAnalysis.upload')}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}

export default ECGAnalysisPage
