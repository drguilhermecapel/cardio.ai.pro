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
import { useAppDispatch, useAppSelector } from '../hooks/redux'
import { uploadECG, fetchAnalyses, clearError } from '../store/slices/ecgSlice'
import { fetchPatients } from '../store/slices/patientSlice'

const ECGAnalysisPage: React.FC = () => {
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
      await dispatch(uploadECG({
        patientId: selectedPatientId as number,
        file: selectedFile,
      }))
      setUploadDialogOpen(false)
      setSelectedFile(null)
      setSelectedPatientId('')
      dispatch(fetchAnalyses({}))
    }
  }

  const getStatusColor = (status: string): 'default' | 'primary' | 'secondary' | 'error' | 'info' | 'success' | 'warning' => {
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

  const getUrgencyColor = (urgency: string): 'default' | 'primary' | 'secondary' | 'error' | 'info' | 'success' | 'warning' => {
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
        <Typography variant="h4">ECG Analysis</Typography>
        <Button
          variant="contained"
          startIcon={<CloudUpload />}
          onClick={() => setUploadDialogOpen(true)}
        >
          Upload ECG
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
            ECG Analyses
          </Typography>
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Analysis ID</TableCell>
                  <TableCell>Patient ID</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Diagnosis</TableCell>
                  <TableCell>Urgency</TableCell>
                  <TableCell>Confidence</TableCell>
                  <TableCell>Date</TableCell>
                  <TableCell>Actions</TableCell>
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
                    <TableCell>{analysis.diagnosis || 'Pending'}</TableCell>
                    <TableCell>
                      <Chip
                        label={analysis.clinicalUrgency}
                        color={getUrgencyColor(analysis.clinicalUrgency)}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      {analysis.confidence ? `${(analysis.confidence * 100).toFixed(1)}%` : 'N/A'}
                    </TableCell>
                    <TableCell>
                      {new Date(analysis.createdAt).toLocaleDateString()}
                    </TableCell>
                    <TableCell>
                      <Button
                        size="small"
                        startIcon={<Visibility />}
                        onClick={() => {
                        }}
                      >
                        View
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      <Dialog open={uploadDialogOpen} onClose={() => setUploadDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Upload ECG File</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Select Patient</InputLabel>
                <Select
                  value={selectedPatientId}
                  onChange={e => setSelectedPatientId(e.target.value as number)}
                  label="Select Patient"
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
              <Button
                variant="outlined"
                component="label"
                fullWidth
                sx={{ height: 56 }}
              >
                {selectedFile ? selectedFile.name : 'Choose ECG File'}
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
                  Uploading... {uploadProgress}%
                </Typography>
              </Grid>
            )}
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setUploadDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleUpload}
            variant="contained"
            disabled={!selectedFile || !selectedPatientId || isLoading}
          >
            Upload
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}

export default ECGAnalysisPage
