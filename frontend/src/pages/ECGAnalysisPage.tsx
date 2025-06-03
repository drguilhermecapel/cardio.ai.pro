import React, { useState, useEffect, useRef } from 'react'
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
  IconButton,
  Tooltip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
} from '@mui/material'
import { 
  CloudUpload, 
  Visibility, 
  CameraAlt, 
  Image as ImageIcon,
  CheckCircle,
  Warning,
  Error as ErrorIcon,
  ExpandMore,
  GridOn,
  Timeline,
  Assessment
} from '@mui/icons-material'
import { useAppDispatch, useAppSelector } from '../hooks/redux'
import { uploadECG, fetchAnalyses, clearError } from '../store/slices/ecgSlice'
import { fetchPatients } from '../store/slices/patientSlice'

interface DocumentScanningMetadata {
  scanner_confidence: number
  document_detected: boolean
  processing_method: string
  grid_detected: boolean
  leads_detected: number
  original_size: [number, number]
  processed_size: [number, number]
  error?: string
}

interface ContextualResponse {
  message: string
  explanation?: string
  tips?: string[]
  visual_guide?: string
  educational_content?: {
    title?: string;
    description?: string;
    key_features?: string[];
    examples?: string[];
  }
  helpful_actions?: string[]
}

interface StructuredError {
  error_code: string
  message: string
  contextual_response?: ContextualResponse
}

type ErrorType = string | StructuredError | null

const ECGAnalysisPage: React.FC = () => {
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [selectedPatientId, setSelectedPatientId] = useState<number | ''>('')
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [cameraMode, setCameraMode] = useState(false)
  const [scanningProgress, setScanningProgress] = useState(0)
  const [scanningMetadata, setScanningMetadata] = useState<DocumentScanningMetadata | null>(null)
  const [isScanning, setIsScanning] = useState(false)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const dispatch = useAppDispatch()
  const { analyses, isLoading, error, uploadProgress } = useAppSelector(state => state.ecg)
  const typedError = error as ErrorType
  const { patients } = useAppSelector(state => state.patient)
  


  useEffect(() => {
    dispatch(fetchAnalyses({}))
    dispatch(fetchPatients({}))
  }, [dispatch])

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>): void => {
    const file = event.target.files?.[0]
    if (file) {
      setSelectedFile(file)
      
      if (file.type.startsWith('image/')) {
        const reader = new FileReader()
        reader.onload = (e): void => {
          setImagePreview(e.target?.result as string)
        }
        reader.readAsDataURL(file)
      } else {
        setImagePreview(null)
      }
      
      setScanningMetadata(null)
    }
  }

  const startCamera = async (): Promise<void> => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          facingMode: 'environment', // Use back camera on mobile
          width: { ideal: 1920 },
          height: { ideal: 1080 }
        } 
      })
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        setCameraMode(true)
      }
    } catch (error) {
      console.error('Error accessing camera:', error)
      alert('Unable to access camera. Please check permissions.')
    }
  }

  const stopCamera = (): void => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream
      stream.getTracks().forEach(track => track.stop())
      videoRef.current.srcObject = null
    }
    setCameraMode(false)
  }

  const capturePhoto = (): void => {
    if (videoRef.current && canvasRef.current) {
      const canvas = canvasRef.current
      const video = videoRef.current
      
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      
      const ctx = canvas.getContext('2d')
      if (ctx) {
        ctx.drawImage(video, 0, 0)
        
        canvas.toBlob((blob) => {
          if (blob) {
            const file = new File([blob], `ecg-capture-${Date.now()}.jpg`, { type: 'image/jpeg' })
            setSelectedFile(file)
            setImagePreview(canvas.toDataURL())
            stopCamera()
          }
        }, 'image/jpeg', 0.9)
      }
    }
  }

  const handleUpload = async (): Promise<void> => {
    if (selectedFile && selectedPatientId) {
      dispatch(clearError())
      
      if (selectedFile.type.startsWith('image/')) {
        setIsScanning(true)
        setScanningProgress(0)
        
        const progressInterval = setInterval(() => {
          setScanningProgress(prev => {
            if (prev >= 90) {
              clearInterval(progressInterval)
              return 90
            }
            return prev + 10
          })
        }, 200)
      }
      
      const result = await dispatch(
        uploadECG({
          patientId: selectedPatientId as number,
          file: selectedFile,
        })
      )
      
      if (result.payload?.document_scanning_metadata) {
        setScanningMetadata(result.payload.document_scanning_metadata)
        setScanningProgress(100)
        
        setTimeout(() => {
          setIsScanning(false)
          setScanningProgress(0)
        }, 2000)
      } else {
        setIsScanning(false)
        setScanningProgress(0)
      }
      
      setUploadDialogOpen(false)
      setSelectedFile(null)
      setSelectedPatientId('')
      setImagePreview(null)
      setScanningMetadata(null)
      dispatch(fetchAnalyses({}))
    }
  }

  const resetUploadDialog = (): void => {
    setUploadDialogOpen(false)
    setSelectedFile(null)
    setSelectedPatientId('')
    setImagePreview(null)
    setScanningMetadata(null)
    setIsScanning(false)
    setScanningProgress(0)
    stopCamera()
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
        <Typography variant="h4">ECG Analysis</Typography>
        <Button
          variant="contained"
          startIcon={<CloudUpload />}
          onClick={() => setUploadDialogOpen(true)}
        >
          Upload ECG
        </Button>
      </Box>

      {typedError && (
        <Alert 
          severity={typeof typedError === 'object' && typedError !== null && 'error_code' in typedError && typedError.error_code === 'NON_ECG_IMAGE_DETECTED' ? 'info' : 'error'} 
          sx={{ mb: 2 }}
        >
          {typeof typedError === 'object' && typedError !== null && 'error_code' in typedError && typedError.error_code === 'NON_ECG_IMAGE_DETECTED' ? (
            <Box>
              <Typography variant="h6" gutterBottom>
                {typeof typedError === 'object' && typedError !== null && 'contextual_response' in typedError && typedError.contextual_response?.message || 'Non-ECG image detected'}
              </Typography>
              {typeof typedError === 'object' && typedError !== null && 'contextual_response' in typedError && typedError.contextual_response?.explanation && (
                <Typography variant="body2" sx={{ mb: 2 }}>
                  {typeof typedError === 'object' && typedError !== null && 'contextual_response' in typedError && typedError.contextual_response.explanation}
                </Typography>
              )}
              {typeof typedError === 'object' && typedError !== null && 'contextual_response' in typedError && typedError.contextual_response?.tips && typedError.contextual_response.tips.length > 0 && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    💡 Tips:
                  </Typography>
                  <List dense>
                    {typeof typedError === 'object' && typedError !== null && 'contextual_response' in typedError && typedError.contextual_response.tips.map((tip: string, index: number) => (
                      <ListItem key={index} sx={{ py: 0 }}>
                        <ListItemText primary={`• ${tip}`} />
                      </ListItem>
                    ))}
                  </List>
                </Box>
              )}
              {typeof typedError === 'object' && typedError !== null && 'contextual_response' in typedError && typedError.contextual_response?.visual_guide && (
                <Box sx={{ mb: 2, p: 2, bgcolor: 'background.paper', borderRadius: 1 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    📋 What an ECG looks like:
                  </Typography>
                  <Typography variant="body2">
                    {typeof typedError === 'object' && typedError !== null && 'contextual_response' in typedError && typedError.contextual_response.visual_guide}
                  </Typography>
                </Box>
              )}
              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mt: 2 }}>
                <Button 
                  variant="contained" 
                  size="small"
                  onClick={() => {
                    dispatch(clearError())
                    setUploadDialogOpen(true)
                  }}
                >
                  📷 Try Again
                </Button>
                {typeof typedError === 'object' && typedError !== null && 'contextual_response' in typedError && typedError.contextual_response?.educational_content && (
                  <Button 
                    variant="outlined" 
                    size="small"
                    onClick={() => {
                      console.log('Educational content:', typeof typedError === 'object' && typedError !== null && 'contextual_response' in typedError && typedError.contextual_response?.educational_content)
                    }}
                  >
                    📚 Learn More
                  </Button>
                )}
              </Box>
            </Box>
          ) : (
            typeof typedError === 'string' ? typedError : (typeof typedError === 'object' && typedError !== null && 'message' in typedError && typedError.message) || 'An error occurred'
          )}
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
                    <TableCell>{new Date(analysis.createdAt).toLocaleDateString()}</TableCell>
                    <TableCell>
                      <Button size="small" startIcon={<Visibility />} onClick={() => {}}>
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

      <Dialog
        open={uploadDialogOpen}
        onClose={() => setUploadDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
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
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Button 
                  variant="outlined" 
                  component="label" 
                  fullWidth 
                  sx={{ height: 56 }}
                  startIcon={<ImageIcon />}
                >
                  {selectedFile ? selectedFile.name : 'Choose ECG File'}
                  <input
                    ref={fileInputRef}
                    type="file"
                    hidden
                    accept=".csv,.txt,.xml,.dat,.jpg,.jpeg,.png"
                    onChange={handleFileSelect}
                  />
                </Button>
                <Tooltip title="Capture with Camera">
                  <IconButton 
                    onClick={startCamera}
                    sx={{ 
                      border: 1, 
                      borderColor: 'divider',
                      width: 56,
                      height: 56
                    }}
                  >
                    <CameraAlt />
                  </IconButton>
                </Tooltip>
              </Box>
            </Grid>

            {/* Camera Mode */}
            {cameraMode && (
              <Grid item xs={12}>
                <Card sx={{ p: 2 }}>
                  <Box sx={{ position: 'relative', textAlign: 'center' }}>
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      style={{ 
                        width: '100%', 
                        maxHeight: '300px',
                        borderRadius: '8px'
                      }}
                    />
                    <Box sx={{ mt: 2, display: 'flex', gap: 1, justifyContent: 'center' }}>
                      <Button variant="contained" onClick={capturePhoto}>
                        Capture ECG
                      </Button>
                      <Button variant="outlined" onClick={stopCamera}>
                        Cancel
                      </Button>
                    </Box>
                  </Box>
                </Card>
                <canvas ref={canvasRef} style={{ display: 'none' }} />
              </Grid>
            )}

            {/* Image Preview */}
            {imagePreview && !cameraMode && (
              <Grid item xs={12}>
                <Card sx={{ p: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Image Preview
                  </Typography>
                  <Box sx={{ textAlign: 'center' }}>
                    <img
                      src={imagePreview}
                      alt="ECG Preview"
                      style={{
                        maxWidth: '100%',
                        maxHeight: '300px',
                        borderRadius: '8px',
                        border: '2px dashed #ccc'
                      }}
                    />
                  </Box>
                </Card>
              </Grid>
            )}

            {/* Document Scanning Progress */}
            {isScanning && (
              <Grid item xs={12}>
                <Card sx={{ p: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Scanning ECG Document...
                  </Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={scanningProgress} 
                    sx={{ mb: 1 }}
                  />
                  <Typography variant="body2" color="text.secondary">
                    {scanningProgress < 30 && 'Detecting document edges...'}
                    {scanningProgress >= 30 && scanningProgress < 60 && 'Applying perspective correction...'}
                    {scanningProgress >= 60 && scanningProgress < 90 && 'Enhancing image quality...'}
                    {scanningProgress >= 90 && 'Validating ECG document...'}
                  </Typography>
                </Card>
              </Grid>
            )}

            {/* Scanning Results */}
            {scanningMetadata && (
              <Grid item xs={12}>
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMore />}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      {scanningMetadata.document_detected ? (
                        <CheckCircle color="success" />
                      ) : scanningMetadata.scanner_confidence > 0.3 ? (
                        <Warning color="warning" />
                      ) : (
                        <ErrorIcon color="error" />
                      )}
                      <Typography variant="subtitle2">
                        Document Scanning Results 
                        ({(scanningMetadata.scanner_confidence * 100).toFixed(1)}% confidence)
                      </Typography>
                    </Box>
                  </AccordionSummary>
                  <AccordionDetails>
                    <List dense>
                      <ListItem>
                        <ListItemIcon>
                          {scanningMetadata.document_detected ? (
                            <CheckCircle color="success" />
                          ) : (
                            <ErrorIcon color="error" />
                          )}
                        </ListItemIcon>
                        <ListItemText 
                          primary="ECG Document Detected"
                          secondary={scanningMetadata.document_detected ? 'Yes' : 'No'}
                        />
                      </ListItem>
                      
                      <ListItem>
                        <ListItemIcon>
                          {scanningMetadata.grid_detected ? (
                            <GridOn color="success" />
                          ) : (
                            <GridOn color="disabled" />
                          )}
                        </ListItemIcon>
                        <ListItemText 
                          primary="Grid Pattern"
                          secondary={scanningMetadata.grid_detected ? 'Detected' : 'Not detected'}
                        />
                      </ListItem>
                      
                      <ListItem>
                        <ListItemIcon>
                          <Timeline color={scanningMetadata.leads_detected > 0 ? 'success' : 'disabled'} />
                        </ListItemIcon>
                        <ListItemText 
                          primary="ECG Leads"
                          secondary={`${scanningMetadata.leads_detected} leads detected`}
                        />
                      </ListItem>
                      
                      <ListItem>
                        <ListItemIcon>
                          <Assessment />
                        </ListItemIcon>
                        <ListItemText 
                          primary="Processing Method"
                          secondary={scanningMetadata.processing_method}
                        />
                      </ListItem>
                      
                      {scanningMetadata.error && (
                        <ListItem>
                          <ListItemIcon>
                            <ErrorIcon color="error" />
                          </ListItemIcon>
                          <ListItemText 
                            primary="Error"
                            secondary={scanningMetadata.error}
                          />
                        </ListItem>
                      )}
                    </List>
                  </AccordionDetails>
                </Accordion>
              </Grid>
            )}
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
          <Button onClick={resetUploadDialog}>Cancel</Button>
          <Button
            onClick={handleUpload}
            variant="contained"
            disabled={!selectedFile || !selectedPatientId || isLoading || isScanning}
          >
            {isScanning ? 'Processing...' : 'Upload'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}

export default ECGAnalysisPage
