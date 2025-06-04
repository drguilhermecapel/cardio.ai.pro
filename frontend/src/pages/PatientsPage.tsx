import React, { useEffect, useState } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  LinearProgress,
  SelectChangeEvent,
} from '@mui/material'
import { Add, Edit } from '@mui/icons-material'
import { useTranslation } from 'react-i18next'
import { useAppDispatch, useAppSelector } from '../hooks/redux'
import { fetchPatients, createPatient, clearError } from '../store/slices/patientSlice'
import { useFormatters } from '../utils/formatters'

interface PatientFormData {
  patientId: string
  firstName: string
  lastName: string
  dateOfBirth: string
  gender: string
  phone: string
  email: string
}

const PatientsPage: React.FC = () => {
  const { t } = useTranslation()
  const formatters = useFormatters()
  const [createDialogOpen, setCreateDialogOpen] = useState(false)
  const [formData, setFormData] = useState<PatientFormData>({
    patientId: '',
    firstName: '',
    lastName: '',
    dateOfBirth: '',
    gender: '',
    phone: '',
    email: '',
  })

  const dispatch = useAppDispatch()
  const { patients, isLoading, error } = useAppSelector(state => state.patient)

  useEffect(() => {
    dispatch(fetchPatients({}))
  }, [dispatch])

  const handleInputChange =
    (field: keyof PatientFormData) =>
    (event: React.ChangeEvent<HTMLInputElement>): void => {
      setFormData(prev => ({
        ...prev,
        [field]: event.target.value,
      }))
    }

  const handleGenderChange = (event: SelectChangeEvent<string>): void => {
    setFormData(prev => ({
      ...prev,
      gender: event.target.value as string,
    }))
  }

  const handleCreatePatient = async (): Promise<void> => {
    dispatch(clearError())
    await dispatch(createPatient(formData))
    setCreateDialogOpen(false)
    setFormData({
      patientId: '',
      firstName: '',
      lastName: '',
      dateOfBirth: '',
      gender: '',
      phone: '',
      email: '',
    })
    dispatch(fetchPatients({}))
  }

  const isFormValid =
    formData.patientId &&
    formData.firstName &&
    formData.lastName &&
    formData.dateOfBirth &&
    formData.gender

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">{t('patients.title')}</Typography>
        <Button variant="contained" startIcon={<Add />} onClick={() => setCreateDialogOpen(true)}>
          {t('patients.addPatient')}
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
            {t('patients.patientList')}
          </Typography>
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>{t('patients.patientId')}</TableCell>
                  <TableCell>{t('patients.name')}</TableCell>
                  <TableCell>{t('patients.dateOfBirth')}</TableCell>
                  <TableCell>{t('patients.gender')}</TableCell>
                  <TableCell>{t('patients.phone')}</TableCell>
                  <TableCell>{t('patients.email')}</TableCell>
                  <TableCell>{t('patients.actions')}</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {patients.map(patient => (
                  <TableRow key={patient.id}>
                    <TableCell>{patient.patientId}</TableCell>
                    <TableCell>
                      {patient.firstName} {patient.lastName}
                    </TableCell>
                    <TableCell>{formatters.formatDate(new Date(patient.dateOfBirth))}</TableCell>
                    <TableCell>{patient.gender}</TableCell>
                    <TableCell>{patient.phone || t('common.notAvailable')}</TableCell>
                    <TableCell>{patient.email || t('common.notAvailable')}</TableCell>
                    <TableCell>
                      <Button size="small" startIcon={<Edit />} onClick={() => {}}>
                        {t('common.edit')}
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
        open={createDialogOpen}
        onClose={() => setCreateDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>{t('patients.addNewPatient')}</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label={t('patients.patientId')}
                value={formData.patientId}
                onChange={handleInputChange('patientId')}
                required
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth required>
                <InputLabel>{t('patients.gender')}</InputLabel>
                <Select value={formData.gender} onChange={handleGenderChange} label={t('patients.gender')}>
                  <MenuItem value="male">{t('patients.male')}</MenuItem>
                  <MenuItem value="female">{t('patients.female')}</MenuItem>
                  <MenuItem value="other">{t('patients.other')}</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label={t('patients.firstName')}
                value={formData.firstName}
                onChange={handleInputChange('firstName')}
                required
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label={t('patients.lastName')}
                value={formData.lastName}
                onChange={handleInputChange('lastName')}
                required
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label={t('patients.dateOfBirth')}
                type="date"
                value={formData.dateOfBirth}
                onChange={handleInputChange('dateOfBirth')}
                InputLabelProps={{ shrink: true }}
                required
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label={t('patients.phone')}
                value={formData.phone}
                onChange={handleInputChange('phone')}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label={t('patients.email')}
                type="email"
                value={formData.email}
                onChange={handleInputChange('email')}
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialogOpen(false)}>{t('common.cancel')}</Button>
          <Button
            onClick={handleCreatePatient}
            variant="contained"
            disabled={!isFormValid || isLoading}
          >
            {t('patients.createPatient')}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}

export default PatientsPage
