import React, { useEffect } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Button,
  LinearProgress,
  Alert,
} from '@mui/material'
import { Assignment, CheckCircle } from '@mui/icons-material'
import { useTranslation } from 'react-i18next'
import { useAppDispatch, useAppSelector } from '../hooks/redux'
import { fetchMyValidations } from '../store/slices/validationSlice'
import { useFormatters } from '../utils/formatters'

const ValidationsPage: React.FC = () => {
  const { t } = useTranslation()
  const formatters = useFormatters()
  const dispatch = useAppDispatch()
  const { validations, isLoading, error } = useAppSelector(state => state.validation)

  useEffect(() => {
    dispatch(fetchMyValidations({}))
  }, [dispatch])

  const getStatusColor = (
    status: string
  ): 'default' | 'primary' | 'secondary' | 'error' | 'info' | 'success' | 'warning' => {
    switch (status) {
      case 'completed':
        return 'success'
      case 'pending':
        return 'warning'
      case 'in_progress':
        return 'info'
      default:
        return 'default'
    }
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        {t('validations.title')}
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {isLoading && <LinearProgress sx={{ mb: 2 }} />}

      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            {t('validations.myAssignments')}
          </Typography>
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>{t('validations.validationId')}</TableCell>
                  <TableCell>{t('validations.analysisId')}</TableCell>
                  <TableCell>{t('validations.status')}</TableCell>
                  <TableCell>{t('validations.approved')}</TableCell>
                  <TableCell>{t('validations.clinicalNotes')}</TableCell>
                  <TableCell>{t('validations.createdDate')}</TableCell>
                  <TableCell>{t('validations.actions')}</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {validations.map(validation => (
                  <TableRow key={validation.id}>
                    <TableCell>{validation.id}</TableCell>
                    <TableCell>{validation.analysisId}</TableCell>
                    <TableCell>
                      <Chip
                        label={validation.status}
                        color={getStatusColor(validation.status)}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      {validation.approved !== undefined ? (
                        <Chip
                          label={validation.approved ? t('common.yes') : t('common.no')}
                          color={validation.approved ? 'success' : 'error'}
                          size="small"
                        />
                      ) : (
                        t('validations.pending')
                      )}
                    </TableCell>
                    <TableCell>
                      {validation.clinicalNotes ? (
                        <Typography variant="body2" noWrap sx={{ maxWidth: 200 }}>
                          {validation.clinicalNotes}
                        </Typography>
                      ) : (
                        t('validations.noNotes')
                      )}
                    </TableCell>
                    <TableCell>{formatters.formatDate(new Date(validation.createdAt))}</TableCell>
                    <TableCell>
                      {validation.status === 'pending' ? (
                        <Button
                          size="small"
                          variant="contained"
                          startIcon={<Assignment />}
                          onClick={() => {}}
                        >
                          {t('validations.review')}
                        </Button>
                      ) : (
                        <Button
                          size="small"
                          variant="outlined"
                          startIcon={<CheckCircle />}
                          onClick={() => {}}
                        >
                          {t('validations.view')}
                        </Button>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>
    </Box>
  )
}

export default ValidationsPage
