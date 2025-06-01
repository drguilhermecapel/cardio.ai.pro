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
import { useAppDispatch, useAppSelector } from '../hooks/redux'
import { fetchMyValidations } from '../store/slices/validationSlice'

const ValidationsPage: React.FC = () => {
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
        Validations
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
            My Validation Assignments
          </Typography>
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Validation ID</TableCell>
                  <TableCell>Analysis ID</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Approved</TableCell>
                  <TableCell>Clinical Notes</TableCell>
                  <TableCell>Created Date</TableCell>
                  <TableCell>Actions</TableCell>
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
                          label={validation.approved ? 'Yes' : 'No'}
                          color={validation.approved ? 'success' : 'error'}
                          size="small"
                        />
                      ) : (
                        'Pending'
                      )}
                    </TableCell>
                    <TableCell>
                      {validation.clinicalNotes ? (
                        <Typography variant="body2" noWrap sx={{ maxWidth: 200 }}>
                          {validation.clinicalNotes}
                        </Typography>
                      ) : (
                        'No notes'
                      )}
                    </TableCell>
                    <TableCell>{new Date(validation.createdAt).toLocaleDateString()}</TableCell>
                    <TableCell>
                      {validation.status === 'pending' ? (
                        <Button
                          size="small"
                          variant="contained"
                          startIcon={<Assignment />}
                          onClick={() => {}}
                        >
                          Review
                        </Button>
                      ) : (
                        <Button
                          size="small"
                          variant="outlined"
                          startIcon={<CheckCircle />}
                          onClick={() => {}}
                        >
                          View
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
