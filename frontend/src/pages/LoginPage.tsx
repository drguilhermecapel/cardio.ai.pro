import React, { useState } from 'react'
import {
  Box,
  Card,
  CardContent,
  TextField,
  Button,
  Typography,
  Alert,
  CircularProgress,
  Container,
} from '@mui/material'
import { MonitorHeart } from '@mui/icons-material'
import { useTranslation } from 'react-i18next'
import { useAppDispatch, useAppSelector } from '../hooks/redux'
import { login, clearError } from '../store/slices/authSlice'
import LanguageSelector from '../components/LanguageSelector'

const LoginPage: React.FC = () => {
  const { t } = useTranslation()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const dispatch = useAppDispatch()
  const { isLoading, error } = useAppSelector(state => state.auth)

  const handleSubmit = async (e: React.FormEvent): Promise<void> => {
    e.preventDefault()
    if (username && password) {
      dispatch(clearError())
      dispatch(login({ username, password }))
    }
  }

  return (
    <Container component="main" maxWidth="sm">
      <Box
        sx={{
          position: 'absolute',
          top: 16,
          right: 16,
          zIndex: 1000,
        }}
      >
        <LanguageSelector />
      </Box>
      <Box
        sx={{
          marginTop: 8,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
        }}
      >
        <Card sx={{ width: '100%', maxWidth: 400 }}>
          <CardContent sx={{ p: 4 }}>
            <Box
              sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                mb: 3,
              }}
            >
              <MonitorHeart sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
              <Typography component="h1" variant="h4" gutterBottom>
                {t('login.title')}
              </Typography>
              <Typography variant="body2" color="text.secondary" align="center">
                {t('login.subtitle')}
              </Typography>
            </Box>

            {error && (
              <Alert severity="error" sx={{ mb: 2 }}>
                {error}
              </Alert>
            )}

            <Box component="form" onSubmit={handleSubmit} sx={{ mt: 1 }}>
              <TextField
                margin="normal"
                required
                fullWidth
                id="username"
                label={t('login.username')}
                name="username"
                autoComplete="username"
                autoFocus
                value={username}
                onChange={e => setUsername(e.target.value)}
                disabled={isLoading}
              />
              <TextField
                margin="normal"
                required
                fullWidth
                name="password"
                label={t('login.password')}
                type="password"
                id="password"
                autoComplete="current-password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                disabled={isLoading}
              />
              <Button
                type="submit"
                fullWidth
                variant="contained"
                sx={{ mt: 3, mb: 2 }}
                disabled={isLoading || !username || !password}
              >
                {isLoading ? <CircularProgress size={24} /> : t('login.signIn')}
              </Button>
            </Box>

            <Typography variant="body2" color="text.secondary" align="center" sx={{ mt: 2 }}>
              {t('login.demoCredentials')}
            </Typography>
          </CardContent>
        </Card>
      </Box>
    </Container>
  )
}

export default LoginPage
