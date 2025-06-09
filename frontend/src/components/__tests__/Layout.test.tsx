import React from 'react'
import { render, screen } from '@testing-library/react'
import { Provider } from 'react-redux'
import { BrowserRouter } from 'react-router-dom'
import { ThemeProvider } from '@mui/material/styles'
import { describe, it, expect, vi } from 'vitest'
import { store } from '../../store'
import { theme } from '../../theme'
import { AuthProvider } from '../../contexts/AuthContext'
import Layout from '../Layout'

vi.mock('../../hooks/useAuth', () => ({
  useAuth: vi.fn(() => ({
    user: { username: 'Test User' },
    isAuthenticated: true,
    login: vi.fn(),
    logout: vi.fn(),
  })),
}))

vi.mock('../../hooks/redux', () => ({
  useAppSelector: vi.fn(() => ({
    auth: {
      isAuthenticated: true,
      user: { firstName: 'Test', lastName: 'User', role: 'physician' },
    },
    ecg: {
      analyses: [],
      isLoading: false,
    },
    notification: {
      unreadCount: 0,
    },
  })),
  useAppDispatch: vi.fn(() => vi.fn()),
}))

const renderWithProviders = (component: React.ReactElement): ReturnType<typeof render> => {
  return render(
    <Provider store={store}>
      <BrowserRouter>
        <ThemeProvider theme={theme}>
          <AuthProvider>{component}</AuthProvider>
        </ThemeProvider>
      </BrowserRouter>
    </Provider>
  )
}

describe('Layout', () => {
  it('renders layout component with navigation', () => {
    renderWithProviders(<Layout />)
    expect(screen.getByRole('main')).toBeDefined()
  })

  it('displays navigation elements', () => {
    renderWithProviders(<Layout />)
    expect(screen.getByText('Dashboard')).toBeDefined()
    expect(screen.getByText('CardioAI Pro')).toBeDefined()
  })
})
