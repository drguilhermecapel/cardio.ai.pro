import React from 'react'
import { render, screen } from '@testing-library/react'
import { Provider } from 'react-redux'
import { BrowserRouter } from 'react-router-dom'
import { ThemeProvider } from '@mui/material/styles'
import { I18nextProvider } from 'react-i18next'
import { describe, it, expect, vi, beforeAll } from 'vitest'
import { theme } from '../../theme'
import DashboardPage from '../DashboardPage'
import { initI18nForTesting } from '../../test-utils/i18n-test-setup'

vi.mock('../../hooks/redux', () => ({
  useAppDispatch: vi.fn(() => vi.fn()),
  useAppSelector: vi.fn(selector => {
    const mockState = {
      ecg: {
        analyses: [],
        isLoading: false,
        error: null,
        currentAnalysis: null,
        uploadProgress: 0,
      },
      notification: {
        unreadCount: 0,
        notifications: [],
        isLoading: false,
        error: null,
      },
      auth: {
        isAuthenticated: true,
        user: {
          id: 1,
          username: 'testuser',
          email: 'test@example.com',
          firstName: 'Test',
          lastName: 'User',
          role: 'physician',
          isActive: true,
        },
        token: 'mock-token',
        refreshToken: 'mock-refresh-token',
        isLoading: false,
        error: null,
      },
    }
    return selector(mockState)
  }),
}))

import { store } from '../../store'

const renderWithProviders = (component: React.ReactElement): ReturnType<typeof render> => {
  const i18n = initI18nForTesting()
  return render(
    <Provider store={store}>
      <BrowserRouter>
        <ThemeProvider theme={theme}>
          <I18nextProvider i18n={i18n}>{component}</I18nextProvider>
        </ThemeProvider>
      </BrowserRouter>
    </Provider>
  )
}

describe('DashboardPage', () => {
  beforeAll(() => {
    initI18nForTesting()
  })

  it('renders dashboard page', () => {
    renderWithProviders(<DashboardPage />)
    expect(screen.getByText('Dashboard')).toBeDefined()
  })

  it('displays dashboard metrics', () => {
    renderWithProviders(<DashboardPage />)
    expect(screen.getByText('Total Analyses')).toBeDefined()
    expect(screen.getByText('Pending')).toBeDefined()
  })
})
