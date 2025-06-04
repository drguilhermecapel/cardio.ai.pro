import React from 'react'
import { render, screen } from '@testing-library/react'
import { Provider } from 'react-redux'
import { BrowserRouter } from 'react-router-dom'
import { ThemeProvider } from '@mui/material/styles'
import { I18nextProvider } from 'react-i18next'
import { describe, it, expect, vi, beforeAll } from 'vitest'
import { store } from '../../store'
import { theme } from '../../theme'
import Layout from '../Layout'
import { initI18nForTesting } from '../../test-utils/i18n-test-setup'

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
  const i18n = initI18nForTesting()
  return render(
    <Provider store={store}>
      <BrowserRouter>
        <ThemeProvider theme={theme}>
          <I18nextProvider i18n={i18n}>
            {component}
          </I18nextProvider>
        </ThemeProvider>
      </BrowserRouter>
    </Provider>
  )
}

describe('Layout', () => {
  beforeAll(() => {
    initI18nForTesting()
  })

  it('renders layout component', () => {
    renderWithProviders(
      <Layout>
        <div>Test Content</div>
      </Layout>
    )
    expect(screen.getByText('Test Content')).toBeDefined()
  })

  it('displays navigation elements', () => {
    renderWithProviders(
      <Layout>
        <div>Test Content</div>
      </Layout>
    )
    expect(screen.getAllByText('CardioAI Pro')[0]).toBeDefined()
  })
})
