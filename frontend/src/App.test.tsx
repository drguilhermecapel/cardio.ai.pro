import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { Provider } from 'react-redux'
import { BrowserRouter } from 'react-router-dom'
import { ThemeProvider, createTheme } from '@mui/material/styles'
import { configureStore } from '@reduxjs/toolkit'
import App from './App'

// Mock do store Redux
const mockStore = configureStore({
  reducer: {
    auth: (state = { isAuthenticated: false, user: null }) => state,
  },
})

const theme = createTheme()

const AllTheProviders = ({ children }: { children: React.ReactNode }) => {
  return (
    <Provider store={mockStore}>
      <BrowserRouter>
        <ThemeProvider theme={theme}>
          {children}
        </ThemeProvider>
      </BrowserRouter>
    </Provider>
  )
}

describe('App', () => {
  it('should render without crashing', () => {
    render(
      <AllTheProviders>
        <App />
      </AllTheProviders>
    )
    
    // Verifica se algum elemento básico está presente
    expect(document.body).toBeTruthy()
  })
})
