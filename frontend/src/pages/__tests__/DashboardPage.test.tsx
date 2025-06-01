import React from 'react';
import { render, screen } from '@testing-library/react';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import { describe, it, expect, vi } from 'vitest';
import { store } from '../../store';
import { theme } from '../../theme';
import DashboardPage from '../DashboardPage';

vi.mock('../../hooks/redux', () => ({
  useAppSelector: vi.fn(() => ({
    auth: {
      isAuthenticated: true,
      user: { firstName: 'Test', lastName: 'User', role: 'physician' }
    },
    dashboard: {
      metrics: {
        analysesToday: 15,
        pendingValidations: 3,
        criticalAlerts: 1,
        systemHealth: 'healthy'
      }
    }
  })),
  useAppDispatch: vi.fn(() => vi.fn())
}));

const renderWithProviders = (component: React.ReactElement) => {
  return render(
    <Provider store={store}>
      <BrowserRouter>
        <ThemeProvider theme={theme}>
          {component}
        </ThemeProvider>
      </BrowserRouter>
    </Provider>
  );
};

describe('DashboardPage', () => {
  it('renders dashboard page', () => {
    renderWithProviders(<DashboardPage />);
    expect(screen.getByText('Dashboard')).toBeDefined();
  });

  it('displays dashboard metrics', () => {
    renderWithProviders(<DashboardPage />);
    expect(screen.getByText('Análises Hoje')).toBeDefined();
    expect(screen.getByText('Validações Pendentes')).toBeDefined();
  });
});
