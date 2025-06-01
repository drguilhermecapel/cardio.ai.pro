import React from 'react';
import { render, screen } from '@testing-library/react';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import { describe, it, expect, vi } from 'vitest';
import { store } from '../../store';
import { theme } from '../../theme';
import Layout from '../Layout';

vi.mock('../../hooks/redux', () => ({
  useAppSelector: vi.fn(() => ({
    auth: {
      isAuthenticated: true,
      user: { firstName: 'Test', lastName: 'User', role: 'physician' }
    },
    notification: {
      unreadCount: 0
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

describe('Layout', () => {
  it('renders layout component', () => {
    renderWithProviders(<Layout><div>Test Content</div></Layout>);
    expect(screen.getByText('Test Content')).toBeDefined();
  });

  it('displays navigation elements', () => {
    renderWithProviders(<Layout><div>Test Content</div></Layout>);
    expect(screen.getByText('CardioAI Pro')).toBeDefined();
  });
});
