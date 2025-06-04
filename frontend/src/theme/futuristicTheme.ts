export const futuristicTheme = {
  colors: {
    background: {
      primary: '#0a0f1c', // Deep navy background
      secondary: '#1a1f2e',
      tertiary: '#2a2f3e'
    },
    data: {
      primary: '#00bfff', // Electric blue primary data
      secondary: '#00ff7f', // Emerald green for normal readings
      warning: '#ffa500', // Amber warnings
      critical: '#ff0000', // Red critical alerts
      holographic: '#00ffff', // Holographic cyan interfaces
      text: '#ffffff' // White clinical text
    },
    neural: {
      connections: '#00ff7f',
      nodes: '#00bfff',
      pathways: '#ffa500'
    },
    ui: {
      glass: 'rgba(255, 255, 255, 0.1)',
      glow: 'rgba(0, 191, 255, 0.3)',
      border: 'rgba(0, 255, 255, 0.2)'
    }
  },
  effects: {
    glow: {
      primary: '0 0 20px rgba(0, 191, 255, 0.5)',
      secondary: '0 0 15px rgba(0, 255, 127, 0.4)',
      critical: '0 0 25px rgba(255, 0, 0, 0.6)'
    },
    blur: {
      glass: 'blur(10px)',
      background: 'blur(5px)'
    }
  },
  typography: {
    fontFamily: {
      primary: '"Orbitron", "Roboto", sans-serif',
      secondary: '"Exo 2", "Arial", sans-serif',
      mono: '"Fira Code", "Courier New", monospace'
    },
    sizes: {
      xs: '0.75rem',
      sm: '0.875rem',
      base: '1rem',
      lg: '1.125rem',
      xl: '1.25rem',
      '2xl': '1.5rem',
      '3xl': '1.875rem',
      '4xl': '2.25rem'
    }
  },
  spacing: {
    xs: '0.25rem',
    sm: '0.5rem',
    md: '1rem',
    lg: '1.5rem',
    xl: '2rem',
    '2xl': '3rem',
    '3xl': '4rem'
  },
  borderRadius: {
    sm: '0.25rem',
    md: '0.5rem',
    lg: '0.75rem',
    xl: '1rem',
    full: '9999px'
  },
  zIndex: {
    background: 0,
    base: 1,
    overlay: 10,
    modal: 20,
    tooltip: 30,
    floating: 40
  }
}

export type FuturisticTheme = typeof futuristicTheme
