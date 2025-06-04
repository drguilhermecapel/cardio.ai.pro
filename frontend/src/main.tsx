import React from 'react'
import ReactDOM from 'react-dom/client'
import { Provider } from 'react-redux'
import { BrowserRouter } from 'react-router-dom'
import { ThemeProvider, createTheme } from '@mui/material/styles'
import CssBaseline from '@mui/material/CssBaseline'
import { CacheProvider } from '@emotion/react'
import createCache from '@emotion/cache'
import rtlPlugin from 'stylis-plugin-rtl'
import { prefixer } from 'stylis'

import App from './App'
import { store } from './store'
import { theme } from './theme'
import './index.css'
import './styles/rtl.css'
import './i18n'

const rtlCache = createCache({
  key: 'muirtl',
  stylisPlugins: [prefixer, rtlPlugin],
})

const ltrCache = createCache({
  key: 'muiltr',
  stylisPlugins: [prefixer],
})

export const AppWrapper: React.FC = () => {
  const [isRtl, setIsRtl] = React.useState(false)

  React.useEffect(() => {
    const handleLanguageChange = (lng: string): void => {
      setIsRtl(lng === 'ar' || lng === 'he')
      
      document.dir = lng === 'ar' || lng === 'he' ? 'rtl' : 'ltr'
      document.documentElement.lang = lng
      
      if (lng === 'ar' || lng === 'he') {
        document.body.classList.add('rtl')
        document.body.classList.remove('ltr')
      } else {
        document.body.classList.add('ltr')
        document.body.classList.remove('rtl')
      }
    }

    import('./i18n').then(() => {
      if (typeof window !== 'undefined' && (window as unknown as Record<string, unknown>).i18n) {
        const i18n = (window as unknown as Record<string, unknown>).i18n as { language: string; on: (event: string, callback: (lng: string) => void) => void; off: (event: string, callback: (lng: string) => void) => void }
        handleLanguageChange(i18n.language)
        i18n.on('languageChanged', handleLanguageChange)
        
        return () => {
          i18n.off('languageChanged', handleLanguageChange)
        }
      }
    }).catch(() => {
      const savedLang = localStorage.getItem('i18nextLng') || navigator.language.split('-')[0]
      handleLanguageChange(savedLang)
    })
  }, [])

  const rtlAwareTheme = React.useMemo(() => {
    return createTheme({
      ...theme,
      direction: isRtl ? 'rtl' : 'ltr',
      typography: {
        ...theme.typography,
        body1: {
          ...theme.typography?.body1,
          textAlign: isRtl ? 'right' : 'left',
        },
        body2: {
          ...theme.typography?.body2,
          textAlign: isRtl ? 'right' : 'left',
        },
      },
      components: {
        ...theme.components,
        MuiTextField: {
          styleOverrides: {
            root: {
              '& .MuiInputBase-input': {
                textAlign: isRtl ? 'right' : 'left',
              },
            },
          },
        },
        MuiButton: {
          styleOverrides: {
            root: {
              textAlign: isRtl ? 'right' : 'left',
            },
          },
        },
        MuiTypography: {
          styleOverrides: {
            root: {
              textAlign: isRtl ? 'right' : 'left',
            },
          },
        },
      },
    })
  }, [isRtl])

  const cache = isRtl ? rtlCache : ltrCache

  return (
    <CacheProvider value={cache}>
      <Provider store={store}>
        <BrowserRouter>
          <ThemeProvider theme={rtlAwareTheme}>
            <CssBaseline />
            <div dir={isRtl ? 'rtl' : 'ltr'}>
              <App />
            </div>
          </ThemeProvider>
        </BrowserRouter>
      </Provider>
    </CacheProvider>
  )
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <AppWrapper />
  </React.StrictMode>
)
