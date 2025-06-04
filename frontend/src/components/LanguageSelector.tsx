import React from 'react'
import { Select, MenuItem, FormControl, InputLabel, Box } from '@mui/material'
import { useTranslation } from 'react-i18next'

const LanguageSelector: React.FC = () => {
  const { i18n, t } = useTranslation()
  
  const languages = [
    { code: 'en', name: 'English', flag: '🇺🇸' },
    { code: 'pt', name: 'Português', flag: '🇧🇷' },
    { code: 'es', name: 'Español', flag: '🇪🇸' },
    { code: 'fr', name: 'Français', flag: '🇫🇷' },
    { code: 'de', name: 'Deutsch', flag: '🇩🇪' },
    { code: 'zh', name: '中文', flag: '🇨🇳' },
    { code: 'ar', name: 'العربية', flag: '🇸🇦' }
  ]

  const handleLanguageChange = (languageCode: string) => {
    i18n.changeLanguage(languageCode)
    localStorage.setItem('i18nextLng', languageCode)
  }

  const currentLanguage = languages.find(lang => lang.code === i18n.language) || languages[0]

  return (
    <FormControl size="small" sx={{ minWidth: 120 }}>
      <InputLabel id="language-selector-label">{t('common.language')}</InputLabel>
      <Select
        labelId="language-selector-label"
        value={i18n.language}
        label={t('common.language')}
        onChange={(e) => handleLanguageChange(e.target.value)}
        sx={{
          '& .MuiSelect-select': {
            display: 'flex',
            alignItems: 'center',
            gap: 1
          }
        }}
      >
        {languages.map((lang) => (
          <MenuItem key={lang.code} value={lang.code}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <span>{lang.flag}</span>
              <span>{lang.name}</span>
            </Box>
          </MenuItem>
        ))}
      </Select>
    </FormControl>
  )
}

export default LanguageSelector
