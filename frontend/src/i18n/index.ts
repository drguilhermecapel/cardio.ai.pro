import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'
import LanguageDetector from 'i18next-browser-languagedetector'
import Backend from 'i18next-http-backend'

i18n
  .use(Backend)
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    fallbackLng: 'en',
    debug: true,
    ns: ['common'],
    defaultNS: 'common',

    interpolation: {
      escapeValue: false,
    },

    backend: {
      loadPath: '/locales/{{lng}}/{{ns}}.json',
    },

    detection: {
      order: ['localStorage', 'navigator', 'htmlTag'],
      caches: ['localStorage'],
      lookupLocalStorage: 'i18nextLng',
      convertDetectedLanguage: (lng: string) => {
        const supportedLanguages = ['en', 'pt', 'es', 'fr', 'de', 'zh', 'ar']
        const langCode = lng.split('-')[0] // Convert 'en-US' to 'en'
        return supportedLanguages.includes(langCode) ? langCode : 'en'
      },
    },

    react: {
      useSuspense: false,
    },
  })

export default i18n
