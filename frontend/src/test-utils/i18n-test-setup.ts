import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'

const testTranslations = {
  en: {
    translation: {
      'app.title': 'CardioAI Pro',
      'login.title': 'CardioAI Pro',
      'login.subtitle': 'AI-powered ECG analysis system for healthcare professionals',
      'login.username': 'Username',
      'login.password': 'Password',
      'login.signIn': 'Sign In',
      'login.demoCredentials': 'For demo purposes, use: admin / admin123',
      'dashboard.title': 'Dashboard',
      'dashboard.totalAnalyses': 'Total Analyses',
      'dashboard.pending': 'Pending',
      'dashboard.critical': 'Critical',
      'dashboard.allTime': 'All time',
      'dashboard.awaitingProcessing': 'Awaiting processing',
      'dashboard.requireImmediateAttention': 'Require immediate attention',
      'dashboard.unreadMessages': 'Unread messages',
      'dashboard.recentECGAnalyses': 'Recent ECG Analyses',
      'dashboard.systemStatus': 'System Status',
      'dashboard.processingQueue': 'Processing Queue',
      'dashboard.completedOf': '0 of 0 completed',
      'dashboard.aiModelStatus': 'AI Model Status',
      'dashboard.online': 'Online',
      'notifications.title': 'Notifications',
      'nav.dashboard': 'Dashboard',
      'nav.patients': 'Patients',
      'nav.ecgAnalysis': 'ECG Analysis',
      'nav.notifications': 'Notifications',
      'nav.validations': 'Validations',
      'nav.logout': 'Logout'
    }
  }
}

export const initI18nForTesting = (): typeof i18n => {
  if (!i18n.isInitialized) {
    i18n
      .use(initReactI18next)
      .init({
        lng: 'en',
        fallbackLng: 'en',
        debug: false,
        interpolation: {
          escapeValue: false,
        },
        resources: testTranslations,
        react: {
          useSuspense: false,
        },
      })
  }
  return i18n
}

export default initI18nForTesting
