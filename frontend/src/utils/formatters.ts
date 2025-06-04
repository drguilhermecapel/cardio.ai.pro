import { useTranslation } from 'react-i18next'

export const useFormatters = () => {
  const { i18n } = useTranslation()
  
  const formatDate = (date: Date, options?: Intl.DateTimeFormatOptions) => {
    const defaultOptions: Intl.DateTimeFormatOptions = {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    }
    return new Intl.DateTimeFormat(i18n.language, { ...defaultOptions, ...options }).format(date)
  }
  
  const formatTime = (date: Date, options?: Intl.DateTimeFormatOptions) => {
    const defaultOptions: Intl.DateTimeFormatOptions = {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    }
    return new Intl.DateTimeFormat(i18n.language, { ...defaultOptions, ...options }).format(date)
  }
  
  const formatDateTime = (date: Date, options?: Intl.DateTimeFormatOptions) => {
    const defaultOptions: Intl.DateTimeFormatOptions = {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    }
    return new Intl.DateTimeFormat(i18n.language, { ...defaultOptions, ...options }).format(date)
  }
  
  const formatNumber = (number: number, options?: Intl.NumberFormatOptions) => {
    return new Intl.NumberFormat(i18n.language, options).format(number)
  }
  
  const formatCurrency = (amount: number, currency: string = 'USD', options?: Intl.NumberFormatOptions) => {
    const defaultOptions: Intl.NumberFormatOptions = {
      style: 'currency',
      currency,
    }
    return new Intl.NumberFormat(i18n.language, { ...defaultOptions, ...options }).format(amount)
  }
  
  const formatPercent = (value: number, options?: Intl.NumberFormatOptions) => {
    const defaultOptions: Intl.NumberFormatOptions = {
      style: 'percent',
      minimumFractionDigits: 1,
      maximumFractionDigits: 2,
    }
    return new Intl.NumberFormat(i18n.language, { ...defaultOptions, ...options }).format(value)
  }
  
  const formatHeartRate = (bpm: number) => {
    return `${formatNumber(bpm)} bpm`
  }
  
  const formatBloodPressure = (systolic: number, diastolic: number) => {
    return `${formatNumber(systolic)}/${formatNumber(diastolic)} mmHg`
  }
  
  const formatTemperature = (temp: number, unit: 'C' | 'F' = 'C') => {
    return `${formatNumber(temp, { minimumFractionDigits: 1, maximumFractionDigits: 1 })}°${unit}`
  }
  
  const formatWeight = (weight: number, unit: 'kg' | 'lb' = 'kg') => {
    return `${formatNumber(weight, { minimumFractionDigits: 1, maximumFractionDigits: 1 })} ${unit}`
  }
  
  const formatHeight = (height: number, unit: 'cm' | 'ft' = 'cm') => {
    if (unit === 'ft') {
      const feet = Math.floor(height / 30.48)
      const inches = Math.round((height % 30.48) / 2.54)
      return `${feet}'${inches}"`
    }
    return `${formatNumber(height)} ${unit}`
  }
  
  const formatMedicalValue = (value: number, unit: string, precision: number = 2) => {
    return `${formatNumber(value, { 
      minimumFractionDigits: precision, 
      maximumFractionDigits: precision 
    })} ${unit}`
  }
  
  const formatECGInterval = (milliseconds: number) => {
    return `${formatNumber(milliseconds)} ms`
  }
  
  const formatECGAmplitude = (microvolts: number) => {
    return `${formatNumber(microvolts, { minimumFractionDigits: 1, maximumFractionDigits: 1 })} μV`
  }
  
  const formatConfidenceScore = (score: number) => {
    return formatPercent(score / 100)
  }
  
  const formatRelativeTime = (date: Date) => {
    const rtf = new Intl.RelativeTimeFormat(i18n.language, { numeric: 'auto' })
    const now = new Date()
    const diffInSeconds = Math.floor((date.getTime() - now.getTime()) / 1000)
    
    if (Math.abs(diffInSeconds) < 60) {
      return rtf.format(diffInSeconds, 'second')
    } else if (Math.abs(diffInSeconds) < 3600) {
      return rtf.format(Math.floor(diffInSeconds / 60), 'minute')
    } else if (Math.abs(diffInSeconds) < 86400) {
      return rtf.format(Math.floor(diffInSeconds / 3600), 'hour')
    } else {
      return rtf.format(Math.floor(diffInSeconds / 86400), 'day')
    }
  }
  
  return {
    formatDate,
    formatTime,
    formatDateTime,
    formatNumber,
    formatCurrency,
    formatPercent,
    formatHeartRate,
    formatBloodPressure,
    formatTemperature,
    formatWeight,
    formatHeight,
    formatMedicalValue,
    formatECGInterval,
    formatECGAmplitude,
    formatConfidenceScore,
    formatRelativeTime,
  }
}

export const createFormatters = (locale: string) => {
  const formatDate = (date: Date, options?: Intl.DateTimeFormatOptions) => {
    const defaultOptions: Intl.DateTimeFormatOptions = {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    }
    return new Intl.DateTimeFormat(locale, { ...defaultOptions, ...options }).format(date)
  }
  
  const formatTime = (date: Date, options?: Intl.DateTimeFormatOptions) => {
    const defaultOptions: Intl.DateTimeFormatOptions = {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    }
    return new Intl.DateTimeFormat(locale, { ...defaultOptions, ...options }).format(date)
  }
  
  const formatDateTime = (date: Date, options?: Intl.DateTimeFormatOptions) => {
    const defaultOptions: Intl.DateTimeFormatOptions = {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    }
    return new Intl.DateTimeFormat(locale, { ...defaultOptions, ...options }).format(date)
  }
  
  const formatNumber = (number: number, options?: Intl.NumberFormatOptions) => {
    return new Intl.NumberFormat(locale, options).format(number)
  }
  
  const formatCurrency = (amount: number, currency: string = 'USD', options?: Intl.NumberFormatOptions) => {
    const defaultOptions: Intl.NumberFormatOptions = {
      style: 'currency',
      currency,
    }
    return new Intl.NumberFormat(locale, { ...defaultOptions, ...options }).format(amount)
  }
  
  const formatPercent = (value: number, options?: Intl.NumberFormatOptions) => {
    const defaultOptions: Intl.NumberFormatOptions = {
      style: 'percent',
      minimumFractionDigits: 1,
      maximumFractionDigits: 2,
    }
    return new Intl.NumberFormat(locale, { ...defaultOptions, ...options }).format(value)
  }
  
  return {
    formatDate,
    formatTime,
    formatDateTime,
    formatNumber,
    formatCurrency,
    formatPercent,
  }
}
